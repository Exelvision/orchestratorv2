from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Annotated, TypeVar

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError
from starlette.responses import JSONResponse

from orchestrator.attachments import format_context_block, normalize_uploads
from orchestrator.config import Settings, get_settings
from orchestrator.flow_registry import DEFAULT_FLOWS, get_flow, list_flow_summaries
from orchestrator.graph import (
    GRAPH,
    generate_plan,
    last_assistant_text,
    run_executor,
    serialize_executor_messages,
)
from orchestrator.models import (
    ExecutePayload,
    ExecuteResponse,
    FlowSummary,
    NamedFlowExecutePayload,
    OrchestratePayload,
    OrchestrateResponse,
    OrchestratorPlan,
    ToolSummary,
)
from orchestrator.logging_setup import configure_logging, get_logger
from orchestrator.tools import DEFAULT_TOOLS

logger = get_logger(__name__)

TBody = TypeVar("TBody", bound=BaseModel)


def create_app(
    *,
    tools: Sequence[BaseTool] | None = None,
    flows: Mapping[str, tuple[str, str, OrchestratorPlan]] | None = None,
) -> FastAPI:
    """Build a FastAPI app with configurable LangChain tools and named-flow registry.

    Host projects can ``pip install`` this package and call ``create_app(tools=my_tools, flows=my_flows)``
    to serve orchestration with domain-specific tools and plans.
    """
    configure_logging()
    app = FastAPI(title="Orchestrator", version="0.1.0")
    app.state.tools = list(tools) if tools is not None else list(DEFAULT_TOOLS)
    app.state.flow_registry = dict(flows) if flows is not None else dict(DEFAULT_FLOWS)

    @app.middleware("http")
    async def log_unhandled_exceptions(request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException:
            raise
        except RequestValidationError:
            raise
        except Exception:
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    def _check_multipart_content_length(request: Request, settings: Settings) -> None:
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > settings.max_total_request_bytes:
            raise HTTPException(413, detail="Request body too large")

    def _model_from_json_str(raw: str, model: type[TBody]) -> TBody:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(400, detail=f"Invalid JSON payload: {e}") from e
        try:
            return model.model_validate(data)
        except ValidationError as e:
            raise HTTPException(400, detail=e.errors()) from e

    async def _json_or_multipart_body(
        request: Request,
        settings: Settings,
        model: type[TBody],
    ) -> tuple[TBody, list[UploadFile]]:
        ct = (request.headers.get("content-type") or "").lower()
        if "multipart/form-data" in ct:
            _check_multipart_content_length(request, settings)
            form = await request.form()
            raw_payload = form.get("payload")
            if raw_payload is None:
                raise HTTPException(400, detail="Missing form field 'payload'")
            if not isinstance(raw_payload, str):
                raise HTTPException(400, detail="Form field 'payload' must be a JSON string")
            body = _model_from_json_str(raw_payload, model)
            files: list[UploadFile] = []
            for item in form.getlist("files"):
                # Starlette form parsing yields starlette.datastructures.UploadFile, not fastapi.UploadFile
                if isinstance(item, StarletteUploadFile):
                    files.append(item)
            return body, files
        if "application/x-www-form-urlencoded" in ct:
            _check_multipart_content_length(request, settings)
            form = await request.form()
            raw_payload = form.get("payload")
            if raw_payload is None:
                raise HTTPException(400, detail="Missing form field 'payload'")
            if not isinstance(raw_payload, str):
                raise HTTPException(400, detail="Form field 'payload' must be a JSON string")
            return _model_from_json_str(raw_payload, model), []
        try:
            data = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(400, detail=f"Invalid JSON body: {e}") from e
        try:
            return model.model_validate(data), []
        except ValidationError as e:
            raise HTTPException(422, detail=e.errors()) from e

    async def _attachment_context_from_parts(
        files: list[UploadFile],
        settings: Settings,
        context: dict | None,
        metadata: dict | None,
    ) -> str:
        upload_block, _ = await normalize_uploads(files, settings)
        return format_context_block(upload_block, context, metadata)

    async def _attachment_context_for_payload(
        body: OrchestratePayload,
        files: list[UploadFile],
        settings: Settings,
    ) -> str:
        return await _attachment_context_from_parts(
            files, settings, body.context, body.metadata
        )

    async def _run_orchestration(
        body: OrchestratePayload,
        files: list[UploadFile],
        settings: Settings,
        request: Request,
    ):
        attachment_context = await _attachment_context_for_payload(body, files, settings)
        state = {
            "messages": [],
            "user_prompt": body.user_prompt,
            "attachment_context": attachment_context,
            "chat_history": [m.model_dump() for m in body.chat_history],
            "model_name": body.model,
            "tools": request.app.state.tools,
        }
        out = await GRAPH.ainvoke(state)
        plan = OrchestratorPlan.model_validate(out["plan"])
        exec_messages = out.get("messages", [])
        answer = last_assistant_text(exec_messages)
        return OrchestrateResponse(
            plan=plan,
            answer=answer,
            messages=serialize_executor_messages(exec_messages),
        )

    @app.post("/orchestrate/execute", response_model=ExecuteResponse)
    async def orchestrate_execute_only(
        request: Request,
        settings: Annotated[Settings, Depends(get_settings)],
    ):
        """Run only the ReAct executor with a supplied plan (no planner call).

        JSON body (`ExecutePayload`) or multipart form: `payload` (JSON string, same shape) and optional `files`.
        """
        body, files = await _json_or_multipart_body(request, settings, ExecutePayload)
        attachment_context = await _attachment_context_from_parts(
            files, settings, body.context, body.metadata
        )
        messages = await run_executor(
            plan=body.plan.model_dump(mode="json"),
            user_prompt=body.user_prompt,
            attachment_context=attachment_context,
            chat_history=[m.model_dump() for m in body.chat_history],
            model_name=body.model,
            tools=request.app.state.tools,
        )
        return ExecuteResponse(
            answer=last_assistant_text(messages),
            messages=serialize_executor_messages(messages),
        )

    @app.get("/orchestrate/tools", response_model=list[ToolSummary])
    def list_orchestrate_tools(request: Request):
        """List tools registered on this app (same set used by plan and execute)."""
        return [
            ToolSummary(name=t.name, description=(t.description or "").strip())
            for t in request.app.state.tools
        ]

    @app.get("/orchestrate/flows", response_model=list[FlowSummary])
    def list_orchestrate_flows(request: Request):
        """List ids and metadata for server-defined plans invokable via POST /orchestrate/flows/{flow_id}."""
        return list_flow_summaries(request.app.state.flow_registry)

    @app.post("/orchestrate/flows/{flow_id}", response_model=ExecuteResponse)
    async def orchestrate_named_flow(
        flow_id: str,
        request: Request,
        settings: Annotated[Settings, Depends(get_settings)],
    ):
        """Run the ReAct executor with a pre-configured plan looked up by `flow_id`.

        JSON body (`NamedFlowExecutePayload`) or multipart: `payload` (JSON string, same shape) and optional `files`.
        """
        plan = get_flow(flow_id, request.app.state.flow_registry)
        if plan is None:
            raise HTTPException(404, detail=f"Unknown flow_id: {flow_id}")
        body, files = await _json_or_multipart_body(request, settings, NamedFlowExecutePayload)
        attachment_context = await _attachment_context_from_parts(
            files, settings, body.context, body.metadata
        )
        messages = await run_executor(
            plan=plan.model_dump(mode="json"),
            user_prompt=body.user_prompt,
            attachment_context=attachment_context,
            chat_history=[m.model_dump() for m in body.chat_history],
            model_name=body.model,
            tools=request.app.state.tools,
        )
        return ExecuteResponse(
            answer=last_assistant_text(messages),
            messages=serialize_executor_messages(messages),
        )

    @app.post("/orchestrate/plan", response_model=OrchestratorPlan)
    async def orchestrate_plan_only(
        request: Request,
        settings: Annotated[Settings, Depends(get_settings)],
    ):
        """Run only the planning LLM (structured `OrchestratorPlan`); does not invoke the ReAct executor.

        JSON body (`OrchestratePayload`) or multipart: `payload` (JSON string, same shape) and optional `files`.
        """
        body, files = await _json_or_multipart_body(request, settings, OrchestratePayload)
        attachment_context = await _attachment_context_for_payload(body, files, settings)
        return await generate_plan(
            user_prompt=body.user_prompt,
            attachment_context=attachment_context,
            chat_history=[m.model_dump() for m in body.chat_history],
            model_name=body.model,
            tools=request.app.state.tools,
        )

    @app.post("/orchestrate", response_model=OrchestrateResponse)
    @app.post("/orchestrate/json", response_model=OrchestrateResponse)
    async def orchestrate(
        request: Request,
        settings: Annotated[Settings, Depends(get_settings)],
    ):
        """Full graph: plan then execute. JSON (`OrchestratePayload`), multipart, or urlencoded `payload`; optional `files` on multipart.

        ``/orchestrate/json`` is the same handler (backward-compatible alias).
        """
        body, files = await _json_or_multipart_body(request, settings, OrchestratePayload)
        return await _run_orchestration(body, files, settings, request)

    return app


app = create_app()


def run():
    import uvicorn

    uvicorn.run("orchestrator.main:app", host="0.0.0.0", port=8000, reload=False)
