from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, NotRequired

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState, START, StateGraph

from orchestrator.llm.factory import get_chat_model
from orchestrator.llm_audit import get_llm_runnable_config
from orchestrator.models import OrchestratorPlan, RegisteredAgent
from orchestrator.tools import DEFAULT_TOOLS


class OrchestratorState(MessagesState):
    user_prompt: str
    attachment_context: str
    chat_history: list[dict[str, str]]
    model_name: NotRequired[str | None]
    plan: NotRequired[dict | None]
    tools: NotRequired[list[BaseTool]]
    agents: NotRequired[list[dict[str, Any]]]
    executor_plan_override: NotRequired[dict[str, Any] | None]
    executor_scratchpad: NotRequired[str]
    executor_step_info: NotRequired[tuple[int, int] | None]


def _resolve_tools(state: OrchestratorState) -> list[BaseTool]:
    t = state.get("tools")
    if t:
        return list(t)
    return list(DEFAULT_TOOLS)


def _history_lines(history: list[dict[str, str]]) -> str:
    if not history:
        return "(no prior messages)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def _tools_manifest(tools: Sequence[BaseTool]) -> str:
    lines = []
    for t in tools:
        desc = t.description or "(no description)"
        lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)


def _agents_manifest(agents: Sequence[RegisteredAgent]) -> str:
    lines = []
    for a in agents:
        desc = a.description or "(no description)"
        tools_part = ", ".join(a.tool_names)
        lines.append(f"- {a.id}: {desc} (tools: {tools_part})")
    return "\n".join(lines)


def resolve_model_name(model_name: str | None) -> str:
    from orchestrator.config import get_settings

    s = get_settings()
    if model_name and str(model_name).strip():
        return str(model_name).strip()
    return s.openai_model


def _resolve_model(state: OrchestratorState) -> str:
    return resolve_model_name(state.get("model_name"))


_agent_cache: dict[tuple[str, tuple[str, ...]], Any] = {}


def _get_agent(model: str, tools: Sequence[BaseTool]):
    sig = tuple(sorted(t.name for t in tools))
    key = (model, sig)
    cached = _agent_cache.get(key)
    if cached is None:
        llm = get_chat_model(model)
        cached = create_agent(llm, list(tools))
        _agent_cache[key] = cached
    return cached


def _planning_prompt(
    *,
    user_prompt: str,
    attachment_context: str,
    chat_history: list[dict[str, str]],
    tools: Sequence[BaseTool],
    agents: Sequence[RegisteredAgent] | None = None,
) -> str:
    tool_block = _tools_manifest(tools)
    agent_block = ""
    if agents:
        agent_block = f"""
Registered agents (set PlanStep.agent_id to run a step with that agent's tool subset during execution):
{_agents_manifest(agents)}

"""
    return f"""You are a planning component for an AI orchestrator.
Available tools (name and description):
{tool_block}
{agent_block}
Prior chat (compact):
{_history_lines(chat_history)}

Attachment / extra context:
{attachment_context or "(none)"}

User request:
{user_prompt}

Produce a concise plan. Reference tools by name when a step should use a specific tool.
Reference agent_id on a step when that step should be executed under a registered agent's tool subset.
"""


def _coerce_agents(raw: list[dict[str, Any]] | None) -> list[RegisteredAgent]:
    if not raw:
        return []
    return [RegisteredAgent.model_validate(x) for x in raw]


async def generate_plan(
    *,
    user_prompt: str,
    attachment_context: str = "",
    chat_history: list[dict[str, str]] | None = None,
    model_name: str | None = None,
    tools: Sequence[BaseTool] | None = None,
    agents: Sequence[RegisteredAgent] | None = None,
) -> OrchestratorPlan:
    """Structured plan only (no tool execution). Used by the graph plan node and by POST /orchestrate/plan."""
    history = chat_history or []
    resolved_tools = list(tools) if tools is not None else list(DEFAULT_TOOLS)
    agent_list = list(agents) if agents else []
    model = resolve_model_name(model_name)
    llm = get_chat_model(model)
    structured = llm.with_structured_output(OrchestratorPlan)
    prompt = _planning_prompt(
        user_prompt=user_prompt,
        attachment_context=attachment_context,
        chat_history=history,
        tools=resolved_tools,
        agents=agent_list or None,
    )
    return await structured.ainvoke(
        [HumanMessage(content=prompt)],
        config=get_llm_runnable_config("plan"),
    )


async def plan_node(state: OrchestratorState):
    tools = _resolve_tools(state)
    agent_models = _coerce_agents(state.get("agents"))
    plan = await generate_plan(
        user_prompt=state["user_prompt"],
        attachment_context=state.get("attachment_context") or "",
        chat_history=state["chat_history"],
        model_name=state.get("model_name"),
        tools=tools,
        agents=agent_models or None,
    )
    return {"plan": plan.model_dump(mode="json")}


def _build_executor_messages(state: OrchestratorState) -> list:
    plan_dict = state.get("executor_plan_override") or state.get("plan") or {}
    plan_text = json.dumps(plan_dict, indent=2)
    scratchpad = state.get("executor_scratchpad") or ""
    raw_agents = state.get("agents") or []
    agent_models = _coerce_agents(raw_agents) if raw_agents else []

    sections: list[str] = [
        "You are the execution agent. Follow the plan below; use tools when they help.",
    ]
    if agent_models:
        sections.append(
            "Registered agents (steps may reference agent_id):\n" + _agents_manifest(agent_models),
        )
    step_info = state.get("executor_step_info")
    if step_info:
        cur, total = step_info
        sections.append(f"You are executing step {cur} of {total}.")
    if scratchpad.strip():
        sections.append("Completed steps summary:\n" + scratchpad.strip())
    sections.append("## Plan\n" + plan_text)
    sections.append("## Attachment / context\n" + (state["attachment_context"] or "(none)"))
    sys_content = "\n\n".join(sections)

    out: list = [SystemMessage(content=sys_content)]
    for m in state["chat_history"]:
        role = str(m.get("role", "user")).lower().strip()
        content = str(m.get("content", ""))
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    out.append(HumanMessage(content=state["user_prompt"]))
    return out


def _plan_has_agent_steps(plan: dict[str, Any]) -> bool:
    for s in plan.get("steps") or []:
        if s.get("agent_id"):
            return True
    return False


def _resolve_step_tools(
    step: dict[str, Any],
    all_tools: list[BaseTool],
    agent_by_id: dict[str, RegisteredAgent],
) -> list[BaseTool]:
    aid = step.get("agent_id")
    name_to_tool = {t.name: t for t in all_tools}
    if not aid:
        return all_tools
    agent = agent_by_id.get(str(aid))
    if agent is None:
        msg = f"Unknown agent_id on plan step: {aid!r}"
        raise ValueError(msg)
    return [name_to_tool[n] for n in agent.tool_names]


def _step_scratchpad_update(messages: list[Any]) -> str:
    parts: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            c = m.content
            text = c if isinstance(c, str) else str(c)
            parts.append(f"tool: {text[:800]}")
        elif isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str) and c.strip():
                parts.append(f"assistant: {c[:800]}")
    if not parts:
        return ""
    return "\n".join(parts) + "\n---\n"


def _normalize_run_agents(
    agents: Sequence[RegisteredAgent] | Sequence[dict[str, Any]] | None,
) -> tuple[RegisteredAgent, ...]:
    if not agents:
        return ()
    out: list[RegisteredAgent] = []
    for a in agents:
        out.append(a if isinstance(a, RegisteredAgent) else RegisteredAgent.model_validate(a))
    return tuple(out)


async def run_executor(
    *,
    plan: dict[str, Any],
    user_prompt: str,
    attachment_context: str = "",
    chat_history: list[dict[str, str]] | None = None,
    model_name: str | None = None,
    tools: Sequence[BaseTool] | None = None,
    agents: Sequence[RegisteredAgent] | Sequence[dict[str, Any]] | None = None,
) -> list:
    """ReAct agent only (no planning). `plan` is the serialized orchestrator plan dict.

    When ``agents`` is non-empty and any step has ``agent_id``, runs one sub-invocation per plan
    step so each step can use a filtered tool set; otherwise a single invocation with optional
    agent manifest in the system prompt.
    """
    normalized_agents = _normalize_run_agents(agents)
    state_core: OrchestratorState = {
        "messages": [],
        "user_prompt": user_prompt,
        "attachment_context": attachment_context,
        "chat_history": chat_history or [],
        "model_name": model_name,
        "plan": plan,
        "agents": [a.model_dump(mode="json") for a in normalized_agents],
    }
    resolved_tools = list(tools) if tools is not None else list(DEFAULT_TOOLS)
    model = _resolve_model(state_core)
    steps = plan.get("steps") or []

    if any(s.get("agent_id") for s in steps) and not normalized_agents:
        msg = "Plan step references agent_id but no agents were registered on the app"
        raise ValueError(msg)

    use_sequential = bool(normalized_agents) and _plan_has_agent_steps(plan) and len(steps) > 0

    if not use_sequential:
        agent = _get_agent(model, resolved_tools)
        messages = _build_executor_messages(state_core)
        out = await agent.ainvoke({"messages": messages})
        return out["messages"]

    agent_by_id = {a.id: a for a in normalized_agents}
    all_messages: list = []
    scratchpad = ""
    for i, step in enumerate(steps):
        step_tools = _resolve_step_tools(step, resolved_tools, agent_by_id)
        mini_plan = {
            "goal_summary": plan.get("goal_summary", ""),
            "steps": [step],
            "final_output_description": plan.get("final_output_description", ""),
        }
        iter_state: OrchestratorState = {
            **state_core,
            "executor_plan_override": mini_plan,
            "executor_scratchpad": scratchpad,
            "executor_step_info": (i + 1, len(steps)),
        }
        sub = _get_agent(model, step_tools)
        sub_messages = _build_executor_messages(iter_state)
        out = await sub.ainvoke({"messages": sub_messages})
        batch = out["messages"]
        all_messages.extend(batch)
        scratchpad += _step_scratchpad_update(batch)
    return all_messages


async def execute_node(state: OrchestratorState):
    tools = _resolve_tools(state)
    agent_payload = _coerce_agents(state.get("agents"))
    messages = await run_executor(
        plan=state.get("plan") or {},
        user_prompt=state["user_prompt"],
        attachment_context=state.get("attachment_context") or "",
        chat_history=state["chat_history"],
        model_name=state.get("model_name"),
        tools=tools,
        agents=agent_payload or None,
    )
    return {"messages": messages}


def build_compiled_graph() -> Any:
    g = StateGraph(OrchestratorState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", END)
    return g.compile()


GRAPH = build_compiled_graph()


def last_assistant_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str) and c.strip():
                return c
            if isinstance(c, list):
                parts = []
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                text = "".join(parts).strip()
                if text:
                    return text
    return ""


def serialize_executor_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Turn LangChain messages from the executor into JSON-safe dicts for API responses."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, BaseMessage):
            out.append(m.model_dump(mode="json", exclude_none=True))
        else:
            out.append({"type": "unknown", "repr": repr(m)})
    return out
