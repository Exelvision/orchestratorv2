from typing import Any

from pydantic import BaseModel, Field


class ChatMessageItem(BaseModel):
    role: str = Field(..., description="user | assistant | system")
    content: str


class OrchestratePayload(BaseModel):
    user_prompt: str
    chat_history: list[ChatMessageItem] = Field(default_factory=list)
    model: str | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class PlanStep(BaseModel):
    step_id: str
    description: str
    tool_name: str | None = Field(None, description="Tool to use for this step, if any")
    agent_id: str | None = Field(
        None,
        description="Optional registered agent id; when set, executor may restrict tools to that agent's subset",
    )
    inputs: str = Field("", description="Inputs this step needs")
    expected_output: str = Field("", description="What this step should produce")


class OrchestratorPlan(BaseModel):
    goal_summary: str
    steps: list[PlanStep] = Field(default_factory=list)
    final_output_description: str = Field(
        "",
        description="What the overall request should deliver to the user",
    )


class OrchestrateResponse(BaseModel):
    plan: OrchestratorPlan
    answer: str = Field("", description="Final assistant text from the executor agent")
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full executor message trace (LangChain messages as JSON-serializable dicts)",
    )
    artifact: dict[str, Any] | None = Field(
        None,
        description="Structured payload from commit_structured_output tool, if present (prefer over answer for machine-grounded data)",
    )


class FlowSummary(BaseModel):
    flow_id: str
    title: str = ""
    description: str = ""


class ToolSummary(BaseModel):
    """Registered LangChain tool as exposed to the planner and executor."""

    name: str
    description: str = ""


class RegisteredAgent(BaseModel):
    """Named agent: a tool subset referenced by ``PlanStep.agent_id``."""

    id: str = Field(..., description="Unique id; must match PlanStep.agent_id when used")
    description: str = ""
    tool_names: list[str] = Field(
        ...,
        description="Names of tools this agent may use (must be a subset of create_app tools)",
    )


class AgentSummary(BaseModel):
    """Registered agent as exposed to the planner (GET /orchestrate/agents)."""

    id: str
    description: str = ""
    tool_names: list[str] = Field(default_factory=list)


class NamedFlowExecutePayload(BaseModel):
    """Body for named pre-defined plan execution (plan is resolved server-side by flow_id)."""

    user_prompt: str
    chat_history: list[ChatMessageItem] = Field(default_factory=list)
    model: str | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ExecutePayload(BaseModel):
    """Body for executor-only: supply a plan (e.g. from /orchestrate/plan) plus the user turn and history.

    Plans may be produced entirely outside this service (e.g. another repo). Tool names in the plan must
    match tools registered on the server (``create_app(tools=...)``); otherwise the executor cannot run those steps.
    If a step sets ``agent_id``, the server must have registered that agent via ``create_app(agents=...)``;
    execution then uses per-step tool filtering when multiple steps reference agents.
    """

    plan: OrchestratorPlan = Field(
        ...,
        description="Orchestrator plan; may originate from any client or service if tool names match the configured tool set.",
    )
    user_prompt: str
    chat_history: list[ChatMessageItem] = Field(default_factory=list)
    model: str | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ExecuteResponse(BaseModel):
    answer: str = Field("", description="Final assistant text from the ReAct executor")
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full executor message trace (LangChain messages as JSON-serializable dicts)",
    )
    artifact: dict[str, Any] | None = Field(
        None,
        description="Structured payload from commit_structured_output tool, if present",
    )
