from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain.tools import tool

from orchestrator.main import create_app
from orchestrator.models import OrchestratorPlan, PlanStep, RegisteredAgent
from orchestrator.tools import DEFAULT_TOOLS


def test_create_app_rejects_agent_unknown_tool():
    with pytest.raises(ValueError, match="unknown tool_name"):
        create_app(
            agents=[
                RegisteredAgent(
                    id="bad",
                    description="d",
                    tool_names=["__not_a_real_tool__"],
                ),
            ],
        )


def test_create_app_rejects_duplicate_agent_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        create_app(
            agents=[
                RegisteredAgent(id="dup", description="a", tool_names=["echo_text"]),
                RegisteredAgent(id="dup", description="b", tool_names=["word_count"]),
            ],
        )


def test_create_app_rejects_empty_tool_names_on_agent():
    with pytest.raises(ValueError, match="at least one tool_name"):
        create_app(
            agents=[
                RegisteredAgent(id="empty", description="d", tool_names=[]),
            ],
        )


def test_get_orchestrate_agents():
    application = create_app(
        agents=[
            RegisteredAgent(
                id="metrics_only",
                description="Text metrics",
                tool_names=["text_metrics"],
            ),
        ],
    )
    with TestClient(application) as tc:
        r = tc.get("/orchestrate/agents")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["id"] == "metrics_only"
    assert data[0]["tool_names"] == ["text_metrics"]


@pytest.mark.asyncio
async def test_run_executor_sequential_uses_filtered_tools(monkeypatch: pytest.MonkeyPatch):
    from langchain_core.messages import AIMessage

    from orchestrator.graph import run_executor

    captured: list[list[str]] = []

    def fake_get_agent(model, tools):
        captured.append([t.name for t in tools])

        class MockAgent:
            async def ainvoke(self, input):
                return {"messages": [AIMessage(content="step done")]}

        return MockAgent()

    monkeypatch.setattr("orchestrator.graph._get_agent", fake_get_agent)

    plan = {
        "goal_summary": "g",
        "steps": [
            {
                "step_id": "1",
                "description": "first",
                "tool_name": "echo_text",
                "agent_id": "agent_a",
                "inputs": "",
                "expected_output": "",
            },
            {
                "step_id": "2",
                "description": "second",
                "tool_name": "word_count",
                "agent_id": "agent_b",
                "inputs": "",
                "expected_output": "",
            },
        ],
        "final_output_description": "f",
    }
    agents = (
        RegisteredAgent(id="agent_a", description="", tool_names=["echo_text"]),
        RegisteredAgent(id="agent_b", description="", tool_names=["word_count"]),
    )
    await run_executor(
        plan=plan,
        user_prompt="hi",
        tools=DEFAULT_TOOLS,
        agents=agents,
    )
    assert captured == [["echo_text"], ["word_count"]]


def test_execute_rejects_plan_with_agent_id_without_registry(client: TestClient):
    r = client.post(
        "/orchestrate/execute",
        json={
            "plan": {
                "goal_summary": "g",
                "steps": [
                    {
                        "step_id": "1",
                        "description": "d",
                        "agent_id": "missing_agent",
                        "inputs": "",
                        "expected_output": "",
                    },
                ],
                "final_output_description": "f",
            },
            "user_prompt": "x",
        },
    )
    assert r.status_code == 400
    assert "agent_id" in r.json()["detail"].lower() or "registered" in r.json()["detail"].lower()
