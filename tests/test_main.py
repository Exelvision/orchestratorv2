from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from orchestrator.main import app
from orchestrator.tools import DEFAULT_TOOLS


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_create_app_replaces_flow_registry():
    from fastapi.testclient import TestClient

    from orchestrator.main import create_app
    from orchestrator.models import OrchestratorPlan

    custom = {
        "custom_only": (
            "Custom",
            "Desc",
            OrchestratorPlan(
                goal_summary="cg",
                steps=[],
                final_output_description="cf",
            ),
        ),
    }
    application = create_app(flows=custom)
    with TestClient(application) as tc:
        r = tc.get("/orchestrate/flows")
        assert r.status_code == 200
        assert [item["flow_id"] for item in r.json()] == ["custom_only"]


def test_create_app_custom_tools_propagate_to_execute(monkeypatch: pytest.MonkeyPatch):
    from langchain.tools import tool
    from langchain_core.messages import AIMessage

    from fastapi.testclient import TestClient

    from orchestrator.main import create_app

    @tool
    def only_external_tool(q: str) -> str:
        """Sole tool for this app."""
        return q

    captured: dict[str, object] = {}

    async def fake_run_executor(**kwargs: object) -> list:
        captured["tool_names"] = [t.name for t in kwargs["tools"]]
        return [AIMessage(content="done")]

    monkeypatch.setattr("orchestrator.main.run_executor", fake_run_executor)
    application = create_app(tools=[only_external_tool])
    with TestClient(application) as tc:
        r = tc.post(
            "/orchestrate/execute",
            json={
                "plan": {
                    "goal_summary": "g",
                    "steps": [],
                    "final_output_description": "f",
                },
                "user_prompt": "hi",
            },
        )
    assert r.status_code == 200
    assert captured["tool_names"] == ["only_external_tool"]


def test_create_app_custom_tools_propagate_to_plan(monkeypatch: pytest.MonkeyPatch):
    from langchain.tools import tool

    from fastapi.testclient import TestClient

    from orchestrator.main import create_app
    from orchestrator.models import OrchestratorPlan

    @tool
    def plan_external_tool(q: str) -> str:
        """Tool visible to planner."""
        return q

    captured: dict[str, object] = {}

    async def fake_generate_plan(**kwargs: object) -> OrchestratorPlan:
        captured["tool_names"] = [t.name for t in kwargs["tools"]]
        return OrchestratorPlan(
            goal_summary="pg",
            steps=[],
            final_output_description="pf",
        )

    monkeypatch.setattr("orchestrator.main.generate_plan", fake_generate_plan)
    application = create_app(tools=[plan_external_tool])
    with TestClient(application) as tc:
        r = tc.post("/orchestrate/plan", json={"user_prompt": "plan this"})
    assert r.status_code == 200
    assert r.json()["goal_summary"] == "pg"
    assert captured["tool_names"] == ["plan_external_tool"]


def test_create_app_custom_tools_in_graph_state_for_json_route(monkeypatch: pytest.MonkeyPatch):
    from unittest.mock import AsyncMock

    from langchain.tools import tool
    from langchain_core.messages import AIMessage

    from fastapi.testclient import TestClient

    from orchestrator.main import create_app

    @tool
    def json_route_tool(x: str) -> str:
        """Injected for JSON orchestrate path."""
        return x

    graph = AsyncMock()
    graph.ainvoke = AsyncMock(
        return_value={
            "plan": {
                "goal_summary": "G",
                "steps": [],
                "final_output_description": "F",
            },
            "messages": [AIMessage(content="from mock graph")],
        }
    )
    monkeypatch.setattr("orchestrator.main.GRAPH", graph)

    application = create_app(tools=[json_route_tool])
    with TestClient(application) as tc:
        r = tc.post("/orchestrate/json", json={"user_prompt": "symptoms"})
    assert r.status_code == 200
    state = graph.ainvoke.call_args[0][0]
    assert [t.name for t in state["tools"]] == ["json_route_tool"]


def test_orchestrate_json_success(client: TestClient, mock_graph, fake_graph_output):
    body = {
        "user_prompt": "Hello",
        "chat_history": [{"role": "user", "content": "Hi"}],
        "context": {"k": "v"},
    }
    r = client.post("/orchestrate", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["plan"]["goal_summary"] == fake_graph_output["plan"]["goal_summary"]
    assert data["answer"] == "Here is the final answer."
    assert isinstance(data["messages"], list)
    assert len(data["messages"]) == 1
    assert data["messages"][0]["type"] == "ai"
    assert data["messages"][0]["content"] == "Here is the final answer."
    mock_graph.ainvoke.assert_awaited_once()
    call_state = mock_graph.ainvoke.call_args[0][0]
    assert call_state["user_prompt"] == "Hello"
    assert call_state["chat_history"] == [{"role": "user", "content": "Hi"}]
    assert "Client context" in call_state["attachment_context"]
    tool_state = call_state.get("tools")
    assert tool_state is not None
    assert [t.name for t in tool_state] == [t.name for t in DEFAULT_TOOLS]


def test_orchestrate_multipart_includes_tools_in_graph_state(client: TestClient, mock_graph):
    payload = json.dumps({"user_prompt": "multipart tools"})
    r = client.post("/orchestrate", data={"payload": payload})
    assert r.status_code == 200
    call_state = mock_graph.ainvoke.call_args[0][0]
    assert [t.name for t in call_state["tools"]] == [t.name for t in DEFAULT_TOOLS]


def test_orchestrate_plan_only(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from orchestrator.models import OrchestratorPlan

    async def fake_generate_plan(**kwargs: object) -> OrchestratorPlan:
        assert kwargs["user_prompt"] == "Plan me"
        assert "Client context" in kwargs["attachment_context"]
        tools = kwargs.get("tools")
        assert tools is not None
        assert [t.name for t in tools] == [t.name for t in DEFAULT_TOOLS]
        return OrchestratorPlan(
            goal_summary="mock goal",
            steps=[],
            final_output_description="mock final",
        )

    monkeypatch.setattr("orchestrator.main.generate_plan", fake_generate_plan)
    r = client.post(
        "/orchestrate/plan",
        json={"user_prompt": "Plan me", "context": {"tenant": "t1"}},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["goal_summary"] == "mock goal"
    assert data["final_output_description"] == "mock final"


def test_orchestrate_execute_only(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from langchain_core.messages import AIMessage

    async def fake_run_executor(**kwargs: object) -> list:
        assert kwargs["user_prompt"] == "Run it"
        assert kwargs["plan"]["goal_summary"] == "g"
        assert "Client context" in kwargs["attachment_context"]
        tools = kwargs.get("tools")
        assert tools is not None
        assert [t.name for t in tools] == [t.name for t in DEFAULT_TOOLS]
        return [AIMessage(content="executor done")]

    monkeypatch.setattr("orchestrator.main.run_executor", fake_run_executor)
    r = client.post(
        "/orchestrate/execute",
        json={
            "plan": {
                "goal_summary": "g",
                "steps": [],
                "final_output_description": "f",
            },
            "user_prompt": "Run it",
            "context": {"x": 1},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "executor done"
    assert len(body["messages"]) == 1
    assert body["messages"][0]["type"] == "ai"
    assert body["messages"][0]["content"] == "executor done"


def test_orchestrate_execute_only_validation_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    called: list[bool] = []

    async def fake_run_executor(**kwargs: object):
        called.append(True)
        raise AssertionError("should not be called")

    monkeypatch.setattr("orchestrator.main.run_executor", fake_run_executor)
    r = client.post("/orchestrate/execute", json={"user_prompt": "missing plan"})
    assert r.status_code == 422
    assert not called


def test_orchestrate_plan_only_validation_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    called: list[bool] = []

    async def fake_generate_plan(**kwargs: object):
        called.append(True)
        raise AssertionError("should not be called")

    monkeypatch.setattr("orchestrator.main.generate_plan", fake_generate_plan)
    r = client.post("/orchestrate/plan", json={})
    assert r.status_code == 422
    assert not called


def test_orchestrate_json_validation_error(client: TestClient, mock_graph):
    r = client.post("/orchestrate", json={})
    assert r.status_code == 422
    mock_graph.ainvoke.assert_not_awaited()


def test_orchestrate_json_legacy_path_alias(client: TestClient, mock_graph):
    """``/orchestrate/json`` is the same handler as ``/orchestrate``."""
    r = client.post("/orchestrate/json", json={"user_prompt": "alias path"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Here is the final answer."


def test_orchestrate_multipart_success(client: TestClient, mock_graph):
    payload = json.dumps({"user_prompt": "From multipart"})
    r = client.post(
        "/orchestrate",
        data={"payload": payload},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "Here is the final answer."
    assert data["messages"][0]["content"] == "Here is the final answer."


def test_orchestrate_multipart_invalid_json(client: TestClient, mock_graph):
    r = client.post("/orchestrate", data={"payload": "not-json{"})
    assert r.status_code == 400
    mock_graph.ainvoke.assert_not_awaited()


def test_orchestrate_multipart_invalid_payload_shape(client: TestClient, mock_graph):
    r = client.post("/orchestrate", data={"payload": json.dumps({"chat_history": []})})
    assert r.status_code == 400
    mock_graph.ainvoke.assert_not_awaited()


def test_orchestrate_multipart_content_length_too_large(client: TestClient, mock_graph):
    from orchestrator.config import Settings, get_settings

    app.dependency_overrides[get_settings] = lambda: Settings(max_total_request_bytes=100)
    try:
        r = client.post(
            "/orchestrate",
            data={"payload": json.dumps({"user_prompt": "x"})},
            headers={"content-length": "999999"},
        )
        assert r.status_code == 413
        mock_graph.ainvoke.assert_not_awaited()
    finally:
        app.dependency_overrides.clear()


def test_list_orchestrate_tools(client: TestClient):
    r = client.get("/orchestrate/tools")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    names = {item["name"] for item in data}
    assert "echo_text" in names
    assert "word_count" in names
    for item in data:
        assert "name" in item and "description" in item


def test_list_orchestrate_tools_matches_create_app_tools():
    from langchain.tools import tool

    from fastapi.testclient import TestClient

    from orchestrator.main import create_app

    @tool
    def catalog_only_tool(x: str) -> str:
        """Only tool on this app instance."""
        return x

    application = create_app(tools=[catalog_only_tool])
    with TestClient(application) as tc:
        r = tc.get("/orchestrate/tools")
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert r.json()[0]["name"] == "catalog_only_tool"
    assert "Only tool" in r.json()[0]["description"]


def test_list_orchestrate_flows(client: TestClient):
    r = client.get("/orchestrate/flows")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    ids = [item["flow_id"] for item in data]
    assert ids == sorted(ids)
    assert "echo_smoke" in ids
    assert "word_stats" in ids
    assert "agent_metrics" in ids
    assert "agent_reverse" in ids
    assert "agent_bullets" in ids
    for item in data:
        assert "title" in item and "description" in item


def test_orchestrate_named_flow_unknown(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    async def fake_run_executor(**kwargs: object):
        raise AssertionError("should not be called")

    monkeypatch.setattr("orchestrator.main.run_executor", fake_run_executor)
    r = client.post(
        "/orchestrate/flows/does_not_exist",
        json={"user_prompt": "hello"},
    )
    assert r.status_code == 404


def test_orchestrate_named_flow_success(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from langchain_core.messages import AIMessage

    captured: dict[str, object] = {}

    async def fake_run_executor(**kwargs: object) -> list:
        captured.update(kwargs)
        tools = kwargs.get("tools")
        assert tools is not None
        assert [t.name for t in tools] == [t.name for t in DEFAULT_TOOLS]
        return [AIMessage(content="named flow done")]

    monkeypatch.setattr("orchestrator.main.run_executor", fake_run_executor)
    r = client.post(
        "/orchestrate/flows/word_stats",
        json={
            "user_prompt": "Count these words",
            "chat_history": [{"role": "user", "content": "prior"}],
            "context": {"c": 1},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "named flow done"
    assert body["messages"][0]["content"] == "named flow done"
    assert captured["user_prompt"] == "Count these words"
    assert captured["chat_history"] == [{"role": "user", "content": "prior"}]
    plan = captured["plan"]
    assert isinstance(plan, dict)
    assert plan["goal_summary"] == "Report how many words are in the user's text"
    assert "Client context" in str(captured.get("attachment_context", ""))


def test_orchestrate_named_flow_multipart_file_in_attachment_context(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    from langchain_core.messages import AIMessage

    captured: dict[str, str] = {}

    async def fake_run_executor(**kwargs: object) -> list:
        captured["attachment_context"] = str(kwargs.get("attachment_context", ""))
        return [AIMessage(content="ok")]

    monkeypatch.setattr("orchestrator.main.run_executor", fake_run_executor)
    r = client.post(
        "/orchestrate/flows/word_stats",
        data={"payload": json.dumps({"user_prompt": "count words"})},
        files=[("files", ("snippet.txt", b"one two three", "text/plain"))],
    )
    assert r.status_code == 200
    ac = captured["attachment_context"]
    assert "Uploaded files" in ac
    assert "snippet.txt" in ac
    assert "one two three" in ac


def test_orchestrate_execute_multipart_file_in_attachment_context(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    from langchain_core.messages import AIMessage

    captured: dict[str, str] = {}

    async def fake_run_executor(**kwargs: object) -> list:
        captured["attachment_context"] = str(kwargs.get("attachment_context", ""))
        return [AIMessage(content="done")]

    monkeypatch.setattr("orchestrator.main.run_executor", fake_run_executor)
    r = client.post(
        "/orchestrate/execute",
        data={
            "payload": json.dumps(
                {
                    "plan": {
                        "goal_summary": "g",
                        "steps": [],
                        "final_output_description": "f",
                    },
                    "user_prompt": "run",
                }
            )
        },
        files=[("files", ("doc.txt", b"body text", "text/plain"))],
    )
    assert r.status_code == 200
    ac = captured["attachment_context"]
    assert "Uploaded files" in ac
    assert "doc.txt" in ac
    assert "body text" in ac


def test_orchestrate_plan_multipart_file_in_attachment_context(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    from orchestrator.models import OrchestratorPlan

    captured: dict[str, str] = {}

    async def fake_generate_plan(**kwargs: object) -> OrchestratorPlan:
        captured["attachment_context"] = str(kwargs.get("attachment_context", ""))
        return OrchestratorPlan(
            goal_summary="g",
            steps=[],
            final_output_description="f",
        )

    monkeypatch.setattr("orchestrator.main.generate_plan", fake_generate_plan)
    r = client.post(
        "/orchestrate/plan",
        data={"payload": json.dumps({"user_prompt": "plan with file"})},
        files=[("files", ("notes.txt", b"line", "text/plain"))],
    )
    assert r.status_code == 200
    ac = captured["attachment_context"]
    assert "Uploaded files" in ac
    assert "notes.txt" in ac
    assert "line" in ac


def test_orchestrate_json_multipart_file_in_graph_state(client: TestClient, mock_graph):
    r = client.post(
        "/orchestrate",
        data={"payload": json.dumps({"user_prompt": "with file"})},
        files=[("files", ("data.txt", b"payload-bytes", "text/plain"))],
    )
    assert r.status_code == 200
    state = mock_graph.ainvoke.call_args[0][0]
    ac = state["attachment_context"]
    assert "Uploaded files" in ac
    assert "data.txt" in ac
    assert "payload-bytes" in ac
