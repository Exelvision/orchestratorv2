# Orchestrator v3

FastAPI service that runs a **LangGraph** pipeline: structured **plan** (Pydantic via `with_structured_output`) then **`create_react_agent`** execution with LangChain `@tool` definitions.

## Setup (use `.venv` only)

```bash
cd /path/to/orchestrator
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`). You can use a `.env` file in the project root.

## Run

```bash
source .venv/bin/activate
uvicorn orchestrator.main:app --reload --host 0.0.0.0 --port 8000
```

## Registering tools and flows

Host applications pass tools and named flows into [`create_app`](orchestrator/main.py). That factory stores them on `app.state`; planning, execution, and `GET /orchestrate/tools` / `GET /orchestrate/flows` all use the same registry.

### Tools

1. Define LangChain tools with the `@tool` decorator from `langchain.tools` (same pattern as this repo under [`orchestrator/tools/`](orchestrator/tools/)). The function **docstring** is exposed to the planner as the tool description; the tool’s **name** is usually the function name (for example `my_search`).
2. Pass a sequence of tool objects to `create_app(tools=[...])`. If you omit `tools` or pass `None`, the app uses the package default list [`DEFAULT_TOOLS`](orchestrator/tools/__init__.py) (stub tools plus sample agents).
3. **Planner and executor** only know about the tools you register. Any `tool_name` in an [`OrchestratorPlan`](orchestrator/models.py) / [`PlanStep`](orchestrator/models.py) must match a registered tool name, or the ReAct agent cannot run that step.

### Flows (named plans)

1. A **flow** is a server-side id mapped to metadata plus a fixed [`OrchestratorPlan`](orchestrator/models.py):  
   `flow_id -> (title, description, plan)`.
2. `title` and `description` appear in `GET /orchestrate/flows`. The `plan` is the same structured shape the planner would emit: `goal_summary`, `steps` (each [`PlanStep`](orchestrator/models.py) with `step_id`, `description`, optional `tool_name`, `inputs`, `expected_output`), and `final_output_description`.
3. Pass a mapping to `create_app(flows={...})`. If you omit `flows` or pass `None`, the app uses [`DEFAULT_FLOWS`](orchestrator/flow_registry.py). Passing your own dict **replaces** the default registry entirely; to keep built-in flows and add yours, merge explicitly, for example `{**DEFAULT_FLOWS, **my_flows}`.
4. Clients run a named flow with `POST /orchestrate/flows/{flow_id}` (JSON body, or form `payload` + optional file uploads — see [API](#api)). The server loads the plan by id and runs the executor with your registered tools.

### Example: custom app module

```python
# myapp.py — run with: uvicorn myapp:app --host 0.0.0.0 --port 8000
from langchain.tools import tool

from orchestrator.flow_registry import DEFAULT_FLOWS
from orchestrator.main import create_app
from orchestrator.models import OrchestratorPlan, PlanStep
from orchestrator.tools import DEFAULT_TOOLS


@tool
def shout(text: str) -> str:
    """Return the input in uppercase for emphasis."""
    return text.upper()


my_flows = {
    "shout_once": (
        "Shout",
        "Uppercase the user’s message using shout.",
        OrchestratorPlan(
            goal_summary="Deliver the user text in uppercase",
            steps=[
                PlanStep(
                    step_id="1",
                    description="Call shout with the user’s prompt",
                    tool_name="shout",
                    inputs="User message text",
                    expected_output="Uppercased string",
                ),
            ],
            final_output_description="The shouted text and a brief confirmation",
        ),
    ),
}

app = create_app(tools=[*DEFAULT_TOOLS, shout], flows={**DEFAULT_FLOWS, **my_flows})
```

If you replace `DEFAULT_FLOWS` entirely with plans that only use your own tools, you can pass `tools=[shout]` and `flows=my_flows` alone.

## API

**Request bodies (orchestration routes):** Each `POST` below accepts either:

- **`application/json`** — the documented JSON object for that route, or
- **`multipart/form-data`** — form field `payload` (a JSON string with the **same** shape as the JSON body) and optional repeated `files` parts for uploads; or
- **`application/x-www-form-urlencoded`** — field `payload` only (JSON string); file uploads are not available on this encoding.

File uploads are always optional. Text-like files are inlined into the planner/executor context; other types are noted as non-text (see [`orchestrator/attachments.py`](orchestrator/attachments.py)).

- `GET /health`
- `GET /orchestrate/tools` — registered tool names and descriptions (matches `create_app(tools=...)`)
- `GET /orchestrate/flows` — named flow ids and metadata (matches `create_app(flows=...)`)
- `POST /orchestrate/flows/{flow_id}` — run the executor with the server-registered plan for that id (`NamedFlowExecutePayload`: `user_prompt`, `chat_history`, optional `model` / `context` / `metadata`)
- `POST /orchestrate/plan` — same payload shape as `POST /orchestrate`, but returns only the structured **plan** (no executor)
- `POST /orchestrate/execute` — body includes an explicit **`plan`** plus `user_prompt` / `chat_history` / optional `context` / `metadata`; runs only the ReAct **executor**
- `POST /orchestrate` — full graph: plan then execute (`OrchestratePayload`). Same handler is also mounted at **`POST /orchestrate/json`** for backward compatibility.

**Orchestration responses** (`/orchestrate`, `/orchestrate/execute`, `/orchestrate/flows/{flow_id}`) include `answer` (final assistant text) and `messages`: a JSON array of LangChain message objects (`type`, `content`, `tool_calls`, `tool_call_id`, etc.) so clients can inspect tool rounds and multimodal content, not only the flattened string.

OpenAPI: `http://localhost:8000/docs`

## Tests

```bash
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Build and use in another project

### Build distributable artifacts

From this repo root, install the standard build tool once (any environment), then produce a wheel and source distribution:

```bash
pip install build
python -m build
```

Outputs land in `dist/`, for example `orchestrator-0.1.0-py3-none-any.whl` and `orchestrator-0.1.0.tar.gz`. Bump `version` in `pyproject.toml` when you cut a new release.

### Install into another project

**From the wheel (typical for a local or internal handoff):**

```bash
pip install /absolute/path/to/orchestratorv2/dist/orchestrator-0.1.0-py3-none-any.whl
```

**From the source tree (editable, good while both projects change):**

```bash
pip install -e /absolute/path/to/orchestratorv2
```

**From a Git URL (after you push the repo):**

```bash
pip install "git+https://example.com/your-org/orchestratorv2.git@v0.1.0"
```

### After it's installed

- **Python import:** use the package name `orchestrator` (see the `orchestrator/` package in this repo).
- **CLI:** `orchestrator-serve` runs the bundled server (defined in `pyproject.toml` as `[project.scripts]`).
- **Environment:** set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`) in the environment or a `.env` file wherever you run the app.
- **As a service:** if the other project only needs the HTTP API, run this package’s FastAPI app (for example `uvicorn orchestrator.main:app` or `orchestrator-serve`) and call the endpoints documented under [API](#api) above.
- **In-process:** import from `orchestrator` and compose or extend the same building blocks this repo uses (FastAPI factory, graph, tools) if you need the logic inside another application.

### Declare it as a dependency (consumer `pyproject.toml`)

Point at a local path during development:

```toml
[project]
dependencies = [
    "orchestrator @ file:///absolute/path/to/orchestratorv2",
]
```

Or at a built wheel:

```toml
dependencies = [
    "orchestrator @ file:///absolute/path/to/orchestratorv2/dist/orchestrator-0.1.0-py3-none-any.whl",
]
```

For a published package, use a normal version constraint once the project is on an index you configure (for example PyPI or a private package index).

## Demos

Example clients live under [`demos/`](demos/README.md); that file documents the **command to run** for each script.
