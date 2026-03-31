# Demos

Small scripts that call the running API. Run them from the **repository root**.

## Prerequisites

1. **Server** (orchestration demos need `OPENAI_API_KEY`):

   ```bash
   source .venv/bin/activate
   export OPENAI_API_KEY=sk-...
   uvicorn orchestrator.main:app --host 127.0.0.1 --port 8000
   ```

2. **httpx** (if not already installed): `pip install httpx` or `pip install -e '.[dev]'`

3. **Shell**: activate `.venv` in the terminal where you run the commands below, or call `.venv/bin/python` explicitly.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `ORCHESTRATOR_URL` | `http://127.0.0.1:8000` | API base URL |
| `ORCHESTRATOR_DEMO_TIMEOUT` | `120` | HTTP timeout (seconds) |

---

### `demo_health.py` — `GET /health`

No API key required.

```bash
python demos/demo_health.py
```

---

### `demo_list_catalog.py` — `GET /orchestrate/tools` and `GET /orchestrate/flows`

Lists tool names/descriptions and named-flow ids (what the server was started with, including `create_app(tools=..., flows=...)`).

No API key required.

```bash
python demos/demo_list_catalog.py
```

---

### `demo_plan_only.py` — `POST /orchestrate/plan`

Planner only (structured plan, no ReAct executor).

```bash
python demos/demo_plan_only.py
```

---

### `demo_executor_only.py` — `POST /orchestrate/execute`

ReAct executor with a fixed plan in the request body (no planner call).

```bash
python demos/demo_executor_only.py
```

---

### `demo_orchestrate_json.py` — `POST /orchestrate`

Full pipeline with `user_prompt` and `chat_history`.

```bash
python demos/demo_orchestrate_json.py
```

---

### `demo_orchestrate_tools.py` — `POST /orchestrate`

Same route; prompt written to encourage `word_count` (or similar tools).

```bash
python demos/demo_orchestrate_tools.py
```

---

### `demo_orchestrate_context.py` — `POST /orchestrate`

Includes `context` and `metadata` in the JSON body.

```bash
python demos/demo_orchestrate_context.py
```

---

### `demo_orchestrate_multipart.py` — `POST /orchestrate`

Multipart form: `payload` (JSON string) plus a text file upload.

```bash
python demos/demo_orchestrate_multipart.py
```

---

Using the venv interpreter without activating:

```bash
.venv/bin/python demos/demo_health.py
```
