"""Stub tools for tagged-form workflows (hosts replace with real DB-backed implementations)."""

import json

from langchain.tools import tool


@tool
def list_form_field_tags(form_schema_ref: str) -> str:
    """Return tagged field ids for a form. ``form_schema_ref`` is an opaque id or URI (stub returns example JSON)."""
    return json.dumps(
        {"form_schema_ref": form_schema_ref, "tags": ["example.field_a", "example.field_b"]},
    )


@tool
def resolve_tags_batch(tags_json: str, context_hint: str) -> str:
    """Resolve many tags in one round-trip. ``tags_json`` is a JSON array of tag strings; ``context_hint`` narrows the query (stub returns empty map)."""
    try:
        tags = json.loads(tags_json)
    except json.JSONDecodeError:
        tags = []
    if not isinstance(tags, list):
        tags = []
    return json.dumps(
        {
            "context_hint": context_hint,
            "resolved": {t: None for t in tags},
            "note": "stub: replace with DB-backed resolution in the host app",
        },
    )


FORM_TOOLS = [list_form_field_tags, resolve_tags_batch]
