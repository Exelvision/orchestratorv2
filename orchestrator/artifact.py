"""Extract tool-grounded structured payloads from executor message traces."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage


def extract_structured_artifact(messages: list[Any]) -> dict[str, Any] | None:
    """Return the latest JSON object from a ``commit_structured_output`` tool result, if any.

    Hosts should prefer this (or tool messages) over parsing free-form ``answer`` text for
    DB-backed or form-fill artifacts.
    """
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        name = getattr(m, "name", None) or ""
        content = m.content
        if not isinstance(content, str):
            continue
        if name == "commit_structured_output":
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        else:
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "artifact" in data:
                art = data.get("artifact")
                if isinstance(art, dict):
                    return art
    return None
