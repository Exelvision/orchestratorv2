"""Validation for registered agents (named tool subsets referenced from plans)."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.tools import BaseTool

from orchestrator.models import RegisteredAgent


def validate_registered_agents(
    tools: Sequence[BaseTool],
    agents: Sequence[RegisteredAgent] | None,
) -> tuple[RegisteredAgent, ...]:
    """Ensure each agent id is unique and every tool_name exists on ``tools``."""
    if not agents:
        return ()
    tool_names = {t.name for t in tools}
    seen: set[str] = set()
    out: list[RegisteredAgent] = []
    for a in agents:
        if a.id in seen:
            msg = f"Duplicate registered agent id: {a.id!r}"
            raise ValueError(msg)
        seen.add(a.id)
        if not a.tool_names:
            msg = f"Registered agent {a.id!r} must list at least one tool_name"
            raise ValueError(msg)
        for tn in a.tool_names:
            if tn not in tool_names:
                msg = f"Agent {a.id!r} references unknown tool_name: {tn!r}"
                raise ValueError(msg)
        out.append(a)
    return tuple(out)
