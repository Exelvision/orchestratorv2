#!/usr/bin/env python3
"""Demo: JSON orchestrate with context + metadata fields."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import support


def main() -> None:
    support.banner("Demo: context + metadata on /orchestrate")
    print(f"Base URL: {support.base_url()}")
    support.require_api()
    payload = {
        "user_prompt": (
            "Summarize what you see in the client context and metadata in one sentence each."
        ),
        "chat_history": [],
        "context": {
            "tenant": "demo-tenant",
            "locale": "en-US",
            "feature_flags": {"beta_tools": True},
        },
        "metadata": {
            "request_id": "demo-req-001",
            "client": "demos/demo_orchestrate_context.py",
        },
    }
    support.print_json("Request body", payload)
    with support.client() as c:
        r = c.post("/orchestrate", json=payload)
    support.print_json(f"Response {r.status_code}", r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
    support.raise_for_status_verbose(r)


if __name__ == "__main__":
    main()
