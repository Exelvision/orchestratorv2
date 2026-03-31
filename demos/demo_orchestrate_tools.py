#!/usr/bin/env python3
"""Demo: JSON orchestrate with a prompt that should trigger word_count or echo_text."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import support


def main() -> None:
    support.banner("Demo: orchestrate with tool-friendly prompt")
    print(f"Base URL: {support.base_url()}")
    support.require_api()
    text = "The quick brown fox jumps over the lazy dog"
    payload = {
        "user_prompt": (
            f'Use the word_count tool on this exact sentence and report the count: "{text}"'
        ),
        "chat_history": [],
    }
    support.print_json("Request body", payload)
    with support.client() as c:
        r = c.post("/orchestrate", json=payload)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
    support.print_json(f"Response {r.status_code}", body)
    support.raise_for_status_verbose(r)


if __name__ == "__main__":
    main()
