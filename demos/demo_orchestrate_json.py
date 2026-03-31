#!/usr/bin/env python3
"""Demo: POST /orchestrate — prompt + chat history (calls OpenAI; needs OPENAI_API_KEY)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import support


def main() -> None:
    support.banner("Demo: POST /orchestrate")
    print(f"Base URL: {support.base_url()}")
    support.require_api()
    payload = {
        "user_prompt": (
            "Say hello in one short sentence, then suggest one way to use the echo_text tool."
        ),
        "chat_history": [
            {"role": "user", "content": "We are testing the orchestrator API."},
            {"role": "assistant", "content": "Understood."},
        ],
    }
    support.print_json("Request body", payload)
    with support.client() as c:
        r = c.post("/orchestrate", json=payload)
    support.print_json(f"Response {r.status_code}", r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
    support.raise_for_status_verbose(r)


if __name__ == "__main__":
    main()
