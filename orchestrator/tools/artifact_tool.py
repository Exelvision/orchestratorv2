"""Tool for submitting a canonical structured artifact (tool-grounded JSON)."""

from langchain.tools import tool


@tool
def commit_structured_output(json_payload: str) -> str:
    """Submit the canonical structured artifact for this run as a JSON object string.

    Prefer this over paraphrasing data in assistant text when the result must be machine-validated
    (for example filled forms or DB-backed field maps). Pass minified JSON.
    """
    return json_payload


ARTIFACT_TOOLS = [commit_structured_output]
