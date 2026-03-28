from langchain_core.messages import ToolMessage

from orchestrator.artifact import extract_structured_artifact


def test_extract_structured_artifact_from_commit_tool():
    msgs = [
        ToolMessage(
            content='{"filled": {"a": "1"}}',
            tool_call_id="c1",
            name="commit_structured_output",
        ),
    ]
    assert extract_structured_artifact(msgs) == {"filled": {"a": "1"}}


def test_extract_structured_artifact_nested_artifact_key():
    msgs = [
        ToolMessage(
            content='{"artifact": {"k": "v"}}',
            tool_call_id="c1",
            name="other_tool",
        ),
    ]
    assert extract_structured_artifact(msgs) == {"k": "v"}


def test_extract_structured_artifact_none_if_missing():
    assert extract_structured_artifact([]) is None
    assert extract_structured_artifact([ToolMessage(content="not json", tool_call_id="x")]) is None
