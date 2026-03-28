import json

from orchestrator.tools.form_stubs import list_form_field_tags, resolve_tags_batch


def test_list_form_field_tags_returns_json():
    raw = list_form_field_tags.invoke({"form_schema_ref": "ref-1"})
    data = json.loads(raw)
    assert "tags" in data


def test_resolve_tags_batch_accepts_json_array():
    raw = resolve_tags_batch.invoke({"tags_json": '["t1","t2"]', "context_hint": "ctx"})
    data = json.loads(raw)
    assert "resolved" in data
    assert data["resolved"]["t1"] is None
