from __future__ import annotations

import contextlib
from pathlib import Path

from flight_recorder import RecordingClient, ScriptedClient, diff_runs, read_run
from flight_recorder.fake import message, text, tool_use


def record(path: Path, script, label=None, model="claude-opus-5-5"):
    with RecordingClient(ScriptedClient(script), path, label=label) as client:
        for _ in script:
            with contextlib.suppress(RuntimeError):
                client.messages.create(model=model, max_tokens=1, messages=[])
    return read_run(path)


def tool_step(name, tool_input, block_id="t", **usage):
    return message(
        [tool_use(block_id, name, tool_input)],
        stop_reason="tool_use",
        input_tokens=usage.get("input_tokens", 10),
        output_tokens=usage.get("output_tokens", 5),
    )


def done(**usage):
    return message([text("ok")], input_tokens=usage.get("input_tokens", 10), output_tokens=5)


def test_identical_behavior_is_same_even_with_different_ids_and_timings(tmp_path):
    a = record(tmp_path / "a", [tool_step("x", {"n": 1}, block_id="toolu_a"), done()])
    b = record(tmp_path / "b", [tool_step("x", {"n": 1}, block_id="toolu_b"), done()])
    result = diff_runs(a, b)
    assert not result.behavior_changed
    assert "same behavior" in result.render()


def test_removed_and_added_tools_show_in_the_sequence(tmp_path):
    a = record(tmp_path / "a", [tool_step("lookup", {}), tool_step("refund", {}), done()])
    b = record(tmp_path / "b", [tool_step("refund", {}), tool_step("notify", {}), done()])
    result = diff_runs(a, b)
    assert result.tool_sequence == [("-", "lookup"), (" ", "refund"), ("+", "notify")]
    assert result.behavior_changed


def test_reordered_tools_are_a_change(tmp_path):
    a = record(tmp_path / "a", [tool_step("x", {}), tool_step("y", {})])
    b = record(tmp_path / "b", [tool_step("y", {}), tool_step("x", {})])
    assert diff_runs(a, b).tool_sequence_changed


def test_input_changes_name_the_tool_and_field(tmp_path):
    a = record(tmp_path / "a", [tool_step("refund", {"amount": 40, "meta": {"why": "broken"}})])
    b = record(tmp_path / "b", [tool_step("refund", {"amount": 400, "meta": {"why": "late"}})])
    result = diff_runs(a, b)
    assert not result.tool_sequence_changed
    assert result.input_changes == [
        {"tool": "refund", "path": "$.amount", "a": 40, "b": 400},
        {"tool": "refund", "path": "$.meta.why", "a": "broken", "b": "late"},
    ]


def test_added_input_field_and_int_vs_string_are_changes(tmp_path):
    a = record(tmp_path / "a", [tool_step("t", {"n": 1})])
    b = record(tmp_path / "b", [tool_step("t", {"n": "1", "extra": True})])
    paths = {change["path"] for change in diff_runs(a, b).input_changes}
    assert paths == {"$.n", "$.extra"}


def test_stop_reason_change_alone_is_behavior(tmp_path):
    a = record(tmp_path / "a", [message([text("ok")])])
    b = record(tmp_path / "b", [message([text("ok")], stop_reason="max_tokens")])
    result = diff_runs(a, b)
    assert result.behavior_changed
    assert "stop reasons: CHANGED" in result.render()


def test_usage_change_alone_is_reported_but_not_behavior(tmp_path):
    a = record(tmp_path / "a", [done(input_tokens=100)])
    b = record(tmp_path / "b", [done(input_tokens=210)])
    result = diff_runs(a, b)
    assert not result.behavior_changed
    assert result.usage_a["input_tokens"] == 100
    assert result.usage_b["input_tokens"] == 210
    assert round(result.usage_increase_pct(), 1) == 104.8
    assert "(+110)" in result.render()


def test_none_usage_fields_from_the_sdk_count_as_zero(tmp_path):
    response = message([text("ok")], input_tokens=1)
    response["usage"]["cache_read_input_tokens"] = None
    a = record(tmp_path / "a", [response])
    assert diff_runs(a, a).usage_a["cache_read_input_tokens"] == 0


def test_model_change_and_new_errors_are_behavior(tmp_path):
    a = record(tmp_path / "a", [done()])
    assert diff_runs(a, record(tmp_path / "b", [done()], model="other")).behavior_changed
    errored = record(tmp_path / "c", [RuntimeError("overloaded")])
    assert diff_runs(a, errored).errors_b == ["RuntimeError"]


def test_call_count_change_is_behavior_even_with_no_tools(tmp_path):
    a = record(tmp_path / "a", [done()])
    b = record(tmp_path / "b", [done(), done()])
    assert diff_runs(a, b).behavior_changed


def test_diff_output_is_deterministic(tmp_path):
    a = record(tmp_path / "a", [tool_step("x", {"n": 1})], label="a")
    b = record(tmp_path / "b", [tool_step("y", {"n": 2})], label="b")
    assert diff_runs(a, b).render() == diff_runs(a, b).render()
