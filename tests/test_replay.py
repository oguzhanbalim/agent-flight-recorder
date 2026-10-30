from __future__ import annotations

import json

import pytest
from refund_agent import run_agent, run_tool

from flight_recorder import (
    RecordedAPIError,
    RecordingClient,
    ReplayClient,
    ReplayDivergence,
    ScriptedClient,
)
from flight_recorder.fake import message, text, tool_use
from flight_recorder.replay import first_difference

SCRIPT = [
    message(
        [tool_use("t1", "lookup_order", {"order_id": "ORD-1"})],
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=2,
    ),
    message([text("done")], input_tokens=20, output_tokens=3),
]


@pytest.fixture
def recorded(run_path):
    with RecordingClient(ScriptedClient(SCRIPT), run_path) as client:
        run_agent(client)
    return run_path


def test_replay_gives_the_agent_the_same_answer_with_no_client(recorded):
    replay = ReplayClient(recorded)
    final = run_agent(replay)
    replay.assert_exhausted()
    assert final.content[0].text == "done"
    assert final.usage.input_tokens == 20


def test_replay_is_deterministic(recorded):
    first = run_agent(ReplayClient(recorded))
    second = run_agent(ReplayClient(recorded))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_changed_tool_output_is_a_divergence_at_the_exact_path(recorded):
    def other_tool(name, tool_input):
        return "different"

    with pytest.raises(ReplayDivergence) as caught:
        run_agent(ReplayClient(recorded), tool_runner=other_tool)
    assert caught.value.seq == 1
    assert caught.value.path == "$.messages[2].content[0].content"
    assert caught.value.actual == "different"
    assert caught.value.recorded == run_tool("lookup_order", {"order_id": "ORD-1"})


def test_changed_prompt_or_model_diverges_on_the_first_call(recorded):
    replay = ReplayClient(recorded)
    request = dict(replay.run.calls[0].request, model="other")
    with pytest.raises(ReplayDivergence, match=r"\$\.model"):
        replay.messages.create(**request)

    replay = ReplayClient(recorded)
    request = replay.run.calls[0].request
    request = dict(request, messages=[{"role": "user", "content": "a different prompt"}])
    with pytest.raises(ReplayDivergence, match=r"\$\.messages\[0\]\.content"):
        replay.messages.create(**request)


def test_too_many_calls_diverges(recorded):
    replay = ReplayClient(recorded)
    run_agent(replay)
    with pytest.raises(ReplayDivergence, match="end of recording"):
        replay.messages.create(model="m", max_tokens=1, messages=[])


def test_stopping_early_is_caught_by_assert_exhausted(recorded):
    replay = ReplayClient(recorded, strict=False)
    replay.messages.create()
    assert replay.remaining == 1
    with pytest.raises(ReplayDivergence, match="agent stopped"):
        replay.assert_exhausted()


def test_recorded_failure_replays_as_a_failure(run_path):
    with (
        RecordingClient(ScriptedClient([RuntimeError("overloaded")]), run_path) as client,
        pytest.raises(RuntimeError),
    ):
        client.messages.create(model="m", max_tokens=1, messages=[])
    with pytest.raises(RecordedAPIError) as caught:
        ReplayClient(run_path).messages.create(model="m", max_tokens=1, messages=[])
    assert caught.value.error_type == "RuntimeError"
    assert caught.value.message == "overloaded"


def test_ignored_keys_may_vary(run_path):
    with RecordingClient(ScriptedClient([message([text("a")])]), run_path) as client:
        client.messages.create(model="m", max_tokens=1, messages=[], metadata={"user_id": "one"})
    replay = ReplayClient(run_path, ignore={"metadata"})
    assert replay.messages.create(model="m", max_tokens=1, messages=[], metadata={"user_id": "two"})


def test_strict_false_skips_request_checks(recorded):
    response = ReplayClient(recorded, strict=False).messages.create(model="anything")
    assert response.stop_reason == "tool_use"


def test_redacted_recording_still_replays_with_the_same_redactor(run_path):
    def scrub(record):
        return json.loads(json.dumps(record).replace("secret", "xxx"))

    request = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "secret"}]}
    with RecordingClient(ScriptedClient([message([text("a")])]), run_path, redact=scrub) as client:
        client.messages.create(**request)
    assert ReplayClient(run_path, redact=scrub).messages.create(**request)
    with pytest.raises(ReplayDivergence):
        ReplayClient(run_path).messages.create(**request)


def test_missing_attribute_raises_like_an_object(recorded):
    response = ReplayClient(recorded, strict=False).messages.create()
    with pytest.raises(AttributeError):
        _ = response.no_such_field


@pytest.mark.parametrize(
    ("a", "b", "path"),
    [
        ({"x": 1}, {"x": 1}, None),
        ({"x": 1}, {"x": 2}, "$.x"),
        ({"x": 1}, {}, "$.x"),
        ({}, {"x": 1}, "$.x"),
        ({"x": [1, 2]}, {"x": [1]}, "$.x.length"),
        ({"x": [{"y": 1}]}, {"x": [{"y": 2}]}, "$.x[0].y"),
        ({"x": 1}, {"x": True}, "$.x"),
        ({"x": 1}, {"x": 1.0}, "$.x"),
    ],
)
def test_first_difference(a, b, path):
    found = first_difference(a, b)
    assert (found[0] if found else None) == path
