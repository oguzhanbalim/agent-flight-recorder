from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from flight_recorder import RecordingClient, ScriptedClient, read_run
from flight_recorder.fake import message, text
from flight_recorder.log import RunFileError, to_jsonable

REQUEST = {
    "model": "claude-opus-5-5",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "x"}],
}


def test_returns_the_wrapped_clients_response_and_records_it(run_path):
    inner = ScriptedClient([message([text("hi")], input_tokens=3, output_tokens=1)])
    with RecordingClient(inner, run_path, label="t") as client:
        response = client.messages.create(**REQUEST)
    assert response.content[0].text == "hi"
    run = read_run(run_path)
    assert run.label == "t"
    assert len(run.calls) == 1
    assert run.calls[0].request == REQUEST
    assert run.calls[0].stop_reason == "end_turn"
    assert run.calls[0].usage["input_tokens"] == 3
    assert run.calls[0].duration_ms is not None


def test_errors_are_recorded_and_reraised(run_path):
    inner = ScriptedClient([RuntimeError("overloaded")])
    with (
        RecordingClient(inner, run_path) as client,
        pytest.raises(RuntimeError, match="overloaded"),
    ):
        client.messages.create(**REQUEST)
    call = read_run(run_path).calls[0]
    assert call.error == {"type": "RuntimeError", "message": "overloaded"}
    assert call.response is None


def test_credentials_in_headers_are_never_written(run_path):
    inner = ScriptedClient([message([text("hi")])])
    with RecordingClient(inner, run_path) as client:
        client.messages.create(**REQUEST, extra_headers={"x-api-key": "secret-value"}, timeout=5)
    assert "secret-value" not in run_path.read_text()
    # The transport settings still reach the wrapped client.
    assert inner.requests[0]["extra_headers"] == {"x-api-key": "secret-value"}


def test_later_mutation_of_messages_does_not_change_the_record(run_path):
    inner = ScriptedClient([message([text("a")]), message([text("b")])])
    messages = [{"role": "user", "content": "first"}]
    with RecordingClient(inner, run_path) as client:
        client.messages.create(model="m", max_tokens=1, messages=messages)
        messages.append({"role": "user", "content": "second"})
        client.messages.create(model="m", max_tokens=1, messages=messages)
    calls = read_run(run_path).calls
    assert len(calls[0].request["messages"]) == 1
    assert len(calls[1].request["messages"]) == 2


def test_stream_is_refused_rather_than_recorded_badly(run_path):
    with (
        RecordingClient(ScriptedClient([]), run_path) as client,
        pytest.raises(NotImplementedError),
    ):
        client.messages.create(**REQUEST, stream=True)


def test_other_client_surfaces_cannot_escape_the_recording(run_path):
    with RecordingClient(ScriptedClient([]), run_path) as client:
        with pytest.raises(AttributeError, match="bypass"):
            _ = client.messages.stream
        with pytest.raises(AttributeError):
            _ = client.beta


def test_async_clients_are_refused_clearly(run_path):
    class AsyncLike:
        class messages:
            @staticmethod
            async def create(**kwargs):
                return {}

    with RecordingClient(AsyncLike, run_path) as client, pytest.raises(NotImplementedError):
        client.messages.create(**REQUEST)


def test_existing_run_is_not_overwritten(run_path):
    run_path.write_text("evidence\n")
    with pytest.raises(FileExistsError):
        RecordingClient(ScriptedClient([]), run_path)
    assert run_path.read_text() == "evidence\n"


def test_redact_applies_to_requests_and_responses(run_path):
    def scrub(record):
        return json.loads(json.dumps(record).replace("ORD-1", "ORD-X"))

    inner = ScriptedClient([message([text("about ORD-1")])])
    with RecordingClient(inner, run_path, redact=scrub) as client:
        response = client.messages.create(
            model="m", max_tokens=1, messages=[{"role": "user", "content": "ORD-1"}]
        )
    assert response.content[0].text == "about ORD-1"
    assert "ORD-1" not in run_path.read_text()


def test_crash_keeps_every_finished_call(tmp_path):
    path = tmp_path / "crash.jsonl"
    script = textwrap.dedent(
        f"""
        import os
        from flight_recorder import RecordingClient, ScriptedClient
        from flight_recorder.fake import message, text
        client = RecordingClient(ScriptedClient([message([text("a")]), message([text("b")])]),
                                 {str(path)!r})
        client.messages.create(model="m", max_tokens=1, messages=[])
        client.messages.create(model="m", max_tokens=1, messages=[])
        os._exit(9)  # No close, no flush at exit: the crash this exists for.
        """
    )
    done = subprocess.run([sys.executable, "-c", script], check=False)
    assert done.returncode == 9
    assert len(read_run(path).calls) == 2


def test_torn_last_line_is_tolerated_and_flagged(run_path):
    with RecordingClient(ScriptedClient([message([text("a")])]), run_path) as client:
        client.messages.create(**REQUEST)
    with run_path.open("a") as handle:
        handle.write('{"type": "call", "seq": 1, "requ')
    run = read_run(run_path)
    assert run.truncated
    assert len(run.calls) == 1


def test_damage_before_the_last_line_is_an_error(run_path):
    with RecordingClient(ScriptedClient([message([text("a")])]), run_path) as client:
        client.messages.create(**REQUEST)
    lines = run_path.read_text().splitlines()
    run_path.write_text("\n".join([lines[0], "{broken", lines[1]]) + "\n")
    with pytest.raises(RunFileError, match="not valid JSON"):
        read_run(run_path)


def test_removed_call_is_detected_by_sequence_gap(run_path):
    inner = ScriptedClient([message([text("a")]), message([text("b")])])
    with RecordingClient(inner, run_path) as client:
        client.messages.create(**REQUEST)
        client.messages.create(**REQUEST)
    lines = run_path.read_text().splitlines()
    run_path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(RunFileError, match="expected seq 0"):
        read_run(run_path)


def test_file_without_header_or_wrong_format_is_rejected(tmp_path):
    bad = tmp_path / "x.jsonl"
    bad.write_text('{"type": "call", "seq": 0}\n')
    with pytest.raises(RunFileError, match="header"):
        read_run(bad)
    bad.write_text('{"type": "run", "format": 99}\n')
    with pytest.raises(RunFileError, match="format"):
        read_run(bad)


def test_unserializable_values_get_a_stable_placeholder():
    class Thing:
        pass

    assert (
        to_jsonable({"a": Thing()})
        == to_jsonable({"a": Thing()})
        == {"a": "<unserializable Thing>"}
    )
