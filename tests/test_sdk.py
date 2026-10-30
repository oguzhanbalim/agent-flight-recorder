"""The real anthropic SDK, with its HTTP layer replaced by a local mock. Nothing leaves the process.

The response body below is hand-written to the documented Messages shape. It shows that the
SDK's pydantic objects round-trip through record and replay. It says nothing about model behavior.
"""

from __future__ import annotations

import json

import pytest
from refund_agent import run_agent

from flight_recorder import RecordingClient, ReplayClient, read_run

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")


def body(content, stop_reason):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


@pytest.fixture
def sdk_client():
    responses = [
        body(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup_order",
                    "input": {"order_id": "A"},
                }
            ],
            "tool_use",
        ),
        body([{"type": "text", "text": "done"}], "end_turn"),
    ]
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=responses[len(seen) - 1])

    client = anthropic.Anthropic(
        api_key="not-a-real-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    client.seen = seen
    return client


def test_record_and_replay_with_real_sdk_objects(sdk_client, run_path):
    with RecordingClient(sdk_client, run_path) as client:
        final = run_agent(client)
    # The caller gets the SDK's own type back, not a wrapper.
    assert isinstance(final, anthropic.types.Message)
    assert len(sdk_client.seen) == 2

    run = read_run(run_path)
    assert run.calls[0].content[0]["name"] == "lookup_order"
    assert run.calls[1].stop_reason == "end_turn"
    assert "not-a-real-key" not in run_path.read_text()

    # Replay makes no HTTP call: the handler is not hit again.
    replay = ReplayClient(run_path)
    replayed = run_agent(replay)
    replay.assert_exhausted()
    assert replayed.content[0].text == "done"
    assert len(sdk_client.seen) == 2


def test_sdk_api_error_is_recorded(run_path):
    def handler(request):
        return httpx2.Response(
            429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}
        )

    sdk = anthropic.Anthropic(
        api_key="not-a-real-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    with RecordingClient(sdk, run_path) as client, pytest.raises(anthropic.RateLimitError):
        client.messages.create(model="claude-opus-5-5", max_tokens=1, messages=[])
    assert read_run(run_path).calls[0].error["type"] == "RateLimitError"
