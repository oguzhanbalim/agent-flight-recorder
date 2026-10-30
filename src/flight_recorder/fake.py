"""A scripted stand-in for a Messages API client, for tests and demos.

No model is involved. It returns the responses it was given, in order, and reads like
the SDK does: ``response.content[0].text``, ``response.usage.input_tokens``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flight_recorder.replay import _wrap


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def tool_use(block_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": block_id, "name": name, "input": tool_input}


def message(
    content: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = "claude-opus-5-5",
) -> dict[str, Any]:
    """Build a response in Messages API shape."""
    return {
        "id": "msg_scripted",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


class _Messages:
    def __init__(self, client: ScriptedClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._next(kwargs)


class ScriptedClient:
    """Return each scripted response in turn. An ``Exception`` in the script is raised."""

    def __init__(self, script: Sequence[dict[str, Any] | Exception]) -> None:
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.messages = _Messages(self)

    def _next(self, kwargs: dict[str, Any]) -> Any:
        self.requests.append(kwargs)
        if not self._script:
            raise IndexError("ScriptedClient ran out of responses")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _wrap(item)
