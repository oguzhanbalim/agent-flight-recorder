"""Serve a recorded run back to the same agent code, with no network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flight_recorder.log import Call, read_run
from flight_recorder.recorder import Redactor, normalize_request


class RecordedAPIError(Exception):
    """The recorded call raised; this replays that failure."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message


class ReplayDivergence(AssertionError):
    """The agent asked for something the recording did not."""

    def __init__(self, seq: int, path: str, recorded: Any, actual: Any) -> None:
        super().__init__(f"call {seq} diverged at {path}: recorded {recorded!r}, got {actual!r}")
        self.seq = seq
        self.path = path
        self.recorded = recorded
        self.actual = actual


class Recorded(dict):
    """A recorded JSON object readable like an SDK object: ``block.text`` or ``block["text"]``."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Recorded({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def first_difference(recorded: Any, actual: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    """Return the first differing JSON path and its two values, or None if they match."""
    if isinstance(recorded, dict) and isinstance(actual, dict):
        for key in sorted(set(recorded) | set(actual)):
            if key not in recorded:
                return f"{path}.{key}", "<absent>", actual[key]
            if key not in actual:
                return f"{path}.{key}", recorded[key], "<absent>"
            found = first_difference(recorded[key], actual[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(recorded, list) and isinstance(actual, list):
        for index, (left, right) in enumerate(zip(recorded, actual, strict=False)):
            found = first_difference(left, right, f"{path}[{index}]")
            if found:
                return found
        if len(recorded) != len(actual):
            return f"{path}.length", len(recorded), len(actual)
        return None
    if type(recorded) is not type(actual) or recorded != actual:
        return path, recorded, actual
    return None


class _Messages:
    def __init__(self, replay: ReplayClient) -> None:
        self._replay = replay

    def create(self, **kwargs: Any) -> Any:
        return self._replay._create(kwargs)


class ReplayClient:
    """Answer ``messages.create`` from a run file, checking each request against the recording.

    With ``strict=False`` requests are not compared, which is useful for walking
    a run while changing the prompt. ``ignore`` names top-level request keys that are
    expected to vary, such as ``metadata``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        strict: bool = True,
        ignore: frozenset[str] | set[str] = frozenset(),
        redact: Redactor | None = None,
    ) -> None:
        self.run = read_run(path)
        self.strict = strict
        self.ignore = frozenset(ignore)
        self.messages = _Messages(self)
        self._redact = redact
        self._next = 0

    def _create(self, kwargs: dict[str, Any]) -> Any:
        seq = self._next
        request = normalize_request(kwargs, self._redact)
        if seq >= len(self.run.calls):
            raise ReplayDivergence(seq, "$", "<end of recording>", request)
        call: Call = self.run.calls[seq]
        self._next += 1
        if self.strict:
            recorded = {k: v for k, v in call.request.items() if k not in self.ignore}
            actual = {k: v for k, v in request.items() if k not in self.ignore}
            difference = first_difference(recorded, actual)
            if difference:
                raise ReplayDivergence(seq, *difference)
        if call.error is not None:
            raise RecordedAPIError(call.error.get("type", "?"), call.error.get("message", ""))
        return _wrap(call.response)

    @property
    def remaining(self) -> int:
        return len(self.run.calls) - self._next

    def assert_exhausted(self) -> None:
        """Raise if the agent stopped before using every recorded call."""
        if self.remaining:
            raise ReplayDivergence(
                self._next, "$", f"{self.remaining} more call(s)", "<agent stopped>"
            )
