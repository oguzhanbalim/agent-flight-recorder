"""Wrap a Messages-API-shaped client and record every call."""

from __future__ import annotations

import inspect
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flight_recorder.log import FORMAT_VERSION, to_jsonable

# Transport settings, not part of what was asked. Headers can hold credentials,
# so they are never written.
TRANSPORT_KEYS = frozenset({"extra_headers", "extra_query", "timeout"})

Redactor = Callable[[dict[str, Any]], dict[str, Any]]


def normalize_request(kwargs: dict[str, Any], redact: Redactor | None = None) -> dict[str, Any]:
    request = to_jsonable({k: v for k, v in kwargs.items() if k not in TRANSPORT_KEYS})
    return redact(request) if redact else request


class _Messages:
    def __init__(self, recorder: RecordingClient) -> None:
        self._recorder = recorder

    def create(self, **kwargs: Any) -> Any:
        return self._recorder._create(kwargs)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"messages.{name} would bypass the recording. Only messages.create is supported."
        )


class RecordingClient:
    """Pass ``messages.create`` calls to ``client`` and append each one to ``path``.

    The caller gets the wrapped client's own response object back, untouched. Errors
    are recorded and re-raised. Other client attributes raise rather than pass through,
    so a call cannot silently escape the recording.
    """

    def __init__(
        self,
        client: Any,
        path: str | Path,
        *,
        label: str | None = None,
        redact: Redactor | None = None,
    ) -> None:
        self._client = client
        self._redact = redact
        self._lock = threading.Lock()
        self._seq = 0
        self.path = Path(path)
        self.run_id = uuid.uuid4().hex
        self.messages = _Messages(self)
        # "x": overwriting an old run would destroy the evidence this exists to keep.
        self._file = self.path.open("x", encoding="utf-8")
        self._write(
            {
                "type": "run",
                "format": FORMAT_VERSION,
                "run_id": self.run_id,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "label": label,
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def _create(self, kwargs: dict[str, Any]) -> Any:
        if kwargs.get("stream"):
            raise NotImplementedError("stream=True cannot be recorded; use non-streaming calls")
        request = normalize_request(kwargs, self._redact)
        started = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            self._record(request, started, error={"type": type(exc).__name__, "message": str(exc)})
            raise
        if inspect.isawaitable(response):
            if inspect.iscoroutine(response):
                response.close()
            raise NotImplementedError("async clients are not supported; wrap a sync client")
        recorded = to_jsonable(response)
        if self._redact:
            recorded = self._redact(recorded)
        self._record(request, started, response=recorded)
        return response

    def _record(self, request: dict[str, Any], started: float, **outcome: Any) -> None:
        duration = round((time.perf_counter() - started) * 1000, 1)
        with self._lock:
            record = {
                "type": "call",
                "seq": self._seq,
                "request": request,
                "duration_ms": duration,
                **outcome,
            }
            self._write(record)
            self._seq += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> RecordingClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
