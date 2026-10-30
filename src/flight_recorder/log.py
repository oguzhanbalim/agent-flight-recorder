"""The JSONL format and its reader.

A run file is one ``run`` header followed by one ``call`` record per API call::

    {"type": "run", "format": 1, "run_id": "...", "created_at": "...", "label": "..."}
    {"type": "call", "seq": 0, "request": {...}, "response": {...}, "duration_ms": 12.3}
    {"type": "call", "seq": 1, "request": {...}, "error": {"type": "...", "message": "..."},
     "duration_ms": 4.0}

Each line is flushed and fsynced as it is written, so a crash loses at most the call
in flight. A torn last line is what a crash looks like, so the reader tolerates that
one case and flags it. Damage anywhere else is an error.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1


class RunFileError(ValueError):
    """The file is not a readable flight recorder run."""


def to_jsonable(value: Any) -> Any:
    """Convert a request or response to plain JSON types.

    Handles dicts, lists, SDK pydantic models and dataclasses. Anything else becomes a
    fixed placeholder rather than its repr, which often holds a memory address that
    would make two identical runs look different.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if callable(getattr(value, "model_dump", None)):
        return to_jsonable(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    return f"<unserializable {type(value).__name__}>"


@dataclass(frozen=True)
class Call:
    seq: int
    request: dict[str, Any]
    response: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    duration_ms: float | None = None

    @property
    def content(self) -> list[dict[str, Any]]:
        return (self.response or {}).get("content") or []

    @property
    def stop_reason(self) -> str | None:
        return (self.response or {}).get("stop_reason")

    @property
    def usage(self) -> dict[str, Any]:
        return (self.response or {}).get("usage") or {}


@dataclass(frozen=True)
class Run:
    header: dict[str, Any]
    calls: list[Call] = field(default_factory=list)
    truncated: bool = False

    @property
    def run_id(self) -> str:
        return str(self.header.get("run_id", "?"))

    @property
    def label(self) -> str | None:
        return self.header.get("label")


def read_run(path: str | Path) -> Run:
    try:
        lines = Path(path).read_text(encoding="utf-8").split("\n")
    except OSError as exc:
        raise RunFileError(f"{path}: {exc}") from exc
    complete_final_line = lines[-1] == ""
    if complete_final_line:
        lines.pop()

    records: list[dict[str, Any]] = []
    truncated = False
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if number == len(lines) and not complete_final_line:
                truncated = True
                break
            raise RunFileError(f"{path}:{number}: not valid JSON: {exc}") from exc
        records.append(record)

    if not records or records[0].get("type") != "run":
        raise RunFileError(f"{path}: missing run header")
    header = records[0]
    if header.get("format") != FORMAT_VERSION:
        raise RunFileError(f"{path}: unsupported format {header.get('format')!r}")

    calls = []
    for index, record in enumerate(records[1:]):
        if record.get("type") != "call":
            raise RunFileError(f"{path}: unexpected record type {record.get('type')!r}")
        if record.get("seq") != index:
            raise RunFileError(f"{path}: expected seq {index}, found {record.get('seq')!r}")
        calls.append(
            Call(
                seq=index,
                request=record.get("request", {}),
                response=record.get("response"),
                error=record.get("error"),
                duration_ms=record.get("duration_ms"),
            )
        )
    return Run(header=header, calls=calls, truncated=truncated)
