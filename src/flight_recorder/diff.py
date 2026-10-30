"""Compare two runs on behavior: what the agent did, not the words it used."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any

from flight_recorder.log import Run

TOOL_BLOCK_TYPES = frozenset({"tool_use", "server_tool_use", "mcp_tool_use"})
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def tool_calls(run: Run) -> list[dict[str, Any]]:
    """Every tool call in order. Block ids are dropped since they differ between runs."""
    return [
        {"seq": call.seq, "name": block.get("name"), "input": block.get("input")}
        for call in run.calls
        for block in call.content
        if block.get("type") in TOOL_BLOCK_TYPES
    ]


def total_usage(run: Run) -> dict[str, int]:
    return {name: sum(call.usage.get(name) or 0 for call in run.calls) for name in USAGE_FIELDS}


def _field_changes(left: Any, right: Any, path: str = "$") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            where = f"{path}.{key}"
            if key not in left:
                changes.append({"path": where, "a": "<absent>", "b": right[key]})
            elif key not in right:
                changes.append({"path": where, "a": left[key], "b": "<absent>"})
            else:
                changes.extend(_field_changes(left[key], right[key], where))
        return changes
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        changes = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            changes.extend(_field_changes(a, b, f"{path}[{index}]"))
        return changes
    if type(left) is not type(right) or left != right:
        return [{"path": path, "a": left, "b": right}]
    return []


@dataclass
class RunDiff:
    label_a: str
    label_b: str
    calls_a: int
    calls_b: int
    # Lines of a unified tool sequence: (" " | "-" | "+", tool name).
    tool_sequence: list[tuple[str, str]] = field(default_factory=list)
    input_changes: list[dict[str, Any]] = field(default_factory=list)
    stop_reasons_a: list[str | None] = field(default_factory=list)
    stop_reasons_b: list[str | None] = field(default_factory=list)
    models_a: list[str] = field(default_factory=list)
    models_b: list[str] = field(default_factory=list)
    errors_a: list[str] = field(default_factory=list)
    errors_b: list[str] = field(default_factory=list)
    usage_a: dict[str, int] = field(default_factory=dict)
    usage_b: dict[str, int] = field(default_factory=dict)
    truncated: bool = False

    @property
    def tool_sequence_changed(self) -> bool:
        return any(mark != " " for mark, _ in self.tool_sequence)

    @property
    def behavior_changed(self) -> bool:
        return (
            self.tool_sequence_changed
            or bool(self.input_changes)
            or self.stop_reasons_a != self.stop_reasons_b
            or self.models_a != self.models_b
            or self.errors_a != self.errors_b
            or self.calls_a != self.calls_b
        )

    def total_tokens(self, usage: dict[str, int]) -> int:
        return sum(usage.values())

    def usage_increase_pct(self) -> float | None:
        """Percent growth in total tokens; None when the baseline used none."""
        before = self.total_tokens(self.usage_a)
        if before == 0:
            return None
        return (self.total_tokens(self.usage_b) - before) * 100 / before

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.label_a,
            "b": self.label_b,
            "behavior_changed": self.behavior_changed,
            "calls": {"a": self.calls_a, "b": self.calls_b},
            "tool_sequence": [{"mark": m, "name": n} for m, n in self.tool_sequence],
            "tool_sequence_changed": self.tool_sequence_changed,
            "input_changes": self.input_changes,
            "stop_reasons": {"a": self.stop_reasons_a, "b": self.stop_reasons_b},
            "models": {"a": self.models_a, "b": self.models_b},
            "errors": {"a": self.errors_a, "b": self.errors_b},
            "usage": {"a": self.usage_a, "b": self.usage_b},
            "truncated": self.truncated,
        }

    def render(self) -> str:
        out = [f"a: {self.label_a}", f"b: {self.label_b}", ""]
        if self.truncated:
            out.extend(["warning: a run file ends in a torn line; it died mid-write", ""])

        status = "CHANGED" if self.calls_a != self.calls_b else "same"
        out.append(f"api calls: {status} ({self.calls_a} -> {self.calls_b})")

        out.append(f"tool sequence: {'CHANGED' if self.tool_sequence_changed else 'same'}")
        out.extend(f"  {mark} {name}" for mark, name in self.tool_sequence)

        out.append(f"tool inputs: {'CHANGED' if self.input_changes else 'same'}")
        out.extend(
            f"  {c['tool']} {c['path']}: {c['a']!r} -> {c['b']!r}" for c in self.input_changes
        )

        same = self.stop_reasons_a == self.stop_reasons_b
        out.append(f"stop reasons: {'same' if same else 'CHANGED'}")
        if not same:
            out.append(f"  a: {self.stop_reasons_a}")
            out.append(f"  b: {self.stop_reasons_b}")

        if self.models_a != self.models_b:
            out.append(f"models: CHANGED ({self.models_a} -> {self.models_b})")
        if self.errors_a != self.errors_b:
            out.append(f"errors: CHANGED ({self.errors_a} -> {self.errors_b})")

        out.append("token usage:")
        for name in USAGE_FIELDS:
            before, after = self.usage_a.get(name, 0), self.usage_b.get(name, 0)
            if before or after:
                out.append(f"  {name:28} {before:>8} -> {after:>8}  ({after - before:+d})")
        increase = self.usage_increase_pct()
        if increase is not None:
            out.append(f"  {'total change':28} {increase:+.1f}%")

        out.append("")
        out.append("result: " + ("BEHAVIOR CHANGED" if self.behavior_changed else "same behavior"))
        return "\n".join(out)


def _describe(run: Run, fallback: str) -> str:
    return run.label or fallback


def diff_runs(a: Run, b: Run, *, name_a: str = "a", name_b: str = "b") -> RunDiff:
    tools_a, tools_b = tool_calls(a), tool_calls(b)
    names_a = [str(tool["name"]) for tool in tools_a]
    names_b = [str(tool["name"]) for tool in tools_b]

    sequence: list[tuple[str, str]] = []
    input_changes: list[dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(a=names_a, b=names_b, autojunk=False)
    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        if tag == "equal":
            for offset, name in enumerate(names_a[a1:a2]):
                sequence.append((" ", name))
                for change in _field_changes(
                    tools_a[a1 + offset]["input"], tools_b[b1 + offset]["input"]
                ):
                    input_changes.append({"tool": name, **change})
        else:
            sequence.extend(("-", name) for name in names_a[a1:a2])
            sequence.extend(("+", name) for name in names_b[b1:b2])

    return RunDiff(
        label_a=_describe(a, name_a),
        label_b=_describe(b, name_b),
        calls_a=len(a.calls),
        calls_b=len(b.calls),
        tool_sequence=sequence,
        input_changes=input_changes,
        stop_reasons_a=[call.stop_reason for call in a.calls],
        stop_reasons_b=[call.stop_reason for call in b.calls],
        models_a=sorted({str(call.request.get("model")) for call in a.calls}),
        models_b=sorted({str(call.request.get("model")) for call in b.calls}),
        errors_a=[str((call.error or {}).get("type")) for call in a.calls if call.error],
        errors_b=[str((call.error or {}).get("type")) for call in b.calls if call.error],
        usage_a=total_usage(a),
        usage_b=total_usage(b),
        truncated=a.truncated or b.truncated,
    )
