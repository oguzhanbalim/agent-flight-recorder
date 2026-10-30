"""Command line: show a run, diff two runs, replay a run through agent code."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Sequence
from typing import Any

from flight_recorder.diff import TOOL_BLOCK_TYPES, diff_runs
from flight_recorder.log import Run, RunFileError, read_run
from flight_recorder.replay import RecordedAPIError, ReplayClient, ReplayDivergence


def render_run(run: Run) -> str:
    title = f"run {run.run_id[:12]}"
    if run.label:
        title += f" ({run.label})"
    lines = [f"{title}: {len(run.calls)} call(s)"]
    if run.truncated:
        lines.append("warning: last line is torn; the process probably died mid-write")
    for call in run.calls:
        usage = f"in={call.usage.get('input_tokens', 0)} out={call.usage.get('output_tokens', 0)}"
        if call.error:
            lines.append(f"#{call.seq} ERROR {call.error.get('type')}: {call.error.get('message')}")
            continue
        lines.append(f"#{call.seq} stop={call.stop_reason} {usage}")
        for block in call.content:
            if block.get("type") in TOOL_BLOCK_TYPES:
                arguments = json.dumps(block.get("input"), sort_keys=True)
                lines.append(f"    tool {block.get('name')} {arguments}")
    return "\n".join(lines)


def _load_agent(spec: str) -> Callable[[Any], Any]:
    module_name, _, attribute = spec.partition(":")
    if not attribute:
        raise ValueError(f"expected module:function, got {spec!r}")
    sys.path.insert(0, "")
    return getattr(importlib.import_module(module_name), attribute)


def _replay(path: str, agent_spec: str, ignore: Sequence[str]) -> int:
    agent = _load_agent(agent_spec)
    client = ReplayClient(path, ignore=set(ignore))
    try:
        agent(client)
        client.assert_exhausted()
    except ReplayDivergence as exc:
        print(f"diverged: {exc}")
        return 1
    except RecordedAPIError as exc:
        print(f"replayed recorded failure: {exc}")
        return 0
    print(f"replayed {len(client.run.calls)} call(s) with no divergence")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flight-recorder", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    show = sub.add_parser("show", help="print a one-screen summary of a run")
    show.add_argument("run")

    diff = sub.add_parser("diff", help="compare two runs; exit 1 if behavior changed")
    diff.add_argument("a")
    diff.add_argument("b")
    diff.add_argument("--json", action="store_true", help="print JSON instead of text")
    diff.add_argument(
        "--max-usage-increase",
        type=float,
        metavar="PCT",
        help="also exit 1 if total tokens grew by more than PCT percent",
    )

    replay = sub.add_parser("replay", help="run agent code against a recording, no network")
    replay.add_argument("run")
    replay.add_argument("--agent", required=True, help="module:function taking a client")
    replay.add_argument("--ignore", action="append", default=[], metavar="KEY")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.action == "show":
            print(render_run(read_run(args.run)))
            return 0
        if args.action == "replay":
            return _replay(args.run, args.agent, args.ignore)
        result = diff_runs(read_run(args.a), read_run(args.b), name_a=args.a, name_b=args.b)
    except (RunFileError, ValueError, ImportError, AttributeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2) if args.json else result.render())
    failed = result.behavior_changed
    if args.max_usage_increase is not None:
        increase = result.usage_increase_pct()
        failed = failed or (increase is not None and increase > args.max_usage_increase)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
