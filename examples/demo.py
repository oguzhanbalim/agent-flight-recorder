"""Record two scripted runs, replay one offline, and show what a regression looks like.

Every response here comes from a script, including the token counts. No model ran.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from refund_agent import run_agent, run_tool

from flight_recorder import RecordingClient, ReplayClient, ReplayDivergence, ScriptedClient
from flight_recorder.fake import message, text, tool_use

BASELINE = [
    message(
        [tool_use("toolu_01", "lookup_order", {"order_id": "ORD-1042"})],
        stop_reason="tool_use",
        input_tokens=412,
        output_tokens=58,
    ),
    message(
        [tool_use("toolu_02", "issue_refund", {"order_id": "ORD-1042", "amount": 40})],
        stop_reason="tool_use",
        input_tokens=530,
        output_tokens=61,
    ),
    message([text("Refunded 40 on ORD-1042.")], input_tokens=640, output_tokens=22),
]

REGRESSED = [
    message(
        [tool_use("toolu_91", "issue_refund", {"order_id": "ORD-1042", "amount": 400})],
        stop_reason="tool_use",
        input_tokens=412,
        output_tokens=64,
    ),
    message(
        [tool_use("toolu_92", "issue_refund", {"order_id": "ORD-1042", "amount": 400})],
        stop_reason="tool_use",
        input_tokens=540,
        output_tokens=61,
    ),
    message(
        [text("I have refunded the order and")],
        stop_reason="max_tokens",
        input_tokens=660,
        output_tokens=2048,
    ),
]


def changed_tool(name: str, tool_input: dict) -> str:
    """The same tool after someone changed its output format."""
    if name == "lookup_order":
        return json.dumps({"id": tool_input["order_id"], "total_cents": 4000})
    return run_tool(name, tool_input)


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline_path, regressed_path = out / "baseline.jsonl", out / "regressed.jsonl"
    for path in (baseline_path, regressed_path):
        path.unlink(missing_ok=True)

    with RecordingClient(ScriptedClient(BASELINE), baseline_path, label="baseline") as client:
        run_agent(client)
    with RecordingClient(ScriptedClient(REGRESSED), regressed_path, label="regressed") as client:
        run_agent(client)
    print(f"recorded {baseline_path} and {regressed_path}")

    replay = ReplayClient(baseline_path)
    final = run_agent(replay)
    replay.assert_exhausted()
    print(f"replay of baseline, no network: {final.content[0].text!r}")

    try:
        run_agent(ReplayClient(baseline_path), tool_runner=changed_tool)
    except ReplayDivergence as exc:
        print(f"replay after changing a tool's output:\n  {exc}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
