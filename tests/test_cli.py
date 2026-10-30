from __future__ import annotations

import json

import pytest
from demo import BASELINE, REGRESSED
from refund_agent import run_agent

from flight_recorder import RecordingClient, ScriptedClient
from flight_recorder.cli import main


@pytest.fixture
def runs(tmp_path):
    baseline, regressed, again = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    with RecordingClient(ScriptedClient(BASELINE), baseline, label="baseline") as client:
        run_agent(client)
    with RecordingClient(ScriptedClient(REGRESSED), regressed, label="regressed") as client:
        run_agent(client)
    with RecordingClient(ScriptedClient(BASELINE), again, label="again") as client:
        run_agent(client)
    return str(baseline), str(regressed), str(again)


def test_show(runs, capsys):
    assert main(["show", runs[1]]) == 0
    out = capsys.readouterr().out
    assert "regressed" in out
    assert "stop=max_tokens" in out
    assert 'tool issue_refund {"amount": 400' in out


def test_diff_exit_status(runs, capsys):
    baseline, regressed, again = runs
    assert main(["diff", baseline, again]) == 0
    assert "same behavior" in capsys.readouterr().out
    assert main(["diff", baseline, regressed]) == 1
    assert "BEHAVIOR CHANGED" in capsys.readouterr().out


def test_diff_json(runs, capsys):
    assert main(["diff", "--json", runs[0], runs[1]]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["tool_sequence_changed"] is True
    assert data["stop_reasons"]["b"][-1] == "max_tokens"


def test_usage_gate(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    for path, tokens in ((a, 100), (b, 150)):
        response = {
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }
        with RecordingClient(ScriptedClient([response]), path) as client:
            client.messages.create(model="m", max_tokens=1, messages=[])
    assert main(["diff", str(a), str(b)]) == 0
    assert main(["diff", str(a), str(b), "--max-usage-increase", "60"]) == 0
    assert main(["diff", str(a), str(b), "--max-usage-increase", "40"]) == 1


def test_replay_ok_and_diverged(runs, tmp_path, capsys, monkeypatch):
    agent = tmp_path / "my_agent.py"
    agent.write_text(
        "from refund_agent import run_agent\n"
        "def good(client):\n    return run_agent(client)\n"
        "def bad(client):\n    return run_agent(client, tool_runner=lambda n, i: 'changed')\n"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["replay", runs[0], "--agent", "my_agent:good"]) == 0
    assert "no divergence" in capsys.readouterr().out
    assert main(["replay", runs[0], "--agent", "my_agent:bad"]) == 1
    assert "diverged" in capsys.readouterr().out


def test_bad_inputs_exit_2(tmp_path, capsys):
    assert main(["show", str(tmp_path / "missing.jsonl")]) == 2
    assert main(["replay", str(tmp_path / "x"), "--agent", "no_colon"]) == 2
