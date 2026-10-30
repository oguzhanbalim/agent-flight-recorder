# agent-flight-recorder

A black box for agents: record every Messages API call, replay the run with no network, and
diff two runs so a behavior regression shows up as a diff.

When an agent starts behaving differently after a prompt, model or tool change, the wrong
answer is usually the last thing to surface. This records what the agent actually did, then
compares runs on behavior: which tools, with what inputs, why it stopped, and at what token cost.

Every response in this README comes from a scripted fake client, including the token counts.
No model has been run through this tool.

## Quickstart

```bash
pip install -e ".[dev]"
python examples/demo.py runs
```

```
recorded runs/baseline.jsonl and runs/regressed.jsonl
replay of baseline, no network: 'Refunded 40 on ORD-1042.'
replay after changing a tool's output:
  call 1 diverged at $.messages[2].content[0].content: recorded '{"order_id": "ORD-1042", "total": 40, "status": "delivered"}', got '{"id": "ORD-1042", "total_cents": 4000}'
```

Replay caught the tool change because the agent's next request no longer matched what was
recorded. Now compare a good run with a bad one:

```
$ flight-recorder diff runs/baseline.jsonl runs/regressed.jsonl
a: baseline
b: regressed

api calls: same (3 -> 3)
tool sequence: CHANGED
  - lookup_order
    issue_refund
  + issue_refund
tool inputs: CHANGED
  issue_refund $.amount: 40 -> 400
stop reasons: CHANGED
  a: ['tool_use', 'tool_use', 'end_turn']
  b: ['tool_use', 'tool_use', 'max_tokens']
token usage:
  input_tokens                     1582 ->     1612  (+30)
  output_tokens                     141 ->     2173  (+2032)
  total change                 +119.7%

result: BEHAVIOR CHANGED
```

The exit status is 1 when behavior changed, so this can gate a CI job. Add
`--max-usage-increase 20` to also fail on token growth, and `--json` for machine output.

```
$ flight-recorder show runs/regressed.jsonl
run cccec22cd7dc (regressed): 3 call(s)
#0 stop=tool_use in=412 out=64
    tool issue_refund {"amount": 400, "order_id": "ORD-1042"}
#1 stop=tool_use in=540 out=61
    tool issue_refund {"amount": 400, "order_id": "ORD-1042"}
#2 stop=max_tokens in=660 out=2048
```

## Use it

Wrap any client with a sync `messages.create`, including `anthropic.Anthropic()`:

```python
from flight_recorder import RecordingClient

with RecordingClient(anthropic.Anthropic(), "run.jsonl", label="after prompt v2") as client:
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": "..."}],
    )
```

The caller gets the SDK's own response object back. Replay it later, with no key and no network:

```python
from flight_recorder import ReplayClient

client = ReplayClient("run.jsonl")
run_my_agent(client)  # Raises ReplayDivergence at the first request that differs.
client.assert_exhausted()  # Raises if the agent stopped early.
```

Or from the command line, with a `module:function` that takes a client:

```
$ PYTHONPATH=examples flight-recorder replay runs/baseline.jsonl --agent refund_agent:run_agent
replayed 3 call(s) with no divergence
```

Replay returns objects you can read like the SDK's (`block.text`, `response.usage.input_tokens`),
but they are recorded dicts, not SDK types. Code that uses `isinstance` on SDK types will notice.

If a request has a field that legitimately varies, pass `ignore={"metadata"}`.

## What it does

- **Records** each request, response or exception, and duration as one JSONL line, flushed and
  fsynced as it happens. If the process dies, every finished call is still there.
- **Keeps credentials out.** `extra_headers`, `extra_query` and `timeout` are never written, and
  the wrapper does not read the client's key. Prompts and responses are written in full, so treat
  the file as sensitive or pass `redact=` a function that scrubs each record.
- **Refuses what it can't record.** `stream=True`, async clients and other client surfaces such as
  `client.beta` raise instead of letting a call go unrecorded.
- **Refuses to overwrite** an existing run file.
- **Diffs** tool sequence, tool inputs by JSON path, stop reasons, models, errors and token usage.
  Block ids, timestamps and durations are ignored. Generated text is not compared, because the
  wording of a good answer can vary without anything being wrong.

## File format

```json
{"type": "run", "format": 1, "run_id": "...", "created_at": "...", "label": "baseline"}
{"type": "call", "seq": 0, "request": {}, "response": {}, "duration_ms": 12.3}
{"type": "call", "seq": 1, "request": {}, "error": {"type": "...", "message": "..."}, "duration_ms": 4.0}
```

A torn final line is tolerated and flagged. Damage anywhere else, or a gap in `seq`, is an error.

## Tests

```
$ pytest
53 passed in 0.48s
```

These are self-written tests. `tests/test_sdk.py` runs the real `anthropic` SDK (1.17.0) with a
local mock HTTP transport and hand-written response bodies, which shows that SDK objects
round-trip through record and replay. It never reaches the API.

## Limits

- Not run against the live API. There was no API key where this was built.
- Sync, non-streaming `messages.create` only.
- Replay checks requests in order, so concurrent calls from threads may diverge even when nothing
  changed.
- Replay is only as deterministic as the agent. Tool code still runs during a replay. A tool that
  reads the clock or the network will show up as a divergence, which is usually what you want to
  know, but you may need to stub it.
- Tool execution is not recorded directly. It is visible only through the `tool_result` blocks in
  later requests.
- The sequence diff aligns tools by name with `difflib`. For a heavily changed run the alignment
  is reasonable but not unique, so inputs are only compared for tools it pairs up.
- The log is a plain file with no tamper protection.
- Tested locally on Python 3.12 only. CI is configured for 3.11 and 3.12 but has not run, since
  this repository has not been pushed.

Not affiliated with Anthropic.

## License

MIT
