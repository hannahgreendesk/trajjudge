# TrajJudge

**Catch broken agent runs before your users do.**

TrajJudge scores LLM **agent trajectories** — the full message + tool-call trace — for the failure modes that quietly burn tokens and trust: loops, orphan tools, secret leaks, and thrashy tool ping-pong.

Rule checks are deterministic and offline. An optional LLM judge is there when you want a second opinion.

```bash
pip install -e .
trajjudge examples/loop_run.jsonl
```

---

## Why this exists

Most “evals” judge the **final answer**. Agents fail in the **middle**:

| What goes wrong | What you see in prod |
| --- | --- |
| Same tool + same args, over and over | Infinite retry / empty progress |
| `tool_calls` with no tool result | Hangs, half-finished plans |
| Secrets echoed in assistant text | Key rotation day |
| Search ↔ read ↔ search thrash | Huge bills, tiny outcomes |

TrajJudge reads the transcript and fails the run in CI — same spirit as a linter, for agent traces.

---

## Features

- **Deterministic rules** — loops, orphan calls/results, secret patterns, empty finals, tool storms, A↔B thrash
- **OpenAI-style JSONL** out of the box (also tolerates common Anthropic-ish shapes)
- **0–100 score + pass/fail** — threshold configurable for CI
- **Markdown + JSON reports** — human-readable and machine-readable
- **Optional LLM-as-judge** — any OpenAI-compatible `/v1/chat/completions` endpoint
- **Zero cloud required** for the default rule path

---

## Quick start

```bash
git clone https://github.com/hannahgreendesk/trajjudge.git
cd trajjudge
python -m pip install -e ".[dev]"

# healthy run → PASS
trajjudge examples/good_run.jsonl

# stuck loop → FAIL
trajjudge examples/loop_run.jsonl --format both

# leaky assistant → FAIL
trajjudge examples/leak_run.jsonl --json-out report.json
```

Exit code `1` means at least one trajectory failed the score threshold (default **70**).

---

## Input format

One JSON object per line (`.jsonl`), OpenAI chat-completions shaped:

```json
{
  "id": "run-42",
  "messages": [
    {"role": "user", "content": "…"},
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {"name": "search", "arguments": "{\"q\":\"docs\"}"}
        }
      ]
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "…"},
    {"role": "assistant", "content": "Here’s what I found…"}
  ]
}
```

A single `.json` file with `messages` / `trajectories` also works.

---

## Built-in rules

| Rule ID | Severity | What it catches |
| --- | --- | --- |
| `tool_loop` | error | Identical tool name+args ≥ 3 times |
| `orphan_tool_call` | error | Assistant tool call never answered |
| `orphan_tool_result` | warn | Tool result with unknown `tool_call_id` |
| `secret_leakage` | error/warn | GitHub PAT, OpenAI `sk-`, AWS `AKIA…`, PEM keys, JWTs… |
| `empty_final_answer` | warn | Last assistant turn has no text and no tools |
| `tool_storm` | warn | More than 25 tool calls in one run |
| `tool_ping_pong` | warn | Alternating tool A↔B thrash pattern |

Scoring starts at **100**, subtracts for findings, small bonus for a clean final answer.

---

## Optional LLM judge

Rules are the default. When you want narrative judgment:

```bash
export OPENAI_API_KEY=sk-...
# or any compatible gateway:
# export OPENAI_BASE_URL=https://your-gateway/v1

trajjudge examples/loop_run.jsonl --llm-judge --model gpt-4o-mini
```

The judge returns `score`, `verdict`, `issues`, and a short `rationale` alongside the rule report. It is **never** required for CI gatekeeping.

---

## Python API

```python
from trajjudge import score_trajectory
from trajjudge.loaders import load_trajectories

for traj in load_trajectories("examples/loop_run.jsonl"):
    report = score_trajectory(traj, pass_threshold=70)
    print(report.trajectory_id, report.score, report.passed)
    for f in report.findings:
        print(" ", f.severity, f.rule_id, f.message)
```

---

## CI sketch

```yaml
- name: Judge agent traces
  run: |
    pip install -e .
    trajjudge artifacts/trajectories.jsonl --fail-on-findings --json-out trajjudge.json
```

Wire this after your agent harness dumps traces. Fail the job before merge — not after a customer ping.

---

## Design notes

- **Offline-first.** Network is only used if you opt into `--llm-judge`.
- **Trace-native.** We score the path, not a single string answer.
- **Boring on purpose.** Small rule set, clear IDs, stable JSON schema (`trajjudge.v1`).
- **Not a full harness.** Bring your own runner (OpenAI Agents, LangGraph, custom). TrajJudge grades the tape.

---

## Project layout

```text
src/trajjudge/     # library + CLI
examples/          # good / loop / leak fixtures
tests/             # rule unit tests
```

---

## Roadmap

- [ ] Pluggable custom rules via entry points
- [ ] OpenTelemetry / Langfuse span import
- [ ] HTML single-file report
- [ ] Benchmark pack: annotated “should fail / should pass” corpus

Ideas and PRs welcome.

---

## License

MIT © 2026 Hannah Green
