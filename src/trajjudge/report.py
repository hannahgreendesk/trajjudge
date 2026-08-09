from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .scoring import ScoreReport


def to_json(reports: Iterable[ScoreReport], *, indent: int = 2) -> str:
    payload = {
        "schema": "trajjudge.v1",
        "results": [r.to_dict() for r in reports],
    }
    return json.dumps(payload, indent=indent, ensure_ascii=False) + "\n"


def to_markdown(reports: list[ScoreReport]) -> str:
    lines = [
        "# TrajJudge report",
        "",
        f"Runs: **{len(reports)}** · "
        f"Passed: **{sum(1 for r in reports if r.passed)}** · "
        f"Failed: **{sum(1 for r in reports if not r.passed)}**",
        "",
    ]
    for r in reports:
        badge = "PASS" if r.passed else "FAIL"
        lines.append(f"## `{r.trajectory_id}` — {badge} ({r.score})")
        lines.append("")
        s = r.summary
        lines.append(
            f"- messages: {s.get('messages')} · tool_calls: {s.get('tool_calls')} · "
            f"errors: {s.get('errors')} · warnings: {s.get('warnings')}"
        )
        if r.llm_judge:
            lines.append(
                f"- llm_judge: {r.llm_judge.get('verdict')} "
                f"({r.llm_judge.get('score')}) — {r.llm_judge.get('rationale', '')[:180]}"
            )
        if not r.findings:
            lines.append("- findings: none")
        else:
            lines.append("- findings:")
            for f in r.findings:
                turn = f"turn {f.turn}" if f.turn is not None else "run"
                lines.append(f"  - **{f.severity}** `{f.rule_id}` ({turn}): {f.message}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    reports: list[ScoreReport],
    *,
    json_path: Path | str | None = None,
    md_path: Path | str | None = None,
) -> None:
    if json_path:
        Path(json_path).write_text(to_json(reports), encoding="utf-8")
    if md_path:
        Path(md_path).write_text(to_markdown(reports), encoding="utf-8")
