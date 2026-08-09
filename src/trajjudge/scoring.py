from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Finding, Trajectory
from .rules import DEFAULT_RULES, RuleFn, run_rules

_SEVERITY_PENALTY = {"error": 25, "warn": 10, "info": 2}


@dataclass
class ScoreReport:
    trajectory_id: str
    score: float
    passed: bool
    findings: list[Finding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    llm_judge: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "score": self.score,
            "passed": self.passed,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "llm_judge": self.llm_judge,
        }


def _clamp(n: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, n))


def score_from_findings(
    traj: Trajectory,
    findings: list[Finding],
    *,
    pass_threshold: float = 70.0,
) -> ScoreReport:
    penalty = 0
    by_sev = {"error": 0, "warn": 0, "info": 0}
    for f in findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
        penalty += _SEVERITY_PENALTY.get(f.severity, 5)

    # light reward for finishing with a non-empty assistant answer
    bonus = 0
    for msg in reversed(traj.messages):
        if msg.role == "assistant":
            if (msg.content or "").strip() and not msg.tool_calls:
                bonus = 5
            break

    score = _clamp(100 - penalty + bonus)
    tool_calls = sum(len(m.tool_calls) for m in traj.messages if m.role == "assistant")
    return ScoreReport(
        trajectory_id=traj.id,
        score=round(score, 1),
        passed=score >= pass_threshold,
        findings=findings,
        summary={
            "messages": len(traj.messages),
            "tool_calls": tool_calls,
            "errors": by_sev.get("error", 0),
            "warnings": by_sev.get("warn", 0),
            "infos": by_sev.get("info", 0),
            "pass_threshold": pass_threshold,
        },
    )


def score_trajectory(
    traj: Trajectory,
    *,
    rules: list[RuleFn] | None = None,
    pass_threshold: float = 70.0,
) -> ScoreReport:
    findings = run_rules(traj, rules or DEFAULT_RULES)
    return score_from_findings(traj, findings, pass_threshold=pass_threshold)
