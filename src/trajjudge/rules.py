from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Callable

from .models import Finding, Trajectory

RuleFn = Callable[[Trajectory], list[Finding]]

# Common secret-ish patterns (high-signal, intentionally narrow).
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("private_key_pem", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("bearer_jwt", re.compile(r"\bBearer\s+eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")),
]


def _args_fingerprint(name: str, arguments: object) -> str:
    if isinstance(arguments, str):
        payload = arguments.strip()
    else:
        try:
            payload = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        except TypeError:
            payload = str(arguments)
    digest = hashlib.sha1(f"{name}|{payload}".encode("utf-8")).hexdigest()[:12]
    return f"{name}:{digest}"


def rule_tool_loop(traj: Trajectory, *, threshold: int = 3) -> list[Finding]:
    """Repeated identical tool calls (same name + args) suggest a stuck agent."""
    findings: list[Finding] = []
    fps: list[str] = []
    turns: list[int] = []
    for i, msg in enumerate(traj.messages):
        if msg.role != "assistant":
            continue
        for tc in msg.tool_calls:
            fp = _args_fingerprint(tc.name, tc.arguments)
            fps.append(fp)
            turns.append(i)
    counts = Counter(fps)
    for fp, n in counts.items():
        if n >= threshold:
            name = fp.split(":", 1)[0]
            idx = next(t for t, f in zip(turns, fps) if f == fp)
            findings.append(
                Finding(
                    rule_id="tool_loop",
                    severity="error",
                    message=f"Tool `{name}` called {n}× with identical arguments (loop).",
                    turn=idx,
                    evidence={"fingerprint": fp, "count": n, "threshold": threshold},
                )
            )
    return findings


def rule_orphan_tool_calls(traj: Trajectory) -> list[Finding]:
    """Assistant tool_calls without a matching tool result."""
    pending: dict[str, int] = {}
    for i, msg in enumerate(traj.messages):
        if msg.role == "assistant":
            for tc in msg.tool_calls:
                if tc.id:
                    pending[tc.id] = i
        elif msg.role == "tool" and msg.tool_call_id:
            pending.pop(msg.tool_call_id, None)
    findings = []
    for call_id, turn in pending.items():
        findings.append(
            Finding(
                rule_id="orphan_tool_call",
                severity="error",
                message=f"Tool call `{call_id}` never received a tool result.",
                turn=turn,
                evidence={"tool_call_id": call_id},
            )
        )
    return findings


def rule_orphan_tool_results(traj: Trajectory) -> list[Finding]:
    """Tool results that do not match any prior assistant tool_call id."""
    known: set[str] = set()
    findings: list[Finding] = []
    for i, msg in enumerate(traj.messages):
        if msg.role == "assistant":
            for tc in msg.tool_calls:
                if tc.id:
                    known.add(tc.id)
        elif msg.role == "tool":
            tid = msg.tool_call_id or ""
            if not tid or tid not in known:
                findings.append(
                    Finding(
                        rule_id="orphan_tool_result",
                        severity="warn",
                        message="Tool result has no matching prior tool_call id.",
                        turn=i,
                        evidence={"tool_call_id": tid or None},
                    )
                )
    return findings


def rule_secret_leakage(traj: Trajectory) -> list[Finding]:
    """Flag high-signal secrets appearing in assistant or tool payloads."""
    findings: list[Finding] = []
    for i, msg in enumerate(traj.messages):
        if msg.role not in ("assistant", "tool", "user"):
            continue
        blobs: list[str] = []
        if msg.content:
            blobs.append(msg.content)
        for tc in msg.tool_calls:
            blobs.append(tc.name)
            if isinstance(tc.arguments, str):
                blobs.append(tc.arguments)
            else:
                try:
                    blobs.append(json.dumps(tc.arguments, ensure_ascii=False))
                except TypeError:
                    blobs.append(str(tc.arguments))
        text = "\n".join(blobs)
        for label, pat in _SECRET_PATTERNS:
            if pat.search(text):
                sev = "error" if msg.role == "assistant" else "warn"
                findings.append(
                    Finding(
                        rule_id="secret_leakage",
                        severity=sev,
                        message=f"Possible secret (`{label}`) in {msg.role} message.",
                        turn=i,
                        evidence={"pattern": label, "role": msg.role},
                    )
                )
    return findings


def rule_empty_final_answer(traj: Trajectory) -> list[Finding]:
    """Last assistant message has no content and no tool_calls."""
    for i in range(len(traj.messages) - 1, -1, -1):
        msg = traj.messages[i]
        if msg.role != "assistant":
            continue
        content = (msg.content or "").strip()
        if not content and not msg.tool_calls:
            return [
                Finding(
                    rule_id="empty_final_answer",
                    severity="warn",
                    message="Final assistant message is empty (no text, no tools).",
                    turn=i,
                )
            ]
        return []
    return [
        Finding(
            rule_id="empty_final_answer",
            severity="warn",
            message="Trajectory has no assistant messages.",
            turn=None,
        )
    ]


def rule_tool_storm(traj: Trajectory, *, max_calls: int = 25) -> list[Finding]:
    """Too many tool calls in a single run — often thrashing."""
    n = sum(len(m.tool_calls) for m in traj.messages if m.role == "assistant")
    if n > max_calls:
        return [
            Finding(
                rule_id="tool_storm",
                severity="warn",
                message=f"Trajectory issued {n} tool calls (limit {max_calls}).",
                evidence={"tool_calls": n, "max_calls": max_calls},
            )
        ]
    return []


def rule_ping_pong(traj: Trajectory, *, window: int = 6) -> list[Finding]:
    """Alternating A/B tool names repeatedly (search↔read thrash)."""
    names: list[str] = []
    for msg in traj.messages:
        if msg.role != "assistant":
            continue
        for tc in msg.tool_calls:
            if tc.name:
                names.append(tc.name)
    if len(names) < window:
        return []
    # look for ababab pattern in a sliding window
    for start in range(0, len(names) - window + 1):
        chunk = names[start : start + window]
        a, b = chunk[0], chunk[1]
        if a == b:
            continue
        if chunk == [a, b] * (window // 2):
            return [
                Finding(
                    rule_id="tool_ping_pong",
                    severity="warn",
                    message=f"Alternating tools `{a}` ↔ `{b}` detected (thrash).",
                    evidence={"pattern": chunk},
                )
            ]
    return []


DEFAULT_RULES: list[RuleFn] = [
    rule_tool_loop,
    rule_orphan_tool_calls,
    rule_orphan_tool_results,
    rule_secret_leakage,
    rule_empty_final_answer,
    rule_tool_storm,
    rule_ping_pong,
]


def run_rules(traj: Trajectory, rules: list[RuleFn] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    for rule in rules or DEFAULT_RULES:
        findings.extend(rule(traj))
    return findings
