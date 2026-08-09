from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str | dict[str, Any]


@dataclass
class Message:
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """One agent run as an ordered message list."""

    id: str
    messages: list[Message]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Finding:
    rule_id: str
    severity: str  # info | warn | error
    message: str
    turn: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "turn": self.turn,
            "evidence": self.evidence,
        }
