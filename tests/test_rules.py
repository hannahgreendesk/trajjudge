from trajjudge.loaders import trajectory_from_obj
from trajjudge.rules import (
    rule_orphan_tool_calls,
    rule_secret_leakage,
    rule_tool_loop,
)
from trajjudge.scoring import score_trajectory


def test_detects_tool_loop():
    traj = trajectory_from_obj(
        {
            "id": "t",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"q":"x"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "1", "content": "none"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "2",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"q":"x"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "2", "content": "none"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "3",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"q":"x"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "3", "content": "none"},
            ],
        }
    )
    findings = rule_tool_loop(traj)
    assert any(f.rule_id == "tool_loop" for f in findings)


def test_orphan_tool_call():
    traj = trajectory_from_obj(
        {
            "id": "t",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "missing",
                            "type": "function",
                            "function": {"name": "x", "arguments": "{}"},
                        }
                    ],
                }
            ],
        }
    )
    assert rule_orphan_tool_calls(traj)


def test_secret_leak_and_score():
    traj = trajectory_from_obj(
        {
            "id": "leak",
            "messages": [
                {
                    "role": "assistant",
                    "content": "key=ghp_abcdefghijklmnopqrstuvwxyz0123456789",
                }
            ],
        }
    )
    assert rule_secret_leakage(traj)
    report = score_trajectory(traj)
    assert report.passed is False
    assert report.score < 70
