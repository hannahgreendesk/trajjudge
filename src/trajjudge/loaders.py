from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import Message, ToolCall, Trajectory


def _as_tool_calls(raw: Any) -> list[ToolCall]:
    out: list[ToolCall] = []
    if not raw:
        return out
    for i, tc in enumerate(raw):
        if not isinstance(tc, dict):
            continue
        # OpenAI chat style
        if "function" in tc and isinstance(tc["function"], dict):
            fn = tc["function"]
            out.append(
                ToolCall(
                    id=str(tc.get("id") or f"call_{i}"),
                    name=str(fn.get("name") or ""),
                    arguments=fn.get("arguments", ""),
                )
            )
            continue
        # Anthropic / simplified style
        name = tc.get("name") or tc.get("tool_name") or ""
        args = tc.get("arguments") or tc.get("input") or tc.get("args") or {}
        out.append(
            ToolCall(
                id=str(tc.get("id") or f"call_{i}"),
                name=str(name),
                arguments=args if not isinstance(args, str) else args,
            )
        )
    return out


def message_from_dict(d: dict[str, Any]) -> Message:
    role = str(d.get("role") or "unknown")
    content = d.get("content")
    if isinstance(content, list):
        # Anthropic content blocks → flatten text
        parts = []
        tool_calls = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(str(block.get("text") or ""))
            elif btype in ("tool_use", "function_call"):
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "arguments": block.get("input") or block.get("arguments") or {},
                    }
                )
            elif btype == "tool_result":
                return Message(
                    role="tool",
                    content=str(block.get("content") or ""),
                    tool_call_id=str(block.get("tool_use_id") or block.get("id") or "")
                    or None,
                    raw=d,
                )
        return Message(
            role=role,
            content="\n".join(parts) if parts else None,
            tool_calls=_as_tool_calls(tool_calls or d.get("tool_calls")),
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            raw=d,
        )
    return Message(
        role=role,
        content=None if content is None else str(content),
        tool_calls=_as_tool_calls(d.get("tool_calls")),
        tool_call_id=d.get("tool_call_id"),
        name=d.get("name"),
        raw=d,
    )


def trajectory_from_obj(obj: dict[str, Any], *, default_id: str = "traj") -> Trajectory:
    tid = str(obj.get("id") or obj.get("run_id") or default_id)
    messages_raw = obj.get("messages") or obj.get("trajectory") or obj.get("trace") or []
    if not isinstance(messages_raw, list):
        raise ValueError(f"trajectory {tid}: messages must be a list")
    messages = [message_from_dict(m) for m in messages_raw if isinstance(m, dict)]
    meta = {k: v for k, v in obj.items() if k not in ("messages", "trajectory", "trace")}
    return Trajectory(id=tid, messages=messages, meta=meta)


def load_jsonl(path: Path | str) -> list[Trajectory]:
    path = Path(path)
    out: list[Trajectory] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                # bare message list
                out.append(
                    Trajectory(
                        id=f"{path.stem}-{i}",
                        messages=[message_from_dict(m) for m in obj if isinstance(m, dict)],
                    )
                )
            elif isinstance(obj, dict):
                out.append(trajectory_from_obj(obj, default_id=f"{path.stem}-{i}"))
            else:
                raise ValueError(f"{path}:{i}: expected object or array")
    return out


def load_json(path: Path | str) -> list[Trajectory]:
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and "role" in obj[0]:
            return [
                Trajectory(
                    id=path.stem,
                    messages=[message_from_dict(m) for m in obj if isinstance(m, dict)],
                )
            ]
        return [
            trajectory_from_obj(x, default_id=f"{path.stem}-{i}")
            for i, x in enumerate(obj, 1)
            if isinstance(x, dict)
        ]
    if isinstance(obj, dict):
        if "messages" in obj or "trajectory" in obj or "trace" in obj:
            return [trajectory_from_obj(obj, default_id=path.stem)]
        if "trajectories" in obj and isinstance(obj["trajectories"], list):
            return [
                trajectory_from_obj(x, default_id=f"{path.stem}-{i}")
                for i, x in enumerate(obj["trajectories"], 1)
                if isinstance(x, dict)
            ]
    raise ValueError(f"unsupported JSON shape in {path}")


def load_trajectories(path: Path | str) -> list[Trajectory]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    if path.suffix.lower() == ".json":
        return load_json(path)
    # try jsonl first
    try:
        return load_jsonl(path)
    except json.JSONDecodeError:
        return load_json(path)


def iter_example_paths(root: Path | str) -> Iterable[Path]:
    root = Path(root)
    for p in sorted(root.glob("*.jsonl")):
        yield p
    for p in sorted(root.glob("*.json")):
        yield p
