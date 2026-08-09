from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .judge import llm_judge
from .loaders import load_trajectories
from .report import to_json, to_markdown, write_reports
from .scoring import score_trajectory


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trajjudge",
        description="Score agent trajectories for loops, tool misuse, and secret leaks.",
    )
    p.add_argument("path", nargs="?", help="Trajectory .jsonl / .json file")
    p.add_argument("--version", action="version", version=f"trajjudge {__version__}")
    p.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Minimum score to pass (default: 70)",
    )
    p.add_argument(
        "--format",
        choices=("markdown", "json", "both"),
        default="markdown",
        help="Stdout report format",
    )
    p.add_argument("--json-out", type=Path, help="Write JSON report to this path")
    p.add_argument("--md-out", type=Path, help="Write Markdown report to this path")
    p.add_argument(
        "--llm-judge",
        action="store_true",
        help="Also call an OpenAI-compatible LLM judge (needs API key)",
    )
    p.add_argument("--model", help="Judge model id (default: gpt-4o-mini / env)")
    p.add_argument("--base-url", help="OpenAI-compatible API base URL")
    p.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Exit 1 if any error-severity finding exists (even if score passes)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.path:
        build_parser().print_help()
        return 2

    trajs = load_trajectories(args.path)
    if not trajs:
        print("No trajectories found.", file=sys.stderr)
        return 2

    reports = []
    for traj in trajs:
        report = score_trajectory(traj, pass_threshold=args.threshold)
        if args.llm_judge:
            try:
                report.llm_judge = llm_judge(
                    traj, model=args.model, base_url=args.base_url
                )
            except Exception as exc:  # noqa: BLE001
                print(f"llm-judge failed for {traj.id}: {exc}", file=sys.stderr)
                return 2
        reports.append(report)

    if args.format in ("markdown", "both"):
        sys.stdout.write(to_markdown(reports))
    if args.format == "json":
        sys.stdout.write(to_json(reports))
    elif args.format == "both":
        sys.stdout.write("\n")
        sys.stdout.write(to_json(reports))

    write_reports(reports, json_path=args.json_out, md_path=args.md_out)

    failed = [r for r in reports if not r.passed]
    if failed:
        return 1
    if args.fail_on_findings:
        if any(f.severity == "error" for r in reports for f in r.findings):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
