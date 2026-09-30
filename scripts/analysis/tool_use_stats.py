"""The tool-use columns (rate, success rate) of stored agent trajectories.

Why this exists
---------------
The dual-loop deliverable is three numbers per arm: tool-call rate, tool
success rate, accuracy. ``arm_metrics.py`` computes accuracy and executed
tool-call totals but not tool *outcomes*; the reward bridge reads outcomes from
``tool_result`` events, and this script is the read-only, CPU-side twin of that
reading: same event kind, same ``ok`` flag, no rerun and no model.

The definitions are fixed here because "tool-call rate" is ambiguous:

* **executed tool call** -- a ``tool_call`` event in ``trajectory.events``. The
  strict envelope only records a call it actually ran; a refused attempt shows
  up as an ``invalid_action``, never here.
* **tool-call rate** -- share of tasks with at least one executed tool call.
* **calls per task** -- executed tool calls / tasks.
* **tool result** -- a ``tool_result`` event; a result is successful when
  ``payload.result.ok`` is ``True`` (the same flag ``reward_bridge`` counts for
  the R4 bonus).
* **tool success rate** -- successful results / all results. ``n/a`` when the
  arm produced no results at all: absent measurement, not a measured zero.

Every count is held to the run's own ``summary.json`` (the same authority
``arm_metrics.py`` uses): executed calls against ``tool_calls_total``, accuracy
against ``correct_count``. A mismatch is fatal rather than a footnote -- a table
that cannot be checked against the run is not evidence. Trajectories without a
sibling summary (a hand-made slice) are reported with the checks skipped and
marked as such, so a table can never be read as checked when it was not.

Usage:
    uv run python scripts/analysis/tool_use_stats.py \
        --arm base=artifacts/mvp/dual_loop_eval/base \
        --arm sft=artifacts/mvp/dual_loop_eval/sft \
        --out artifacts/results/dual-loop/tool-use.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class StatsError(RuntimeError):
    """A check failed; the caller prints it and exits 1."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _event_kind(event: dict[str, Any]) -> str | None:
    kind = event.get("kind", event.get("type"))
    return str(kind) if kind is not None else None


def tool_stats(arm_dir: Path) -> dict[str, Any]:
    """The tool-use row for one arm directory, checked against its summary."""
    trajectories = arm_dir / "trajectories.jsonl"
    if not trajectories.is_file():
        raise StatsError(f"{trajectories} is missing: the run did not finish")

    tasks = 0
    correct = 0
    tasks_with_call = 0
    executed_calls = 0
    results = 0
    successes = 0
    for record in _read_jsonl(trajectories):
        tasks += 1
        if (record.get("verdict") or {}).get("status") == "correct":
            correct += 1
        calls_here = 0
        for event in record["trajectory"]["events"]:
            kind = _event_kind(event)
            if kind == "tool_call":
                calls_here += 1
            elif kind == "tool_result":
                results += 1
                result = (event.get("payload") or {}).get("result") or {}
                successes += int(result.get("ok") is True)
        executed_calls += calls_here
        tasks_with_call += int(calls_here > 0)
    if tasks == 0:
        raise StatsError(f"{arm_dir}: no trajectories found")

    checked = False
    summary_path = arm_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        for name, counted, recorded in (
            ("executed tool calls", executed_calls, summary["tool_calls_total"]),
            ("correct", correct, summary["correct_count"]),
        ):
            if counted != int(recorded):
                raise StatsError(
                    f"{arm_dir}: {name} counted {counted} from the trajectories, "
                    f"but summary.json records {recorded}: the table would not be "
                    "the run's own numbers"
                )
        checked = True

    return {
        "tasks": tasks,
        "correct": correct,
        "accuracy": correct / tasks,
        "tasks_with_executed_tool_call": tasks_with_call,
        "tool_call_rate": tasks_with_call / tasks,
        "executed_tool_calls": executed_calls,
        "calls_per_task": executed_calls / tasks,
        "tool_results": results,
        "tool_successes": successes,
        "tool_success_rate": (successes / results) if results else None,
        "checked_against_summary": checked,
    }


def parse_arm(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f"expected label=dir, got {value!r}")
    return label, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--arm",
        type=parse_arm,
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="an arm's merged run directory (repeatable, order preserved)",
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the table as JSON")
    return parser


def render_markdown(stats: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| arm | n | tool-call rate | calls/task | tool success rate | accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in stats.items():
        rate = f"{row['tool_success_rate']:.1%}" if row["tool_success_rate"] is not None else "n/a"
        lines.append(
            f"| `{label}` | {row['tasks']} | {row['tool_call_rate']:.1%} | "
            f"{row['calls_per_task']:.3f} | {rate} | {row['accuracy']:.1%} "
            f"({row['correct']}/{row['tasks']}) |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats: dict[str, dict[str, Any]] = {}
    for label, arm_dir in args.arm:
        try:
            stats[label] = tool_stats(arm_dir)
        except StatsError as error:
            print(f"TOOL-STATS FAIL: {error}")
            return 1
    payload = {"arms": stats}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(render_markdown(stats), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
