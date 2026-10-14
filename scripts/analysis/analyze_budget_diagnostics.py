"""Round-1 budget diagnostics analyzer (thesis).

Reads run_rollout_health.py output dirs (trajectories.jsonl + manifest.json)
and emits the three pre-registered tables -- docs/experiments/
thesis-round1-prereg.md:

  table_a.csv  budget effect, one row per run: accuracy (per-question macro
               and episode micro), mean calls, calls-per-episode shares
               0..4, termination-reason shares, mean generated tokens
  table_b.csv  within-question room, per (budget, task): correct n of 4,
               call range among correct samples, among wrong samples,
               failure-then-later-success presence (two operationalizations:
               tool-error recovery and invalid-action recovery)
  table_c.csv  reward visibility, per (budget, task): answer-reward
               variance, total-reward variance, call-cost variance,
               all-equal-reward flag, correct-but-different-cost flag
  tables.md / summary.json  the three tables plus group aggregates ready
               for the budget comparison

Aggregation rules (pre-registered): every inferential reading aggregates
per question -- four same-question samples are never four independent
questions, and Table B/C rows are per (budget, task), never pooled across
budgets. Any GRPO-style advantage recomputed offline (this script, or the
runner's own groups.jsonl) is a diagnostic value (重算诊断值), never the
trainer's actual advantage. Table C variances are population variances
(ddof=0), matching the runner's group_advantages convention.

The budget label of a run comes from its manifest's agent_config
(max_tool_calls); without a manifest the directory name is used.

Usage:
  python scripts/analysis/analyze_budget_diagnostics.py \
      artifacts/diagnostics/r1/sft_b0_pilot artifacts/diagnostics/r1/sft_b1_pilot \
      [more run dirs ...] [--out artifacts/diagnostics/r1/analysis]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

BUDGET_ORDER = {"b0": 0, "b1": 1, "b2": 2, "b4": 4}
TERMINATIONS = (
    "final",
    "max_steps",
    "max_tool_calls",
    "python_time_budget",
    "model_error",
    "infrastructure_error",
    "cancelled",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


def _budget_label(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        agent_config = _load_json(manifest_path).get("agent_config")
        if isinstance(agent_config, str):
            config_path = Path(agent_config)
            if not config_path.is_absolute():
                config_path = Path.cwd() / config_path
            if config_path.is_file():
                for line in config_path.read_text().splitlines():
                    key, sep, value = line.partition(":")
                    if sep and key.strip() == "max_tool_calls":
                        return f"b{value.strip()}"
    name = run_dir.name
    marker = name.find("b")
    while marker != -1:
        digits = ""
        for char in name[marker + 1 :]:
            if char.isdigit():
                digits += char
            else:
                break
        if digits and (marker == 0 or not name[marker - 1].isalnum()):
            return f"b{digits}"
        marker = name.find("b", marker + 1)
    return name


def _episode_fields(record: dict[str, Any]) -> dict[str, Any]:
    trajectory = record.get("trajectory") or {}
    usage = trajectory.get("usage") or {}
    breakdown = record.get("reward_breakdown") or {}
    components = breakdown.get("components") or {}
    verdict = record.get("verdict") or {}
    correct_component = components.get("correct")
    if correct_component is None:
        correct = 1.0 if verdict.get("status") == "correct" else 0.0
    else:
        correct = float(correct_component)
    total_reward = breakdown.get("total")
    if total_reward is None:
        total_reward = record.get("reward", 0.0)
    return {
        "task_id": record.get("task_id"),
        "sample_index": record.get("sample_index"),
        "correct": 1.0 if correct >= 0.5 else 0.0,
        "answer_reward": correct,
        "total_reward": float(total_reward),
        "tool_calls": int(usage.get("tool_calls", 0)),
        "python_seconds": float(usage.get("python_seconds", 0.0)),
        "generated_tokens": int(usage.get("generated_tokens", 0)),
        "invalid_actions": int(usage.get("invalid_actions", 0)),
        "termination": trajectory.get("termination_reason"),
        "recovered": _recovered_from_failure(trajectory.get("events") or []),
        "recovered_invalid": bool(
            correct >= 0.5
            and any(
                event.get("kind") == "invalid_action"
                for event in (trajectory.get("events") or [])
            )
        ),
    }


def _recovered_from_failure(events: list[dict[str, Any]]) -> bool:
    """True when a failed tool result is followed later by a successful one."""
    failed = False
    for event in events:
        if event.get("kind") != "tool_result":
            continue
        payload = event.get("payload") or {}
        result = payload.get("result") or {}
        if result.get("ok"):
            if failed:
                return True
        else:
            failed = True
    return False


def _variance(values: list[float]) -> float:
    """Population variance (ddof=0), matching the runner's group_advantages."""
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _call_range(values: list[int]) -> str:
    if not values:
        return ""
    return f"{min(values)}-{max(values)}"


def _load_run(run_dir: Path) -> dict[str, Any]:
    trajectories_path = run_dir / "trajectories.jsonl"
    if not trajectories_path.is_file():
        raise SystemExit(f"no trajectories.jsonl under {run_dir}")
    records = _load_jsonl(trajectories_path)
    episodes = [_episode_fields(record) for record in records]
    return {
        "dir": str(run_dir),
        "label": _budget_label(run_dir),
        "episodes": episodes,
    }


def _table_a(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda r: BUDGET_ORDER.get(r["label"], 99)):
        episodes = run["episodes"]
        by_task: dict[str, list[dict[str, Any]]] = {}
        for episode in episodes:
            by_task.setdefault(str(episode["task_id"]), []).append(episode)
        macro_accuracy = statistics.fmean(
            statistics.fmean(e["correct"] for e in group) for group in by_task.values()
        )
        micro_accuracy = statistics.fmean(e["correct"] for e in episodes)
        call_shares = {
            f"share_calls_{count}": sum(
                1 for e in episodes if e["tool_calls"] == count
            )
            / len(episodes)
            for count in range(5)
        }
        termination_shares = {
            f"term_{name}": sum(1 for e in episodes if e["termination"] == name)
            / len(episodes)
            for name in TERMINATIONS
        }
        rows.append(
            {
                "budget": run["label"],
                "questions": len(by_task),
                "episodes": len(episodes),
                "accuracy_macro_by_question": round(macro_accuracy, 4),
                "accuracy_micro_by_episode": round(micro_accuracy, 4),
                "mean_tool_calls": round(
                    statistics.fmean(e["tool_calls"] for e in episodes), 3
                ),
                **{key: round(value, 4) for key, value in call_shares.items()},
                "mean_generated_tokens": round(
                    statistics.fmean(e["generated_tokens"] for e in episodes), 1
                ),
                "mean_python_seconds": round(
                    statistics.fmean(e["python_seconds"] for e in episodes), 3
                ),
                "mean_invalid_actions": round(
                    statistics.fmean(e["invalid_actions"] for e in episodes), 3
                ),
                **{key: round(value, 4) for key, value in termination_shares.items()},
            }
        )
    return rows


def _table_b(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda r: BUDGET_ORDER.get(r["label"], 99)):
        by_task: dict[str, list[dict[str, Any]]] = {}
        for episode in run["episodes"]:
            by_task.setdefault(str(episode["task_id"]), []).append(episode)
        for task_id in sorted(by_task):
            group = by_task[task_id]
            correct_calls = [e["tool_calls"] for e in group if e["correct"]]
            wrong_calls = [e["tool_calls"] for e in group if not e["correct"]]
            rows.append(
                {
                    "budget": run["label"],
                    "task_id": task_id,
                    "samples": len(group),
                    "correct_n": len(correct_calls),
                    "call_range_correct": _call_range(correct_calls),
                    "call_range_wrong": _call_range(wrong_calls),
                    "recovered_n": sum(1 for e in group if e["recovered"]),
                    "recovered_invalid_n": sum(
                        1 for e in group if e["recovered_invalid"]
                    ),
                }
            )
    return rows


def _table_c(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda r: BUDGET_ORDER.get(r["label"], 99)):
        by_task: dict[str, list[dict[str, Any]]] = {}
        for episode in run["episodes"]:
            by_task.setdefault(str(episode["task_id"]), []).append(episode)
        for task_id in sorted(by_task):
            group = by_task[task_id]
            total_rewards = [e["total_reward"] for e in group]
            answer_rewards = [e["answer_reward"] for e in group]
            calls = [float(e["tool_calls"]) for e in group]
            correct_group = [e for e in group if e["correct"]]
            correct_costs = {e["tool_calls"] for e in correct_group}
            rows.append(
                {
                    "budget": run["label"],
                    "task_id": task_id,
                    "answer_reward_var": round(_variance(answer_rewards), 6),
                    "total_reward_var": round(_variance(total_rewards), 6),
                    "call_cost_var": round(_variance(calls), 6),
                    "all_equal_reward": len({round(r, 9) for r in total_rewards}) == 1,
                    "correct_but_diff_cost": len(correct_costs) >= 2,
                }
            )
    return rows


def _group_summary(rows_a: list[dict[str, Any]], rows_c: list[dict[str, Any]]) -> None:
    print("\n== group-level summary (per budget) ==")
    labels = [row["budget"] for row in rows_a]
    for label in labels:
        groups = [row for row in rows_c if row["budget"] == label]
        if not groups:
            continue
        all_equal = sum(1 for row in groups if row["all_equal_reward"]) / len(groups)
        all_correct_equal = (
            statistics.fmean(row["answer_reward_var"] for row in groups) == 0.0
        )
        print(
            f"{label}: groups={len(groups)} "
            f"all_equal_reward={all_equal:.3f} "
            f"answer_var_all_zero={all_correct_equal}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(empty)"
    header = "| " + " | ".join(rows[0].keys()) + " |"
    divider = "|" + "|".join("---" for _ in rows[0]) + "|"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row.values()) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = [_load_run(run_dir) for run_dir in args.run_dirs]
    for run in runs:
        print(f"loaded {run['dir']} label={run['label']} episodes={len(run['episodes'])}")

    rows_a = _table_a(runs)
    rows_b = _table_b(runs)
    rows_c = _table_c(runs)

    out_dir = args.out or (Path(args.run_dirs[0]).parent / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "table_a.csv", rows_a)
    _write_csv(out_dir / "table_b.csv", rows_b)
    _write_csv(out_dir / "table_c.csv", rows_c)

    notes = (
        "Aggregation: table A accuracy is per-question macro (primary) and "
        "per-episode micro; shares are episode-level descriptive. Tables B/C "
        "are per (budget, task) -- never pooled across budgets, and four "
        "same-question samples are not four independent questions.\n"
        "Table B recovery has two operationalizations: recovered_n = a failed "
        "tool result followed later by a successful one; recovered_invalid_n "
        "= an episode with invalid-action steps that still ends correct.\n"
        "Any GRPO-style advantage recomputed offline (here or in the "
        "runner's groups.jsonl) is a diagnostic value (重算诊断值), not the "
        "trainer's actual advantage. Table C variances are population "
        "variances (ddof=0), matching the runner's group_advantages.\n"
    )
    markdown = (
        "# Round-1 budget diagnostics tables\n\n" + notes + "\n"
        "## Table A -- budget effect\n\n"
        + _to_markdown(rows_a)
        + "\n\n## Table B -- within-question room (per budget, task)\n\n"
        + _to_markdown(rows_b)
        + "\n\n## Table C -- reward visibility (per budget, task)\n\n"
        + _to_markdown(rows_c)
        + "\n"
    )
    (out_dir / "tables.md").write_text(markdown)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {"table_a": rows_a, "table_b": rows_b, "table_c": rows_c},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print("\n== table A ==")
    for row in rows_a:
        print(
            f"{row['budget']}: acc_macro={row['accuracy_macro_by_question']} "
            f"acc_micro={row['accuracy_micro_by_episode']} "
            f"mean_calls={row['mean_tool_calls']} "
            f"calls0={row['share_calls_0']} calls1={row['share_calls_1']} "
            f"term_final={row['term_final']} term_max_steps={row['term_max_steps']}"
        )
    _group_summary(rows_a, rows_c)
    print(f"\nwritten: {out_dir}/table_a.csv, table_b.csv, table_c.csv, tables.md, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
