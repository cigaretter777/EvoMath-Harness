"""P1 analysis: what the R2 rollout-health diagnostic actually shows.

Reads a completed run_rollout_health output directory and reports the
pre-specified quantities from docs/results/campaign-2026-09-24/preregistration.md:
termination reasons, tool events, per-turn parse error taxonomy, protocol loop
detection, group structure (GRPO advantage availability) and sandbox liveness
attribution. Read-only: it never modifies the run it analyses.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_math.agent.parser import parse_action


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sandbox_health(path: Path) -> dict:
    if not path.is_file():
        return {"available": False}
    ok = bad = 0
    bad_lines: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if 'ping="pong"' in line and "exec_stdout='42\\n'" in line:
            ok += 1
        else:
            bad += 1
            bad_lines.append(line.strip())
    return {"available": True, "healthy_probes": ok, "unhealthy_probes": bad, "unhealthy": bad_lines[:10]}


def analyze(run_dir: Path, health_log: Path | None) -> dict:
    trajectories = _read_jsonl(run_dir / "trajectories.jsonl")
    groups = _read_jsonl(run_dir / "groups.jsonl")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").is_file() else {}

    termination = Counter()
    parse_errors = Counter()
    model_outputs = 0
    tool_events = 0
    invalid_actions = 0
    unclosed_think = 0
    tokens: list[int] = []
    rewards: list[float] = []
    rollout_seconds: list[float] = []
    loop_trajectories = 0
    repeat_details: list[dict] = []
    max_steps_with_final_shaped_output = 0

    for record in trajectories:
        trajectory = record["trajectory"]
        termination[str(trajectory.get("termination_reason"))] += 1
        rewards.append(float(record["reward"]))
        rollout_seconds.append(float(record.get("rollout_seconds", 0.0)))
        usage = trajectory.get("usage") or {}
        invalid_actions += int(usage.get("invalid_actions", 0))
        tokens.append(int(usage.get("generated_tokens", 0)))

        raws: list[str] = []
        for event in trajectory["events"]:
            kind = event["kind"]
            if kind == "model_output":
                model_outputs += 1
                raw = event["payload"]["raw"]
                raws.append(raw)
                parsed = parse_action(raw)
                parse_errors[str(parsed.error) if parsed.error else "ok"] += 1
                stripped = raw.strip()
                if stripped.startswith("<think>") and "</think>" not in stripped:
                    unclosed_think += 1
            elif kind == "tool_call":
                tool_events += 1

        repeats = Counter(raws)
        duplicated = {raw: count for raw, count in repeats.items() if count > 1}
        if duplicated:
            loop_trajectories += 1
            repeat_details.append(
                {
                    "task_index": record["task_index"],
                    "sample_index": record["sample_index"],
                    "termination": str(trajectory.get("termination_reason")),
                    "reward": record["reward"],
                    "repeated_output": next(iter(duplicated))[:120],
                    "times": max(duplicated.values()),
                }
            )
        if str(trajectory.get("termination_reason")) == "max_steps" and any(
            "</final>" in raw for raw in raws
        ):
            max_steps_with_final_shaped_output += 1

    group_stats = []
    for group in groups:
        group_stats.append(
            {
                "task_index": group.get("task_index"),
                "rewards": group.get("rewards"),
                "mean": group.get("mean"),
                "std": group.get("std"),
                "effective": group.get("effective"),
            }
        )
    effective = [g for g in group_stats if g["effective"]]

    report = {
        "run_dir": str(run_dir),
        "manifest": {
            "model": manifest.get("model"),
            "adapter": manifest.get("adapter"),
            "reward_config": manifest.get("reward_config"),
            "sandbox_url": manifest.get("sandbox_url"),
            "generation": manifest.get("generation"),
            "status": manifest.get("status"),
        },
        "trajectory_count": len(trajectories),
        "termination_counts": dict(termination),
        "tool_events_total": tool_events,
        "invalid_actions_total": invalid_actions,
        "model_output_turns": model_outputs,
        "parse_error_taxonomy": dict(parse_errors),
        "unclosed_think_turns": unclosed_think,
        "unclosed_think_rate": round(unclosed_think / model_outputs, 4) if model_outputs else None,
        "loop_trajectories": loop_trajectories,
        "loop_details": repeat_details,
        "max_steps_with_final_shaped_output": max_steps_with_final_shaped_output,
        "generated_tokens": {
            "mean": round(statistics.mean(tokens), 1) if tokens else None,
            "p50": statistics.median(tokens) if tokens else None,
            "max": max(tokens) if tokens else None,
        },
        "reward": {
            "mean": round(statistics.mean(rewards), 4) if rewards else None,
            "std": round(statistics.pstdev(rewards), 4) if len(rewards) > 1 else None,
            "positive": sum(1 for r in rewards if r > 0),
            "zero": sum(1 for r in rewards if r == 0),
            "negative": sum(1 for r in rewards if r < 0),
        },
        "rollout_seconds_mean": round(statistics.mean(rollout_seconds), 1) if rollout_seconds else None,
        "groups": {
            "count": len(group_stats),
            "effective": len(effective),
            "effective_rate": round(len(effective) / len(group_stats), 4) if group_stats else None,
            "all_zero": sum(1 for g in group_stats if g["mean"] == 0),
            "detail": group_stats,
        },
        "run_summary_json": summary,
        "sandbox_health": _sandbox_health(health_log) if health_log else {"available": False},
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--health-log", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = analyze(args.run_dir, args.health_log)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
