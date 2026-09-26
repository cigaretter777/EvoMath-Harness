"""Quantify the reward inversion for trajectories that exhaust the step budget.

Mechanism (verified in code, not inferred):
  agent/environment.py:142   evaluate() returns None when final_answer is None
  training/verl_environment.py:139,151  reward stays 0.0 and reward_for_trajectory
                                        is never called when the verdict is None
  evaluation/model_eval.py:209          the same situation is scored as an
                                        explicit INVALID_PREDICTION

So under R1/R2 the invalid-action penalty cannot fire on the trajectories that
accumulate the most invalid actions, because those are exactly the ones that run
out of steps. They score 0.0, while an honest wrong answer that also had invalid
actions scores negative. Under GRPO's group-relative advantage that makes
protocol failure relatively *better* than honest termination.

This script recomputes rewards and group advantages for a stored rollout-health
run under the aligned rule (no final answer -> INVALID_PREDICTION verdict, reward
config still applied) and reports how the training signal changes. Read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.types import Budget
from adaptive_math.reward import RewardConfig
from adaptive_math.training.reward_bridge import (
    group_advantages,
    reward_for_trajectory,
)
from adaptive_math.verifier.service import VerifierResult, VerifierStatus


def _invalid_verdict() -> VerifierResult:
    return VerifierResult(
        status=VerifierStatus.INVALID_PREDICTION,
        reward=0.0,
        normalized_prediction=None,
        normalized_reference=None,
        details={"reason": "terminated without a final answer"},
    )


def analyze(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    budget = Budget.model_validate(yaml.safe_load(Path(manifest["agent_config"]).read_text()))
    config = RewardConfig.model_validate(yaml.safe_load(Path(manifest["reward_config"]).read_text()))
    generation = manifest.get("generation") or {}
    max_generated_tokens = int(generation.get("max_new_tokens", 1024)) * budget.max_steps

    records = [
        json.loads(line)
        for line in (run_dir / "trajectories.jsonl").read_text().splitlines()
        if line.strip()
    ]

    rows = []
    for record in records:
        trajectory = Trajectory.model_validate(record["trajectory"])
        actual = float(record["reward"])
        verdict = record.get("verdict")
        if verdict is None:
            fixed_breakdown = reward_for_trajectory(
                trajectory,
                _invalid_verdict(),
                budget,
                max_generated_tokens=max_generated_tokens,
                config=config,
            )
            fixed = float(fixed_breakdown.total)
            components = fixed_breakdown.components
        else:
            fixed = actual
            components = (record.get("reward_breakdown") or {}).get("components")
        rows.append(
            {
                "task_index": record["task_index"],
                "sample_index": record["sample_index"],
                "termination": trajectory.termination_reason.value,
                "invalid_actions": trajectory.usage.invalid_actions,
                "actual_reward": actual,
                "fixed_reward": round(fixed, 4),
                "fixed_components": components,
                "verdict_was_none": verdict is None,
            }
        )

    group_size = manifest.get("group_size") or 4
    groups_actual: list[list[float]] = []
    groups_fixed: list[list[float]] = []
    for start in range(0, len(rows), group_size):
        chunk = rows[start : start + group_size]
        groups_actual.append([r["actual_reward"] for r in chunk])
        groups_fixed.append([r["fixed_reward"] for r in chunk])

    adv_actual = [group_advantages(g) for g in groups_actual]
    adv_fixed = [group_advantages(g) for g in groups_fixed]

    flips = []
    for gi, (ga, gf) in enumerate(zip(adv_actual, adv_fixed)):
        for si, (a, f) in enumerate(zip(ga.advantages, gf.advantages)):
            row = rows[gi * group_size + si]
            if row["verdict_was_none"] and a > 0 >= f:
                flips.append(
                    {
                        "group": gi,
                        "sample": si,
                        "termination": row["termination"],
                        "invalid_actions": row["invalid_actions"],
                        "advantage_actual": round(a, 4),
                        "advantage_fixed": round(f, 4),
                    }
                )

    def _effective(advantages) -> int:
        return sum(1 for a in advantages if a.effective)

    return {
        "run_dir": str(run_dir),
        "reward_config": manifest.get("reward_config"),
        "reward_variant": config.variant,
        "invalid_weight": config.invalid_weight,
        "invalid_cap": config.invalid_cap,
        "max_generated_tokens": max_generated_tokens,
        "trajectory_count": len(rows),
        "no_verdict_trajectories": sum(1 for r in rows if r["verdict_was_none"]),
        "reward_sum_actual": round(sum(r["actual_reward"] for r in rows), 4),
        "reward_sum_fixed": round(sum(r["fixed_reward"] for r in rows), 4),
        "groups": {
            "count": len(groups_actual),
            "effective_actual": _effective(adv_actual),
            "effective_fixed": _effective(adv_fixed),
            "effective_rate_actual": round(_effective(adv_actual) / len(groups_actual), 4),
            "effective_rate_fixed": round(_effective(adv_fixed) / len(groups_fixed), 4),
        },
        "budget_exhausted_samples_with_positive_advantage_actual": sum(
            1
            for gi, ga in enumerate(adv_actual)
            for si, a in enumerate(ga.advantages)
            if rows[gi * group_size + si]["verdict_was_none"] and a > 0
        ),
        "advantage_sign_flips_after_fix": flips,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    reports = [analyze(d) for d in args.run_dir]
    text = json.dumps(reports, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
