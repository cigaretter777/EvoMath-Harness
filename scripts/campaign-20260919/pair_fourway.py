"""Offline four-way pairing for the 2026-09-19 overnight campaign.

Joins per-task verifier verdicts of three eval runs on the same frozen
OmniMath-200 pool (task_ids_sha256 1fc257f2…):

- SFT baseline (09-14): ``artifacts/eval/sft_dp_v1_omnimath_200/sft_predictions.jsonl``
- R0 adapter-only:      ``artifacts/eval/r0_omnimath_200/sft_predictions.jsonl``
- R2 adapter-only:      ``artifacts/eval/r2_omnimath_200/sft_predictions.jsonl``

Reuses ``_paired_statistics`` from ``model_eval`` so McNemar/bootstrap
conventions are identical to the online pipeline (seed 20260913, 10k
resamples). Pair delta convention matches the online one: ``B - A``.

Usage:
    python scripts/campaign-20260919/pair_fourway.py \
        [--output-dir artifacts/eval/fourway_omnimath_200]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from adaptive_math.evaluation.model_eval import _paired_statistics

REPO = Path(__file__).resolve().parents[2]

ARM_PATHS = {
    "sft": REPO / "artifacts/eval/sft_dp_v1_omnimath_200/sft_predictions.jsonl",
    "r0": REPO / "artifacts/eval/r0_omnimath_200/sft_predictions.jsonl",
    "r2": REPO / "artifacts/eval/r2_omnimath_200/sft_predictions.jsonl",
}

PAIRS = [("sft", "r0"), ("sft", "r2"), ("r0", "r2")]

CORRECT = "correct"
VALID = {CORRECT, "incorrect"}


def load_verdicts(path: Path) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            verdicts[row["task_id"]] = row["verifier_status"]
    return verdicts


def summarize_arm(verdicts: dict[str, str]) -> dict[str, object]:
    statuses = Counter(verdicts.values())
    total = len(verdicts)
    correct = statuses[CORRECT]
    valid = sum(count for status, count in statuses.items() if status in VALID)
    return {
        "total": total,
        "correct": correct,
        "incorrect": statuses["incorrect"],
        "invalid_prediction": statuses["invalid_prediction"],
        "valid_answer_rate": round(valid / total, 3),
        "verifier_accuracy": round(correct / total, 3),
    }


def pair_stats(arm_a: dict[str, str], arm_b: dict[str, str]) -> dict[str, object]:
    rows = [
        {"base_correct": arm_a[task_id] == CORRECT, "sft_correct": arm_b[task_id] == CORRECT}
        for task_id in arm_a
    ]
    outcomes = Counter(
        "improved" if row["sft_correct"] and not row["base_correct"]
        else "regressed" if row["base_correct"] and not row["sft_correct"]
        else "unchanged"
        for row in rows
    )
    return {"paired": dict(outcomes), **_paired_statistics(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "artifacts/eval/fourway_omnimath_200",
    )
    args = parser.parse_args()

    verdicts = {arm: load_verdicts(path) for arm, path in ARM_PATHS.items()}
    task_ids = set(verdicts["sft"])
    for arm, arm_verdicts in verdicts.items():
        if len(arm_verdicts) != 200:
            raise SystemExit(f"{arm}: expected 200 rows, got {len(arm_verdicts)}")
        if set(arm_verdicts) != task_ids:
            raise SystemExit(f"{arm}: task_id set mismatch (join would be partial)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = [
        {
            "task_id": task_id,
            **{arm: verdicts[arm][task_id] for arm in verdicts},
            **{f"{arm}_correct": verdicts[arm][task_id] == CORRECT for arm in verdicts},
        }
        for task_id in sorted(task_ids)
    ]
    with (args.output_dir / "comparison.jsonl").open("w") as handle:
        for row in comparison_rows:
            handle.write(json.dumps(row) + "\n")

    pairs = {f"{a}_vs_{b}": pair_stats(verdicts[a], verdicts[b]) for a, b in PAIRS}
    summary = {
        "task_count": len(task_ids),
        "arms": {arm: summarize_arm(arm_verdicts) for arm, arm_verdicts in verdicts.items()},
        "pairs": pairs,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Compact console table
    print(f"{'arm':>6} {'correct':>8} {'invalid':>8} {'valid%':>7}")
    for arm, arm_summary in summary["arms"].items():
        print(
            f"{arm:>6} {arm_summary['correct']:>7}/{arm_summary['total']:<3}"
            f" {arm_summary['invalid_prediction']:>8}"
            f" {100 * arm_summary['valid_answer_rate']:>6.1f}%"
        )
    print()
    for pair_name, stats in pairs.items():
        paired = stats["paired"]
        print(
            f"{pair_name:>11}: delta={stats['accuracy_delta']:+.3f} "
            f"improved={paired['improved']} regressed={paired['regressed']} "
            f"unchanged={paired['unchanged']} "
            f"mcnemar_p={stats['mcnemar_pvalue']:.4g} "
            f"ci95={[round(v, 3) for v in stats['paired_bootstrap_ci95']]}"
        )

    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
