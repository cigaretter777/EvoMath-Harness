"""Paired per-protocol comparisons between stored arms, from a rescore artifact.

Why this exists
---------------
``tolerance_rescore.py`` re-scores every stored arm under three escalating
extraction protocols and keeps, for each, the *task ids* it counted correct. A
headline delta is not two accuracies next to each other, though: it is a paired
comparison on the same tasks, and under a tolerant protocol it is not even the
same comparison -- the strict rung and the lenient rung disagree about which
tasks were read at all, in either arm.

So this reads that artifact back and runs the repo's own paired test (the exact
McNemar and paired bootstrap ``evaluate_pair`` reports, same seed and resample
count) once per protocol, so the p-values behind the claim come from a committed
artifact instead of from an ad-hoc probe. Nothing is re-verified and no model is
called: every number here is derived from task ids that are already on disk.

The comparison set is the intersection of the two arms' scored tasks. An arm
that covered only part of the frozen set (the rule arm was rolled out on 124 of
200) is therefore compared on what the two arms actually share, with the
excluded tasks and the base arm's score on them reported rather than dropped
silently -- an accuracy computed over a different task set is not the same
number wearing a different denominator.

Usage:
    python scripts/analysis/paired_protocols.py \
        --from-json artifacts/results/mvp-20260929/tolerance-rescore.json \
        --base base_direct \
        --against base_tool --against sft_direct \
        --out artifacts/results/mvp-20260929/paired-protocols.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

from adaptive_math.core.hashing import sha256_hex
from adaptive_math.evaluation.model_eval import _paired_statistics

REPO = Path(__file__).resolve().parents[2]

DEFAULT_TASK_IDS = REPO / "artifacts/eval/thesis_e0_base_direct_b1/task_ids.txt"

# The frozen Omni-MATH-200, as recorded by the arms' own summaries: sha256 of the
# ids joined by newlines with no trailing newline.
EXPECTED_TASK_IDS_SHA256 = "1fc257f26fbf80ab877223c6bef8b9eda3e8cbfb47bd38b68f4ba6ccddddaee2"

PROTOCOLS = ("strict", "coerce", "lenient")


def paired_row(
    protocol: str,
    base_correct: set[str],
    arm_correct: set[str],
    common: list[str],
) -> dict[str, Any]:
    """One paired comparison, with both sides scored on the same task list."""
    comparison = [
        {
            "task_id": task_id,
            "base_correct": task_id in base_correct,
            "sft_correct": task_id in arm_correct,
        }
        for task_id in common
    ]
    stats = _paired_statistics(comparison)
    improved = sum(1 for row in comparison if row["sft_correct"] and not row["base_correct"])
    regressed = sum(1 for row in comparison if row["base_correct"] and not row["sft_correct"])
    if improved + regressed == 0:
        # Every paired test below is exact McNemar on the discordant pairs; with
        # none, p is 1.0 by definition and the delta is 0, which is a real (if
        # dull) result rather than an error.
        assert stats["mcnemar_pvalue"] == 1.0
    return {
        "protocol": protocol,
        "n": len(common),
        "base_correct": sum(1 for row in comparison if row["base_correct"]),
        "arm_correct": sum(1 for row in comparison if row["sft_correct"]),
        "improved": improved,
        "regressed": regressed,
        "accuracy_delta": stats["accuracy_delta"],
        "paired_bootstrap_ci95": stats["paired_bootstrap_ci95"],
        "mcnemar_pvalue": stats["mcnemar_pvalue"],
    }


def compare(
    base: dict[str, Any], arm: dict[str, Any], base_label: str, arm_label: str
) -> dict[str, Any]:
    base_ids = set(base["scored_task_ids"])
    arm_ids = set(arm["scored_task_ids"])
    if not base_ids & arm_ids:
        raise RuntimeError(f"{base_label} and {arm_label} share no scored tasks")
    if not (base_ids <= arm_ids or arm_ids <= base_ids):
        # Overlapping-but-not-nested means the two arms were scored on task sets
        # nothing in the run configured; comparing the overlap would hide it.
        raise RuntimeError(
            f"{base_label} and {arm_label} cover crossing task sets "
            f"({len(base_ids - arm_ids)} only in the first, {len(arm_ids - base_ids)} only in "
            "the second)"
        )
    common = sorted(base_ids & arm_ids)
    excluded = sorted((base_ids | arm_ids) - set(common))
    base_correct = {protocol: set(base["correct_task_ids"][protocol]) for protocol in PROTOCOLS}
    arm_correct = {protocol: set(arm["correct_task_ids"][protocol]) for protocol in PROTOCOLS}
    return {
        "base": base_label,
        "arm": arm_label,
        "excluded_tasks": len(excluded),
        "excluded_base_correct": {
            protocol: sum(1 for task_id in excluded if task_id in base_correct[protocol])
            for protocol in PROTOCOLS
        },
        "protocols": {
            protocol: paired_row(protocol, base_correct[protocol], arm_correct[protocol], common)
            for protocol in PROTOCOLS
        },
    }


def _format_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| base | arm | protocol | n | base | arm | delta | 95% CI | up/down | p |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for data in results:
        for protocol in PROTOCOLS:
            row = data["protocols"][protocol]
            ci = row["paired_bootstrap_ci95"]
            lines.append(
                f"| {data['base']} | {data['arm']} | {protocol} | {row['n']} | "
                f"{row['base_correct']} | {row['arm_correct']} | {row['accuracy_delta']:+.3f} | "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {row['improved']}/{row['regressed']} | "
                f"{row['mcnemar_pvalue']:.3g} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-json", type=Path, required=True, help="tolerance_rescore.py output")
    parser.add_argument("--base", required=True, help="arm the others are compared against")
    parser.add_argument(
        "--against",
        action="append",
        required=True,
        metavar="ARM",
        help="repeatable; must name arms present in --from-json",
    )
    parser.add_argument("--task-ids", type=Path, default=DEFAULT_TASK_IDS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    task_ids = args.task_ids.read_text().splitlines()
    canonical = sha256_hex("\n".join(task_ids).encode())
    if canonical != EXPECTED_TASK_IDS_SHA256:
        raise RuntimeError(
            f"task set sha256 {canonical} != expected {EXPECTED_TASK_IDS_SHA256}; "
            "these are not the frozen 200 tasks"
        )

    payload = json.loads(args.from_json.read_text())
    if payload["task_ids_sha256"] != canonical:
        raise RuntimeError(
            f"{args.from_json} was scored against task set {payload['task_ids_sha256']}, "
            f"not against {canonical}"
        )
    arms: dict[str, Any] = payload["arms"]
    if args.base not in arms:
        raise RuntimeError(f"base arm {args.base!r} is not in {args.from_json}")
    for label in args.against:
        if label not in arms:
            raise RuntimeError(f"arm {label!r} is not in {args.from_json}")
        if label == args.base:
            raise RuntimeError(f"arm {label!r} cannot be compared against itself")

    results = [compare(arms[args.base], arms[label], args.base, label) for label in args.against]

    print(_format_table(results))
    for data in results:
        if data["excluded_tasks"]:
            print(
                f"\nnote: {data['base']} vs {data['arm']} is paired on {data['excluded_tasks']} "
                f"fewer tasks than the wider arm: base scored "
                f"{data['excluded_base_correct']['strict']} of them correct at strict"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"task_ids_sha256": canonical, "comparisons": results}, indent=2, sort_keys=True)
        + "\n"
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
