"""Classify invalid predictions of the four-way campaign (audit step 1).

Runs on the AutoDL raw prediction artifacts (NOT committed — raw answer text
stays local). Writes aggregates + a per-category sample file for human review.

Categories (first match wins):
- truncated_at_cap            output_tokens == 1024 (max_new_tokens). The
                              finish_reason field is a stub ("stop" is
                              hardcoded in transformers_client), so the token
                              count is the only truncation signal.
- verifier_rejected_parseable extract_status == "ok": a value was extracted
                              but the verifier returned invalid_prediction.
- final_malformed             a `<final` envelope exists but the extractor
                              could not parse a value out of it.
- ambiguous_tail              extractor saw answer-like candidates but could
                              not pick one (extract_status == "ambiguous").
- no_submission               reasoning ran to a natural stop before the cap
                              with no answer-like tail (extract_status ==
                              "missing").
- empty_or_garbage            no thinking block or very short raw output.
- other                       anything left (should be empty in practice).

Usage:
    python scripts/campaign-20260919/classify_invalid.py \
        --output-dir artifacts/audit/invalid-classification-20260919
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

ARM_PATHS = {
    "sft": REPO / "artifacts/eval/sft_dp_v1_omnimath_200/sft_predictions.jsonl",
    "r0": REPO / "artifacts/eval/r0_omnimath_200/sft_predictions.jsonl",
    "r2": REPO / "artifacts/eval/r2_omnimath_200/sft_predictions.jsonl",
}

MAX_NEW_TOKENS = 1024
TAIL_CHARS = 300

# Answer-ish tail: "= <expr>" near the end, or the raw ends with a number /
# math fragment right after a newline or space (rough "unsubmitted answer").
ANSWER_TAIL_RE = re.compile(
    r"(?:=\s*[^\n]{1,80}|\\boxed\{|因此.{0,60}答案.{0,60})\s*$", re.DOTALL
)


def classify(row: dict) -> str:
    if row["output_tokens"] >= MAX_NEW_TOKENS:
        return "truncated_at_cap"
    if row["extract_status"] == "ok":
        return "verifier_rejected_parseable"
    if "<final" in row["raw_output"]:
        return "final_malformed"
    if row["extract_status"] == "ambiguous":
        return "ambiguous_tail"
    if "<think" not in row["raw_output"] or len(row["raw_output"]) < 50:
        return "empty_or_garbage"
    if row["extract_status"] == "missing":
        return "no_submission"
    return "other"


def tail(raw: str) -> str:
    return raw[-TAIL_CHARS:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "artifacts/audit/invalid-classification-20260919",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[dict]] = {}
    invalid: dict[str, list[dict]] = {}
    for arm, path in ARM_PATHS.items():
        with path.open() as handle:
            rows[arm] = [json.loads(line) for line in handle]
        invalid[arm] = [r for r in rows[arm] if r["verifier_status"] == "invalid_prediction"]

    classified = []
    for arm, inv_rows in invalid.items():
        for row in inv_rows:
            category = classify(row)
            classified.append(
                {
                    "task_id": row["task_id"],
                    "arm": arm,
                    "category": category,
                    "output_tokens": row["output_tokens"],
                    "extract_status": row["extract_status"],
                    "prediction": row["prediction"],
                    "raw_tail": tail(row["raw_output"]),
                }
            )
    with (args.output_dir / "classification.jsonl").open("w") as handle:
        for row in classified:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_arm = {
        arm: dict(Counter(r["category"] for r in classified if r["arm"] == arm))
        for arm in ARM_PATHS
    }
    # Tasks invalid in how many arms
    invalid_sets = {arm: {r["task_id"] for r in inv} for arm, inv in invalid.items()}
    consistency = Counter()
    for task_id in {t for s in invalid_sets.values() for t in s}:
        consistency[sum(task_id in s for s in invalid_sets.values())] += 1
    summary = {
        "per_arm_category_counts": per_arm,
        "per_arm_invalid_total": {arm: len(inv) for arm, inv in invalid.items()},
        "invalid_in_n_arms": dict(consistency),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Per-category review samples (up to 10, spread across arms)
    samples_by_category: dict[str, list[dict]] = defaultdict(list)
    for row in classified:
        if len(samples_by_category[row["category"]]) < 10:
            samples_by_category[row["category"]].append(row)
    lines = ["# Invalid prediction review samples\n"]
    for category, samples in sorted(samples_by_category.items()):
        lines.append(f"\n## {category} ({len(samples)} shown)\n")
        for sample in samples:
            lines.append(
                f"- **{sample['arm']}** {sample['task_id']} "
                f"tokens={sample['output_tokens']} extract={sample['extract_status']} "
                f"prediction={sample['prediction']!r}\n"
            )
            snippet = sample["raw_tail"].replace("\n", "\\n")
            lines.append(f"  tail: `{snippet}`\n")
    (args.output_dir / "samples.md").write_text("\n".join(lines))

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
