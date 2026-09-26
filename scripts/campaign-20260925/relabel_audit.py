"""Re-score every stored raw output with the current verifier and report drift.

Why this exists: the 2026-09-25 verifier fixes changed how answers are *judged*, so
every headline number in the campaign (SFT 27/200, R0 22/200, R2 25/200, the E2
pair) is now produced by different code than the one that measured it. Nothing else
would have caught that. Re-running generation is not required -- the raw outputs are
on disk -- so this is a label question answered offline, at zero GPU cost.

Two directions of drift matter differently:

  invalid -> correct      the old labels understated a model; conclusions about
                          "RL did not help" could flip.
  correct -> invalid/incorrect  the old labels overstated a model, and the
                          false-positive class found today (float equality on
                          1000-digit numbers) predicts this direction is real.

Reads only artifacts/, writes only artifacts/. Never regenerates text.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from adaptive_math.core.types import AnswerType, ReferenceAnswer
from adaptive_math.verifier import ExtractStatus, VerifierStatus, extract, verify_answer

REPO = Path(__file__).resolve().parents[2]
EVAL_PARQUET = REPO / "data" / "processed" / "v1" / "frozen_eval.parquet"

ARMS = {
    "sft": "artifacts/eval/sft_dp_v1_omnimath_200/sft_predictions.jsonl",
    "r0": "artifacts/eval/r0_omnimath_200/sft_predictions.jsonl",
    "r2": "artifacts/eval/r2_omnimath_200/sft_predictions.jsonl",
    "e2_champion_cap1024": "artifacts/eval/harness_e2_champion_cap1024/sft_predictions.jsonl",
    "e2_candidate_cap2048": "artifacts/eval/harness_e2_candidate_cap2048/sft_predictions.jsonl",
}


def _references() -> dict[str, ReferenceAnswer]:
    frame = pd.read_parquet(EVAL_PARQUET)
    out: dict[str, ReferenceAnswer] = {}
    for row in frame.itertuples():
        # parquet hands back an empty numpy array, whose truth value raises, so the
        # fallback has to be a length test rather than `or ()`.
        forms = row.reference_acceptable_forms
        out[row.task_id] = ReferenceAnswer(
            value=row.reference_value,
            answer_type=AnswerType(row.answer_type),
            acceptable_forms=tuple(forms) if forms is not None and len(forms) else (),
        )
    return out


def relabel(arm: str, path: Path, references: dict[str, ReferenceAnswer]) -> dict[str, object]:
    transitions: Counter[str] = Counter()
    changed: list[dict[str, str]] = []
    counts_old: Counter[str] = Counter()
    counts_new: Counter[str] = Counter()
    rows = 0

    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            reference = references.get(row["task_id"])
            old = row["verifier_status"]
            counts_old[old] += 1
            if reference is None:
                # Not in the frozen pool: keep the stored label rather than guess.
                counts_new[old] += 1
                transitions[f"{old}:(no reference)"] += 1
                continue
            extraction = extract(row["raw_output"])
            if extraction.status is not ExtractStatus.OK or extraction.value is None:
                new = VerifierStatus.INVALID_PREDICTION.value
            else:
                new = verify_answer(extraction.value, reference, task_id=row["task_id"]).status.value
            counts_new[new] += 1
            if new != old:
                transitions[f"{old} -> {new}"] += 1
                changed.append({"arm": arm, "task_id": row["task_id"], "old": old, "new": new})

    return {
        "arm": arm,
        "rows": rows,
        "correct_old": counts_old[VerifierStatus.CORRECT.value],
        "correct_new": counts_new[VerifierStatus.CORRECT.value],
        "invalid_old": counts_old[VerifierStatus.INVALID_PREDICTION.value],
        "invalid_new": counts_new[VerifierStatus.INVALID_PREDICTION.value],
        "transitions": dict(transitions),
        "changed": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "artifacts/audit/relabel-20260925/summary.json")
    args = parser.parse_args()

    references = _references()
    results = []
    for arm, relative in ARMS.items():
        path = REPO / relative
        if not path.is_file():
            print(f"{arm:24s} SKIP (no stored outputs)")
            continue
        result = relabel(arm, path, references)
        results.append(result)
        print(
            f"{arm:24s} rows={result['rows']:4d} correct {result['correct_old']}->{result['correct_new']} "
            f"invalid {result['invalid_old']}->{result['invalid_new']} changed={len(result['changed'])}"
        )
        for transition, count in sorted(result["transitions"].items()):
            print(f"{'':24s}   {transition:46s} {count}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for result in results:
        for item in result["changed"]:
            grouped[item["task_id"]].append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "arms": [{key: value for key, value in result.items() if key != "changed"} for result in results],
                "changed_rows": [item for result in results for item in result["changed"]],
                "tasks_changed_in_more_than_one_arm": {
                    task: items for task, items in grouped.items() if len(items) > 1
                },
            },
            indent=1,
            sort_keys=True,
        )
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
