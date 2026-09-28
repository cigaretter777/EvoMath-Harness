"""Re-score stored direct-arm generations under escalating extraction protocols.

Why this exists
---------------
The direct arms record accuracy under one extractor: ``<final>`` tags, then
``\\boxed{}``, then ``$...$``, with ``{"answer": <non-empty str>}`` required in a
final tag. The agent arms show that a large share of discarded turns fail on the
envelope, not on the mathematics -- most dramatically ``{"answer":71}``, a
well-formed final block thrown away because the value is a JSON number where the
schema wants a string. The direct arms have the same generations on disk, so the
same question can be asked of them without spending a GPU hour: **how much of
each arm's accuracy is a property of the extractor rather than of the model?**

Symmetry is the whole game here, so the protocol is applied identically to every
arm and the rungs nest: a task counted correct at ``strict`` stays correct at
``coerce`` and ``lenient``. What changes between rungs is only how much
interpretation is allowed in reading the recorded text.

Both stored genres are accepted, because both are in the headline claim:
prediction files (one generation per task) and trajectory files (a recorded
final answer plus the turns that led to it). For a trajectory the strict rung is
the answer the loop closed with, and the rungs above it read the last model
turn -- the model's own final word, not a value it mentioned while exploring.
The runtime's own parser is then applied to that last turn and must agree with
what the trajectory closed with; if it does not, the run was read differently
here than when it ran (a tolerant parser, say) and the harness stops rather than
reporting a number measured against the wrong ruler.

The rungs
---------
* ``strict`` -- the ruler the runs used. Re-run here and required to reproduce
  every recorded status; a mismatch is fatal, because then this harness is not
  the one that scored the file and no rung above it means anything.
* ``coerce`` -- only where strict found nothing: read the literal text of a
  numeric ``answer`` inside a single well-formed ``<final>`` pair. Mechanical,
  adds no information, no search.
* ``lenient`` -- only where both found nothing: ``extract_solution_answer``, the
  repo's own dataset extractor (last boxed group in the tail, prose anchor,
  whole-text math span). This one *does* interpret: on a reasoning-heavy
  response it can pick up a value the model mentioned while exploring. Treat it
  as an upper bound, not as a score the arm would have earned.

Verification is the frozen one: same parquet references, same ``verify_answer``,
same task ids. The task-set identity gate runs first and is fatal.

One deliberate tolerance in the identity gate, because the arms were scored by
different commits: the extractor and the prompt are byte-identical to the ones
the runs used (their source hashes are in each arm's ``eval_manifest.json``), but
``verifier/service.py`` is not -- the r0/r2 arms were evaluated at ``b4ca1604``
and three later commits touched the verifier. So a recorded verdict this
verifier cannot reproduce is fatal **only when either side says correct**, since
that is the only case where an accuracy would depend on the version. Rows that
move between two non-correct statuses are counted in ``verifier_drift`` and
reported; nothing is silently reconciled.

Usage:
    python scripts/analysis/tolerance_rescore.py \
        --arm base_direct=artifacts/eval/thesis_e0_base_direct_b1/base_predictions.jsonl \
        --arm sft_direct=artifacts/eval/thesis_e0_base_direct_b1/sft_predictions.jsonl \
        --arm base_tool=artifacts/rollout_health/thesis_e0_base_tool/trajectories.jsonl \
        --out artifacts/results/mvp-20260929/tolerance-rescore.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from adaptive_math.agent.actions import FinalAction
from adaptive_math.agent.parser import parse_action
from adaptive_math.core.hashing import sha256_hex
from adaptive_math.verifier import ExtractStatus, extract, verify_answer
from adaptive_math.verifier.extractor import extract_solution_answer

REPO = Path(__file__).resolve().parents[2]

DEFAULT_TASK_IDS = REPO / "artifacts/eval/thesis_e0_base_direct_b1/task_ids.txt"
DEFAULT_PARQUET = REPO / "data/processed/v1/frozen_eval.parquet"

# Recorded in artifacts/eval/r0_omnimath_200/summary.json and r2_.../summary.json
# as task_ids_sha256: sha256 of the ids joined by newlines, no trailing newline.
EXPECTED_TASK_IDS_SHA256 = "1fc257f26fbf80ab877223c6bef8b9eda3e8cbfb47bd38b68f4ba6ccddddaee2"

PROTOCOLS = ("strict", "coerce", "lenient")
VALID_STATUSES = frozenset({"correct", "incorrect"})

# The literal text of a numeric answer, which is what makes coercion mechanical:
# the repair writes the digits the model wrote, not a re-rendering of a float.
_NUMERIC_ANSWER = re.compile(r'"answer"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[,}]')


def load_references(parquet: Path, task_ids: list[str]) -> dict[str, Any]:
    """References for the frozen task set, through the eval script's own loader."""
    import importlib.util

    import pyarrow.parquet as pq

    spec = importlib.util.spec_from_file_location(
        "run_model_eval", REPO / "scripts/eval/run_model_eval.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    references: dict[str, Any] = {}
    for row in pq.read_table(parquet).to_pylist():
        labeled = module._labeled_task_from_row(row)
        references[labeled.task.task_id] = labeled.reference
    missing = [task_id for task_id in task_ids if task_id not in references]
    if missing:
        raise RuntimeError(f"{len(missing)} task ids have no reference, first: {missing[0]}")
    return references


def numeric_answer_candidate(raw: str) -> str | None:
    """The numeric answer literal from the turn's single ``<final>`` block.

    Handles both closure states -- a missing closing tag is the shape the agent
    arms produced most -- and returns the literal digits the model wrote rather
    than a re-rendering of a float, which is what keeps the repair mechanical.
    The block body must be exactly one JSON object, so trailing prose or a
    stacked second action is never silently read as an answer.
    """
    start = raw.find("<final>")
    if start == -1 or raw.find("<final>", start + 1) != -1:
        return None
    body = raw[start + len("<final>") :]
    if "</final>" in body:
        body = body.split("</final>", 1)[0]
    body = body.strip()
    if not body.startswith("{"):
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    answer = payload.get("answer")
    if not isinstance(answer, (int, float)) or isinstance(answer, bool):
        return None
    match = _NUMERIC_ANSWER.search(body)
    return match.group(1) if match else None


def arm_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Read one arm's stored records into the shape the rungs consume.

    Two stored genres, one reading each:

    * **direct** arms -- one generation per task, and the rungs above ``strict``
      re-read that generation's text.
    * **agent** arms -- a recorded final answer the loop closed with, and the
      rungs above ``strict`` re-read the *last model turn's* text, the model's
      own final word on the task. Earlier turns are deliberately not scanned:
      a value mentioned while exploring is not an answer the run produced, and
      scanning them would make the upper rungs measure the search rather than
      the answer.
    """
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if records and "trajectory" in records[0]:
        rows = []
        for record in records:
            status = (record["verdict"] or {}).get("status")
            if status is None:
                raise RuntimeError(f"{path}: trajectory {record['task_id']} carries no verdict")
            turns = [
                event["payload"]["raw"]
                for event in record["trajectory"]["events"]
                if event["kind"] == "model_output"
            ]
            if not turns:
                raise RuntimeError(f"{path}: trajectory {record['task_id']} has no model turns")
            rows.append(
                {
                    "task_id": record["task_id"],
                    "recorded_status": status,
                    "recorded_extract_status": None,
                    "strict_answer": record["trajectory"]["final_answer"],
                    "text": turns[-1],
                }
            )
        return "agent", rows
    rows = [
        {
            "task_id": record["task_id"],
            "recorded_status": record["verifier_status"],
            "recorded_extract_status": record["extract_status"],
            "strict_answer": None,
            "text": record["raw_output"],
        }
        for record in records
    ]
    return "direct", rows


def score_arm(label: str, rows: list[dict[str, Any]], references: dict[str, Any]) -> dict[str, Any]:
    """Run the three rungs over one arm and hold the strict rung to the record.

    Each rung is stored as what it *adds* and then returned cumulatively: the
    rungs are escalating ceilings, so a task the strict rung already read must
    also appear at ``coerce`` and ``lenient``. Summing the increments instead
    would under-report the upper rungs by the strict-correct count.
    """
    genres = {row["recorded_extract_status"] is None for row in rows}
    if len(genres) > 1:
        raise RuntimeError(f"{label}: the file mixes direct generations and trajectories")
    mode = "agent" if genres == {True} else "direct"

    added: dict[str, set[str]] = {protocol: set() for protocol in PROTOCOLS}
    provenance: dict[str, str] = {}
    mismatches: list[str] = []
    drift: list[str] = []
    for row in rows:
        task_id = row["task_id"]
        reference = references[task_id]
        text = row["text"]
        if row["recorded_extract_status"] is not None:
            # Direct arm: the strict rung is the stored extractor, re-run.
            exact = extract(text)
            if exact.status.value != row["recorded_extract_status"]:
                mismatches.append(
                    f"{task_id}: extract {exact.status.value} != {row['recorded_extract_status']}"
                )
            strict_answer = exact.value if exact.status is ExtractStatus.OK else None
        else:
            # Agent arm: the strict rung is what the loop closed with, so the
            # rung is never re-derived. But the runtime's own parser is applied
            # to the last turn, because if the two disagree then this file is
            # not read the way the run read it and the rungs above would be
            # measured against the wrong ruler.
            strict_answer = row["strict_answer"]
            action = parse_action(text).action
            parsed_answer = action.answer if isinstance(action, FinalAction) else None
            if parsed_answer != strict_answer:
                mismatches.append(
                    f"{task_id}: the runtime parser reads {parsed_answer!r} from the last turn "
                    f"but the trajectory closed with {strict_answer!r}"
                )

        if strict_answer is not None:
            status = verify_answer(strict_answer, reference, task_id=task_id).status.value
            if status != row["recorded_status"]:
                # A verdict that only moves between two non-correct statuses
                # cannot change an accuracy, so it is measured rather than
                # fatal. The recorded arms were scored by an older verifier
                # (see the doc); crossing the correct boundary is still fatal,
                # because then a headline number would depend on the version.
                if "correct" in (status, row["recorded_status"]):
                    mismatches.append(f"{task_id}: verifier {status} != {row['recorded_status']}")
                else:
                    drift.append(f"{task_id}: {status} != {row['recorded_status']}")
            if status == "correct":
                added["strict"].add(task_id)
        elif row["recorded_status"] != "invalid_prediction":
            mismatches.append(
                f"{task_id}: nothing extracted but the record says {row['recorded_status']}"
            )

        if task_id not in added["strict"]:
            candidate = numeric_answer_candidate(text)
            if candidate is not None and (
                verify_answer(candidate, reference, task_id=task_id).status.value == "correct"
            ):
                added["coerce"].add(task_id)
                provenance[task_id] = "coerce"

        if task_id not in added["strict"] and task_id not in added["coerce"]:
            lenient = extract_solution_answer(text)
            lenient_ok = lenient.status is ExtractStatus.OK and lenient.value is not None
            if lenient_ok and (
                verify_answer(lenient.value, reference, task_id=task_id).status.value == "correct"
            ):
                added["lenient"].add(task_id)
                provenance[task_id] = "lenient"

    if mismatches:
        raise RuntimeError(
            f"{label}: {len(mismatches)} rows disagree with the record, so this harness is not "
            f"the one that scored the file; first: {mismatches[0]}"
        )
    if drift:
        print(
            f"note: {label}: {len(drift)} rows whose recorded verdict the current verifier "
            f"does not reproduce (non-correct statuses only, so no accuracy moves); "
            f"first: {drift[0]}"
        )

    cumulative = {
        "strict": added["strict"],
        "coerce": added["strict"] | added["coerce"],
        "lenient": added["strict"] | added["coerce"] | added["lenient"],
    }
    n = len(rows)
    return {
        "mode": mode,
        "n": n,
        "protocols": {
            protocol: {
                "correct": len(cumulative[protocol]),
                "accuracy": len(cumulative[protocol]) / n,
                "gained_vs_strict": len(cumulative[protocol]) - len(cumulative["strict"]),
            }
            for protocol in PROTOCOLS
        },
        # Kept per task, not just counted: a paired comparison between two arms
        # under a tolerant protocol needs to know *which* tasks moved, and
        # re-deriving that would mean paying for every verification twice.
        "correct_task_ids": {protocol: sorted(cumulative[protocol]) for protocol in PROTOCOLS},
        # Which tasks this arm was scored on at all -- an arm that covers a
        # subset of the frozen set (the rule arm was rolled out on 124) can only
        # be paired against another arm on the tasks they share.
        "scored_task_ids": sorted(row["task_id"] for row in rows),
        "n_coerced_answers": sum(1 for value in provenance.values() if value == "coerce"),
        "verifier_addressable": sum(1 for row in rows if row["recorded_status"] in VALID_STATUSES),
        # Recorded verdicts this verifier does not reproduce, restricted above to
        # rows where neither side says correct. Empty means the recorded
        # accuracies hold verbatim under the current verifier.
        "verifier_drift": drift,
    }


def _format_table(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| arm | mode | n | strict | +coerce | +lenient | coerced answers | verifier-valid |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, data in results.items():
        p = data["protocols"]
        lines.append(
            f"| {label} | {data['mode']} | {data['n']} | "
            f"{p['strict']['correct']} ({p['strict']['accuracy']:.1%}) | "
            f"{p['coerce']['correct']} ({p['coerce']['accuracy']:.1%}) | "
            f"{p['lenient']['correct']} ({p['lenient']['accuracy']:.1%}) | "
            f"{data['n_coerced_answers']} | {data['verifier_addressable']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=PREDICTIONS_JSONL",
        help="repeatable; NAME labels the arm in the output",
    )
    parser.add_argument("--task-ids", type=Path, default=DEFAULT_TASK_IDS)
    parser.add_argument("--eval-parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N rows (smoke)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    task_ids = args.task_ids.read_text().splitlines()
    canonical = sha256_hex("\n".join(task_ids).encode())
    if canonical != EXPECTED_TASK_IDS_SHA256:
        raise RuntimeError(
            f"task set sha256 {canonical} != expected {EXPECTED_TASK_IDS_SHA256}; "
            "these are not the frozen 200 tasks"
        )

    arms: list[tuple[str, Path]] = []
    for spec in args.arm:
        name, _, raw_path = spec.partition("=")
        if not name or not raw_path:
            print(f"usage error: --arm expects NAME=PATH, got {spec!r}")
            return 2
        arms.append((name, Path(raw_path)))

    references = load_references(args.eval_parquet, task_ids)

    results: dict[str, dict[str, Any]] = {}
    for name, path in arms:
        mode, rows = arm_rows(path)
        if args.limit:
            rows = rows[: args.limit]
        unknown = [row["task_id"] for row in rows if row["task_id"] not in references]
        if unknown:
            raise RuntimeError(f"{name}: {len(unknown)} rows outside the frozen task set")
        ids = [row["task_id"] for row in rows]
        if len(set(ids)) != len(ids):
            # A repeated task would be scored twice and inflate n and every
            # accuracy on this arm without any visible symptom.
            raise RuntimeError(f"{name}: {len(ids) - len(set(ids))} duplicate task rows")
        results[name] = score_arm(name, rows, references)
        if mode != results[name]["mode"]:
            raise RuntimeError(f"{name}: read as {mode} but scored as {results[name]['mode']}")
        strict, coerce, lenient = (
            results[name]["protocols"][protocol]["correct"] for protocol in PROTOCOLS
        )
        if not strict <= coerce <= lenient:
            raise RuntimeError(f"{name}: the rungs must nest, got {strict}/{coerce}/{lenient}")

    print(_format_table(results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"task_ids_sha256": canonical, "arms": results}, indent=2, sort_keys=True) + "\n"
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
