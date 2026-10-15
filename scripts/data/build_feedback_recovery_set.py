"""Build the feedback-recovery A/B task pools (2026-10-14 protocol, section 4.1).

Deterministic and reproducible from pinned inputs:

  pool    = frozen_eval.parquet
            minus the 200 tasks spent by the E0/E1 + MVP arms
            minus the 100 tasks spent by the round-1 diagnostic
            minus the 2 tasks spent by this study's model pre-flight
  window  = the two adjacent difficulty layers L1 = {3.5, 3.75, 4.0} and
            L2 = {4.25, 4.375, 4.5, 4.75} -- the dataset's own labels
  numeric = reference_value is a plain number (see NUMERIC_PATTERN); the pool's
            answer_type is "expression" for every row, so this value-form rule
            is the closest honest proxy for the report's "numeric answers
            preferred"
  rank    = sha256(main_seed : layer : task_id), ascending
  A       = ranks 1..5 of each layer   (10 debug tasks, never in B/C stats)
  B       = ranks 6..25 of each layer  (40 development tasks)

Every input is pinned by sha256 in the manifest, which also lists every selected
task id, the exclusions and the overlap with the training split. The same commit
with the same inputs reproduces the same ids.

Layer choice is evidence-based but outcome-blind with respect to the new pools:
the difficulty window was picked from the TIR model's prior accuracy and
execution-error rate on the *already spent* 200 E0/E1 tasks (see the manifest's
"layer_rationale"), never from results of tasks selected here.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_math.recovery.seeds import MAIN_SEED, stable_order_key

DEFAULT_POOL = "data/processed/v1/frozen_eval.parquet"
DEFAULT_TRAIN = "data/processed/v1/train.parquet"
DEFAULT_OUT_DIR = "data/processed/diagnostics/feedback_recovery"
DEFAULT_MANIFEST = "data/manifests/diagnostic_feedback_recovery.json"
DEFAULT_LAYERS = ("3.5,3.75,4.0", "4.25,4.375,4.5,4.75")

# Used-task sources, all excluded before selection.
EXCLUSION_SOURCES = (
    "artifacts/eval/thesis_e0_base_direct_b1/task_ids.txt",
    "artifacts/diagnostics/r1/*/task_pool.jsonl",
    "artifacts/diagnostics/feedback_recovery/*/preflight/*/task_pool.jsonl",
)

# A plain decimal number, optionally negative, with $ and \-escapes stripped:
# "1168", "296", "-15", "10.5". Fractions, radicals and symbolic answers are
# excluded rather than guessed at.
NUMERIC_PATTERN = re.compile(r"^[+-]?\d+(\.\d+)?$")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--layer", action="append", default=None, metavar="D1,D2,...")
    parser.add_argument("--a-per-layer", type=int, default=5)
    parser.add_argument("--b-per-layer", type=int, default=20)
    parser.add_argument("--seed", type=int, default=MAIN_SEED)
    parser.add_argument(
        "--allow-non-numeric",
        action="store_true",
        help="keep symbolic reference values too (disclosed as a deviation)",
    )
    return parser.parse_args()


def _used_ids(sources: tuple[str, ...]) -> dict[str, list[str]]:
    """Every already-spent task id, grouped by the file that proves it."""
    found: dict[str, list[str]] = {}
    for pattern in sources:
        for name in sorted(glob.glob(pattern)):
            path = Path(name)
            ids: list[str] = []
            if path.suffix == ".txt":
                ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            else:
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    task = record.get("task", record)
                    ids.append(str(task["task_id"]))
            found[str(path)] = ids
    return found


def _is_numeric(value: Any) -> bool:
    text = str(value).strip().replace("$", "").replace(",", "")
    text = re.sub(r"\\left\(|\\right\)|\{|\}", "", text)
    return bool(NUMERIC_PATTERN.match(text))


def main() -> int:
    args = _parse_args()
    layers = [
        sorted({float(item) for item in spec.split(",") if item})
        for spec in (args.layer or list(DEFAULT_LAYERS))
    ]
    if len(layers) < 2:
        raise SystemExit("at least two adjacent layers are required")
    for index in range(len(layers) - 1):
        if min(layers[index + 1]) <= max(layers[index]):
            raise SystemExit("layers must be adjacent, non-overlapping difficulty bands")

    pool_path = Path(args.pool)
    train_path = Path(args.train)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)

    frame = pd.read_parquet(pool_path)
    used_by_source = _used_ids(EXCLUSION_SOURCES)
    used: set[str] = set()
    duplicate_across_sources: set[str] = set()
    for ids in used_by_source.values():
        fresh = set(ids) - used
        duplicate_across_sources |= set(ids) - fresh
        used |= set(ids)

    missing_used = sorted(used - set(frame["task_id"]))
    work = frame[~frame["task_id"].isin(used)].copy()
    work["_difficulty"] = work["metadata"].map(
        lambda raw: json.loads(raw).get("difficulty") if isinstance(raw, str) else None
    )
    if work["_difficulty"].isna().any():
        raise SystemExit("pool contains rows without a numeric difficulty")
    work["_difficulty"] = work["_difficulty"].astype(float)
    if not args.allow_non_numeric:
        work = work[work["reference_value"].map(_is_numeric)]
    work = work.reset_index(drop=True)

    layer_records: list[dict[str, Any]] = []
    a_ids: list[str] = []
    b_ids: list[str] = []
    for index, difficulty_set in enumerate(layers):
        subset = work[work["_difficulty"].isin(difficulty_set)].copy()
        subset["_rank"] = [
            stable_order_key("task_selection", str(index), task_id)
            for task_id in subset["task_id"]
        ]
        subset = subset.sort_values("_rank", kind="mergesort").reset_index(drop=True)
        needed = args.a_per_layer + args.b_per_layer
        if len(subset) < needed:
            raise SystemExit(
                f"layer {index} has {len(subset)} eligible tasks, {needed} needed"
            )
        layer_a = subset["task_id"].tolist()[: args.a_per_layer]
        layer_b = subset["task_id"].tolist()[
            args.a_per_layer : args.a_per_layer + args.b_per_layer
        ]
        a_ids.extend(layer_a)
        b_ids.extend(layer_b)
        layer_records.append(
            {
                "index": index,
                "difficulties": sorted(difficulty_set),
                "eligible_rows": len(subset),
                "a_ids": layer_a,
                "b_ids": layer_b,
                "difficulty_counts": {
                    str(key): int(value)
                    for key, value in subset["_difficulty"].value_counts().sort_index().items()
                },
            }
        )

    if set(a_ids) & set(b_ids):
        raise SystemExit("A and B pools overlap")
    if (set(a_ids) | set(b_ids)) & used:
        raise SystemExit("selection overlaps an excluded task")

    def _write(ids: list[str], name: str) -> tuple[Path, str]:
        positions = {task_id: rank for rank, task_id in enumerate(ids)}
        rows = work[work["task_id"].isin(set(positions))].copy()
        rows["_order"] = rows["task_id"].map(positions)
        rows = rows.sort_values("_order", kind="mergesort").drop(
            columns=["_order", "_difficulty"]
        )
        rows = rows.reset_index(drop=True)
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(path, index=False)
        (out_dir / f"{path.stem}_task_ids.txt").write_text("\n".join(ids) + "\n")
        return path, _sha256_file(path)

    a_path, a_sha = _write(a_ids, "fr_a10.parquet")
    b_path, b_sha = _write(b_ids, "fr_b40.parquet")

    train_ids: set[str] = set()
    train_hashes: set[str] = set()
    train_note = "not read"
    if train_path.is_file():
        train = pd.read_parquet(train_path, columns=["task_id", "source_hash"])
        train_ids = set(train["task_id"])
        train_hashes = set(train["source_hash"])
        train_note = f"read {len(train_ids)} ids from {train_path}"

    selected = set(a_ids) | set(b_ids)
    selected_hashes = set(work[work["task_id"].isin(selected)]["source_hash"])
    r1_ids = set()
    for name, ids in used_by_source.items():
        if "/r1/" in name:
            r1_ids |= set(ids)

    manifest = {
        "name": "diagnostic_feedback_recovery",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "source_pool": {
            "path": str(pool_path),
            "sha256": _sha256_file(pool_path),
            "rows": len(frame),
        },
        "excluded": {
            "sources": {name: len(ids) for name, ids in used_by_source.items()},
            "union": len(used),
            "duplicates_across_sources": len(duplicate_across_sources),
            "not_found_in_pool": missing_used,
        },
        "pool_rows_after_exclusion": int((~frame["task_id"].isin(used)).sum()),
        "numeric_filter": {
            "enabled": not args.allow_non_numeric,
            "pattern": NUMERIC_PATTERN.pattern,
            "note": (
                "every row of the pool has answer_type=expression, so the "
                "report's numeric-answer preference is applied as a value-form "
                "rule on reference_value"
            ),
        },
        "selection": {
            "method": (
                "rank each layer by sha256(main_seed:layer:task_id); A = ranks "
                "1..a_per_layer, B = the next b_per_layer; outcome-blind, no "
                "resampling at run time"
            ),
            "seed": args.seed,
            "layers": layer_records,
        },
        "layer_rationale": {
            "source": "artifacts/mvp/dual_loop_eval/sft (TIR model on the spent E0/E1 200)",
            "observed": (
                "difficulty 3.5-4.5: 12.1% final accuracy, 32% of tasks hit an "
                "execution error; 1-3: 45.2% accuracy (too easy to leave room for "
                "a recovery effect); 5-7: 9-12% with the same error rate but a "
                "higher chance of an unrecoverable floor"
            ),
            "note": "outcome-blind for the tasks selected here; used only to fix the window",
        },
        "overlap_with_spent_tasks": {
            "round1_diagnostic_ids": {
                "count": len(r1_ids),
                "overlap": sorted(selected & r1_ids),
            },
            "e0_e1_200": {
                "count": len(used_by_source.get(EXCLUSION_SOURCES[0], [])),
                "overlap": sorted(
                    selected & set(used_by_source.get(EXCLUSION_SOURCES[0], []))
                ),
            },
        },
        "overlap_with_training_data": {
            "note": train_note,
            "train_rows": len(train_ids),
            "task_id_overlap": sorted(selected & train_ids),
            "source_hash_overlap": sorted(selected_hashes & train_hashes),
        },
        "a": {"path": str(a_path), "sha256": a_sha, "rows": len(a_ids), "task_ids": a_ids},
        "b": {"path": str(b_path), "sha256": b_sha, "rows": len(b_ids), "task_ids": b_ids},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"[fr-set] pool {len(frame)} - used {len(used)} -> {len(work)} eligible rows")
    for record in layer_records:
        print(
            f"  layer {record['index']} difficulty={record['difficulties']} "
            f"eligible={record['eligible_rows']} A={len(record['a_ids'])} B={len(record['b_ids'])}"
        )
    print(f"  A -> {a_path} ({len(a_ids)} rows, sha {a_sha[:12]})")
    print(f"  B -> {b_path} ({len(b_ids)} rows, sha {b_sha[:12]})")
    print(
        f"  train overlap: {len(manifest['overlap_with_training_data']['task_id_overlap'])} ids, "
        f"{len(manifest['overlap_with_training_data']['source_hash_overlap'])} hashes"
    )
    print(f"  manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
