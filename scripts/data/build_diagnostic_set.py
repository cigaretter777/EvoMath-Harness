"""Build the round-1 budget-diagnostic task set (thesis).

Deterministic, reproducible from the pinned inputs:

  pool      = frozen_eval.parquet minus the 200 tasks already spent by the
              E0/E1 eval arms (thesis_e0_base_direct_b1/task_ids.txt)
  strata    = pool sorted by (difficulty, task_id), split into 4 equal chunks
  formal100 = 25 sampled per stratum (seeded, uniform within stratum)
  pilot20   = first 5 of each stratum's formal 25 (strict subset of formal100)

The parquets land under data/processed/ (gitignored build artifacts); the
committed record is this script plus data/manifests/diagnostic_r1.json, which
pins every input by sha256 and lists every selected task id. The same commit
with the same inputs reproduces the same task ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_POOL = "data/processed/v1/frozen_eval.parquet"
DEFAULT_USED_IDS = "artifacts/eval/thesis_e0_base_direct_b1/task_ids.txt"
DEFAULT_OUT_DIR = "data/processed/diagnostics/r1"
DEFAULT_MANIFEST = "data/manifests/diagnostic_r1.json"
SELECTION_SEED = 20261014


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
    parser.add_argument("--used-ids", default=DEFAULT_USED_IDS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--strata", type=int, default=4)
    parser.add_argument("--formal-size", type=int, default=100)
    parser.add_argument("--pilot-size", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pool_path = Path(args.pool)
    used_path = Path(args.used_ids)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)

    if args.formal_size % args.strata or args.pilot_size % args.strata:
        raise SystemExit("formal/pilot sizes must divide evenly across strata")
    formal_per = args.formal_size // args.strata
    pilot_per = args.pilot_size // args.strata
    if pilot_per > formal_per:
        raise SystemExit("pilot must be a subset of the formal set")

    frame = pd.read_parquet(pool_path)
    used = {
        line.strip() for line in used_path.read_text().splitlines() if line.strip()
    }
    missing_used = sorted(used - set(frame["task_id"]))
    work = frame[~frame["task_id"].isin(used)].copy()
    difficulty = work["metadata"].map(
        lambda raw: json.loads(raw).get("difficulty") if isinstance(raw, str) else None
    )
    if difficulty.isna().any():
        raise SystemExit("pool contains rows without a numeric difficulty")
    work["_difficulty"] = difficulty.astype(float)
    work = work.sort_values(["_difficulty", "task_id"], kind="mergesort")
    work = work.reset_index(drop=True)

    ids = work["task_id"].tolist()
    chunk_indices = np.array_split(np.arange(len(work)), args.strata)

    strata_record: list[dict[str, Any]] = []
    formal_ids: list[str] = []
    pilot_ids: list[str] = []
    for index, indices in enumerate(chunk_indices):
        chunk_ids = [ids[i] for i in indices]
        rng = random.Random(args.seed * 100 + index)
        selected = rng.sample(chunk_ids, formal_per)
        formal_ids.extend(selected)
        pilot_ids.extend(selected[:pilot_per])
        stratum_difficulty = work["_difficulty"].iloc[indices]
        strata_record.append(
            {
                "index": index,
                "pool_rows": len(chunk_ids),
                "difficulty_min": float(stratum_difficulty.min()),
                "difficulty_max": float(stratum_difficulty.max()),
                "formal_ids": selected,
                "pilot_ids": selected[:pilot_per],
            }
        )

    if set(pilot_ids) > set(formal_ids):
        raise SystemExit("pilot is not a subset of the formal set")
    if set(formal_ids) & used:
        raise SystemExit("formal selection overlaps the excluded used-200 set")

    def _write_subset(ordered_ids: list[str], path: Path) -> str:
        positions = {task_id: rank for rank, task_id in enumerate(ordered_ids)}
        rows = work[work["task_id"].isin(set(positions))].copy()
        rows["_order"] = rows["task_id"].map(positions)
        rows = rows.sort_values("_order", kind="mergesort")
        rows = rows.drop(columns=["_order", "_difficulty"]).reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(path, index=False)
        return _sha256_file(path)

    formal_path = out_dir / f"diagnostic_r1_{args.formal_size}.parquet"
    pilot_path = out_dir / f"diagnostic_r1_pilot{args.pilot_size}.parquet"
    formal_sha = _write_subset(formal_ids, formal_path)
    pilot_sha = _write_subset(pilot_ids, pilot_path)
    (out_dir / f"task_ids_{args.formal_size}.txt").write_text(
        "\n".join(formal_ids) + "\n"
    )
    (out_dir / f"task_ids_pilot{args.pilot_size}.txt").write_text(
        "\n".join(pilot_ids) + "\n"
    )

    manifest = {
        "name": "diagnostic_r1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "source_pool": {
            "path": str(pool_path),
            "sha256": _sha256_file(pool_path),
            "rows": len(frame),
        },
        "excluded_used_ids": {
            "path": str(used_path),
            "sha256": _sha256_file(used_path),
            "count": len(used),
            "not_found_in_pool": missing_used,
        },
        "pool_rows_after_exclusion": len(work),
        "selection": {
            "method": (
                "sort pool by (difficulty, task_id); equal contiguous strata; "
                "uniform seeded sample within stratum; pilot = first pilot_per "
                "of each stratum's formal sample"
            ),
            "seed": args.seed,
            "strata_count": args.strata,
        },
        "strata": strata_record,
        "formal": {
            "path": str(formal_path),
            "sha256": formal_sha,
            "rows": len(formal_ids),
            "task_ids": formal_ids,
        },
        "pilot": {
            "path": str(pilot_path),
            "sha256": pilot_sha,
            "rows": len(pilot_ids),
            "task_ids": pilot_ids,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"[diagnostic_r1] pool {len(frame)} - used {len(used)} -> {len(work)} rows")
    for record in strata_record:
        print(
            f"  stratum {record['index']}: pool={record['pool_rows']} "
            f"difficulty=[{record['difficulty_min']}, {record['difficulty_max']}] "
            f"formal={len(record['formal_ids'])} pilot={len(record['pilot_ids'])}"
        )
    print(f"  formal -> {formal_path} ({len(formal_ids)} rows, sha {formal_sha[:12]})")
    print(f"  pilot  -> {pilot_path} ({len(pilot_ids)} rows, sha {pilot_sha[:12]})")
    print(f"  manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
