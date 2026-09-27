"""Stream the dp_v1 pool out of the raw arrow shard and census the compiler.

The container is capped at 2 GiB (``/sys/fs/cgroup/memory.max``), and the
obvious route — ``load_source_records(spec, sample=10000)`` — materializes 10000
rows *including* the multi-kilobyte ``solution`` column as Python dicts, which
the OOM killer ends before the census starts.  Memory-mapping the shard and
reading one batch at a time keeps the peak at a few megabytes.

The identity check is the point of the first half: the dp_v1 builder recorded
``raw_records_sha256`` over exactly these records, so recomputing it turns "the
probes ran on the same pool" into a checked fact.  SHA-256 is updated
incrementally, which is equivalent because ``json.dumps`` of a list is the
bracket-joined dump of its elements.

Compiler only: no tool runs, no answer is verified, nothing is accepted here.

Usage:
    python sympy_pool_census.py --limit 10000 --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/root/autodl-tmp/Adaptive-Solver-main-git")
sys.path.insert(0, str(REPO / "src"))

import pyarrow as pa
import yaml

from adaptive_math.core.hashing import make_source_hash
from adaptive_math.data.sources import (
    SourceRegistry,
    _to_json,
    load_source_records,
)
from adaptive_math.training.sympy_compiler import compile_sympy_candidates

SHARD = (
    "/root/autodl-tmp/hf-cache/datasets/open-r1___open_r1-math-220k/default/"
    "0.0.0/e4e141ec9dea9f8326f4d347be56105859b2bd68/"
    "open_r1-math-220k-train-00000-of-00010.arrow"
)
DP_V1_RAW_SHA256 = "969d778943acdc8b5845c02bb1b1f409af9ff22a18270cbffcf8e4cd92f46fa3"
DP_V1_RAW_COUNT = 10000
DP_V1_CANONICALIZED = 2944


def stream_records(limit: int):
    """Yield the first ``limit`` rows of the train split as plain dicts."""
    with pa.memory_map(SHARD, "r") as source:
        reader = pa.ipc.open_stream(source)
        seen = 0
        for batch in reader:
            for row in batch.to_pylist():
                if seen >= limit:
                    return
                seen += 1
                yield row


def canonical_json(value: object) -> bytes:
    """The exact byte form the dp_v1 builder hashed: compact separators."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def source_hash_of(record: dict[str, object]) -> str:
    """The hash ``canonicalize_source`` keys tasks by: *default* separators."""
    return make_source_hash(json.dumps(record, sort_keys=True, ensure_ascii=False).encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="openr1_math_220k")
    parser.add_argument("--limit", type=int, default=DP_V1_RAW_COUNT)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry.model_validate(
        yaml.safe_load((REPO / "configs/data/sources.yaml").read_text())
    )
    spec = registry.sources[args.source]

    # 1. Cross-check the streaming reader against the library path on 200 rows.
    library_rows, _ = load_source_records(spec, sample=200)
    library_sha = hashlib.sha256(canonical_json(library_rows)).hexdigest()
    streamed_200 = [
        _to_json(row) for row in stream_records(limit=200)
    ]
    streamed_sha = hashlib.sha256(canonical_json(streamed_200)).hexdigest()
    reader_agrees = library_sha == streamed_sha
    print(
        json.dumps(
            {
                "reader_agrees_on_200_rows": reader_agrees,
                "library_sha256": library_sha,
                "streamed_sha256": streamed_sha,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not reader_agrees:
        raise SystemExit("streaming reader does not reproduce the library path")

    # 2. Identity over the whole pool, hashed incrementally.
    digest = hashlib.sha256()
    reasons: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    scanned = 0
    with_candidates = 0
    rows: list[dict[str, object]] = []
    digest.update(b"[")
    for index, raw_row in enumerate(stream_records(limit=args.limit)):
        record = _to_json(raw_row)
        digest.update(canonical_json(record) if index == 0 else b"," + canonical_json(record))
        problem = record.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            reasons["missing_problem"] += 1
            continue
        scanned += 1
        candidates = compile_sympy_candidates(problem, limit=args.candidates)
        built = [item for item in candidates if item.arguments is not None]
        if built:
            with_candidates += 1
            candidate_counts[str(len(built))] += 1
        else:
            reasons[candidates[0].reason] += 1
        rows.append(
            {
                "index": index,
                "source_hash": source_hash_of(record),
                "reasons": [item.reason for item in candidates],
                "candidates": [
                    {
                        "operation": item.arguments.operation,
                        "expression": item.arguments.expression,
                        "variables": item.arguments.variables,
                    }
                    for item in built
                ],
            }
        )
    digest.update(b"]")
    raw_sha256 = digest.hexdigest()

    summary = {
        "raw_records": scanned,
        "raw_records_sha256": raw_sha256,
        "matches_dp_v1_raw_sha256": raw_sha256 == DP_V1_RAW_SHA256,
        "dp_v1_canonicalized": DP_V1_CANONICALIZED,
        "scanned": scanned,
        "with_candidates": with_candidates,
        "ceiling_over_pool": round(with_candidates / scanned, 4) if scanned else 0.0,
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "reasons": dict(sorted(reasons.items())),
    }
    (args.out / "pool-census.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.out / "pool-census.rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
