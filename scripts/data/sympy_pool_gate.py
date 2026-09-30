"""Decisive Phase 0 gate: verified sympy demos obtainable from the dp_v1 pool.

Reads the candidate list the census produced (so nothing is recompiled), then,
for a stratified sample of the candidate-bearing problems, does what the plan
requires and nothing less:

* ``canonicalize_source`` decides whether the record is even in the pool — a
  reference that fails its own self-check is quarantined and can never become a
  demo;
* the production ``SympyTool`` actually executes each candidate in rank order;
* the production ``verify_answer`` decides, against the hidden reference, which
  execution counts.

A record is accepted only when both gates pass.  The denominator is reported
alongside every count, and the sample is stratified across the whole 10k pool
rather than taken from its head.

Resumable: accepted/rejected indices are appended to a jsonl and skipped on a
re-run, because a full pass costs hours of CPU and must survive an interruption.

Usage:
    python sympy_pool_gate.py --sample 300 --out <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path("/root/autodl-tmp/Adaptive-Solver-main-git")
sys.path.insert(0, str(REPO / "src"))

import pyarrow as pa
import yaml

from adaptive_math.data.canonicalize import canonicalize_source
from adaptive_math.data.sources import SourceRegistry, _to_json
from adaptive_math.tools.base import ToolContext
from adaptive_math.tools.sympy_tool import SympyTool
from adaptive_math.training.sympy_compiler import compile_sympy_candidates
from adaptive_math.verifier import VerifierStatus, verify_answer

SHARD = (
    "/root/autodl-tmp/hf-cache/datasets/open-r1___open_r1-math-220k/default/"
    "0.0.0/e4e141ec9dea9f8326f4d347be56105859b2bd68/"
    "open_r1-math-220k-train-00000-of-00010.arrow"
)
_TRAILING_ZEROS = re.compile(r"^(-?\d+)\.0+$")


def tool_output_to_answer(output: str) -> str:
    """Normalize a tool result into the answer surface the verifier compares.

    ``numeric`` evaluates at 50 significant digits, so a whole number comes back
    as ``888.000...0``; ``solve`` returns a Python list; and sympy prints powers
    as ``x**2`` where the verifier's expression parser reads only ``x^2``.
    """
    text = output.strip()
    whole = _TRAILING_ZEROS.match(text)
    if whole is not None:
        return whole.group(1)
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return text.replace("**", "^")


def stream_records(limit: int):
    """One row at a time, never the whole batch.

    A row of this shard is ~52 KB (``solution`` and ``generations`` carry full
    worked solutions), and the shard arrives as a single batch of 10374 rows:
    ``batch.to_pylist()`` materializes ~550 MB of Python objects at once, which
    the 2 GiB cgroup cap ends with SIGKILL.  Slicing keeps the row count
    materialized at exactly one while the arrow batch stays memory-mapped.
    """
    with pa.memory_map(SHARD, "r") as source:
        reader = pa.ipc.open_stream(source)
        seen = 0
        for batch in reader:
            for index in range(batch.num_rows):
                if seen >= limit:
                    return
                seen += 1
                yield batch.slice(index, 1).to_pylist()[0]


def extract(indices: list[int], limit: int) -> dict[int, dict[str, object]]:
    """Materialize only the sampled rows, one at a time, keyed by pool index."""
    wanted = set(indices)
    rows: dict[int, dict[str, object]] = {}
    with pa.memory_map(SHARD, "r") as source:
        reader = pa.ipc.open_stream(source)
        seen = 0
        for batch in reader:
            for position in range(batch.num_rows):
                if seen >= limit or not wanted:
                    return rows
                index = seen
                seen += 1
                if index not in wanted:
                    continue
                wanted.discard(index)
                rows[index] = _to_json(batch.slice(position, 1).to_pylist()[0])
    return rows


def stratified(indices: list[int], sample: int) -> list[int]:
    """Evenly spaced across the whole pool, so the head cannot dominate."""
    if sample >= len(indices):
        return list(indices)
    stride = len(indices) / sample
    return [indices[int(index * stride)] for index in range(sample)]


async def run(args: argparse.Namespace) -> int:
    registry = SourceRegistry.model_validate(
        yaml.safe_load((REPO / "configs/data/sources.yaml").read_text())
    )
    spec = registry.sources[args.source]
    rows = [
        json.loads(line)
        for line in (args.census / "pool-census.rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_index = {int(row["index"]): row for row in rows}
    candidate_indices = sorted(
        index for index, row in by_index.items() if row["candidates"]
    )
    selected = stratified(candidate_indices, args.sample)
    print(
        json.dumps(
            {
                "pool_raw_records": 10000,
                "candidate_bearing": len(candidate_indices),
                "sampled": len(selected),
                "first_index": selected[0],
                "last_index": selected[-1],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)

    # Pull the sampled rows out of the memory-mapped shard *before* any worker
    # process exists, then let the mapping go.  Holding it across the tool loop
    # keeps ~550 MB of page cache charged to a cgroup that already runs at
    # 1.4 GiB of 2 GiB, and the next spawned child is what tips it over.
    named = extract(selected, args.limit)
    print(
        json.dumps({"extracted": len(named)}, sort_keys=True),
        flush=True,
    )

    done_path = args.out / "pool-gate.done.jsonl"
    done: set[int] = set()
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["index"]))
    handle = done_path.open("a")

    tool = SympyTool()
    counters: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    accepted: list[dict[str, object]] = []
    started = time.monotonic()

    scanned = 0
    for index in selected:
        if index in done:
            continue
        scanned += 1
        kept, quarantined = canonicalize_source(spec, [named[index]])
        if not kept:
            reason = str(quarantined[0].reason) if quarantined else "unknown"
            counters[f"quarantine_{reason}"] += 1
            handle.write(json.dumps({"index": index, "outcome": "quarantined"}) + "\n")
            handle.flush()
            continue
        labeled = kept[0]
        candidates = compile_sympy_candidates(
            labeled.task.problem, limit=args.candidates
        )
        verdict: str | None = None
        for rank, candidate in enumerate(candidates):
            if candidate.arguments is None:
                continue
            context = ToolContext(
                trace_id=f"gate:{labeled.task.source_hash[:20]}",
                task_id=labeled.task.task_id,
                remaining_observation_chars=8000,
                remaining_python_seconds=args.timeout,
            )
            try:
                result = await asyncio.wait_for(
                    tool.execute(candidate.arguments, context), timeout=args.timeout
                )
            except TimeoutError:
                counters["tool_timeout"] += 1
                continue
            if not result.ok:
                counters[f"tool_{result.error_code}"] += 1
                continue
            counters["executed"] += 1
            answer = tool_output_to_answer(result.output)
            outcome = verify_answer(
                answer, labeled.reference, task_id=labeled.task.task_id
            )
            verdicts[str(outcome.status)] += 1
            if outcome.status is VerifierStatus.CORRECT:
                verdict = f"correct_at_{rank}"
                counters["accepted"] += 1
                accepted.append(
                    {
                        "index": index,
                        "task_id": labeled.task.task_id,
                        "rank": rank,
                        "operation": candidate.arguments.operation,
                        "expression": candidate.arguments.expression,
                        "tool_output": result.output,
                        "answer": answer,
                        "reference": labeled.reference.value,
                        "problem": labeled.task.problem[:300],
                    }
                )
                break
        if verdict is None:
            counters["no_candidate_verified"] += 1
        handle.write(
            json.dumps(
                {
                    "index": index,
                    "outcome": "accepted" if verdict else "rejected",
                    "verdict": verdict or "",
                }
            )
            + "\n"
        )
        handle.flush()
        if scanned % 10 == 0:
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "progress": scanned,
                        "sample": len(selected),
                        "accepted": counters["accepted"],
                        "seconds_per_record": round(elapsed / scanned, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    elapsed = time.monotonic() - started
    sampled = scanned
    accepted_count = counters["accepted"]
    summary = {
        "pool_raw_records": args.limit,
        "candidate_bearing": len(candidate_indices),
        "sampled": len(selected),
        "scanned": sampled,
        "accepted": accepted_count,
        "accept_rate": round(accepted_count / sampled, 4) if sampled else 0.0,
        "projected_over_candidate_bearing": (
            round(accepted_count / sampled * len(candidate_indices), 1)
            if sampled
            else 0.0
        ),
        "verdicts": dict(sorted(verdicts.items())),
        "counters": dict(sorted(counters.items())),
        "elapsed_seconds": round(elapsed, 1),
    }
    (args.out / "pool-gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.out / "pool-gate.accepted.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted)
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="openr1_math_220k")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--census",
        type=Path,
        default=Path("/root/autodl-tmp/tmp/census"),
        help="directory holding pool-census.rows.jsonl",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
