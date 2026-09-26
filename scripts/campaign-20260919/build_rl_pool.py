"""Build the formal RL task pool (jsonl + verl parquet) from v1 rl_dev.

Campaign tool (2026-09-19): samples N tasks deterministically from
data/processed/v1/rl_dev.parquet, checks leakage against frozen_eval and the
SFT training set, then writes:
  - pool jsonl in rl_smoke.jsonl format ({task, reference} per line)
  - verl-format parquet: one row per task, env_kwargs.task_ids = group repeats
"""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path("/root/autodl-tmp/Adaptive-Solver-main-git")


def _stable_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    ordered = sorted(rows, key=lambda r: r["task_id"])
    return ordered[seed % max(len(ordered) - n, 1) :][:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-parquet", type=Path, required=True)
    args = ap.parse_args()

    rl_dev = pq.read_table(REPO / "data/processed/v1/rl_dev.parquet").to_pylist()
    frozen = pq.read_table(REPO / "data/processed/v1/frozen_eval.parquet").to_pylist()
    frozen_ids = {r["task_id"] for r in frozen}
    sft_ids = {
        json.loads(r["record_json"])["task_id"]
        for r in pq.read_table(REPO / "data/processed/sft_dp_v1/train.parquet").to_pylist()
    }
    clean = [r for r in rl_dev if r["task_id"] not in frozen_ids and r["task_id"] not in sft_ids]
    if len(clean) < args.n:
        raise SystemExit(f"only {len(clean)} leakage-free rl_dev tasks, need {args.n}")
    sample = _stable_sample(clean, args.n, args.seed)
    ids = [r["task_id"] for r in sample]
    assert len(ids) == len(set(ids)), "duplicate task ids in sample"

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as fh:
        for r in sample:
            meta = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
            row = {
                "task": {
                    "task_id": r["task_id"],
                    "problem": r["problem"],
                    "answer_type": r["answer_type"],
                    "dataset": r["dataset"],
                    "split": "rl_dev",
                    "source_hash": r["source_hash"],
                    "pipeline_version": r["pipeline_version"],
                    "metadata": meta,
                },
                "reference": {
                    "value": r["reference_value"],
                    "answer_type": r["answer_type"],
                    "acceptable_forms": list(r["reference_acceptable_forms"] or []),
                },
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = []
    for i, r in enumerate(sample):
        rows.append(
            {
                "prompt": [{"role": "user", "content": r["problem"]}],
                "data_source": "adaptive_math",
                "env_kwargs": {"task_ids": [r["task_id"]] * args.group_size},
                "task_id": r["task_id"],
                "extra_info": {"index": i},
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), args.out_parquet, compression="zstd")

    print(
        f"pool: {len(ids)} tasks (seed {args.seed}, group {args.group_size})\n"
        f"jsonl: {args.out_jsonl}\nparquet: {args.out_parquet}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
