"""Merge sharded rule_baseline output dirs into the final arm directory.

Each shard writes its own trajectories.jsonl + summary.json + COMPLETE; this
script concatenates them (deterministic order: sorted by task_id) and writes
the final summary.json + manifest.json + COMPLETE that the queue and the
analysis script consume. routing.jsonl is identical across shards (each shard
routes the full pool) and is copied from the first shard.

CPU-only. Never regenerates anything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    for shard in args.shards:
        if not (shard / "COMPLETE").is_file():
            raise SystemExit(f"shard not complete: {shard}")
    if not (args.shards[0] / "routing.jsonl").is_file():
        raise SystemExit(f"no routing.jsonl in {args.shards[0]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for shard in args.shards:
        with (shard / "trajectories.jsonl").open() as handle:
            for line in handle:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["task_id"])

    with (args.output_dir / "trajectories.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # routing.jsonl is identical across shards; copy the first.
    (args.output_dir / "routing.jsonl").write_bytes(
        (args.shards[0] / "routing.jsonl").read_bytes()
    )

    summary = {
        "mode": None,
        "task_count": None,
        "rolled_out": 0,
        "direct_reused": 0,
        "correct_count": 0,
        "tool_calls_total": 0,
        "invalid_actions_total": 0,
        "generated_tokens_total": 0,
        "routing_distribution": None,
    }
    for shard in args.shards:
        part = json.loads((shard / "summary.json").read_text())
        summary["mode"] = part["mode"]
        summary["task_count"] = part["task_count"]
        summary["rolled_out"] += part["rolled_out"]
        summary["direct_reused"] += part["direct_reused"]
        summary["correct_count"] += part["correct_count"]
        summary["tool_calls_total"] += part["tool_calls_total"]
        summary["invalid_actions_total"] += part["invalid_actions_total"]
        summary["generated_tokens_total"] += part["generated_tokens_total"]
        summary["routing_distribution"] = part["routing_distribution"]

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "merged_from_shards": [str(s) for s in args.shards],
                "git_sha": git_sha(),
                "trajectory_count": len(records),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (args.output_dir / "COMPLETE").write_text("")
    print(f"merged {len(records)} trajectories from {len(args.shards)} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
