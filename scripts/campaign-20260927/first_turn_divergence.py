"""Did the adapter arm actually load its adapter? §8b's post-run check.

An adapter arm that silently runs without its adapter produces a base arm
wearing an SFT label, and no artifact of the run would show it: the frozen
manifest has no adapter field, and the identity sidecar records what the
wrapper *intended*, not what the loader received. The run itself is the only
witness, so this reads it.

The rule, fixed in the pre-registration before any of these arms ran
(docs/results/campaign-2026-09-27/preregistration.md §8b): compare the first
``model_output`` of each shared task between the arm that must differ from the
candidate.  At or above 50% byte-identical, the adapter did not load and the
arm is **void**.

The threshold is set against measured poles on these 200 tasks under this
decoding: two genuinely different weight sets agree byte-for-byte on 0/200
(base vs sft) and 12/200 (sft vs r0) of their direct generations, while
identical weights and configuration reproduce 200/200. 50% is more than eight
times the worst observed different-weights rate.

CPU-only: reads trajectories.jsonl from dirs already on disk, generates
nothing.

Exit status: 0 = below threshold (pass), 1 = alarm (arm void), 2 = cannot
evaluate (nothing shared, unreadable input).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_THRESHOLD = 0.5


def first_outputs(dirs: list[Path]) -> dict[str, str]:
    """task_id -> raw text of that task's first ``model_output`` event.

    A task appears once even when several dirs carry it (shards overlap only if
    the caller passes overlapping dirs); the first dir wins.
    """
    outputs: dict[str, str] = {}
    for directory in dirs:
        path = directory / "trajectories.jsonl"
        if not path.is_file():
            raise SystemExit(f"cannot evaluate: no trajectories.jsonl in {directory}")
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                task_id = record["task_id"]
                if task_id in outputs:
                    continue
                for event in record["trajectory"]["events"]:
                    if event.get("kind") == "model_output":
                        outputs[task_id] = event["payload"]["raw"]
                        break
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        nargs="+",
        required=True,
        help="arm the candidate must differ from (its first outputs are the "
        "baseline); one or more trajectory dirs",
    )
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    reference = first_outputs(args.reference)
    candidate = first_outputs(args.candidate)

    shared = sorted(set(reference) & set(candidate))
    if not shared:
        print(
            f"CANNOT EVALUATE: {args.candidate_label} and {args.reference_label} "
            "share no tasks",
            file=sys.stderr,
        )
        return 2

    identical = [task_id for task_id in shared if reference[task_id] == candidate[task_id]]
    rate = len(identical) / len(shared)
    only_reference = len(reference) - len(shared)
    only_candidate = len(candidate) - len(shared)

    print(
        f"first model_output: {args.candidate_label} vs {args.reference_label}\n"
        f"  reference  {args.reference_label}: {len(reference)} tasks\n"
        f"  candidate  {args.candidate_label}: {len(candidate)} tasks\n"
        f"  shared {len(shared)} tasks "
        f"(only in reference {only_reference}, only in candidate {only_candidate})\n"
        f"  byte-identical first outputs: {len(identical)}/{len(shared)} "
        f"= {rate:.1%}  (threshold {args.threshold:.0%})"
    )

    if rate >= args.threshold:
        print(
            f"ALARM: {args.candidate_label} agrees with {args.reference_label} on "
            f"{rate:.1%} of shared tasks' first outputs, at or above "
            f"{args.threshold:.0%} -- the adapter did not load. "
            f"{args.candidate_label} is VOID.",
            file=sys.stderr,
        )
        return 1

    print(
        f"PASS: {args.candidate_label} diverges from {args.reference_label} as an "
        "arm with different weights must."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
