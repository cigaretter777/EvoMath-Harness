"""CPU-only identity gate: a candidate eval arm must be pairable with the
stored arm it will be compared against, BEFORE any GPU minute is spent.

Why this exists
---------------
The 2026-09-26 campaign spent ~3.5 GPU hours running agent arms on the RL
training pool while the direct arms evaluated the frozen eval parquet. Zero
task overlap, so no pairing was possible. The failure was an identity
assumption nobody verified. ``verify_eval_alignment.sh`` covers the task-set
half of that; this covers the rest of the identity, which the same incident
showed cannot be left to good intentions:

* the same decoding configuration,
* the same tool set and agent/reward configs,
* the same task set,
* the same evaluation source, or a drift from the baseline's commit that has
  been argued for explicitly.

The one thing that is *not* required to match is the arm's weights. Two paired
arms differ there by construction -- the stored ``base+tool`` arm runs the base
snapshot, the arms it is paired against run an adapter or a merged checkpoint --
so ``WEIGHTS_FIELDS`` are reported in the gate's output rather than blocking.
The gate cannot know which adapter was intended; it can make sure the
difference is never silent.

The last one is not a formality. Between the commit that produced the stored
``base+tool`` arm (``fdf9276``) and this branch's HEAD, eleven files under the
evaluation path changed. They were audited line by line and every one of them
adds an optional field that is excluded from the canonical trajectory hash,
adds an API, or adds a type annotation -- no trajectory-generating behaviour
changed. That audit is exactly the kind of thing that must not live only in
someone's memory, so this script refuses to pass silently when the diff is
non-empty; it requires the drift to be named in a justification file.

Usage
-----
    verify_arm_identity.py --baseline <shard-manifest.json> \
                           --candidate <identity.json> \
                           [--repo <path>] [--justify <drift-justification.md>]

``--baseline`` is the stored arm's per-shard ``manifest.json`` (the merged
manifest drops the fields this gate needs -- see the merged ``manifest.json``
for ``base_tool``, which keeps only ``git_sha`` and counts). ``--candidate`` is
what the new arm will run with, emitted on CPU by
``scripts/eval/emit_arm_identity.py`` from the runner's own loaders and hashes.
Run it again after the run against the arm's real ``manifest.json``: the
pre-flight prediction is what makes the gate fail on CPU rather than after two
GPU hours, and the post-run pass is the authoritative one.

One operational consequence worth knowing before editing anything. The agent
runner ``scripts/campaign-20260926/rule_baseline.py`` writes ``git hash-object``
of *its own committed file* into every manifest as ``router_source_sha256``,
which is a blocking field. While that file stays byte-identical to the blob the
stored arm pinned (``e591c886…``), the stored arm is pairable; edit it -- even
to add a flag -- and every new agent arm records a different hash and the gate
blocks the pair. The runner is immutable by contract, and
``tests/unit/eval/test_emit_arm_identity.py`` is what notices if someone
changes it.

Exit codes: 0 aligned, 1 not aligned, 2 usage/setup error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Fields that are expected to differ between two arms being paired: they name
# the arm or its bookkeeping, not its identity. Everything else must match.
ALLOWED_TO_DIFFER = frozenset(
    {
        "arm",
        "run_id",
        # The candidate runs at a later commit than the baseline by
        # construction; source drift is checked separately, and more strictly,
        # by diffing the evaluation path.
        "git_sha",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "shard_id",
        "shard_count",
        "status",
        "merged_from_shards",
        "trajectory_count",
        "task_count",
        "rolled_out",
        "direct_reused",
        "correct_count",
        "tool_calls_total",
        "invalid_actions_total",
        "generated_tokens_total",
        "routing_distribution",
    }
)

# The arm's weights. Two arms being paired differ in these *by construction* --
# comparing adapters is the entire point of having two arms -- so a difference
# must not block, or the gate could never pass a pair it was written for.
#
# They are reported rather than silently skipped, because "which weights is this
# arm" is the first question asked of a paired result. A wrong adapter is still
# not fatal here (the gate cannot know which adapter was intended), but it is
# printed in the gate's output instead of being invisible.
WEIGHTS_FIELDS = frozenset({"model", "adapter", "adapter_path", "adapter_sha256", "adapter_kind"})

# Source trees whose behaviour determines what a trajectory means. A drift here
# is what makes two arms unpairable.
EVAL_PATH = (
    "src/adaptive_math/agent",
    "src/adaptive_math/verifier",
    "src/adaptive_math/tools",
    "src/adaptive_math/evaluation",
    "src/adaptive_math/tasks",
    # Config contents, not just the paths recorded in the manifest: the paths
    # are compared as identity fields, but a path can stay identical while the
    # file it points at changes underneath it.
    "configs",
)


def compare_identities(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    allowed: frozenset[str] = ALLOWED_TO_DIFFER,
    weights: frozenset[str] = WEIGHTS_FIELDS,
) -> tuple[list[str], list[str]]:
    """Return ``(blocking, advisory)`` differences; blocking empty == aligned.

    Keys are compared over the union of both objects, because a field one arm
    records and the other silently drops is a weaker gate, not a passing one.
    Three categories, and the reason each is where it is:

    * **blocking** -- the harness identity: the task set, the decoding, the
      configs, the prompt/verifier/router sources. A difference here means the
      two arms are not measuring the same thing.
    * **advisory** -- the weights (reported, not fatal: they are the variable
      under study), and any field the candidate records that the baseline never
      wrote down. The second is the asymmetry that matters: a candidate
      recording *more* than the 09-26 arm is strengthening the gate and only
      deserves a note, whereas a candidate that *omits* a field the baseline
      recorded has regressed the evidence and blocks -- which is why omission
      stays blocking for every field, weights included.
    * **skipped** -- names and bookkeeping, listed in ``allowed``.
    """
    blocking: list[str] = []
    advisory: list[str] = []
    for key in sorted(set(baseline) | set(candidate)):
        if key in allowed:
            continue
        if key not in baseline:
            note = f"{key}: candidate records {candidate[key]!r}, baseline does not"
            if key in weights:
                note += " (weights: the baseline arm recorded none)"
            advisory.append(note)
        elif key not in candidate:
            blocking.append(f"{key}: baseline has {baseline[key]!r}, candidate omits it")
        elif baseline[key] != candidate[key]:
            problem = f"{key}: baseline {baseline[key]!r} != candidate {candidate[key]!r}"
            if key in weights:
                advisory.append(f"{problem} (weights: expected to differ between arms)")
            else:
                blocking.append(problem)
    return blocking, advisory


def source_drift(repo: Path, baseline_sha: str, paths: tuple[str, ...] = EVAL_PATH) -> str:
    """Return the diff --stat of the evaluation path between two revisions.

    Raises RuntimeError if the baseline revision is not present in this clone.
    A shallow clone cannot prove identity alignment, and an unprovable gate is
    a failing gate -- the caller turns this into GATE FAIL, never into a pass.
    """
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{baseline_sha}^{{commit}}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode != 0:
        raise RuntimeError(
            f"baseline revision {baseline_sha!r} is not present in {repo} "
            f"(shallow clone? fetch it before gating)"
        )

    result = subprocess.run(
        ["git", "diff", "--stat", baseline_sha, "HEAD", "--", *paths],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed: {result.stderr.strip()}")
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--justify", type=Path, default=None)
    args = parser.parse_args(argv)

    for label, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        if not path.is_file():
            print(f"GATE FAIL: {label} {path} missing", file=sys.stderr)
            return 2

    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())

    blocking, advisory = compare_identities(baseline, candidate)
    for note in advisory:
        print(f"GATE NOTE: {note}")
    if blocking:
        print("GATE FAIL: identity mismatch", file=sys.stderr)
        for problem in blocking:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    # Identity fields agree. Now ask whether the code that produced the baseline
    # still exists in this tree; a silent drift here invalidates the pairing
    # even when every recorded field matches.
    baseline_sha = baseline.get("git_sha")
    if not isinstance(baseline_sha, str):
        print("GATE FAIL: baseline records no git_sha to diff against", file=sys.stderr)
        return 1

    try:
        drift = source_drift(args.repo, baseline_sha)
    except RuntimeError as exc:
        print(f"GATE FAIL: cannot verify source drift -- {exc}", file=sys.stderr)
        return 1
    if drift:
        justification = args.justify
        if justification is None or not justification.is_file():
            print(
                "GATE FAIL: evaluation source drifted since the baseline arm ran, "
                "and no --justify file argues the drift is behaviour-preserving",
                file=sys.stderr,
            )
            print(drift, file=sys.stderr)
            return 1
        print(f"GATE NOTE: drift since {baseline_sha[:8]} justified by {justification}")
        print(drift)
    else:
        print(f"GATE OK: evaluation source byte-identical to baseline commit {baseline_sha[:8]}")

    matched = set(baseline) - ALLOWED_TO_DIFFER - WEIGHTS_FIELDS
    print(f"GATE OK: candidate aligned with baseline on {len(matched)} identity fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
