"""Emit a candidate arm identity for the CPU gate, before any GPU minute.

Why this is a separate file instead of a flag on the runner
-----------------------------------------------------------
``scripts/campaign-20260926/rule_baseline.py`` writes ``git hash-object`` of its
*own committed file* into every manifest it produces
(``router_source_sha256``), and the stored ``base+tool`` arm's manifest pins that
blob (``e591c886…``). The runner is therefore immutable by contract: editing it
-- even to add a flag -- changes the hash it would write, ``router_source_sha256``
is a blocking field in ``verify_arm_identity.py``, and the stored arm stops
being pairable. So the runner stays byte-for-byte as it was when it produced the
arm, and this emitter lives beside it.

It **imports** the runner rather than reimplementing it, so the two cannot
disagree about what "the same configuration" means: the task loader, the router
hash and the git hash below are the runner's own functions, not copies. The
remaining coupling -- that the emitted keys are what the runner really writes --
cannot be structural without editing the runner, so a test enforces it the
honest way: it runs this emitter and compares the result against the stored
arm's manifest *through the gate itself*, and fails if there is any blocking
difference.

What this is, and what it is not
--------------------------------
A **prediction** of the identity the run will write. The runner's own
``--dry-run`` cannot serve: it returns before the manifest is written, one line
above the GPU section it must never reach. The authoritative check is the same
gate run again *after* the run, against the arm's real ``manifest.json``; this
pre-flight exists so the gate fails on a CPU instead of after two GPU hours.

Usage
-----
::

    ADAPTIVE_MATH_SANDBOX_URL=http://localhost:8080 \\
    .venv/bin/python scripts/eval/emit_arm_identity.py --out identity.json \\
        --mode all-tools --shard-id 0 --shard-count 3 \\
        --data data/processed/v1/frozen_eval.parquet \\
        --task-ids-file artifacts/eval/thesis_e0_base_direct_b1/task_ids.txt \\
        --model <the exact --model the run will use> \\
        --agent-config configs/agent/default.yaml \\
        --reward-config configs/reward/r0.yaml \\
        --output-dir <the run's output dir; parsed for parity, not created here>

The argv is the run's argv, including ``--output-dir``: the point of a
pre-flight is that the configuration gated is the configuration that runs. Two
string-level traps, both refused here rather than at gate time:

* ``ADAPTIVE_MATH_SANDBOX_URL`` must be set exactly as the run will set it. The
  stored arm recorded ``http://localhost:8080`` where the runner's fallback
  spells the same endpoint ``http://127.0.0.1:8080``.
* Every recorded path must be **absolute**. The gate compares these as strings
  and the stored arm recorded absolute paths.

The task set is checked here, not assumed: the loaded ids must equal the frozen
list, once each, and hash to the frozen ``task_ids_sha256``. The runner's loader
catches a *missing* id but not a short list, a duplicate, or a pool that is
simply not the frozen one -- the three ways an arm quietly stops being
comparable, and the failure the 2026-09-26 campaign paid 3.5 GPU hours for.

Exit codes: 0 emitted, 1 the arm is not the frozen task set, 2 usage/setup error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

from adaptive_math.core.hashing import sha256_hex

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "campaign-20260926" / "rule_baseline.py"
# The frozen task list and its canonical hash live with the re-scoring scripts;
# this file imports the constant rather than restating it, so there is exactly
# one place where the frozen set is written down.
PAIRED_PROTOCOLS = REPO / "scripts" / "analysis" / "paired_protocols.py"
DEFAULT_TASK_IDS = REPO / "artifacts" / "eval" / "thesis_e0_base_direct_b1" / "task_ids.txt"
FROZEN_TASK_COUNT = 200


def load_script(name: str, path: Path) -> ModuleType:
    """Import a sibling script, registering it before executing it.

    Registration is not optional: a module executed without a ``sys.modules``
    entry breaks ``from __future__ import annotations`` resolution at class
    creation, which is how this repository's scripts fail when loaded any other
    way.
    """
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


runner = load_script("rule_baseline", RUNNER)
protocols = load_script("paired_protocols", PAIRED_PROTOCOLS)

# The runner's own functions, bound here so the emitted values and the values
# the run will write come from one implementation.
load_tasks = runner.load_tasks
route = runner.route
git_sha = runner.git_sha
router_source_sha256 = runner.router_source_sha256

EXPECTED_TASK_IDS_SHA256 = protocols.EXPECTED_TASK_IDS_SHA256


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def check_frozen(task_ids: list[str], expected: list[str]) -> None:
    """Refuse anything that is not the frozen 200 ids, once each, in order."""
    if len(expected) != FROZEN_TASK_COUNT:
        raise ValueError(
            f"the reference list {len(expected)} ids, expected {FROZEN_TASK_COUNT}"
        )
    duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate task ids, e.g. {duplicates[:3]}")
    if task_ids != expected:
        missing = [task_id for task_id in expected if task_id not in set(task_ids)]
        extra = [task_id for task_id in task_ids if task_id not in set(expected)]
        raise ValueError(
            f"loaded {len(task_ids)} ids which are not the frozen list "
            f"({len(missing)} missing, e.g. {missing[:3]}; "
            f"{len(extra)} unexpected, e.g. {extra[:3]})"
        )
    canonical = sha256_hex("\n".join(task_ids).encode())
    if canonical != EXPECTED_TASK_IDS_SHA256:
        raise ValueError(
            f"task set sha256 {canonical} != expected {EXPECTED_TASK_IDS_SHA256}; "
            "the frozen task list itself changed"
        )


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")


def recorded_path(value: object, label: str) -> str:
    """The string the manifest will carry, required to be absolute.

    The gate compares these as strings, and the stored arm recorded absolute
    paths, so ``data/processed/v1/frozen_eval.parquet`` blocks a pair that
    ``/root/autodl-tmp/.../frozen_eval.parquet`` would pass. Same file, same
    run, different spelling -- the kind of mismatch that costs a gate round
    trip and teaches nothing. Refused here, where the fix is one re-invocation.
    """
    text = str(value)
    if not Path(text).is_absolute():
        raise ValueError(
            f"{label} must be absolute: the manifest records this string and the "
            f"gate compares it against the stored arm's, which is absolute. "
            f"Got {text!r}."
        )
    return text


ADAPTER_KINDS = ("sft", "rl")


def adapter_identity(adapter: Path, kind: str) -> dict[str, str]:
    """The adapter's own identity, hashed the way the direct eval hashes it.

    Same files, same checks, same algorithm as ``run_model_eval.py``, which is
    the point: the agent arm's adapter has to be the *same bytes* as the direct
    arm's adapter for the two rows to be two views of one model, and a sha256 of
    ``adapter_model.safetensors`` is the only thing that can say so. The rl
    branch requires the provenance file for the same reason the direct path
    does -- an RL adapter without training provenance is not evidence.
    """
    if kind not in ADAPTER_KINDS:
        raise ValueError(f"--adapter-kind must be one of {ADAPTER_KINDS}, got {kind!r}")
    weights = adapter / "adapter_model.safetensors"
    for name in ("COMPLETE", "adapter_config.json", "adapter_model.safetensors"):
        require_file(adapter / name, "--adapter")
    if kind == "rl":
        require_file(adapter / "rl_provenance.json", "--adapter (rl)")
    return {
        "adapter": recorded_path(adapter, "--adapter"),
        "adapter_path": str(adapter.resolve()),
        "adapter_sha256": sha256_hex(weights.read_bytes()),
        "adapter_kind": kind,
    }


def emit(args: argparse.Namespace) -> dict[str, Any]:
    """Build the identity dict, mirroring the runner's manifest key for key."""
    pool_path = Path(args.pool)
    agent_config = Path(args.agent_config)
    reward_config = Path(args.reward_config)

    if args.data is None:
        require_file(pool_path, "--pool")
    else:
        require_file(Path(args.data), "--data")
        if args.task_ids_file is None:
            raise ValueError("--data requires --task-ids-file")
        require_file(Path(args.task_ids_file), "--task-ids-file")
    require_file(agent_config, "--agent-config")
    require_file(reward_config, "--reward-config")
    if not Path(args.model).is_dir():
        raise FileNotFoundError(f"--model: {args.model}")

    pool_text = recorded_path(pool_path, "--pool")
    data_text = recorded_path(args.data, "--data") if args.data is not None else None
    ids_text = (
        recorded_path(args.task_ids_file, "--task-ids-file")
        if args.task_ids_file is not None
        else None
    )
    agent_text = recorded_path(agent_config, "--agent-config")
    reward_text = recorded_path(reward_config, "--reward-config")
    model_text = recorded_path(args.model, "--model")

    if (args.shard_id is None) != (args.shard_count is None):
        raise ValueError("--shard-id and --shard-count must be given together")

    tasks = load_tasks(args)
    task_ids = [task.task.task_id for task in tasks]
    reference = Path(args.task_ids_file) if args.task_ids_file is not None else DEFAULT_TASK_IDS
    require_file(reference, "the frozen task list")
    check_frozen(task_ids, read_ids(reference))

    # The runner computes this for both modes; in all-tools mode it is inert
    # (every task runs with both tools) and it is allowed to differ anyway. It
    # is emitted so the two artifacts stay the same shape.
    distribution = Counter(route(task.task) for task in tasks)

    payload: dict[str, Any] = {
        # "dry-run" rather than "running": this is a prediction, and a reader
        # comparing it with a real manifest should not have to guess which is
        # which. ``status`` is bookkeeping and never blocks.
        "status": "dry-run",
        "mode": args.mode,
        "pool": pool_text,
        "data": data_text,
        "task_ids_file": ids_text,
        "model": model_text,
        "agent_config": agent_text,
        "reward_config": reward_text,
        "task_count": len(tasks),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
        "sandbox_url": os.environ.get(
            "ADAPTIVE_MATH_SANDBOX_URL", "http://127.0.0.1:8080"
        ),
        "git_sha": git_sha(),
        "router_source_sha256": router_source_sha256(),
        "routing_distribution": dict(sorted(distribution.items())),
        "shard_id": args.shard_id,
        "shard_count": args.shard_count,
    }

    # The runner's manifest has no adapter field, so these are the only record of
    # which weights the arm ran -- which is why the gate reports weights instead
    # of comparing them, and why the adapter is pinned here rather than assumed.
    # Absent rather than null when there is no adapter: a null would read as a
    # claim about the arm, and the base arm's manifest makes no such claim.
    adapter = getattr(args, "adapter", None)
    if adapter is not None:
        kind = getattr(args, "adapter_kind", None)
        if kind is None:
            raise ValueError("--adapter requires --adapter-kind (one of 'sft', 'rl')")
        payload.update(adapter_identity(Path(adapter), kind))
    return payload


def build_parser() -> argparse.ArgumentParser:
    """The runner's parser, so a flag cannot mean two different things.

    The runner cannot be given a ``--adapter`` flag (see the module docstring),
    so the two flags that describe an arm's weights are added here, and
    ``scripts/campaign-20260927/run_arm_with_adapter.py`` parses its argv with
    *this* parser: the arm that is gated and the arm that runs are then the same
    argv through the same code, which is the only reason a pre-flight is worth
    running.
    """
    parser = runner.build_parser()
    parser.description = (__doc__ or "").splitlines()[0]
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the identity JSON here as well as to stdout",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="PEFT adapter directory the run will load; recorded, and its weights hashed",
    )
    parser.add_argument(
        "--adapter-kind",
        choices=ADAPTER_KINDS,
        default=None,
        help="'sft' or 'rl'; required with --adapter (an rl adapter must carry provenance)",
    )
    parser.epilog = (
        "--dry-run is accepted and ignored by the emitter: it never touches the GPU, "
        "and --output-dir is parsed for parity with the run and not created."
    )
    return parser


def dumps(identity: dict[str, Any]) -> str:
    """The identity as it is written, in one place.

    ``run_arm_with_adapter.py`` writes the same bytes to a sidecar path before it
    starts a run, so the file the gate sees and the file the emitter's own CLI
    writes cannot drift into two spellings of the same identity.
    """
    return json.dumps(identity, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        identity = emit(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"EMIT FAIL: {exc}", file=sys.stderr)
        return 1
    payload = dumps(identity)
    if args.out is not None:
        args.out.write_text(payload)
    sys.stdout.write(payload)
    print(
        f"EMIT OK: {identity['task_count']} frozen tasks, mode={identity['mode']}, "
        f"model={identity['model']}, sandbox={identity['sandbox_url']}",
        file=sys.stderr,
    )
    print(
        "Next: verify_arm_identity.py --baseline <stored shard manifest> "
        "--candidate <this file> --justify <drift justification>",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
