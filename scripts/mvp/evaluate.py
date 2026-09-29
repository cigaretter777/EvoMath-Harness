#!/usr/bin/env python3
"""MVP evaluation: run the three arms (Base / SFT / SFT+GRPO) and tabulate them.

Why this exists
---------------
Design doc §7 exposes exactly four scripts as the MVP's outside surface; this is
the evaluation one. It runs the three arms the design names, against the frozen
200 tasks, and prints the four headline metrics from each run's own artifacts.
It is orchestration, not new evaluation logic: the runner, the identity emitter
and the metrics collator are the same three that produced and verified the
released numbers, and this script only wires them in the order that was proven
on this host.

Two traps it exists to close:

* **The task set.** The runner's ``--pool`` default is the RL *training* pool
  (``openr1_math_220k`` ids) while the frozen eval set is ``frozen_eval.parquet``
  (``omni_math`` ids); they share zero task_ids. The 2026-09-26 campaign spent
  ~3.5 GPU hours pairing nothing because of exactly that. This script always
  passes ``--data`` and ``--task-ids-file``, so the wrong default is not
  reachable from here.
* **Which weights each arm loads.** ``base`` = the base snapshot, no adapter;
  ``sft`` = base snapshot + the released SFT adapter; ``grpo`` = the merged SFT
  model + the released R0 adapter. The base arm therefore runs the frozen
  ``rule_baseline.py`` directly (an arm with no adapter is not the adapter
  wrapper's business, by its own design), while the other two go through
  ``run_arm_with_adapter.py``, which emits the arm's identity *before* loading
  any weights.

Every arm runs in its own subprocess -- one fresh interpreter per arm, the way
the campaign ran them. In-process would be cheaper, but the runner caches its
client and the adapter wrapper patches ``from_pretrained``; one process per arm
means a leftover patch or a cached base model can never be the next arm's.

Run it with the interpreter that has torch (the training one); the subprocesses
inherit ``--python`` (default: this interpreter). Real runs also need the
sandbox reachable via ``ADAPTIVE_MATH_SANDBOX_URL``. ``--dry-run`` needs
neither: it does each arm's CPU pre-flight (task list, paths, adapter hashes)
and stops before the weights.

Metrics are computed only for a single-shard run: the runner's ``summary.json``
always reports the full 200 tasks, so a shard's rolled-out count legitimately
differs and the per-arm counts could not be checked against it. Multi-shard runs
are still launched (``--shard-id``/``--shard-count``); merging is
``scripts/campaign-20260926/merge_rule_shards.py``'s job. Pairwise comparisons
(p-values, CIs) are ``scripts/analysis/paired_protocols.py``'s job, off the
artifacts this writes.

Usage:
    uv run python scripts/mvp/evaluate.py --dry-run
    uv run python scripts/mvp/evaluate.py --arms sft grpo --out-root artifacts/mvp/eval
    uv run python scripts/mvp/evaluate.py --metrics-only   # re-derive from finished runs
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RULE_BASELINE = REPO / "scripts" / "campaign-20260926" / "rule_baseline.py"
RUN_ARM = REPO / "scripts" / "campaign-20260927" / "run_arm_with_adapter.py"
ARM_METRICS = REPO / "scripts" / "analysis" / "arm_metrics.py"
EMITTER = REPO / "scripts" / "eval" / "emit_arm_identity.py"

DEFAULT_DATA = REPO / "data" / "processed" / "v1" / "frozen_eval.parquet"
DEFAULT_TASK_IDS = REPO / "artifacts" / "eval" / "thesis_e0_base_direct_b1" / "task_ids.txt"
DEFAULT_AGENT_CONFIG = REPO / "configs" / "agent" / "default.yaml"
DEFAULT_REWARD_CONFIG = REPO / "configs" / "reward" / "r0.yaml"
DEFAULT_SFT_ADAPTER = REPO / "artifacts" / "sft" / "qwen3_1_7b_sft_dp_v1" / "adapter"
DEFAULT_GRPO_MODEL = REPO / "artifacts" / "models" / "qwen3_1_7b_sft_dp_v1_merged"
DEFAULT_RL_ADAPTER = REPO / "artifacts" / "runs" / "grpo_qwen3_1_7b_r0" / "r0_adapter"
DEFAULT_OUT_ROOT = REPO / "artifacts" / "mvp" / "eval"

ARMS = ("base", "sft", "grpo")
# Hard-coded, not a flag: all three released agent arms ran with both tools
# available, and the stored base arm this pairs against is an all-tools run. The
# runner's other mode ("rule") reuses the stored direct arm for the tasks the
# rule routes straight, which needs a direct pool and a different metrics path --
# not an MVP arm, and not something a flag should be able to swap in silently.
MODE = "all-tools"


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


rule_baseline = load_script("rule_baseline", RULE_BASELINE)
emitter = load_script("emit_arm_identity", EMITTER)
arm_metrics = load_script("arm_metrics", ARM_METRICS)

# The base snapshot the released SFT adapter was trained from, taken from the
# runner's own default so there is exactly one place it is spelled.
DEFAULT_BASE_MODEL: Path = rule_baseline.build_parser().get_default("model")


def arm_weights(arm: str, args: argparse.Namespace) -> tuple[Path, Path | None, str | None]:
    """(model, adapter, adapter_kind) for one arm -- the released weights by default."""
    if arm == "base":
        return Path(args.base_model), None, None
    if arm == "sft":
        return Path(args.base_model), Path(args.sft_adapter), "sft"
    return Path(args.grpo_model), Path(args.rl_adapter), "rl"


def run_dir_for(arm: str, args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    if args.shard_count == 1:
        return out_root / arm
    return out_root / f"{arm}.shard{args.shard_id}"


def runner_flags(args: argparse.Namespace, run_dir: Path) -> list[str]:
    """The runner's flags, always with the task set pinned explicitly.

    ``--pool`` is passed even though ``--data`` makes it inert: the emitter
    records it, and the stored arms' identities record it, so leaving it to a
    default would make two identities differ in a field that means nothing.
    """
    flags = [
        "--mode", MODE,
        "--pool", str(args.pool),
        "--data", str(args.data),
        "--task-ids-file", str(args.task_ids),
        "--agent-config", str(args.agent_config),
        "--reward-config", str(args.reward_config),
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--output-dir", str(run_dir),
    ]
    if args.shard_count != 1:
        flags += ["--shard-id", str(args.shard_id), "--shard-count", str(args.shard_count)]
    return flags


def arm_command(
    arm: str, args: argparse.Namespace, run_dir: Path
) -> tuple[Path, list[str]]:
    """(script, argv) for one arm's run -- the frozen runner, or the adapter wrapper."""
    model, adapter, kind = arm_weights(arm, args)
    flags = runner_flags(args, run_dir)
    if adapter is None:
        return RULE_BASELINE, ["--model", str(model), *flags]
    return RUN_ARM, [
        "--model", str(model),
        "--adapter", str(adapter),
        "--adapter-kind", str(kind),
        *flags,
    ]


def base_preflight(args: argparse.Namespace, run_dir: Path) -> tuple[dict[str, Any], Path]:
    """The base arm's identity, emitted in-process before its run starts.

    Same emitter the adapter wrapper uses, minus the adapter: it refuses a task
    list that is not the frozen 200, and it is what makes the base arm's sidecar
    the same shape as the stored base arm's manifest. Written beside the run
    directory, never inside it -- the runner refuses a non-empty output dir.
    """
    parse = emitter.build_parser().parse_args(runner_flags(args, run_dir))
    identity = emitter.emit(parse)
    sidecar = Path(f"{run_dir}.identity.json")
    sidecar.write_text(emitter.dumps(identity))
    return identity, sidecar


def derived_verifier_valid(trajectories: Path) -> int:
    """Count verifier-addressable answers the way arm_metrics defines them.

    The runner's ``summary.json`` records nine counts and this is not one of
    them, so unlike the others it has no run-level number to be checked against;
    the trajectories are the only source. It is recomputed here rather than
    imported from arm_metrics so that a drift in how that script reads
    ``verdict.status`` shows up as a mismatch instead of agreeing with itself.
    """
    count = 0
    for line in trajectories.read_text().splitlines():
        if not line.strip():
            continue
        status = (json.loads(line).get("verdict") or {}).get("status")
        count += int(status in arm_metrics.VALID_STATUSES)
    return count


def compute_metrics(arm: str, run_dir: Path) -> dict[str, Any]:
    """Four metrics for one finished arm, checked against its own summary.json."""
    summary = json.loads((run_dir / "summary.json").read_text())
    trajectories = run_dir / "trajectories.jsonl"
    if not trajectories.is_file():
        raise FileNotFoundError(f"{trajectories} is missing: the run did not finish")
    spec = arm_metrics.ArmSpec(
        label=arm,
        mode="agent",
        expected_n=int(summary["task_count"]),
        expected_correct=int(summary["correct_count"]),
        expected_verifier_valid=derived_verifier_valid(trajectories),
        expected_rolled_out=int(summary["rolled_out"]),
        expected_direct_reused=int(summary["direct_reused"]),
        expected_tool_calls=int(summary["tool_calls_total"]),
        expected_invalid_actions=int(summary["invalid_actions_total"]),
        expected_generated_tokens=int(summary["generated_tokens_total"]),
        trajectories=trajectories,
    )
    return arm_metrics.compute_arm(spec)


def read_identity(run_dir: Path) -> dict[str, Any] | None:
    """The arm's weights identity: inside the run dir, else the sibling sidecar.

    Absent is allowed rather than fatal -- the two places the identity lands are
    written at different moments, and the metrics are computed from the run's own
    summary and trajectories either way.
    """
    for path in (run_dir / "identity.json", Path(f"{run_dir}.identity.json")):
        if path.is_file():
            return json.loads(path.read_text())
    return None


def task_list_sha256(path: Path) -> str:
    """The repository's task-list hash: ids joined by newlines, no trailing newline."""
    from adaptive_math.core.hashing import sha256_hex

    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return sha256_hex("\n".join(ids).encode("utf-8"))


def dry_run_arm(
    arm: str, args: argparse.Namespace, run_dir: Path, script: Path, command: list[str]
) -> int:
    if arm != "base":
        # The adapter wrapper's own --dry-run: emit the identity, print the gate
        # hint, stop before the weights. Same code the real arm runs, so the
        # pre-flight cannot describe a command the run would not use.
        completed = subprocess.run(
            [str(args.python), str(script), *command, "--dry-run"], check=False
        )
        return completed.returncode
    try:
        identity, sidecar = base_preflight(args, run_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[base] PRE-FLIGHT FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        f"[base] PRE-FLIGHT OK: {identity['task_count']} frozen tasks, "
        f"model={identity['model']}, identity at {sidecar}",
        file=sys.stderr,
    )
    # The stored base arm ran before the 2026-09-27 agent-loop source changes,
    # so its gate needs the drift justification that campaign wrote; without it
    # the gate exits 1 on the source diff alone. Verified on this host: with the
    # file, the identity above reaches GATE OK on all nine aligned fields.
    print(
        "[base] gate it: scripts/eval/verify_arm_identity.py --baseline "
        "<stored base arm manifest> "
        f"--candidate {sidecar} "
        "--justify docs/results/campaign-2026-09-27/source-drift-justification.md",
        file=sys.stderr,
    )
    return 0


def run_arm_for_real(
    arm: str, args: argparse.Namespace, run_dir: Path, script: Path, command: list[str]
) -> int:
    sidecar: Path | None = None
    if arm == "base":
        try:
            _, sidecar = base_preflight(args, run_dir)
        except (ValueError, FileNotFoundError) as exc:
            print(f"[base] PRE-FLIGHT FAIL: {exc}", file=sys.stderr)
            return 1
    completed = subprocess.run([str(args.python), str(script), *command], check=False)
    if completed.returncode == 0 and sidecar is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sidecar, run_dir / "identity.json")
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=list(ARMS),
        help="arms to run, in this order (default: all three)",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--task-ids", type=Path, default=DEFAULT_TASK_IDS)
    parser.add_argument(
        "--pool",
        type=Path,
        default=rule_baseline.build_parser().get_default("pool"),
        help="recorded in each arm's identity; inert while --data is passed",
    )
    parser.add_argument("--agent-config", type=Path, default=DEFAULT_AGENT_CONFIG)
    parser.add_argument("--reward-config", type=Path, default=DEFAULT_REWARD_CONFIG)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=DEFAULT_BASE_MODEL,
        help="base snapshot for the base and sft arms (the runner's own default)",
    )
    parser.add_argument("--sft-adapter", type=Path, default=DEFAULT_SFT_ADAPTER)
    parser.add_argument(
        "--grpo-model",
        type=Path,
        default=DEFAULT_GRPO_MODEL,
        help="merged SFT model the R0 adapter was trained on",
    )
    parser.add_argument("--rl-adapter", type=Path, default=DEFAULT_RL_ADAPTER)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter for the arm subprocesses (needs torch; default: this one)",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="skip the runs and re-derive metrics from the finished run directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="per-arm CPU pre-flight (identity, paths, adapters); no weights loaded",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.metrics_only and args.dry_run:
        build_parser().error("--metrics-only and --dry-run are different questions; pick one")
    # Every arm is scored on this list, and every identity is emitted against it;
    # failing here beats failing 200 rollouts later.
    if not Path(args.task_ids).is_file():
        print(
            f"EVALUATE FAIL: {args.task_ids} is missing (run scripts/mvp/prepare_data.py "
            "to regenerate the frozen task list)",
            file=sys.stderr,
        )
        return 1
    # This script owns --out-root, and the arms' identity sidecars land directly
    # inside it (the runner's own output dirs are created by the runner).
    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for arm in args.arms:
        run_dir = run_dir_for(arm, args)
        script, command = arm_command(arm, args, run_dir)
        print(f"[{arm}] {shlex.join([str(args.python), str(script), *command])}")

        if args.metrics_only:
            rc = 0 if (run_dir / "summary.json").is_file() else 1
            if rc:
                print(f"[{arm}] no summary.json in {run_dir}", file=sys.stderr)
        elif args.dry_run:
            rc = dry_run_arm(arm, args, run_dir, script, command)
        else:
            rc = run_arm_for_real(arm, args, run_dir, script, command)
        if rc != 0:
            failures.append(f"{arm} (rc={rc})")
            continue
        if args.dry_run:
            continue

        if args.shard_count != 1:
            print(
                f"[{arm}] {args.shard_count} shards: metrics need the merged run "
                "(scripts/campaign-20260926/merge_rule_shards.py), skipping",
                file=sys.stderr,
            )
            continue
        try:
            metrics = compute_metrics(arm, run_dir)
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            print(f"[{arm}] METRICS FAIL: {exc}", file=sys.stderr)
            failures.append(f"{arm} (metrics)")
            continue
        identity = read_identity(run_dir)
        results[arm] = {"identity": identity, "metrics": metrics}
        print(
            f"[{arm}] {metrics['correct']}/{metrics['n']} strict, "
            f"final action {metrics['final_action_rate']:.1%}, "
            f"tool calls {metrics['tool_calls_total']}",
            file=sys.stderr,
        )

    if results:
        print(arm_metrics._format_table({arm: data["metrics"] for arm, data in results.items()}))
        out = Path(args.out_root) / "four-metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": MODE,
            "data": str(args.data),
            "task_ids_file": str(args.task_ids),
            "task_ids_sha256": task_list_sha256(Path(args.task_ids)),
            "shard": {"id": args.shard_id, "count": args.shard_count},
            "arms": results,
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out}")

    if failures:
        print(f"EVALUATE FAIL: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"EVALUATE OK: {', '.join(args.arms)}" + (" (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
