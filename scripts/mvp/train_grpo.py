#!/usr/bin/env python3
"""MVP GRPO entry: the released R0 config by default, plus an optional LoRA export.

Why this exists
---------------
Design doc §7 exposes exactly four scripts as the MVP's outside surface; this is
the RL one. It fills in ``--config configs/mvp/grpo.yaml`` (a mirror of the
released R0 config) when the caller does not, and forwards everything else to
``scripts/train/run_grpo_direct.py`` -- the launch path verified on this host,
where torchrun hangs in the register-center handshake. ``run_grpo.py`` keeps
owning config validation, the pinned upstream checkout and the trainer itself.

``--export-adapter`` closes the gap that used to sit between "GRPO finished"
and "GRPO is evaluable": verl saves the LoRA inline in its FSDP checkpoint, and
the export that lifts it into a PEFT adapter directory was a campaign script
with no place in the MVP path. The key-mapping rules are easy to get subtly
wrong, so this delegates to that script rather than restating them, and finds
the newest checkpoint itself (the launch config trains one GPU, so the file is
``global_step_N/actor/model_world_size_1_rank_0.pt``).

Usage:
    uv run python scripts/mvp/train_grpo.py --dry-run
    uv run python scripts/mvp/train_grpo.py \
        --export-adapter artifacts/runs/grpo_qwen3_1_7b_r0/r0_adapter

Run it with the training interpreter -- the one that has torch and peft. The
export runs as a subprocess of the same interpreter, so it inherits that
guarantee; the LoRA shape (rank/alpha) is read from the config's own upstream
overrides, not guessed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "mvp" / "grpo.yaml"
RUN_GRPO_DIRECT = REPO / "scripts" / "train" / "run_grpo_direct.py"
RUN_GRPO = REPO / "scripts" / "train" / "run_grpo.py"
EXPORT_VERL_LORA = REPO / "scripts" / "campaign-20260919" / "export_verl_lora.py"


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


run_grpo_direct = load_script("run_grpo_direct", RUN_GRPO_DIRECT)
run_grpo = load_script("run_grpo", RUN_GRPO)


def find_latest_checkpoint(checkpoints_dir: Path) -> Path:
    """The newest ``global_step_*`` actor checkpoint, as the single file to export.

    The export script loads one ``.pt`` state dict, not a directory, and the
    launch config trains on one GPU (``trainer.n_gpus_per_node=1``), so verl
    writes ``global_step_N/actor/model_world_size_1_rank_0.pt``. Picking by
    parsed step number, not lexicographic name: ``global_step_100`` sorts
    before ``global_step_50`` as a string.
    """
    candidates = sorted(
        checkpoints_dir.glob("global_step_*/actor/model_world_size_*_rank_0.pt"),
        key=lambda path: int(path.parent.parent.name.removeprefix("global_step_")),
    )
    if not candidates:
        raise FileNotFoundError(
            f"no global_step_*/actor/model_world_size_*_rank_0.pt under {checkpoints_dir}: "
            "nothing to export (did the run reach a save_freq checkpoint?)"
        )
    return candidates[-1]


def lora_rank_alpha(config: object) -> tuple[int, int]:
    """The LoRA shape, read from the config's own ``upstream_overrides``.

    The overrides are ``key=value`` strings destined for Hydra; scanning them
    keeps the export's rank/alpha tied to the values the run actually trained
    with instead of a default that happens to agree today.
    """
    overrides: dict[str, str] = {}
    for item in config.upstream_overrides:  # type: ignore[attr-defined]
        key, sep, value = item.partition("=")
        if sep:
            overrides[key] = value
    rank = int(overrides.get("actor_rollout_ref.model.lora_rank", "16"))
    alpha = int(overrides.get("actor_rollout_ref.model.lora_alpha", "32"))
    return rank, alpha


def export_adapter(config_path: Path, out_dir: Path) -> Path:
    """Export the newest checkpoint's inline LoRA into ``out_dir``.

    A subprocess, not an import: the export loads a multi-GB state dict under
    torch and is a standalone entry with its own argv, and ``sys.executable`` is
    the interpreter that just trained, so it is the one that has torch.
    """
    config = run_grpo.load_config(config_path)
    # ``output_dir`` is written repo-relative and the trainer's checkpoints land
    # in its ``checkpoints/`` subdirectory (``trainer.default_local_dir`` in the
    # config). Anchored at the repository root rather than the caller's cwd,
    # which is what the launcher does too.
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO / output_dir
    checkpoint = find_latest_checkpoint(output_dir / "checkpoints")
    rank, alpha = lora_rank_alpha(config)
    completed = subprocess.run(
        [
            sys.executable,
            str(EXPORT_VERL_LORA),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(out_dir),
            "--rank",
            str(rank),
            "--alpha",
            str(alpha),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"export failed with rc={completed.returncode}: {EXPORT_VERL_LORA}"
        )
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--export-adapter",
        type=Path,
        default=None,
        help="after training, export the newest checkpoint's LoRA into this adapter dir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="forwarded to run_grpo_direct.py; also skips --export-adapter",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, forwarded = build_parser().parse_known_args(argv)

    rc = run_grpo_direct.main(
        ["--config", str(args.config), *forwarded, *(["--dry-run"] if args.dry_run else [])]
    )
    if rc != 0 or args.export_adapter is None:
        return rc
    if args.dry_run:
        print(f"dry-run: skipping adapter export into {args.export_adapter}", file=sys.stderr)
        return rc
    export_adapter(args.config, args.export_adapter)
    print(f"exported adapter -> {args.export_adapter}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
