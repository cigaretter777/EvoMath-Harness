#!/usr/bin/env python3
"""MVP SFT entry: the released config by default, plus an optional merge step.

Why this exists
---------------
Design doc §7 exposes exactly four scripts as the MVP's outside surface; this is
the training one. It is a wrapper, not a second trainer: it fills in
``--config configs/mvp/sft.yaml`` (a mirror of the config that produced the
released adapter) when the caller does not, and forwards every other argument
to ``scripts/train/run_sft.py``, which owns config validation, the manifest
hash gate and the training loop. The flags that reach it are therefore its own,
documented there, and cannot mean two different things.

The one thing this wrapper adds is ``--merge-to``. The GRPO stage trains on a
*merged* model (its ``actor_rollout_ref.model.path`` points at a merged SFT
checkpoint), and nothing in the repository produced that directory -- it was a
one-off on the training host. Merging is therefore part of "SFT finished", not
a separate tool, and it stays opt-in because it doubles the disk the run
writes and a base-only arm never needs it.

Usage:
    uv run python scripts/mvp/train_sft.py --dry-run
    uv run python scripts/mvp/train_sft.py \
        --data data/processed/sft_dp_v1/train.parquet \
        --data-manifest data/manifests/sft_dp_v1_split.json
    # ... and, when the merged checkpoint is wanted (it is, before GRPO):
    uv run python scripts/mvp/train_sft.py \
        --data data/processed/sft_dp_v1/train.parquet \
        --data-manifest data/manifests/sft_dp_v1_split.json \
        --merge-to artifacts/models/qwen3_1_7b_sft_dp_v1_merged

Exit codes are run_sft.py's own (0 ran or dry-ran, 2 config/usage error); a
failed merge raises after a successful training run, deliberately: the adapter
is still on disk and only the merge has to be repeated.
"""

from __future__ import annotations

import argparse
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "mvp" / "sft.yaml"
RUN_SFT = REPO / "scripts" / "train" / "run_sft.py"


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


run_sft = load_script("run_sft", RUN_SFT)


def merge_adapter(config_path: Path, out_dir: Path) -> Path:
    """Merge the finished run's adapter into the base weights and save the result.

    Order is deliberate: the config is validated and the adapter's ``COMPLETE``
    marker is checked *before* the heavy imports and the multi-GB load, so the
    failure modes that do not need a GPU stack fail without one -- and the CPU
    test environment can pin them. Weights load in the config's own precision
    (bf16), matching the dtype of the released merged checkpoint.
    """
    config = run_sft.load_config(config_path)
    adapter = Path(config.output_dir) / "adapter"
    complete = adapter / "COMPLETE"
    if not complete.is_file():
        raise FileNotFoundError(
            f"{adapter} has no COMPLETE marker: this is not a finished SFT run "
            "(or a different output_dir was meant)"
        )

    # Deferred on purpose: the CPU environment imports this module for the
    # --dry-run tests and has no torch; only an actual merge needs the stack.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    base = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        torch_dtype=dtypes[config.precision],
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(
        config.model_id, revision=config.tokenizer_revision
    ).save_pretrained(out_dir)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--merge-to",
        type=Path,
        default=None,
        help="after training, merge the new adapter into the base weights and save here",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="forwarded to run_sft.py; also skips --merge-to",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, forwarded = build_parser().parse_known_args(argv)

    rc = run_sft.main(
        ["--config", str(args.config), *forwarded, *(["--dry-run"] if args.dry_run else [])]
    )
    if rc != 0 or args.merge_to is None:
        return rc
    if args.dry_run:
        print(f"dry-run: skipping merge into {args.merge_to}", file=sys.stderr)
        return rc
    merge_adapter(args.config, args.merge_to)
    print(f"merged adapter -> {args.merge_to}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
