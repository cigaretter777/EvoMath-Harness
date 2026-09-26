#!/usr/bin/env python3
"""SFT LoRA training entry: Transformers causal LM + PEFT + Accelerate.

Heavyweight imports (torch/transformers/peft) are deferred until after
config, override, placeholder and data-manifest validation, so the whole
config workflow is testable without a GPU stack.

Usage:
    uv run python scripts/train/run_sft.py \
        --config configs/sft/qwen3_1_7b_lora.yaml \
        --data data/processed/sft_v1.parquet \
        --data-manifest data/manifests/sft_v1.json

    accelerate launch --multi_gpu scripts/train/run_sft.py --config ...
    --set key=value        documented pilot overrides (repeatable), recorded
                           in resolved_config.yaml
    --dry-run              print the resolved config; write nothing
"""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from adaptive_math.core.hashing import sha256_hex
from adaptive_math.training.config import SFTConfig

PLACEHOLDER_HASH = "0" * 64
TRAINING_PACKAGES = ("torch", "transformers", "peft", "accelerate", "datasets")


class ConfigError(SystemExit):
    pass


def load_config(path: Path) -> SFTConfig:
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        return SFTConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"invalid SFT config {path}: {exc}") from exc


def apply_overrides(config: SFTConfig, pairs: list[str]) -> tuple[SFTConfig, dict[str, object]]:
    updates: dict[str, object] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not raw:
            raise ConfigError(f"override must be key=value, got {pair!r}")
        if key not in SFTConfig.model_fields:
            valid = ", ".join(sorted(SFTConfig.model_fields))
            raise ConfigError(f"unknown config key {key!r}; valid keys: {valid}")
        try:
            updates[key] = json.loads(raw)
        except json.JSONDecodeError:
            updates[key] = raw
    if not updates:
        return config, {}
    try:
        return SFTConfig.model_validate({**config.model_dump(), **updates}), updates
    except ValueError as exc:
        raise ConfigError(f"override produced an invalid config: {exc}") from exc


def check_data_manifest_hash_resolved(config: SFTConfig) -> None:
    if config.data_manifest_sha256 == PLACEHOLDER_HASH:
        raise ConfigError(
            "data_manifest_sha256 is the all-zero placeholder: materialize the "
            "SFT dataset (Training Plan Task 2), then pin sha256 of its manifest "
            "into the config or pass --set data_manifest_sha256=<hash>"
        )


def check_data_manifest(config: SFTConfig, manifest_path: Path) -> None:
    try:
        digest = sha256_hex(manifest_path.read_bytes())
    except OSError as exc:
        raise ConfigError(f"cannot read data manifest {manifest_path}: {exc}") from exc
    if digest != config.data_manifest_sha256:
        raise ConfigError(
            f"data manifest hash mismatch: config pins {config.data_manifest_sha256}, "
            f"{manifest_path} hashes to {digest}"
        )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_replace_dir(tmp_dir: Path, final_dir: Path) -> None:
    backup: Path | None = None
    if final_dir.exists():
        backup = final_dir.with_name(final_dir.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(final_dir, backup)
    os.replace(tmp_dir, final_dir)
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def collect_environment() -> dict[str, object]:
    packages: dict[str, str] = {}
    for name in TRAINING_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "packages": packages,
    }


def write_run_headers(output_dir: Path, config: SFTConfig, overrides: dict[str, object]) -> None:
    resolved = {
        "config": config.model_dump(mode="json"),
        "overrides": overrides,
        "config_hash": sha256_hex(config.model_dump_json().encode()),
    }
    atomic_write_text(output_dir / "resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=True))
    atomic_write_text(output_dir / "environment.json", json.dumps(collect_environment(), indent=2))


def run_training(config: SFTConfig, data_path: Path) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    from adaptive_math.training.collator import CausalLMCollator
    from adaptive_math.training.sft_io import read_records
    from adaptive_math.training.tokenization import tokenize_trajectory

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.tokenizer_revision)
    if not getattr(tokenizer, "chat_template", None):
        raise ConfigError(f"{config.model_id} tokenizer has no chat template")
    tokenizer.revision = config.tokenizer_revision  # consumed by tokenize_trajectory
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )

    records = read_records(data_path)
    tokenized = []
    rejects: list[dict[str, str]] = []
    for record in records:
        try:
            tokenized.append(
                tokenize_trajectory(record.messages, tokenizer, max_length=config.max_length)
            )
        except ValueError as exc:
            rejects.append({"task_id": record.task_id, "reason": str(exc)})
    if not tokenized:
        atomic_write_text(output_dir / "rejects.jsonl", _jsonl(rejects))
        raise ConfigError(f"every record was rejected during tokenization ({len(rejects)} rows)")
    if rejects:
        atomic_write_text(output_dir / "rejects.jsonl", _jsonl(rejects))

    dataset = _make_dataset_class(torch)(tokenized)
    inner = CausalLMCollator(pad_token_id=int(tokenizer.pad_token_id), max_length=config.max_length)
    collator = _TensorCollator(torch, inner)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
    )
    model = get_peft_model(model, lora)
    if config.gradient_checkpointing:
        # Frozen base embeddings + checkpointed blocks: the first checkpointed
        # backward needs input grads or step 1 dies with
        # "element 0 of tensors does not require grad and does not have a grad_fn".
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    class _MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            row = {"step": state.global_step, "epoch": state.epoch, **(logs or {})}
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    run_tmp = output_dir / "trainer_tmp"
    args = TrainingArguments(
        output_dir=str(run_tmp),
        logging_dir=str(output_dir / "tensorboard"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=config.precision == "bf16",
        fp16=config.precision != "bf16",
        logging_steps=config.logging_steps,
        save_strategy="no",
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        data_seed=config.seed,
        report_to=["tensorboard"],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset, data_collator=collator,
        callbacks=[_MetricsCallback()],
    )
    train_output = trainer.train()
    shutil.rmtree(run_tmp, ignore_errors=True)

    adapter_tmp = output_dir / "adapter.tmp"
    shutil.rmtree(adapter_tmp, ignore_errors=True)
    model.save_pretrained(str(adapter_tmp))
    tokenizer.save_pretrained(str(adapter_tmp))
    (adapter_tmp / "COMPLETE").write_text(
        datetime.now(UTC).isoformat() + "\n" + sha256_hex(config.model_dump_json().encode())
    )
    atomic_replace_dir(adapter_tmp, output_dir / "adapter")

    final_row = final_metrics_row(
        train_output.metrics,
        trainer.state.log_history,
        records=len(tokenized),
        rejected=len(rejects),
    )
    with metrics_path.open("a") as handle:
        handle.write(json.dumps(final_row, sort_keys=True) + "\n")


def final_metrics_row(
    train_metrics: dict[str, object], log_history: list[dict[str, object]], *,
    records: int, rejected: int,
) -> dict[str, object]:
    """Keep Trainer's aggregate loss distinct from its last logged step loss."""
    last_logged_loss = next(
        (entry["loss"] for entry in reversed(log_history) if "loss" in entry),
        None,
    )
    return {
        "event": "final", "train_loss": train_metrics.get("train_loss"),
        "last_logged_loss": last_logged_loss, "records": records, "rejected": rejected,
    }


def _make_dataset_class(torch_module):
    """Build a torch Dataset subclass lazily (torch is imported inside run_training)."""

    class TokenizedDataset(torch_module.utils.data.Dataset):
        def __init__(self, items: list) -> None:
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, index: int):
            return self._items[index]

    return TokenizedDataset


class _TensorCollator:
    """Adapt CausalLMCollator output to torch tensors, dropping audit-only keys."""

    def __init__(self, torch_module, inner) -> None:
        self._torch = torch_module
        self._inner = inner

    def __call__(self, examples: list) -> dict:
        batch = self._inner(examples)
        return {
            key: self._torch.tensor(batch[key], dtype=self._torch.long)
            for key in ("input_ids", "attention_mask", "labels")
        }


def _jsonl(rows: list[dict[str, str]]) -> str:
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None, help="SFT trajectories parquet")
    parser.add_argument("--data-manifest", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config, overrides = apply_overrides(config, args.overrides)
    check_data_manifest_hash_resolved(config)
    if args.dry_run:
        print(json.dumps({"config": config.model_dump(mode="json"), "overrides": overrides}, indent=2))
        return 0
    if args.data is None or args.data_manifest is None:
        raise ConfigError("--data and --data-manifest are required unless --dry-run")
    check_data_manifest(config, args.data_manifest)
    write_run_headers(Path(config.output_dir), config, overrides)
    run_training(config, args.data)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
