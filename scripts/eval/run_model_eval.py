#!/usr/bin/env python3
"""Paired greedy Base/SFT evaluation on Omni-MATH frozen_eval.

Example: python scripts/eval/run_model_eval.py --sft-parquet DATA --sft-manifest MANIFEST
  --adapter ADAPTER --output-dir artifacts/eval/sft_v1_omnimath_200 --limit 200
"""

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import yaml

import adaptive_math
from adaptive_math.agent.model_client import ChatMessage, ModelTurn
from adaptive_math.core.hashing import sha256_hex
from adaptive_math.core.types import LabeledMathTask, MathTask
from adaptive_math.evaluation.model_eval import EvaluationJournal, evaluate_pair, write_artifacts
from adaptive_math.training.sft_io import read_records
from adaptive_math.training.sft_materialization import _labeled_task_from_row

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


class HFGenerator:
    """Greedy HF/PEFT generator with a one-item compatibility call and batch API."""

    def __init__(self, model_id: str, tokenizer, model, torch, max_new_tokens: int) -> None:
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device
        self._peak_gpu_memory_bytes: int | None = None

    def __call__(self, arm: str, item: MathTask, messages: tuple[ChatMessage, ...]) -> ModelTurn:
        return self.generate_batch(arm, [(item, messages)])[0]

    def generate_batch(
        self, arm: str, requests: list[tuple[MathTask, tuple[ChatMessage, ...]]]
    ) -> list[ModelTurn]:
        if not requests:
            return []
        if arm not in {"base", "sft"}:
            raise ValueError(f"unknown evaluation arm: {arm}")
        rendered = [
            self.tokenizer.apply_chat_template(
                [message.model_dump() for message in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            for _item, messages in requests
        ]
        encoded = self.tokenizer(
            rendered, return_tensors="pt", padding=True, return_attention_mask=True,
        ).to(self.device)
        input_ids = encoded["input_ids"]

        def produce():
            with self.torch.inference_mode():
                return self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        if arm == "base":
            with self.model.disable_adapter():
                output = produce()
        else:
            output = produce()
        prompt_width = input_ids.shape[-1]
        return [
            ModelTurn(
                text=self.tokenizer.decode(output[index, prompt_width:], skip_special_tokens=False),
                prompt_tokens=int(input_ids.shape[-1]),
                generated_tokens=int(output[index, prompt_width:].shape[-1]),
                finish_reason="stop",
                model_id=self.model_id,
            )
            for index in range(len(requests))
        ]

    def start_arm(self, _arm: str) -> None:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            self._peak_gpu_memory_bytes = None
            return
        cuda.reset_peak_memory_stats(self.device)
        self._peak_gpu_memory_bytes = None

    def resource_metrics(self) -> dict[str, object]:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            self._peak_gpu_memory_bytes = int(cuda.max_memory_allocated(self.device))
        return {"peak_gpu_memory_bytes": self._peak_gpu_memory_bytes}


def preflight(
    *, eval_parquet: Path, data_manifest: Path, sft_parquet: Path, sft_manifest: Path,
    adapter: Path, limit: int, model_id: str, model_revision: str,
    tokenizer_revision: str, max_new_tokens: int,
    batch_size: int = 1,
    expected_adapter_sha256: str | None = None,
    rl_adapter: bool = False,
) -> tuple[list[LabeledMathTask], set[str], dict[str, object]]:
    """Validate immutable inputs and leakage before importing the GPU stack.

    ``rl_adapter=True`` evaluates a GRPO-trained adapter: instead of the SFT
    manifest binding it requires an ``rl_provenance.json`` written by the
    checkpoint export tool (verl checkpoint path + SHA-256).
    """
    if limit <= 0 or max_new_tokens <= 0 or batch_size <= 0:
        raise ValueError("limit, max_new_tokens and batch_size must be positive")
    imported_package = Path(adaptive_math.__file__).resolve().parent
    expected_package = (ROOT / "src/adaptive_math").resolve()
    if imported_package != expected_package:
        raise ValueError(f"adaptive_math import path {imported_package} does not match {expected_package}")
    for label, revision in (("model", model_revision), ("tokenizer", tokenizer_revision)):
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError(f"{label} revision must be a pinned 40-character commit SHA")
    data_meta = json.loads(data_manifest.read_text())
    split = data_meta["splits"]["frozen_eval"]
    eval_hash = sha256_hex(eval_parquet.read_bytes())
    if eval_hash != split["file_hash"]:
        raise ValueError("eval parquet hash does not match data manifest")
    if eval_parquet.name != split.get("file", eval_parquet.name):
        raise ValueError("eval parquet filename does not match data manifest")
    sft_meta = json.loads(sft_manifest.read_text())
    sft_manifest_hash = sha256_hex(sft_manifest.read_bytes())
    sft_hash = sha256_hex(sft_parquet.read_bytes())
    if sft_hash != (sft_meta.get("parquet_sha256") or sft_meta["train"]["sha256"]):
        raise ValueError("SFT parquet hash does not match SFT manifest")
    table = pq.read_table(eval_parquet)
    if table.num_rows != split["count"]:
        raise ValueError("eval row count does not match data manifest")
    rows = sorted(table.to_pylist(), key=lambda row: str(row["task_id"]))
    if len(rows) < limit:
        raise ValueError(f"requested {limit} eval tasks, only {len(rows)} available")
    tasks = [_labeled_task_from_row(row) for row in rows[:limit]]
    ids = [item.task.task_id for item in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate eval task IDs")
    if any(item.task.dataset != "omni_math" or item.task.split != "frozen_eval" for item in tasks):
        raise ValueError("eval parquet contains non-Omni-MATH frozen_eval rows")
    sft_records = read_records(sft_parquet)
    if len(sft_records) != (sft_meta.get("total_records") or sft_meta["train"]["count"]):
        raise ValueError("SFT row count does not match SFT manifest")
    sft_ids = {record.task_id for record in sft_records}
    if set(ids) & sft_ids:
        raise ValueError("SFT/eval task ID leakage")
    adapter_config = adapter / "adapter_config.json"
    adapter_weights = adapter / "adapter_model.safetensors"
    complete_path = adapter / "COMPLETE"
    if not complete_path.is_file():
        raise ValueError("adapter COMPLETE marker is missing")
    rl_provenance_hash: str | None = None
    if rl_adapter:
        provenance_path = adapter / "rl_provenance.json"
        if not provenance_path.is_file():
            raise ValueError("rl adapter requires rl_provenance.json (run the export tool)")
        rl_provenance_hash = sha256_hex(provenance_path.read_bytes())
    else:
        _validate_training_config(adapter, complete_path, sft_manifest_hash)
    peft_meta = json.loads(adapter_config.read_text())
    adapter_base = peft_meta.get("base_model_name_or_path")

    selected_model = model_id
    if "/models--" in model_id and "/snapshots/" in model_id:
        encoded = model_id.split("/models--", 1)[1].split("/snapshots/", 1)[0]
        if "--" in encoded:
            namespace, repo = encoded.split("--", 1)
            selected_model = f"{namespace}/{repo}"

    if adapter_base != selected_model and not rl_adapter:
        # RL adapters record their true training base (the GRPO actor model
        # path) in rl_provenance.json; the hub-id base in adapter_config is
        # only a fallback for standalone loading.
        raise ValueError(
            f"adapter base model does not match selected model: "
            f"{adapter_base!r} != {selected_model!r}"
        )
    adapter_hash = sha256_hex(adapter_weights.read_bytes())
    if expected_adapter_sha256 is not None and adapter_hash != expected_adapter_sha256:
        raise ValueError("adapter SHA-256 does not match expected value")
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                 capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    manifest: dict[str, object] = {
        "schema_version": "paired-model-eval-v1", "git_sha": git_sha,
        "adaptive_math_import_path": str(imported_package),
        "base_model": model_id, "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision, "adapter_path": str(adapter.resolve()),
        "adapter_sha256": adapter_hash,
        "adapter_kind": "rl" if rl_adapter else "sft",
        "rl_provenance_sha256": rl_provenance_hash,
        "adapter_config_sha256": sha256_hex(adapter_config.read_bytes()),
        "eval_dataset": "omni_math", "eval_split": "frozen_eval",
        "eval_parquet_sha256": eval_hash,
        "data_manifest_sha256": sha256_hex(data_manifest.read_bytes()),
        "sft_parquet_sha256": sft_hash,
        "sft_manifest_sha256": sft_manifest_hash,
        "generation_config": {"do_sample": False, "max_new_tokens": max_new_tokens,
                              "batch_size": batch_size},
        "verifier_source_sha256": sha256_hex((ROOT / "src/adaptive_math/verifier/service.py").read_bytes()),
        "extractor_source_sha256": sha256_hex((ROOT / "src/adaptive_math/verifier/extractor.py").read_bytes()),
        "prompt_source_sha256": sha256_hex((ROOT / "src/adaptive_math/agent/prompts.py").read_bytes()),
    }
    return tasks, sft_ids, manifest


def _validate_training_config(
    adapter: Path, complete_path: Path, sft_manifest_sha256: str
) -> None:
    """Bind an adapter to the immutable SFT manifest it was trained against."""
    resolved_path = adapter.parent / "resolved_config.yaml"
    try:
        resolved = yaml.safe_load(resolved_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read adapter training config: {resolved_path}") from exc
    if not isinstance(resolved, dict) or not isinstance(resolved.get("config"), dict):
        raise TypeError("adapter training config has no config object")
    config = resolved["config"]
    if config.get("data_manifest_sha256") != sft_manifest_sha256:
        raise ValueError("adapter training config is bound to another SFT manifest")
    config_hash = resolved.get("config_hash")
    complete_lines = complete_path.read_text().splitlines()
    if not isinstance(config_hash, str) or len(complete_lines) < 2:
        raise ValueError("adapter COMPLETE marker lacks a training config hash")
    if complete_lines[-1] != config_hash:
        raise ValueError("adapter COMPLETE marker does not match training config")


def make_generator(
    model_id: str, model_revision: str, tokenizer_revision: str, adapter: Path,
    max_new_tokens: int,
):
    """Load one pinned model; disable its adapter for the Base arm."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install runtime extras (torch, transformers, peft) on a GPU host") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tokenizer_revision)
    if not tokenizer.chat_template:
        raise ValueError("tokenizer has no chat template")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=model_revision, dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()

    return HFGenerator(model_id, tokenizer, model, torch, max_new_tokens)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-parquet", type=Path, default=Path("data/processed/v1/frozen_eval.parquet"))
    parser.add_argument("--data-manifest", type=Path, default=Path("data/manifests/v1.json"))
    parser.add_argument("--sft-parquet", type=Path, required=True)
    parser.add_argument("--sft-manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, choices=(200, 500), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rl-adapter", action="store_true",
                        help="adapter comes from GRPO training; require rl_provenance.json instead of an SFT manifest binding")
    parser.add_argument("--adapter-only", action="store_true",
                        help="evaluate only the adapter arm; write predictions for offline pairing")
    args = parser.parse_args(argv)
    arms: tuple[str, ...] = ("sft",) if args.adapter_only else ("base", "sft")
    tasks, sft_ids, manifest = preflight(
        eval_parquet=args.eval_parquet, data_manifest=args.data_manifest,
        sft_parquet=args.sft_parquet, sft_manifest=args.sft_manifest,
        adapter=args.adapter, limit=args.limit, model_id=args.model_id,
        model_revision=args.model_revision, tokenizer_revision=args.tokenizer_revision,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        expected_adapter_sha256=args.expected_adapter_sha256,
        rl_adapter=args.rl_adapter,
    )
    if args.dry_run:
        print(json.dumps({"ok": True, "task_count": len(tasks), "manifest": manifest}, sort_keys=True))
        return 0
    if args.output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output_dir}")
    journal = EvaluationJournal(
        args.output_dir.with_name(args.output_dir.name + ".in_progress"), manifest
    )
    initial_predictions = {arm: journal.rows(arm) for arm in arms}
    manifest["started_at"] = datetime.now(UTC).isoformat()
    generate = make_generator(args.model_id, args.model_revision, args.tokenizer_revision,
                              args.adapter, args.max_new_tokens)
    from tqdm.auto import tqdm

    bars = {
        arm: tqdm(total=len(tasks), initial=len(initial_predictions[arm]), desc=f"eval:{arm}")
        for arm in arms
    }

    def checkpoint(arm: str, row: dict[str, object]) -> None:
        journal.append(arm, row)
        bars[arm].update(1)
        bars[arm].set_postfix(task=str(row["task_id"])[-12:])

    try:
        result = evaluate_pair(
            tasks,
            sft_ids,
            generate,
            batch_size=args.batch_size,
            initial_predictions=initial_predictions,
            on_prediction=checkpoint,
            arms=arms,
        )
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        write_artifacts(args.output_dir, result, manifest)
    finally:
        for bar in bars.values():
            bar.close()
    shutil.rmtree(journal.directory, ignore_errors=True)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
