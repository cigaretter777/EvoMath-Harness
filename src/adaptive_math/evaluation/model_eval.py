"""Deterministic paired, direct-answer Base versus SFT evaluation."""

import json
import math
import os
import random
import shutil
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypedDict, cast, runtime_checkable

from adaptive_math.agent.model_client import ChatMessage, ModelTurn
from adaptive_math.agent.prompts import PROMPT_VERSION, render_initial_messages
from adaptive_math.core.hashing import sha256_hex
from adaptive_math.core.types import Budget, LabeledMathTask, MathTask
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.verifier import ExtractStatus, VerifierStatus, extract, verify_answer

Generator = Callable[[str, MathTask, tuple[ChatMessage, ...]], ModelTurn]


@runtime_checkable
class BatchGenerator(Protocol):
    """Optional evaluation generator capability for same-arm prompt batches."""

    def generate_batch(
        self, arm: str, requests: list[tuple[MathTask, tuple[ChatMessage, ...]]]
    ) -> list[ModelTurn]: ...


@runtime_checkable
class ResourceReportingGenerator(Protocol):
    """Optional generator hooks for per-arm accelerator measurements."""

    def start_arm(self, arm: str) -> None: ...

    def resource_metrics(self) -> dict[str, object]: ...


class EvalResult(TypedDict):
    base_predictions: list[dict[str, object]]
    sft_predictions: list[dict[str, object]]
    comparison: list[dict[str, object]]
    summary: dict[str, object]


_DIRECT_BUDGET = Budget(max_steps=1, max_tool_calls=0, max_python_seconds=0,
                        max_observation_chars=1)


class EvaluationJournal:
    """Append-only, manifest-bound progress state for an interruptible evaluation."""

    def __init__(self, directory: Path, immutable_manifest: dict[str, object]) -> None:
        self.directory = directory
        self._manifest_path = directory / "run_manifest.json"
        self._progress_path = directory / "progress.json"
        if self._manifest_path.exists():
            stored = json.loads(self._manifest_path.read_text())
            if stored != immutable_manifest:
                raise ValueError("evaluation journal immutable manifest does not match")
        else:
            directory.mkdir(parents=True, exist_ok=False)
            _atomic_json(self._manifest_path, immutable_manifest)
            _atomic_json(self._progress_path, {"completed": {"base": 0, "sft": 0}})

    @property
    def progress(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self._progress_path.read_text()))

    def rows(self, arm: str) -> list[dict[str, object]]:
        path = self._rows_path(arm)
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        task_ids = [row.get("task_id") for row in rows]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"evaluation journal has duplicate {arm} task IDs")
        return cast(list[dict[str, object]], rows)

    def append(self, arm: str, row: dict[str, object]) -> None:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise TypeError("journal row requires a non-empty task_id")
        if task_id in {item["task_id"] for item in self.rows(arm)}:
            raise ValueError(f"evaluation journal already contains {arm}:{task_id}")
        path = self._rows_path(arm)
        with path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        progress = self.progress
        completed = progress.get("completed")
        if not isinstance(completed, dict):
            raise TypeError("evaluation journal progress is malformed")
        completed[arm] = len(self.rows(arm))
        _atomic_json(self._progress_path, progress)

    def _rows_path(self, arm: str) -> Path:
        if arm not in {"base", "sft"}:
            raise ValueError(f"unknown evaluation arm: {arm}")
        return self.directory / f"{arm}_predictions.jsonl"


def _atomic_json(path: Path, value: object) -> None:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(rendered)
    os.replace(temporary, path)


def evaluate_pair(
    tasks: list[LabeledMathTask],
    sft_task_ids: set[str],
    generate: Generator | BatchGenerator,
    *,
    batch_size: int = 1,
    initial_predictions: dict[str, list[dict[str, object]]] | None = None,
    on_prediction: Callable[[str, dict[str, object]], None] | None = None,
    arms: tuple[str, ...] = ("base", "sft"),
) -> EvalResult:
    """Evaluate both arms on the exact same ordered tasks and public prompts."""
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    ordered = sorted(tasks, key=lambda item: item.task.task_id)
    ids = [item.task.task_id for item in ordered]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("evaluation requires nonempty, unique task IDs")
    if any(item.task.split != "frozen_eval" or item.task.dataset != "omni_math" for item in ordered):
        raise ValueError("evaluation requires Omni-MATH frozen_eval tasks")
    overlap = set(ids) & sft_task_ids
    if overlap:
        raise ValueError(f"SFT/eval task ID leakage: {sorted(overlap)[:5]}")
    prompts = {
        item.task.task_id: render_initial_messages(item.public_view(), _DIRECT_BUDGET, ToolRegistry([]))
        for item in ordered
    }
    initial_predictions = initial_predictions or {"base": [], "sft": []}
    predictions: dict[str, list[dict[str, object]]] = {}
    arm_summaries: dict[str, dict[str, object]] = {}
    for arm in arms:
        if isinstance(generate, ResourceReportingGenerator):
            generate.start_arm(arm)
        existing = initial_predictions.get(arm, [])
        rows_by_task_id = {str(row["task_id"]): row for row in existing}
        if len(rows_by_task_id) != len(existing) or not set(rows_by_task_id).issubset(ids):
            raise ValueError(f"invalid resumed {arm} predictions")
        pending = [item for item in ordered if item.task.task_id not in rows_by_task_id]
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            requests = [(item.public_view(), prompts[item.task.task_id]) for item in batch]
            started = time.perf_counter()
            if isinstance(generate, BatchGenerator):
                turns = generate.generate_batch(arm, requests)
            else:
                turns = [generate(arm, item, messages) for item, messages in requests]
            elapsed_ms = (time.perf_counter() - started) * 1_000
            if len(turns) != len(batch):
                raise ValueError(f"generator returned {len(turns)} turns for {len(batch)} requests")
            batch_id = sha256_hex("\n".join(item.task.task_id for item in batch).encode())
            for item, turn in zip(batch, turns, strict=True):
                row = _prediction_row(item, turn, elapsed_ms, len(batch), batch_id)
                rows_by_task_id[item.task.task_id] = row
                if on_prediction is not None:
                    on_prediction(arm, row)
        predictions[arm] = [rows_by_task_id[task_id] for task_id in ids]
        arm_summaries[arm] = _summarize(predictions[arm])
        if isinstance(generate, ResourceReportingGenerator):
            arm_summaries[arm].update(generate.resource_metrics())
    comparison = []
    if "base" in arms and "sft" in arms:
        for base, sft in zip(predictions["base"], predictions["sft"], strict=True):
            base_correct = base["verifier_status"] == VerifierStatus.CORRECT.value
            sft_correct = sft["verifier_status"] == VerifierStatus.CORRECT.value
            outcome = "improved" if sft_correct and not base_correct else (
                "regressed" if base_correct and not sft_correct else "unchanged"
            )
            comparison.append({"task_id": base["task_id"], "base_correct": base_correct,
                               "sft_correct": sft_correct, "outcome": outcome})
    paired_statistics = _paired_statistics(comparison) if comparison else {}
    summary: dict[str, object] = {
        "task_count": len(ids), "task_ids_sha256": sha256_hex("\n".join(ids).encode()),
        "prompt_version": PROMPT_VERSION,
    }
    for arm in arms:
        summary[arm] = arm_summaries[arm]
    if comparison:
        summary.update({"paired": dict(Counter(row["outcome"] for row in comparison)), **paired_statistics})
    else:
        summary.update({"paired": None, "paired_bootstrap_ci95": None, "mcnemar_pvalue": None})
    return {
        "base_predictions": predictions.get("base", []),
        "sft_predictions": predictions.get("sft", []),
        "comparison": comparison,
        "summary": summary,
    }


def _prediction_row(
    item: LabeledMathTask, turn: ModelTurn, elapsed_ms: float, batch_size: int, batch_id: str
) -> dict[str, object]:
    extraction = extract(turn.text)
    verdict = (
        verify_answer(extraction.value, item.reference, task_id=item.task.task_id)
        if extraction.status is ExtractStatus.OK and extraction.value is not None
        else None
    )
    status = verdict.status.value if verdict else VerifierStatus.INVALID_PREDICTION.value
    return {
        "task_id": item.task.task_id, "dataset": item.task.dataset,
        "split": item.task.split, "source_hash": item.task.source_hash,
        "prediction": extraction.value, "raw_output": turn.text,
        "extract_status": extraction.status.value, "verifier_status": status,
        "prompt_tokens": turn.prompt_tokens, "output_tokens": turn.generated_tokens,
        "finish_reason": turn.finish_reason,
        "generation_latency_ms": elapsed_ms, "batch_size": batch_size,
        "generation_batch_id": batch_id,
    }


def _paired_statistics(comparison: list[dict[str, object]]) -> dict[str, object]:
    """Task-level paired bootstrap interval and exact McNemar p-value."""
    deltas = [int(bool(row["sft_correct"])) - int(bool(row["base_correct"]))
              for row in comparison]
    n = len(deltas)
    rng = random.Random(20260913)
    samples = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(10_000))
    improved = deltas.count(1)
    regressed = deltas.count(-1)
    discordant = improved + regressed
    tail = sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
    pvalue = min(1.0, 2 * tail / 2**discordant)
    return {
        "accuracy_delta": sum(deltas) / n,
        "paired_bootstrap_ci95": [samples[249], samples[9749]],
        "bootstrap_seed": 20260913, "bootstrap_resamples": 10_000,
        "mcnemar_pvalue": pvalue,
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    statuses = Counter(str(row["verifier_status"]) for row in rows)
    n = len(rows)
    output_lengths = sorted(cast(int, row["output_tokens"]) for row in rows)
    valid = statuses[VerifierStatus.CORRECT.value] + statuses[VerifierStatus.INCORRECT.value]
    latencies = [float(latency) for row in rows
                 if isinstance(latency := row.get("generation_latency_ms"), (int, float))]
    batch_durations: dict[str, float] = {}
    for row in rows:
        batch_id = row.get("generation_batch_id")
        latency = row.get("generation_latency_ms")
        if isinstance(batch_id, str) and isinstance(latency, (int, float)):
            batch_durations[batch_id] = float(latency)
    elapsed_seconds = sum(batch_durations.values()) / 1_000
    return {
        "total": n, "correct": statuses[VerifierStatus.CORRECT.value],
        "incorrect": statuses[VerifierStatus.INCORRECT.value],
        "invalid_prediction": statuses[VerifierStatus.INVALID_PREDICTION.value],
        "invalid_reference": statuses[VerifierStatus.INVALID_REFERENCE.value],
        "timeout": statuses[VerifierStatus.TIMEOUT.value],
        "internal_error": statuses[VerifierStatus.INTERNAL_ERROR.value],
        "verifier_accuracy": statuses[VerifierStatus.CORRECT.value] / n,
        "valid_answer_rate": valid / n,
        "mean_output_tokens": sum(output_lengths) / n,
        "median_output_tokens": (output_lengths[(n - 1) // 2] + output_lengths[n // 2]) / 2,
        "p50_generation_latency_ms": _percentile(latencies, 0.5),
        "p95_generation_latency_ms": _percentile(latencies, 0.95),
        "generation_elapsed_seconds": elapsed_seconds if batch_durations else None,
        "tokens_per_second": sum(output_lengths) / elapsed_seconds if elapsed_seconds > 0 else None,
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def write_artifacts(output_dir: Path, result: EvalResult, manifest: dict[str, object]) -> None:
    """Publish complete results only after all records and hashes are written."""
    if output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = output_dir.with_name(output_dir.name + f".tmp-{os.getpid()}")
    if staged.exists():
        raise FileExistsError(f"staging directory already exists: {staged}")
    staged.mkdir()
    try:
        hashes = {}
        for key in ("base_predictions", "sft_predictions", "comparison"):
            payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                              for row in result[key]).encode()
            path = staged / f"{key}.jsonl"
            path.write_bytes(payload)
            hashes[path.name] = sha256_hex(payload)
        summary = json.dumps(result["summary"], ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        (staged / "summary.json").write_text(summary)
        hashes["summary.json"] = sha256_hex(summary.encode())
        report = _render_evaluation_report(result["summary"])
        (staged / "evaluation_report.md").write_text(report)
        hashes["evaluation_report.md"] = sha256_hex(report.encode())
        manifest = {**manifest, "artifact_sha256": hashes,
                    "task_ids_sha256": result["summary"]["task_ids_sha256"]}
        (staged / "eval_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        (staged / "COMPLETE").write_text(sha256_hex((staged / "eval_manifest.json").read_bytes()) + "\n")
        os.replace(staged, output_dir)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def _render_evaluation_report(summary: dict[str, object]) -> str:
    """Render a small human-readable companion to the machine-readable summary."""
    if "base" not in summary or "sft" not in summary:
        # Adapter-only mode: report the evaluated arm without paired statistics.
        lines = [
            "# Adapter-only Evaluation",
            "",
            f"- Tasks: {summary['task_count']}",
            f"- Prompt version: `{summary['prompt_version']}`",
            "",
            "| Arm | Correct | Valid-answer rate | Verifier accuracy | p50 latency (ms) | p95 latency (ms) | Tokens/s | Peak GPU memory |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for arm in ("base", "sft"):
            if isinstance(summary.get(arm), dict):
                lines.append(_report_arm_row(arm.title(), cast(dict[str, object], summary[arm])))
        lines.append("")
        lines.append("_Paired statistics are computed offline against the frozen Base/SFT predictions._")
        return "\n".join(lines)
    base = cast(dict[str, object], summary["base"])
    sft = cast(dict[str, object], summary["sft"])
    paired = cast(dict[str, object], summary["paired"])
    lines = [
        "# Paired Base vs SFT Evaluation",
        "",
        f"- Tasks: {summary['task_count']}",
        f"- Prompt version: `{summary['prompt_version']}`",
        f"- Accuracy delta (SFT − Base): {float(cast(float, summary['accuracy_delta'])):.4f}",
        f"- Paired bootstrap 95% CI: {summary['paired_bootstrap_ci95']}",
        f"- McNemar p-value: {float(cast(float, summary['mcnemar_pvalue'])):.6g}",
        "",
        "| Arm | Correct | Valid-answer rate | Verifier accuracy | p50 latency (ms) | p95 latency (ms) | Tokens/s | Peak GPU memory |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _report_arm_row("Base", base),
        _report_arm_row("SFT", sft),
        "",
        "## Paired outcomes",
        "",
        f"- Improved: {paired.get('improved', 0)}",
        f"- Regressed: {paired.get('regressed', 0)}",
        f"- Unchanged: {paired.get('unchanged', 0)}",
        "",
        "Machine-readable provenance and checksums are in `eval_manifest.json`; per-task outputs are JSONL.",
        "",
    ]
    return "\n".join(lines)


def _report_arm_row(name: str, arm: dict[str, object]) -> str:
    def metric(key: str) -> str:
        value = arm.get(key)
        return "n/a" if value is None else f"{float(cast(float, value)):.2f}"

    peak_bytes = arm.get("peak_gpu_memory_bytes")
    peak_memory = "n/a" if peak_bytes is None else f"{int(cast(int, peak_bytes)) / 2**30:.2f} GiB"

    return (
        f"| {name} | {arm['correct']}/{arm['total']} | "
        f"{float(cast(float, arm['valid_answer_rate'])):.2%} | "
        f"{float(cast(float, arm['verifier_accuracy'])):.2%} | "
        f"{metric('p50_generation_latency_ms')} | {metric('p95_generation_latency_ms')} | "
        f"{metric('tokens_per_second')} | {peak_memory} |"
    )
