"""Run the feedback-recovery diagnostic stages A, B and C (2026-10-14 protocol).

  --stage a  10 tasks x 2 samples: the interaction-validity gate (section 5)
  --stage b  40 tasks x 4 samples: natural errors and the anchor pool (section 6)
  --stage c  anchors x 3 arms x 4 continuations: same-state branches (section 7)

Stage A/B call the frozen runner path (real AgentLoop, real sandbox, real
verifier) with two additions the protocol needs: a per-generation seed derived
from the frozen main seed, and a conversation journal that captures the exact
model-visible prefix at every step so stage C can re-enter the episode.

Stage C never replays prefix tool calls. It reloads the anchor's journal
snapshot, checks its content hash, edits exactly one observation, and continues
with the inherited counters.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from adaptive_math.agent.environment import OfflineMathEnv
from adaptive_math.agent.loop import AgentLoop, ConversationJournal, StepSnapshot
from adaptive_math.agent.model_client import ChatMessage, GenerationConfig
from adaptive_math.agent.parser import TOLERANCE_RULES, parse_action
from adaptive_math.agent.prompts import render_initial_messages
from adaptive_math.agent.state import EventKind, StepErrorCode, TerminationReason
from adaptive_math.agent.trace import Trajectory
from adaptive_math.agent.transformers_client import TransformersModelClient
from adaptive_math.core.types import Budget, LabeledMathTask
from adaptive_math.recovery.anchors import (
    AnchorCandidate,
    build_candidate,
    first_exec_error_snapshot,
    pool_verdict,
    select_one_per_task,
    summarise_candidates,
)
from adaptive_math.recovery.arms import ARMS, Arm
from adaptive_math.recovery.resume import run_continuation
from adaptive_math.recovery.seeds import generation_seed, stable_order_key
from adaptive_math.recovery.taxonomy import (
    ErrorClass,
    classify_tool_result,
    is_call_form_error,
)
from adaptive_math.tools.base import Tool
from adaptive_math.tools.python_tool import PythonTool
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.tools.sandboxfusion import SandboxFusionClient
from adaptive_math.tools.sympy_tool import SympyTool
from adaptive_math.training.rollout_health import row_to_labeled_task
from adaptive_math.verifier.service import VerifierStatus, verify_answer

STAGE_A_TASKS = 10
STAGE_A_SAMPLES = 2
STAGE_B_TASKS = 40
STAGE_B_SAMPLES = 4
CONTINUATIONS_PER_ARM = 4


# --------------------------------------------------------------------------- io


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_fingerprint(model_path: Path) -> dict[str, Any]:
    """Content hashes for a model directory, cached per (path, inventory)."""
    inventory = sorted(
        (str(item.relative_to(model_path)), item.stat().st_size)
        for item in model_path.rglob("*")
        if item.is_file()
    )
    cached: dict[str, Any] = {}
    if CACHE_PATH.is_file():
        cached = json.loads(CACHE_PATH.read_text())
    key = f"{model_path}:{hash(json.dumps(inventory))}"
    if key in cached:
        return cached[key]
    files = {
        name: {"size": size, "sha256": _sha256_file(model_path / name)}
        for name, size in inventory
    }
    payload = {
        "path": str(model_path),
        "file_count": len(files),
        "total_bytes": sum(size for _, size in inventory),
        "files": files,
    }
    cached[key] = payload
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cached, indent=2))
    return payload


def _chat_template_fingerprint(tokenizer: Any) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    return {
        "chat_template_sha256": _sha256_text(str(template)) if template else None,
        "chat_template_chars": len(str(template)) if template else 0,
        "enable_thinking_passed": False,
        "note": (
            "apply_chat_template is called without enable_thinking, so the "
            "template default decides; recorded as an unresolved uncertainty"
        ),
    }


# ------------------------------------------------------------------------ probe


async def _sandbox_probe(client: SandboxFusionClient) -> dict[str, Any]:
    """Canonical pong+42 probe plus the statelessness check of section 3.5.

    Two calls in sequence: if the interpreter kept state, the second one would
    see ``x``. The measured answer is recorded, not assumed."""
    probe: dict[str, Any] = {
        "sandbox_url": os.environ.get("ADAPTIVE_MATH_SANDBOX_URL", "http://127.0.0.1:8080"),
        "ping": await client.health(),
    }
    first = await client.run_code("print(6*7)")
    probe["pong42"] = first.ok and "42" in first.output
    probe["pong42_output"] = first.output[:200]
    define = await client.run_code("probe_marker = 1234")
    read = await client.run_code("print('probe_marker' in dir())")
    probe["stateless_first_call_ok"] = define.ok
    probe["stateless_second_call_output"] = read.output[:200]
    probe["persistent_across_calls"] = "True" in read.output
    return probe


# ---------------------------------------------------------------------- stages


@dataclass
class StageContext:
    args: argparse.Namespace
    run_dir: Path
    model: TransformersModelClient
    registry: ToolRegistry
    sandbox: SandboxFusionClient
    budget: Budget
    generation: GenerationConfig
    agent: AgentLoop
    model_version: str
    protocol_hash: str


def _load_budget(path: Path) -> Budget:
    return Budget.model_validate(yaml.safe_load(path.read_text()))


def _load_tasks(data_path: Path) -> list[LabeledMathTask]:
    frame = pd.read_parquet(data_path)
    return [row_to_labeled_task(row) for row in frame.to_dict(orient="records")]


def _load_model(args: argparse.Namespace) -> tuple[TransformersModelClient, str]:
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if args.adapter is not None and not Path(args.adapter).is_dir():
        raise FileNotFoundError(args.adapter)
    model = TransformersModelClient.from_pretrained(
        str(model_path),
        device=args.device,
        dtype=args.dtype,
        adapter=str(args.adapter) if args.adapter is not None else None,
    )
    fingerprint = _model_fingerprint(model_path)
    version = f"{model_path}@{fingerprint['files'][min(fingerprint['files'])]['sha256'][:12]}"
    return model, version


def _build_context(args: argparse.Namespace) -> StageContext:
    model, model_version = _load_model(args)
    sandbox = SandboxFusionClient()
    registry = ToolRegistry([cast(Tool, SympyTool()), cast(Tool, PythonTool(sandbox))])
    budget = _load_budget(Path(args.agent_config))
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature
    )
    return StageContext(
        args=args,
        run_dir=Path(args.run_dir),
        model=model,
        registry=registry,
        sandbox=sandbox,
        budget=budget,
        generation=generation,
        agent=AgentLoop(parser_tolerance=args.parser_tolerance or ()),
        model_version=model_version,
        protocol_hash=args.protocol_hash,
    )


@dataclass
class Rollout:
    task: LabeledMathTask
    sample_index: int
    trajectory: Trajectory
    verdict: Any
    journal: ConversationJournal
    seconds: float


async def _roll_one(
    context: StageContext,
    task: LabeledMathTask,
    *,
    stage: str,
    sample_index: int,
) -> Rollout:
    task_id = task.task.task_id
    sample_id = f"{task_id}::s{sample_index}"
    seeds = [
        generation_seed(
            stage=stage, task_id=task_id, sample_id=sample_id, arm="none", turn=turn
        )
        for turn in range(context.budget.max_steps)
    ]
    environment = OfflineMathEnv(
        task, context.budget, context.registry, trace_id=f"{context.args.run_id}:{sample_id}"
    )
    journal = ConversationJournal()
    started = time.perf_counter()
    trajectory = await context.agent.run(
        environment,
        context.model,
        context.generation,
        journal=journal,
        seed_schedule=seeds,
    )
    seconds = time.perf_counter() - started
    return Rollout(
        task=task,
        sample_index=sample_index,
        trajectory=trajectory,
        verdict=environment.evaluate(),
        journal=journal,
        seconds=seconds,
    )


def _classify_trajectory(trajectory: Trajectory) -> dict[str, Any]:
    """Per-trajectory error inventory used by the stage B report (section 6)."""
    counts: Counter[str] = Counter()
    for event in trajectory.events:
        if event.kind is EventKind.TOOL_RESULT:
            error = classify_tool_result(event.payload)
            if error is None and is_call_form_error(event.payload):
                counts["call_form_errors"] += 1
            elif error is not None:
                counts[error.value] += 1
        elif event.kind is EventKind.TOOL_CALL:
            counts["executed_calls"] += 1
    for event in trajectory.events:
        if event.kind is EventKind.MODEL_OUTPUT and event.payload.get("finish_reason") == "length":
            counts["length_truncated_turns"] += 1
    return dict(counts)


def _rollout_record(rollout: Rollout, *, stage: str, index: int, args: argparse.Namespace) -> dict[str, Any]:
    trajectory = rollout.trajectory
    verdict = rollout.verdict
    counts = _classify_trajectory(trajectory)
    return {
        "stage": stage,
        "task_index": index,
        "sample_index": rollout.sample_index,
        "task_id": rollout.task.task.task_id,
        "trace_id": trajectory.trace_id,
        "rollout_seconds": rollout.seconds,
        "trajectory": trajectory.model_dump(mode="json"),
        "verdict": verdict.model_dump(mode="json") if verdict is not None else None,
        "correct": bool(verdict is not None and verdict.status is VerifierStatus.CORRECT),
        "error_counts": counts,
        "finish_reasons": [
            event.payload.get("finish_reason")
            for event in trajectory.events
            if event.kind is EventKind.MODEL_OUTPUT
        ],
        "generation_seed_count": sum(
            1 for event in trajectory.events if event.kind is EventKind.MODEL_OUTPUT
        ),
        "parser_tolerance": list(args.parser_tolerance or ()),
    }


async def run_stage_ab(context: StageContext, *, stage: str) -> dict[str, Any]:
    args = context.args
    data_path = Path(args.data)
    tasks = _load_tasks(data_path)
    expected = STAGE_A_TASKS if stage == "a" else STAGE_B_TASKS
    if args.limit_tasks is not None:
        tasks = tasks[: args.limit_tasks]
    elif len(tasks) != expected:
        raise ValueError(f"stage {stage} expects {expected} tasks, pool has {len(tasks)}")
    samples = args.group_size or (STAGE_A_SAMPLES if stage == "a" else STAGE_B_SAMPLES)

    stage_dir = context.run_dir / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = stage_dir / "trajectories.jsonl"
    task_pool_path = stage_dir / "task_pool.jsonl"
    candidates_path = (
        context.run_dir / "anchors" / "candidates.jsonl" if stage == "b" else None
    )
    snapshots_path = context.run_dir / "anchors" / "snapshots.jsonl" if stage == "b" else None

    if trajectories_path.exists():
        raise FileExistsError(f"{trajectories_path} exists; refuse to overwrite a stage")

    with task_pool_path.open("w") as handle:
        for task in tasks:
            handle.write(task.model_dump_json() + "\n")

    rollouts: list[Rollout] = []
    started = time.perf_counter()
    for index, task in enumerate(tasks):
        for sample_index in range(samples):
            rollout = await _roll_one(context, task, stage=stage, sample_index=sample_index)
            rollouts.append(rollout)
            record = _rollout_record(rollout, stage=stage, index=index, args=args)
            _append_jsonl(trajectories_path, record)
            usage = rollout.trajectory.usage
            print(
                f"[fr-{stage}] {index + 1}/{len(tasks)} s{sample_index} "
                f"correct={record['correct']} term={rollout.trajectory.termination_reason.value} "
                f"calls={usage.tool_calls} steps={usage.steps} "
                f"errors={record['error_counts']}",
                flush=True,
            )

    summary: dict[str, Any] = {
        "stage": stage,
        "task_count": len(tasks),
        "samples_per_task": samples,
        "trajectory_count": len(rollouts),
        "elapsed_seconds": time.perf_counter() - started,
        "generated_tokens_total": sum(r.trajectory.usage.generated_tokens for r in rollouts),
        "tool_calls_total": sum(r.trajectory.usage.tool_calls for r in rollouts),
        "python_seconds_total": sum(r.trajectory.usage.python_seconds for r in rollouts),
    }

    if stage == "a":
        summary["verifier_smoke"] = _verifier_smoke(tasks)
        summary["gate"] = _stage_a_gate(context, rollouts, tasks)
    else:
        summary.update(await _stage_b_pool(context, rollouts, candidates_path, snapshots_path))
        summary["natural"] = _stage_b_summary(rollouts)

    _write_json(stage_dir / "summary.json", summary)
    return summary


def _verifier_smoke(tasks: Sequence[LabeledMathTask]) -> dict[str, Any]:
    """Judge smoke test required by section 3.4 / section 5.5: the reference
    must be accepted as correct and a wrong answer must not be. A verdict of
    INVALID_PREDICTION is an honest "not correct"; an INTERNAL_ERROR is a fault
    and fails the smoke test."""
    task = tasks[0]
    same = verify_answer(task.reference.value, task.reference, task_id=task.task.task_id)
    wrong = verify_answer(
        "deliberately-not-the-answer-42", task.reference, task_id=task.task.task_id
    )
    return {
        "task_id": task.task.task_id,
        "answer_type": task.task.answer_type.value,
        "pipeline_version": task.task.pipeline_version,
        "reference_self_status": same.status.value,
        "reference_self_reward": same.reward,
        "wrong_status": wrong.status.value,
        "wrong_reward": wrong.reward,
        "pass": (
            same.reward == 1.0
            and wrong.reward == 0.0
            and VerifierStatus.INTERNAL_ERROR not in {same.status, wrong.status}
        ),
    }


def _stage_a_gate(
    context: StageContext, rollouts: Sequence[Rollout], tasks: Sequence[LabeledMathTask]
) -> dict[str, Any]:
    """Section 5's operational gate, computed from the recorded artifacts."""
    generation_steps = 0
    parse_error_steps = 0
    truncated_without_action = 0
    infra_trajectories = 0
    executed_trajectories = 0
    successful_trajectories = 0

    for rollout in rollouts:
        trajectory = rollout.trajectory
        if trajectory.termination_reason in {
            TerminationReason.INFRASTRUCTURE_ERROR,
            TerminationReason.MODEL_ERROR,
        } or any(
            classify_tool_result(event.payload) is ErrorClass.E_INFRA
            for event in trajectory.events
            if event.kind is EventKind.TOOL_RESULT
        ):
            infra_trajectories += 1
        if trajectory.usage.tool_calls > 0:
            executed_trajectories += 1
            if any(
                event.kind is EventKind.TOOL_RESULT
                and classify_tool_result(event.payload) is None
                and not is_call_form_error(event.payload)
                for event in trajectory.events
            ):
                successful_trajectories += 1
        for event in trajectory.events:
            if event.kind is not EventKind.MODEL_OUTPUT:
                continue
            generation_steps += 1
            raw = str(event.payload.get("raw", ""))
            parsed = parse_action(raw, tolerate=context.args.parser_tolerance or ())
            if parsed.action is None:
                if event.payload.get("finish_reason") == "length":
                    truncated_without_action += 1
                else:
                    parse_error_steps += 1

    env_parse_errors = sum(
        1
        for rollout in rollouts
        for event in rollout.trajectory.events
        if event.kind is EventKind.INVALID_ACTION
        and event.error_code is StepErrorCode.ACTION_PARSE_ERROR
    )

    leak_hits: list[dict[str, str]] = []
    for task in tasks:
        rendered = " ".join(
            message.content for message in render_initial_messages(task.task, context.budget, context.registry)
        )
        reference = task.reference.value.strip()
        if len(reference) >= 4 and reference in rendered:
            leak_hits.append({"task_id": task.task.task_id, "match": reference})

    checks = {
        "executed_tool_call_trajectories": {
            "value": executed_trajectories,
            "threshold": 5,
            "total": len(rollouts),
            "pass": executed_trajectories >= 5,
        },
        "infrastructure_failures": {
            "value": infra_trajectories,
            "threshold": 0,
            "pass": infra_trajectories == 0,
        },
        "parse_error_step_rate": {
            "value": parse_error_steps / max(1, generation_steps),
            "threshold": 0.10,
            "pass": parse_error_steps / max(1, generation_steps) <= 0.10,
        },
        "length_truncated_without_action_rate": {
            "value": truncated_without_action / max(1, generation_steps),
            "threshold": 0.10,
            "pass": truncated_without_action / max(1, generation_steps) <= 0.10,
        },
        "hidden_label_leakage": {"value": leak_hits, "pass": not leak_hits},
    }
    return {
        "checks": checks,
        "generation_steps": generation_steps,
        "parse_error_steps": parse_error_steps,
        "environment_parse_error_steps": env_parse_errors,
        "reparse_agrees_with_environment": parse_error_steps == env_parse_errors,
        "length_truncated_without_action_steps": truncated_without_action,
        "successful_tool_call_trajectories": successful_trajectories,
        "automated_pass": all(item["pass"] for item in checks.values()),
        "manual_checks_required": [
            "read at least 3 full call-feedback-continuation trajectories",
            "confirm the rendered input matches the log",
            "answer-judge smoke test (see verifier_smoke in the stage summary)",
        ],
    }


async def _stage_b_pool(
    context: StageContext,
    rollouts: Sequence[Rollout],
    candidates_path: Path | None,
    snapshots_path: Path | None,
) -> dict[str, Any]:
    """Section 6.1: candidate anchors, exclusions, and at most one per task."""
    assert candidates_path is not None and snapshots_path is not None
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidates: list[AnchorCandidate] = []
    for rollout in rollouts:
        snapshot = first_exec_error_snapshot(rollout.journal.snapshots)
        if snapshot is None:
            continue
        candidate = build_candidate(
            snapshot=snapshot,
            task_id=rollout.task.task.task_id,
            source_trajectory_id=rollout.trajectory.trace_id,
            sample_index=rollout.sample_index,
        )
        candidates.append(candidate)
        _append_jsonl(candidates_path, candidate.public_view())

    selected = select_one_per_task(candidates)
    with snapshots_path.open("w") as handle:
        for candidate in selected:
            assert candidate.snapshot is not None
            handle.write(
                json.dumps(_snapshot_record(candidate), ensure_ascii=False, default=str) + "\n"
            )
    selected_path = candidates_path.parent / "selected.jsonl"
    with selected_path.open("w") as handle:
        for candidate in selected:
            handle.write(json.dumps(candidate.public_view(), ensure_ascii=False, default=str) + "\n")

    verdict = pool_verdict(len(selected))
    return {
        "candidates": len(candidates),
        "candidate_summary": summarise_candidates(candidates),
        "selected_anchors": len(selected),
        "selected_by_task": [candidate.task_id for candidate in selected],
        "pool_verdict": verdict,
        "pool_verdict_note": (
            ">=20 anchors: run stage C; 8-19: small-sample exploration; "
            "<8: INSUFFICIENT_ERROR_STATES (report section 6.1)"
        ),
        "selected_path": str(selected_path),
        "snapshots_path": str(snapshots_path),
    }


def _snapshot_record(candidate: AnchorCandidate) -> dict[str, Any]:
    snapshot = candidate.snapshot
    assert snapshot is not None
    return {
        "anchor_id": candidate.anchor_id,
        "task_id": candidate.task_id,
        "source_trajectory_id": candidate.source_trajectory_id,
        "sample_index": candidate.sample_index,
        "step_index": candidate.step_index,
        "state_hash": candidate.state_hash,
        "prefix_event_count": candidate.prefix_event_count,
        "remaining_steps": candidate.remaining_steps,
        "remaining_tool_calls": candidate.remaining_tool_calls,
        "remaining_python_seconds": candidate.remaining_python_seconds,
        "observation": candidate.observation,
        "state": snapshot.state.model_dump(mode="json"),
        "messages": [message.model_dump(mode="json") for message in snapshot.messages],
        "terminated": snapshot.terminated,
    }


def _stage_b_summary(rollouts: Sequence[Rollout]) -> dict[str, Any]:
    """Section 6's natural-run descriptive statistics."""
    total = len(rollouts)
    no_call = 0
    executed = 0
    exec_error_trajectories = 0
    timeout_trajectories = 0
    infra_trajectories = 0
    correct = 0
    correct_after_error = 0
    after_error = 0
    terminations: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    per_task: dict[str, list[bool]] = {}

    for rollout in rollouts:
        trajectory = rollout.trajectory
        counts = _classify_trajectory(trajectory)
        for key, value in counts.items():
            errors[key] += value
        terminations[trajectory.termination_reason.value] += 1
        is_correct = bool(
            rollout.verdict is not None and rollout.verdict.status is VerifierStatus.CORRECT
        )
        correct += is_correct
        per_task.setdefault(rollout.task.task.task_id, []).append(is_correct)
        if counts.get("executed_calls", 0) == 0:
            no_call += 1
        else:
            executed += 1
        if counts.get(ErrorClass.E_EXEC.value, 0) > 0:
            exec_error_trajectories += 1
            after_error += 1
            correct_after_error += is_correct
        if counts.get(ErrorClass.E_TIMEOUT.value, 0) > 0:
            timeout_trajectories += 1
        if counts.get(ErrorClass.E_INFRA.value, 0) > 0:
            infra_trajectories += 1

    return {
        "trajectories": total,
        "accuracy": correct / max(1, total),
        "no_tool_call_rate": no_call / max(1, total),
        "executed_tool_call_rate": executed / max(1, total),
        "exec_error_trajectory_rate": exec_error_trajectories / max(1, total),
        "timeout_trajectory_rate": timeout_trajectories / max(1, total),
        "infrastructure_trajectory_rate": infra_trajectories / max(1, total),
        "conditional_accuracy_after_exec_error": (
            correct_after_error / after_error if after_error else None
        ),
        "trajectories_with_exec_error": after_error,
        "termination_counts": dict(sorted(terminations.items())),
        "error_counts": dict(sorted(errors.items())),
        "tasks_with_any_exec_error": sum(
            1 for values in per_task.values() if values
        ),
    }


# ------------------------------------------------------------------- stage C


def _load_snapshots(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _rebuild_anchor(record: dict[str, Any]) -> AnchorCandidate:
    """Rehydrate a selected anchor and verify it reloaded byte-identically."""
    from adaptive_math.agent.state import AgentState

    state = AgentState.model_validate(record["state"])
    messages = tuple(ChatMessage.model_validate(item) for item in record["messages"])
    snapshot = StepSnapshot(
        step_index=record["step_index"],
        messages=messages,
        state=state,
        terminated=record["terminated"],
    )
    candidate = build_candidate(
        snapshot=snapshot,
        task_id=record["task_id"],
        source_trajectory_id=record["source_trajectory_id"],
        sample_index=record["sample_index"],
    )
    if candidate.state_hash != record["state_hash"] or not candidate.eligible:
        raise ValueError(
            f"anchor {record['anchor_id']} did not reload intact "
            f"(hash {candidate.state_hash} vs {record['state_hash']}, "
            f"reason {candidate.exclusion_reason})"
        )
    return candidate


def _arm_order(anchor_id: str) -> list[Arm]:
    """Rotate the arm order per anchor so no arm is systematically first or
    last (section 7.1), fixed by the run-order hash rather than by timing."""
    return sorted(ARMS, key=lambda arm: stable_order_key("run_order", anchor_id, arm.value))


async def run_stage_c(context: StageContext) -> dict[str, Any]:
    args = context.args
    snapshots = _load_snapshots(Path(args.anchors))
    if args.limit_anchors is not None:
        snapshots = snapshots[: args.limit_anchors]
    tasks = {task.task.task_id: task for task in _load_tasks(Path(args.data))}
    continuations = args.continuations_per_arm or CONTINUATIONS_PER_ARM

    stage_dir = context.run_dir / "stage_c"
    stage_dir.mkdir(parents=True, exist_ok=True)
    output_path = stage_dir / "continuations.jsonl"
    if output_path.exists() and not args.append:
        raise FileExistsError(f"{output_path} exists; refuse to overwrite a stage")
    if output_path.exists():
        done = {
            json.loads(line)["logical_sample_id"]
            for line in output_path.read_text().splitlines()
            if line.strip()
        }
    else:
        done = set()

    executed = 0
    started = time.perf_counter()
    for anchor_index, record in enumerate(snapshots):
        anchor = _rebuild_anchor(record)
        task = tasks[record["task_id"]]
        for arm in _arm_order(anchor.anchor_id):
            for sample_index in range(continuations):
                sample_id = f"{anchor.anchor_id}::a{arm.value}::{sample_index}"
                if sample_id in done:
                    continue
                for attempt in range(2):  # one retry, same seeds (section 7.2)
                    outcome = await run_continuation(
                        run_id=args.run_id,
                        stage="C",
                        anchor=anchor,
                        labeled_task=task,
                        arm=arm,
                        sample_index=sample_index,
                        attempt=attempt,
                        model=context.model,
                        agent=context.agent,
                        generation=context.generation,
                        registry=context.registry,
                        model_version=context.model_version,
                        protocol_hash=context.protocol_hash,
                    )
                    record_out = outcome.record
                    record_out["anchor_index"] = anchor_index
                    record_out["execution_order"] = executed
                    _append_jsonl(output_path, record_out)
                    executed += 1
                    print(
                        f"[fr-c] {anchor_index + 1}/{len(snapshots)} {anchor.anchor_id} "
                        f"arm={arm.value} s{sample_index} attempt={attempt} "
                        f"correct={record_out['correct']} "
                        f"term={record_out['termination_reason']} "
                        f"infra={record_out['infra_failure']}",
                        flush=True,
                    )
                    if not outcome.infra_failure:
                        break

    summary = {
        "stage": "c",
        "anchors": len(snapshots),
        "arms": [arm.value for arm in ARMS],
        "continuations_per_arm": continuations,
        "executed_this_invocation": executed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(stage_dir / "summary.json", summary)
    return summary


def _stage_c_integrity(path: Path) -> dict[str, Any]:
    """Section 10.6: the C log must be internally consistent and complete."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_sample.setdefault(record["logical_sample_id"], []).append(record)
    problems: list[str] = []
    for sample_id, attempts in by_sample.items():
        if len(attempts) > 2:
            problems.append(f"{sample_id}: {len(attempts)} attempts")
        if len(attempts) > 1 and not attempts[0]["infra_failure"]:
            problems.append(f"{sample_id}: retried after a non-infrastructure failure")
        if len({attempt["generation_seeds"][0] for attempt in attempts}) != 1:
            problems.append(f"{sample_id}: retry changed the seed")
    return {
        "records": len(records),
        "logical_samples": len(by_sample),
        "problems": problems,
        "complete": not problems,
    }


# ---------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("a", "b", "c"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--data", default="data/processed/diagnostics/feedback_recovery/fr_a10.parquet"
    )
    parser.add_argument("--anchors", default=None)
    parser.add_argument(
        "--model", default="artifacts/models/qwen3_1_7b_sft_tir_merged"
    )
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--agent-config", default="configs/agent/default.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--continuations-per-arm", type=int, default=0)
    parser.add_argument("--parser-tolerance", default="unclosed_think,bare_final_scalar")
    parser.add_argument("--protocol-hash", default="unfrozen")
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--limit-anchors", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--require-sandbox", action="store_true", default=True)
    return parser


def _parse_tolerance(value: str) -> tuple[str, ...]:
    rules = tuple(rule for rule in value.split(",") if rule)
    unknown = sorted(set(rules) - set(TOLERANCE_RULES))
    if unknown:
        raise SystemExit(f"unknown parser tolerance rules: {unknown}")
    return rules


async def _main() -> int:
    args = build_parser().parse_args()
    args.parser_tolerance = _parse_tolerance(args.parser_tolerance)
    context = _build_context(args)
    context.run_dir.mkdir(parents=True, exist_ok=True)

    probe = await _sandbox_probe(context.sandbox)
    if not probe["pong42"] and args.require_sandbox:
        _write_json(context.run_dir / "sandbox_probe.json", probe)
        raise SystemExit(
            f"sandbox probe failed ({probe}); refusing to run a stage that would "
            "record infrastructure faults as model behaviour"
        )
    _write_json(context.run_dir / "sandbox_probe.json", probe)

    manifest = {
        "run_id": args.run_id,
        "stage": args.stage,
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": str(args.model),
        "model_version": context.model_version,
        "model_fingerprint": _model_fingerprint(Path(args.model)),
        "chat_template": _chat_template_fingerprint(context.model._tokenizer),
        "adapter": args.adapter,
        "data": str(args.data),
        "budget": context.budget.model_dump(mode="json"),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed_enforced_by_transformers_client": True,
            "seed_key": "stage,task_id,sample_id,arm,turn -> sha256(main_seed)",
        },
        "parser_tolerance": list(args.parser_tolerance),
        "protocol_hash": args.protocol_hash,
        "sandbox_probe": probe,
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "status": "running",
    }
    manifest_path = context.run_dir / f"manifest.{args.stage}.json"
    _write_json(manifest_path, manifest)

    try:
        if args.stage in {"a", "b"}:
            summary = await run_stage_ab(context, stage=args.stage)
        else:
            if not args.anchors:
                raise SystemExit("--anchors is required for stage c")
            summary = await run_stage_c(context)
            summary["integrity"] = _stage_c_integrity(
                context.run_dir / "stage_c" / "continuations.jsonl"
            )
        manifest["status"] = "complete"
        manifest["summary"] = summary
        _write_json(manifest_path, manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
        return 0
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        _write_json(manifest_path, manifest)
        raise
    finally:
        await context.sandbox.aclose()


CACHE_PATH = Path("artifacts/diagnostics/feedback_recovery/.model_fingerprint_cache.json")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
