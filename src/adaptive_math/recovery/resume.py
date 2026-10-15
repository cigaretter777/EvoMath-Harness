"""Prefix-resume continuations for stage C (report sections 7 and 7.1).

A branch re-enters the episode from the anchor's snapshot: the model sees the
exact conversation it had already produced, the environment inherits the
counters it had already spent, and no prefix tool call is re-executed. That
last point matters for fairness -- replaying the prefix would either charge the
policy twice or hand the branch a free retry -- and it is safe here because the
sandbox is stateless per call (recorded as a pre-flight check).
"""

from dataclasses import dataclass
from typing import Any

from adaptive_math.agent.environment import OfflineMathEnv
from adaptive_math.agent.loop import AgentLoop, ConversationJournal, StepSnapshot
from adaptive_math.agent.model_client import ChatMessage, GenerationConfig, ModelClient
from adaptive_math.agent.state import TerminationReason
from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.types import Budget, LabeledMathTask
from adaptive_math.recovery.anchors import AnchorCandidate, state_hash
from adaptive_math.recovery.arms import Arm, is_faithful_full_arm, render_arm_observation
from adaptive_math.recovery.seeds import generation_seed
from adaptive_math.recovery.taxonomy import ErrorClass, classify_tool_result
from adaptive_math.tools.registry import ToolRegistry

_INFRA_TERMINATIONS = {
    TerminationReason.INFRASTRUCTURE_ERROR,
    TerminationReason.MODEL_ERROR,
}


@dataclass(frozen=True)
class ContinuationOutcome:
    """Everything one logical branch produced, ready to serialise."""

    record: dict[str, Any]
    trajectory: Trajectory
    infra_failure: bool
    infra_reason: str | None


def build_prefix_messages(
    snapshot: StepSnapshot, arm: Arm
) -> tuple[list[ChatMessage], str, str]:
    """Return (messages, original_observation, intervened_observation).

    Only the final tool message differs between arms; the guard keeps arm F
    honest by asserting it reproduced the frozen text verbatim."""
    messages = list(snapshot.messages)
    last = messages[-1]
    if last.role != "tool":
        raise ValueError(
            f"anchor prefix must end on the tool observation, got role={last.role!r}"
        )
    original = last.content
    intervened = render_arm_observation(original, arm)
    if arm is Arm.FULL and not is_faithful_full_arm(original, intervened):
        raise ValueError("arm F must keep the original observation verbatim")
    messages[-1] = ChatMessage(role="tool", content=intervened)
    return messages, original, intervened


def _suffix_events(trajectory: Trajectory, prefix_event_count: int) -> list[dict[str, Any]]:
    suffix = trajectory.events[prefix_event_count:]
    return [event.model_dump(mode="json") for event in suffix]


def _infra_assessment(trajectory: Trajectory, prefix_event_count: int) -> tuple[bool, str | None]:
    """Infrastructure faults are separated from the model's own mistakes.

    A branch that hit an unavailable sandbox never exercised the intervention,
    so per section 7.2 it is a missing sample, not a wrong answer."""
    if trajectory.termination_reason in _INFRA_TERMINATIONS:
        return True, f"termination:{trajectory.termination_reason.value}"
    for event in trajectory.events[prefix_event_count:]:
        if event.kind.value != "tool_result":
            continue
        if classify_tool_result(event.payload) is ErrorClass.E_INFRA:
            return True, "tool_result:unavailable"
    return False, None


async def run_continuation(
    *,
    run_id: str,
    stage: str,
    anchor: AnchorCandidate,
    labeled_task: LabeledMathTask,
    arm: Arm,
    sample_index: int,
    attempt: int,
    model: ModelClient,
    agent: AgentLoop,
    generation: GenerationConfig,
    registry: ToolRegistry,
    model_version: str,
    protocol_hash: str,
) -> ContinuationOutcome:
    """Run one branch from the anchor state; never resets the budget."""
    snapshot = anchor.snapshot
    if snapshot is None:
        raise ValueError("anchor carries no snapshot; it cannot be resumed")
    budget: Budget = snapshot.state.budget
    sample_id = f"{anchor.anchor_id}::a{arm.value}::{sample_index}"
    attempt_id = f"{sample_id}::{attempt}"
    trace_id = f"{run_id}:{attempt_id}"

    messages, original, intervened = build_prefix_messages(snapshot, arm)

    remaining_steps = max(0, budget.max_steps - snapshot.state.usage.steps)
    seed_schedule = [
        generation_seed(
            stage=stage, task_id=anchor.task_id, sample_id=sample_id, arm=arm.value, turn=turn
        )
        for turn in range(remaining_steps + 1)
    ]

    environment = OfflineMathEnv(
        labeled_task,
        budget,
        registry,
        trace_id=trace_id,
        state=snapshot.state,
    )
    journal = ConversationJournal()
    trajectory = await agent.run(
        environment,
        model,
        generation,
        prefix_messages=messages,
        journal=journal,
        seed_schedule=seed_schedule,
    )
    verdict = environment.evaluate()
    infra_failure, infra_reason = _infra_assessment(trajectory, anchor.prefix_event_count)
    suffix_state_hash = (
        state_hash(journal.snapshots[-1]) if journal.snapshots else None
    )

    suffix_usage = {
        "steps": trajectory.usage.steps - snapshot.state.usage.steps,
        "tool_calls": trajectory.usage.tool_calls - snapshot.state.usage.tool_calls,
        "python_seconds": trajectory.usage.python_seconds - snapshot.state.usage.python_seconds,
        "generated_tokens": trajectory.usage.generated_tokens - snapshot.state.usage.generated_tokens,
        "invalid_actions": trajectory.usage.invalid_actions - snapshot.state.usage.invalid_actions,
    }

    record: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "logical_sample_id": sample_id,
        "attempt_id": attempt_id,
        "attempt": attempt,
        "task_id": anchor.task_id,
        "anchor_id": anchor.anchor_id,
        "source_trajectory_id": anchor.source_trajectory_id,
        "anchor_sample_index": anchor.sample_index,
        "anchor_step_index": anchor.step_index,
        "arm": arm.value,
        "sample_index": sample_index,
        "trace_id": trace_id,
        "generation_seeds": seed_schedule,
        "model_version": model_version,
        "protocol_hash": protocol_hash,
        "prefix_event_count": anchor.prefix_event_count,
        "prefix_message_count": anchor.prefix_message_count,
        "state_hash": anchor.state_hash,
        "suffix_state_hash": suffix_state_hash,
        "suffix_step_count": len(journal.snapshots),
        "observation_before": original,
        "observation_after": intervened,
        "observation_chars_before": len(original),
        "observation_chars_after": len(intervened),
        "remaining_budget_at_anchor": {
            "steps": remaining_steps,
            "tool_calls": max(0, budget.max_tool_calls - snapshot.state.usage.tool_calls),
            "python_seconds": max(
                0.0, budget.max_python_seconds - snapshot.state.usage.python_seconds
            ),
        },
        "budget": budget.model_dump(mode="json"),
        "suffix_events": _suffix_events(trajectory, anchor.prefix_event_count),
        "suffix_usage": suffix_usage,
        "final_answer": trajectory.final_answer,
        "verdict": verdict.model_dump(mode="json") if verdict is not None else None,
        "correct": bool(verdict is not None and verdict.reward > 0),
        "termination_reason": trajectory.termination_reason.value,
        "total_usage": trajectory.usage.model_dump(mode="json"),
        "infra_failure": infra_failure,
        "infra_reason": infra_reason,
        "status": "missing" if infra_failure else "complete",
    }
    return ContinuationOutcome(
        record=record,
        trajectory=trajectory,
        infra_failure=infra_failure,
        infra_reason=infra_reason,
    )
