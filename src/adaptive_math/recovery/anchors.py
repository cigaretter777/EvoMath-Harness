"""Anchor extraction and selection (report section 6.1).

An anchor is a *state*, not a trajectory: the moment after the first natural
E_EXEC completed and before the next model generation. It is captured live as a
:class:`~adaptive_math.agent.loop.StepSnapshot`, because the events alone do not
retain the model-visible observations a resume has to replay.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from adaptive_math.agent.loop import StepSnapshot
from adaptive_math.agent.state import EventKind
from adaptive_math.core.types import JSONValue
from adaptive_math.recovery.seeds import stable_order_key
from adaptive_math.recovery.taxonomy import ErrorClass, classify_tool_result

MIN_REMAINING_STEPS = 3
MIN_REMAINING_TOOL_CALLS = 2
MIN_REMAINING_PYTHON_SECONDS = 3.0


@dataclass(frozen=True)
class AnchorCandidate:
    """One trajectory's first E_EXEC, with everything the pool rules read."""

    anchor_id: str
    task_id: str
    source_trajectory_id: str
    sample_index: int
    step_index: int
    eligible: bool
    exclusion_reason: str | None
    remaining_steps: int
    remaining_tool_calls: int
    remaining_python_seconds: float
    prefix_event_count: int
    prefix_message_count: int
    state_hash: str
    observation: str
    error_class: ErrorClass = ErrorClass.E_EXEC
    snapshot: StepSnapshot | None = field(default=None, repr=False, compare=False)

    def public_view(self) -> dict[str, JSONValue]:
        """The JSONL record; the snapshot stays out of it (it is code objects)."""
        return {
            "anchor_id": self.anchor_id,
            "task_id": self.task_id,
            "source_trajectory_id": self.source_trajectory_id,
            "sample_index": self.sample_index,
            "step_index": self.step_index,
            "error_class": self.error_class.value,
            "eligible": self.eligible,
            "exclusion_reason": self.exclusion_reason,
            "remaining_steps": self.remaining_steps,
            "remaining_tool_calls": self.remaining_tool_calls,
            "remaining_python_seconds": self.remaining_python_seconds,
            "prefix_event_count": self.prefix_event_count,
            "prefix_message_count": self.prefix_message_count,
            "state_hash": self.state_hash,
            "observation": self.observation,
        }


def state_hash(snapshot: StepSnapshot) -> str:
    """Content hash of the state a branch inherits. Two branches of the same
    anchor must report the same value; the arm edit is not part of it, because
    the intervention lives in the message list, which this deliberately does
    not cover."""
    return hashlib.sha256(snapshot.state.model_dump_json().encode()).hexdigest()


def first_exec_error_snapshot(
    snapshots: Sequence[StepSnapshot],
) -> StepSnapshot | None:
    """The first snapshot whose step ended in an executed failure.

    The snapshot's state ends with the ``tool_result`` event of that call, so
    its own messages already carry the original observation."""
    for snapshot in snapshots:
        events = snapshot.state.events
        if not events:
            continue
        last = events[-1]
        if last.kind is not EventKind.TOOL_RESULT:
            continue
        if classify_tool_result(last.payload) is ErrorClass.E_EXEC:
            return snapshot
    return None


def build_candidate(
    *,
    snapshot: StepSnapshot,
    task_id: str,
    source_trajectory_id: str,
    sample_index: int,
) -> AnchorCandidate:
    """Score one first-E_EXEC snapshot against the pool rules."""
    state = snapshot.state
    remaining_steps = max(0, state.budget.max_steps - state.usage.steps)
    remaining_calls = max(0, state.budget.max_tool_calls - state.usage.tool_calls)
    remaining_python = max(0.0, state.budget.max_python_seconds - state.usage.python_seconds)
    observation = snapshot.messages[-1].content

    reason: str | None = None
    if snapshot.terminated:
        reason = "terminated_at_anchor"
    elif remaining_steps < MIN_REMAINING_STEPS:
        reason = "insufficient_remaining_steps"
    elif remaining_calls < MIN_REMAINING_TOOL_CALLS:
        reason = "insufficient_remaining_tool_calls"
    elif remaining_python < MIN_REMAINING_PYTHON_SECONDS:
        reason = "insufficient_remaining_python_seconds"

    anchor_id = f"{task_id}::s{sample_index}::step{snapshot.step_index}"
    return AnchorCandidate(
        anchor_id=anchor_id,
        task_id=task_id,
        source_trajectory_id=source_trajectory_id,
        sample_index=sample_index,
        step_index=snapshot.step_index,
        eligible=reason is None,
        exclusion_reason=reason,
        remaining_steps=remaining_steps,
        remaining_tool_calls=remaining_calls,
        remaining_python_seconds=remaining_python,
        prefix_event_count=len(state.events),
        prefix_message_count=len(snapshot.messages),
        state_hash=state_hash(snapshot),
        observation=observation,
        snapshot=snapshot,
    )


def select_one_per_task(
    candidates: Sequence[AnchorCandidate],
) -> list[AnchorCandidate]:
    """At most one anchor per task, chosen by stable hash order.

    The order is fixed by task id, trajectory id and step index only -- never
    by how the trajectory turned out, nor by whether the error looks easy."""
    eligible = [candidate for candidate in candidates if candidate.eligible]
    ordered = sorted(
        eligible,
        key=lambda candidate: stable_order_key(
            "state_selection",
            candidate.task_id,
            candidate.source_trajectory_id,
            str(candidate.step_index),
        ),
    )
    seen: set[str] = set()
    selected: list[AnchorCandidate] = []
    for candidate in ordered:
        if candidate.task_id in seen:
            continue
        seen.add(candidate.task_id)
        selected.append(candidate)
    return selected


def pool_verdict(anchor_count: int) -> str:
    """Report section 6.1's operational threshold."""
    if anchor_count >= 20:
        return "C"
    if anchor_count >= 8:
        return "C_SMALL_SAMPLE"
    return "INSUFFICIENT_ERROR_STATES"


def summarise_candidates(
    candidates: Sequence[AnchorCandidate],
) -> Mapping[str, int]:
    counts: dict[str, int] = {"candidates": len(candidates), "eligible": 0}
    for candidate in candidates:
        if candidate.eligible:
            counts["eligible"] += 1
        else:
            reason = candidate.exclusion_reason or "unknown"
            counts[f"excluded:{reason}"] = counts.get(f"excluded:{reason}", 0) + 1
    return counts
