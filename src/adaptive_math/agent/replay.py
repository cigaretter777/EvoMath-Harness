"""Content-addressed trace envelopes and offline deterministic validation."""

import hashlib

import orjson
from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.agent.state import EventKind, TerminationReason, Usage
from adaptive_math.agent.trace import Trajectory

TRACE_SCHEMA_VERSION = "trace-v1"


class TraceReplayError(ValueError):
    """Raised when a stored trace cannot be faithfully replayed."""


class TraceEnvelope(BaseModel):
    """Portable trace storage; creation time is deliberately outside the hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = TRACE_SCHEMA_VERSION
    runtime_version: str
    prompt_version: str
    created_at: str
    trajectory: Trajectory
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls, trajectory: Trajectory, *, prompt_version: str, created_at: str
    ) -> "TraceEnvelope":
        return cls(
            runtime_version=trajectory.runtime_version,
            prompt_version=prompt_version,
            created_at=created_at,
            trajectory=trajectory,
            content_hash=trajectory_hash(trajectory),
        )


def trajectory_hash(trajectory: Trajectory) -> str:
    """Hash canonical JSON without non-deterministic envelope metadata.

    ``harness_spec_hash`` is dropped when None so that trajectories recorded
    before the harness field existed keep their original content hash.
    """
    payload = trajectory.model_dump(mode="json")
    if payload["harness_spec_hash"] is None:
        del payload["harness_spec_hash"]
    if payload["model_version"] is None:
        del payload["model_version"]
    for event in payload["events"]:
        if event["error_code"] is None:
            del event["error_code"]
    encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(encoded).hexdigest()


def verify_hash(envelope: TraceEnvelope) -> bool:
    return trajectory_hash(envelope.trajectory) == envelope.content_hash


def replay(envelope: TraceEnvelope, *, verify_content_hash: bool = True) -> Trajectory:
    """Validate recorded transitions without invoking any model, tool or verifier."""
    if verify_content_hash and not verify_hash(envelope):
        raise TraceReplayError("trace content hash mismatch")
    trajectory = envelope.trajectory
    previous_ms = -1
    usage = Usage()
    final_answer: str | None = None
    for expected_sequence, event in enumerate(trajectory.events):
        if event.sequence != expected_sequence:
            raise TraceReplayError("event sequence mismatch")
        if event.monotonic_ms < previous_ms:
            raise TraceReplayError("event time is not monotonic")
        previous_ms = event.monotonic_ms
        if event.kind is EventKind.MODEL_OUTPUT:
            tokens = event.payload.get("generated_tokens", 0)
            if not isinstance(tokens, int) or tokens < 0:
                raise TraceReplayError("invalid generated token count")
            usage = usage.model_copy(update={"generated_tokens": usage.generated_tokens + tokens})
        elif event.kind is EventKind.TOOL_CALL:
            usage = usage.model_copy(
                update={"steps": usage.steps + 1, "tool_calls": usage.tool_calls + 1}
            )
        elif event.kind is EventKind.INVALID_ACTION:
            usage = usage.model_copy(
                update={"steps": usage.steps + 1, "invalid_actions": usage.invalid_actions + 1}
            )
        elif event.kind is EventKind.FINAL:
            answer = event.payload.get("answer")
            if not isinstance(answer, str) or not answer:
                raise TraceReplayError("invalid final answer")
            final_answer = answer
            usage = usage.model_copy(update={"steps": usage.steps + 1})
        elif event.kind is EventKind.TOOL_RESULT and event.payload.get("name") == "python":
            result = event.payload.get("result")
            if not isinstance(result, dict):
                raise TraceReplayError("invalid python tool result")
            metadata = result.get("metadata")
            seconds = metadata.get("execution_time", 0) if isinstance(metadata, dict) else 0
            if not isinstance(seconds, int | float) or seconds < 0:
                raise TraceReplayError("invalid Python execution time")
            usage = usage.model_copy(update={"python_seconds": usage.python_seconds + seconds})
    if usage != trajectory.usage:
        raise TraceReplayError("usage counters do not match events")
    if trajectory.termination_reason is TerminationReason.FINAL and final_answer != trajectory.final_answer:
        raise TraceReplayError("final answer does not match final event")
    return trajectory
