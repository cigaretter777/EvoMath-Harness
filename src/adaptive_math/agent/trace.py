"""Portable deterministic trajectory contract shared by runtime and replay."""

from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.agent.state import TerminationReason, TraceEvent, Usage


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    task_id: str
    events: tuple[TraceEvent, ...]
    final_answer: str | None
    termination_reason: TerminationReason
    usage: Usage
    runtime_version: str
    # Content address of the HarnessSpec that produced this trajectory; None
    # for pre-harness (legacy two-part runtime_version) trajectories. Excluded
    # from the canonical hash when None so legacy content hashes never change.
    harness_spec_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Checkpoint hash or training run id of the policy model; None for legacy
    # trajectories. Excluded from the canonical hash when None, same rule as
    # harness_spec_hash, so pre-existing content hashes never change.
    model_version: str | None = Field(default=None, min_length=1, max_length=128)
