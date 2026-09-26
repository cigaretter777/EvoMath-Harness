"""Immutable runtime state, event, usage and termination contracts."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.core.types import Budget, JSONValue, MathTask


class TerminationReason(StrEnum):
    FINAL = "final"
    MAX_STEPS = "max_steps"
    MAX_TOOL_CALLS = "max_tool_calls"
    PYTHON_TIME_BUDGET = "python_time_budget"
    MODEL_ERROR = "model_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    CANCELLED = "cancelled"


class EventKind(StrEnum):
    MODEL_OUTPUT = "model_output"
    INVALID_ACTION = "invalid_action"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL = "final"
    TERMINATION = "termination"


class StepErrorCode(StrEnum):
    """Machine-readable cause of a failed step; tool-level failures keep their
    own ``ToolErrorCode`` inside the TOOL_RESULT payload instead."""

    ACTION_PARSE_ERROR = "action_parse_error"
    TOOL_CALL_BUDGET_EXHAUSTED = "tool_call_budget_exhausted"
    PYTHON_TIME_BUDGET_EXHAUSTED = "python_time_budget_exhausted"


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    python_seconds: float = Field(default=0.0, ge=0)
    generated_tokens: int = Field(default=0, ge=0)
    invalid_actions: int = Field(default=0, ge=0)


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    kind: EventKind
    monotonic_ms: int = Field(ge=0)
    payload: dict[str, JSONValue] = Field(default_factory=dict)
    # Machine-readable failure cause; None for successful or legacy events.
    # Excluded from the canonical trajectory hash when None so pre-existing
    # content hashes never change.
    error_code: StepErrorCode | None = None


class AgentState(BaseModel):
    """Public state only; hidden references never enter this object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: MathTask
    budget: Budget
    usage: Usage = Field(default_factory=Usage)
    events: tuple[TraceEvent, ...] = ()
    final_answer: str | None = None
    termination_reason: TerminationReason | None = None

    def append_event(
        self,
        kind: EventKind,
        payload: dict[str, JSONValue],
        monotonic_ms: int,
        *,
        error_code: StepErrorCode | None = None,
    ) -> "AgentState":
        self._ensure_active()
        if self.events and monotonic_ms < self.events[-1].monotonic_ms:
            raise ValueError("event time must be monotonic")
        event = TraceEvent(
            sequence=len(self.events),
            kind=kind,
            monotonic_ms=monotonic_ms,
            payload=payload,
            error_code=error_code,
        )
        return self.model_copy(update={"events": (*self.events, event)})

    def with_usage(
        self,
        *,
        steps: int = 0,
        tool_calls: int = 0,
        python_seconds: float = 0.0,
        generated_tokens: int = 0,
        invalid_actions: int = 0,
    ) -> "AgentState":
        self._ensure_active()
        if any(value < 0 for value in (steps, tool_calls, python_seconds, generated_tokens, invalid_actions)):
            raise ValueError("usage deltas must be non-negative")
        usage = Usage(
            steps=self.usage.steps + steps,
            tool_calls=self.usage.tool_calls + tool_calls,
            python_seconds=self.usage.python_seconds + python_seconds,
            generated_tokens=self.usage.generated_tokens + generated_tokens,
            invalid_actions=self.usage.invalid_actions + invalid_actions,
        )
        return self.model_copy(update={"usage": usage})

    def is_budget_exhausted(self) -> bool:
        return self.budget_reason() is not None

    def budget_reason(self) -> TerminationReason | None:
        if self.usage.steps >= self.budget.max_steps:
            return TerminationReason.MAX_STEPS
        # A zero Python budget disables Python execution; it must not terminate
        # an otherwise valid direct/SymPy trajectory before Python is used.
        if self.usage.python_seconds > 0 and self.usage.python_seconds >= self.budget.max_python_seconds:
            return TerminationReason.PYTHON_TIME_BUDGET
        return None

    def terminate(self, reason: TerminationReason, final_answer: str | None = None) -> "AgentState":
        self._ensure_active()
        return self.model_copy(update={"termination_reason": reason, "final_answer": final_answer})

    def _ensure_active(self) -> None:
        if self.termination_reason is not None:
            raise ValueError("cannot transition a terminated agent state")
