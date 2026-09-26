"""Budgeted product and offline math environments with a hidden-label boundary."""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.agent.actions import FinalAction, ToolAction
from adaptive_math.agent.state import AgentState, EventKind, StepErrorCode, TerminationReason
from adaptive_math.core.types import Budget, LabeledMathTask, MathTask
from adaptive_math.tools.base import ToolContext
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.verifier.hidden import HiddenVerifier
from adaptive_math.verifier.service import VerifierResult


class Observation(BaseModel):
    """Public model-visible result of one environment transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["tool_result", "action_error", "budget_warning"]
    content: str
    remaining_steps: int = Field(ge=0)
    remaining_tool_calls: int = Field(ge=0)
    remaining_python_seconds: float = Field(ge=0)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: AgentState
    observation: Observation | None
    terminated: bool


class ProductMathEnv:
    """Serving environment: its constructor deliberately has no label input."""

    def __init__(self, task: MathTask, budget: Budget, registry: ToolRegistry, trace_id: str = "runtime") -> None:
        self._registry = registry
        self._trace_id = trace_id
        self._state = AgentState(task=task, budget=budget)

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def registry(self) -> ToolRegistry:
        """Live public tool schemas used for prompt rendering."""
        return self._registry

    def record_model_output(self, raw: str, generated_tokens: int, monotonic_ms: int) -> AgentState:
        if self._state.termination_reason is not None:
            raise ValueError("cannot record output for a terminated environment")
        self._state = self._state.append_event(
            EventKind.MODEL_OUTPUT, {"raw": raw, "generated_tokens": generated_tokens}, monotonic_ms
        ).with_usage(generated_tokens=generated_tokens)
        return self._state

    def abort(self, reason: TerminationReason) -> AgentState:
        if self._state.termination_reason is None:
            self._state = self._state.terminate(reason)
        return self._state

    async def step(self, action: ToolAction | FinalAction | None, monotonic_ms: int) -> StepResult:
        if self._state.termination_reason is not None:
            raise ValueError("cannot step a terminated environment")
        if isinstance(action, FinalAction):
            state = self._state.append_event(EventKind.FINAL, {"answer": action.answer}, monotonic_ms)
            state = state.with_usage(steps=1).terminate(TerminationReason.FINAL, action.answer)
            return self._store(state, None)
        if not isinstance(action, ToolAction):
            return self._invalid(
                "Use exactly one <tool_call> or <final> action.",
                monotonic_ms,
                error_code=StepErrorCode.ACTION_PARSE_ERROR,
            )
        call = action.call
        if self._state.usage.tool_calls >= self._state.budget.max_tool_calls:
            return self._invalid(
                "Tool-call budget exhausted; submit a final answer.",
                monotonic_ms,
                error_code=StepErrorCode.TOOL_CALL_BUDGET_EXHAUSTED,
            )
        if call.name == "python" and self._state.usage.python_seconds >= self._state.budget.max_python_seconds:
            return self._invalid(
                "Python-time budget exhausted; use another tool or submit a final answer.",
                monotonic_ms,
                error_code=StepErrorCode.PYTHON_TIME_BUDGET_EXHAUSTED,
            )

        state = self._state.append_event(
            EventKind.TOOL_CALL,
            {"name": call.name, "arguments": call.arguments},
            monotonic_ms,
        ).with_usage(steps=1, tool_calls=1)
        context = ToolContext(
            trace_id=self._trace_id,
            task_id=state.task.task_id,
            remaining_observation_chars=state.budget.max_observation_chars,
            remaining_python_seconds=max(
                0.0, state.budget.max_python_seconds - state.usage.python_seconds
            ),
        )
        result = await self._registry.execute(call.name, call.arguments, context)
        python_seconds = _python_seconds(call.name, result.metadata)
        state = state.with_usage(python_seconds=python_seconds).append_event(
            EventKind.TOOL_RESULT,
            {"name": call.name, "result": result.model_dump()},
            monotonic_ms,
        )
        observation = self._observation("tool_result", result.output, state)
        state = self._terminate_if_exhausted(state)
        return self._store(state, observation)

    def _invalid(
        self, content: str, monotonic_ms: int, *, error_code: StepErrorCode
    ) -> StepResult:
        state = self._state.append_event(
            EventKind.INVALID_ACTION, {"message": content}, monotonic_ms, error_code=error_code
        )
        state = state.with_usage(steps=1, invalid_actions=1)
        observation = self._observation("action_error", content, state)
        return self._store(self._terminate_if_exhausted(state), observation)

    @staticmethod
    def _observation(kind: Literal["tool_result", "action_error", "budget_warning"], content: str, state: AgentState) -> Observation:
        return Observation(
            kind=kind,
            content=content,
            remaining_steps=max(0, state.budget.max_steps - state.usage.steps),
            remaining_tool_calls=max(0, state.budget.max_tool_calls - state.usage.tool_calls),
            remaining_python_seconds=max(0.0, state.budget.max_python_seconds - state.usage.python_seconds),
        )

    def _terminate_if_exhausted(self, state: AgentState) -> AgentState:
        reason = state.budget_reason()
        return state.terminate(reason) if reason is not None else state

    def _store(self, state: AgentState, observation: Observation | None) -> StepResult:
        self._state = state
        return StepResult(state=state, observation=observation, terminated=state.termination_reason is not None)


class OfflineMathEnv(ProductMathEnv):
    """Training environment whose label is inaccessible to the public parent API."""

    def __init__(self, labeled_task: LabeledMathTask, budget: Budget, registry: ToolRegistry, trace_id: str = "runtime") -> None:
        super().__init__(labeled_task.public_view(), budget, registry, trace_id)
        self.__hidden_verifier = HiddenVerifier(labeled_task.reference, labeled_task.task.task_id)

    def evaluate(self) -> VerifierResult | None:
        """One verdict for every terminated rollout, none for an in-flight one.

        A rollout that ran out of budget without an answer is an invalid
        prediction, not a missing verdict: returning None here used to route the
        caller around reward_for_trajectory, so exhausting the budget paid more
        than terminating with a penalised wrong answer.
        """
        if self.state.termination_reason is None:
            return None
        if self.state.final_answer is None:
            return self.__hidden_verifier.no_final_answer()
        return self.__hidden_verifier.evaluate(self.state.final_answer)


def _python_seconds(tool_name: str, metadata: Mapping[str, object]) -> float:
    value = metadata.get("execution_time") if tool_name == "python" else None
    return float(value) if isinstance(value, int | float) and value >= 0 else 0.0
