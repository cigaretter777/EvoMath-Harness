import pytest

from adaptive_math.agent.state import EventKind, TerminationReason, TraceEvent, Usage
from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.types import Budget
from adaptive_math.reward import R2_DEFAULT, R4_DEFAULT, RewardContext, compute_reward
from adaptive_math.training.reward_bridge import reward_for_trajectory
from adaptive_math.verifier.service import VerifierResult, VerifierStatus


def test_reward_bridge_matches_production_reward_function() -> None:
    trajectory = Trajectory(trace_id="t", task_id="task", events=(), final_answer="2", termination_reason=TerminationReason.FINAL, usage=Usage(steps=1, tool_calls=1, generated_tokens=10), runtime_version="v1")
    verdict = VerifierResult(status=VerifierStatus.CORRECT, reward=1.0, normalized_prediction="2", normalized_reference="2")
    budget = Budget(max_steps=2, max_tool_calls=4, max_python_seconds=12, max_observation_chars=100)

    actual = reward_for_trajectory(trajectory, verdict, budget, max_generated_tokens=100, config=R2_DEFAULT)
    expected = compute_reward(RewardContext(verifier_result=verdict, tool_calls=1, tool_successes=0, python_seconds=0, invalid_action_count=0, generated_tokens=10, max_generated_tokens=100, budget=budget), R2_DEFAULT)

    assert actual == expected


def _tool_result(sequence: int, *, ok: bool) -> TraceEvent:
    return TraceEvent(
        sequence=sequence,
        kind=EventKind.TOOL_RESULT,
        monotonic_ms=sequence,
        payload={"name": "python", "result": {"ok": ok, "output": "2" if ok else ""}},
    )


def test_bridge_counts_only_successful_tool_executions_for_the_bonus() -> None:
    events = (
        TraceEvent(
            sequence=0,
            kind=EventKind.TOOL_CALL,
            monotonic_ms=0,
            payload={"name": "python", "arguments": {"code": "1+1"}},
        ),
        _tool_result(1, ok=True),
        TraceEvent(
            sequence=2,
            kind=EventKind.TOOL_CALL,
            monotonic_ms=2,
            payload={"name": "python", "arguments": {"code": "boom"}},
        ),
        _tool_result(3, ok=False),
    )
    trajectory = Trajectory(
        trace_id="t",
        task_id="task",
        events=events,
        final_answer="2",
        termination_reason=TerminationReason.FINAL,
        usage=Usage(steps=2, tool_calls=2, generated_tokens=10),
        runtime_version="v1",
    )
    verdict = VerifierResult(
        status=VerifierStatus.INCORRECT, reward=0.0, normalized_prediction=None, normalized_reference="2"
    )
    budget = Budget(max_steps=4, max_tool_calls=4, max_python_seconds=12, max_observation_chars=100)

    breakdown = reward_for_trajectory(
        trajectory, verdict, budget, max_generated_tokens=100, config=R4_DEFAULT
    )

    # One ok result (not two calls, not the failed one) pays the flat bonus,
    # and it pays despite the wrong final answer.
    assert breakdown.components["tool_success_bonus"] == pytest.approx(0.10)
    assert breakdown.total == pytest.approx(0.10)
