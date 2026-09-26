"""Failure taxonomy：确定性机器标签的逐规则锚定测试。"""

import pytest

from adaptive_math.agent.state import (
    EventKind,
    StepErrorCode,
    TerminationReason,
    TraceEvent,
    Usage,
)
from adaptive_math.agent.trace import Trajectory
from adaptive_math.evolution.taxonomy import (
    ATTRIBUTION_DIRECTION,
    AttributionDirection,
    FailureLabel,
    label_trajectory,
)
from adaptive_math.verifier.service import VerifierStatus


def _event(
    sequence: int,
    kind: EventKind,
    payload: dict[str, object] | None = None,
    *,
    error_code: StepErrorCode | None = None,
) -> TraceEvent:
    return TraceEvent(
        sequence=sequence,
        kind=kind,
        monotonic_ms=sequence,
        payload=payload or {},  # type: ignore[arg-type]
        error_code=error_code,
    )


def _tool_call(sequence: int, name: str = "python", arguments: dict[str, object] | None = None) -> TraceEvent:
    return _event(sequence, EventKind.TOOL_CALL, {"name": name, "arguments": arguments or {}})


def _tool_result(
    sequence: int,
    *,
    ok: bool = True,
    output: str = "observed",
    error_code: str | None = None,
) -> TraceEvent:
    result = {
        "ok": ok,
        "output": output,
        "error_code": error_code,
        "latency_ms": 1,
        "truncated": False,
        "metadata": {},
    }
    return _event(sequence, EventKind.TOOL_RESULT, {"name": "python", "result": result})


def _final(sequence: int, answer: str = "42") -> TraceEvent:
    return _event(sequence, EventKind.FINAL, {"answer": answer})


def _trajectory(
    events: list[TraceEvent],
    *,
    final_answer: str | None = "42",
    termination: TerminationReason = TerminationReason.FINAL,
) -> Trajectory:
    return Trajectory(
        trace_id="tax-1",
        task_id="unit:tax",
        events=tuple(events),
        final_answer=final_answer,
        termination_reason=termination,
        usage=Usage(steps=1),
        runtime_version="runtime-v1+agent-v1",
    )


@pytest.mark.parametrize(
    ("tool_error", "expected"),
    [
        ("timeout", FailureLabel.TOOL_TIMEOUT),
        ("execution_error", FailureLabel.TOOL_CODE_ERROR),
        ("output_limit", FailureLabel.TOOL_CODE_ERROR),
        ("invalid_arguments", FailureLabel.FORMAT_INVALID),
        ("unknown_tool", FailureLabel.WRONG_TOOL_CHOICE),
        ("unavailable", FailureLabel.INFRASTRUCTURE_FAILURE),
    ],
)
def test_tool_result_error_codes_map_to_labels(tool_error: str, expected: FailureLabel) -> None:
    trajectory = _trajectory(
        [
            _tool_call(0),
            _tool_result(1, ok=False, error_code=tool_error),
            _tool_call(1 + 1, arguments={"x": 1}),
            _tool_result(3, ok=True),
            _final(4),
        ]
    )

    assert label_trajectory(trajectory, correct=False) == (expected,)


def test_action_parse_error_maps_to_format_invalid() -> None:
    trajectory = _trajectory(
        [_event(0, EventKind.INVALID_ACTION, {"message": "bad"}, error_code=StepErrorCode.ACTION_PARSE_ERROR), _final(1)]
    )

    assert label_trajectory(trajectory, correct=False) == (FailureLabel.FORMAT_INVALID,)


@pytest.mark.parametrize(
    "step_error",
    [StepErrorCode.TOOL_CALL_BUDGET_EXHAUSTED, StepErrorCode.PYTHON_TIME_BUDGET_EXHAUSTED],
)
def test_step_budget_error_codes_map_to_budget_exhausted(step_error: StepErrorCode) -> None:
    trajectory = _trajectory(
        [_event(0, EventKind.INVALID_ACTION, {"message": "budget"}, error_code=step_error), _final(1)]
    )

    assert label_trajectory(trajectory, correct=False) == (FailureLabel.BUDGET_EXHAUSTED,)


def test_infrastructure_termination_maps_to_infrastructure_failure() -> None:
    trajectory = _trajectory([_final(0)], termination=TerminationReason.INFRASTRUCTURE_ERROR)

    assert label_trajectory(trajectory, correct=False) == (FailureLabel.INFRASTRUCTURE_FAILURE,)


@pytest.mark.parametrize(
    "termination",
    [
        TerminationReason.MAX_STEPS,
        TerminationReason.MAX_TOOL_CALLS,
        TerminationReason.PYTHON_TIME_BUDGET,
    ],
)
def test_budget_terminations_map_to_budget_exhausted(termination: TerminationReason) -> None:
    trajectory = _trajectory([_final(0)], termination=termination)

    assert label_trajectory(trajectory, correct=False) == (FailureLabel.BUDGET_EXHAUSTED,)


def test_missing_final_answer_maps_to_no_final() -> None:
    trajectory = _trajectory(
        [_event(0, EventKind.MODEL_OUTPUT, {"raw": "thinking", "generated_tokens": 1})],
        final_answer=None,
        termination=TerminationReason.MODEL_ERROR,
    )

    assert label_trajectory(trajectory, correct=False) == (FailureLabel.NO_FINAL,)


def test_repeated_identical_tool_call_maps_to_loop_or_redundancy() -> None:
    trajectory = _trajectory(
        [
            _tool_call(0, arguments={"a": 1}),
            _tool_result(1),
            _tool_call(2, arguments={"a": 1}),
            _tool_result(3),
            _final(4),
        ]
    )

    assert label_trajectory(trajectory, correct=False) == (FailureLabel.LOOP_OR_REDUNDANCY,)


def test_non_consecutive_or_distinct_tool_calls_do_not_loop() -> None:
    trajectory = _trajectory(
        [
            _tool_call(0, arguments={"a": 1}),
            _tool_result(1),
            _tool_call(2, arguments={"a": 2}),
            _tool_result(3),
            _tool_call(4, arguments={"a": 1}),
            _tool_result(5),
            _final(6),
        ]
    )

    assert label_trajectory(trajectory, correct=False) == (FailureLabel.WRONG_REASONING,)


def test_giving_up_after_failed_tool_result_maps_to_premature_stop() -> None:
    trajectory = _trajectory(
        [_tool_call(0), _tool_result(1, ok=False), _final(2, answer="7")],
        final_answer="7",
    )

    assert label_trajectory(trajectory, correct=False) == (FailureLabel.PREMATURE_STOP,)


def test_tool_result_ignored_when_tool_had_reference_and_final_differs() -> None:
    trajectory = _trajectory(
        [_tool_call(0), _tool_result(1, ok=True, output="computed value: 42"), _final(2, answer="7")],
        final_answer="7",
    )

    assert label_trajectory(trajectory, correct=False, reference_answer="42") == (
        FailureLabel.TOOL_RESULT_IGNORED,
    )


def test_tool_result_not_ignored_when_final_matches_reference() -> None:
    trajectory = _trajectory(
        [_tool_call(0), _tool_result(1, ok=True, output="computed value: 42"), _final(2, answer="42")]
    )

    assert label_trajectory(trajectory, correct=False, reference_answer="42") == (
        FailureLabel.WRONG_REASONING,
    )


def test_tool_result_ignored_requires_reference_answer() -> None:
    trajectory = _trajectory(
        [_tool_call(0), _tool_result(1, ok=True, output="computed value: 42"), _final(2, answer="7")],
        final_answer="7",
    )

    assert label_trajectory(trajectory, correct=False) == (FailureLabel.WRONG_REASONING,)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (VerifierStatus.TIMEOUT, FailureLabel.VERIFIER_TIMEOUT),
        (VerifierStatus.INVALID_REFERENCE, FailureLabel.VERIFIER_UNSUPPORTED),
        (VerifierStatus.INTERNAL_ERROR, FailureLabel.INFRASTRUCTURE_FAILURE),
        (VerifierStatus.INVALID_PREDICTION, FailureLabel.FORMAT_INVALID),
    ],
)
def test_verifier_status_maps_to_labels(status: VerifierStatus, expected: FailureLabel) -> None:
    trajectory = _trajectory([_final(0)])

    assert label_trajectory(trajectory, correct=False, verifier_status=status) == (expected,)


def test_clean_incorrect_trajectory_falls_back_to_wrong_reasoning() -> None:
    trajectory = _trajectory([_final(0, answer="7")])

    assert label_trajectory(trajectory, correct=False, verifier_status=VerifierStatus.INCORRECT) == (
        FailureLabel.WRONG_REASONING,
    )


def test_correct_trajectory_has_no_failure_labels() -> None:
    trajectory = _trajectory(
        [_tool_call(0), _tool_result(1, ok=False, error_code="timeout"), _tool_call(2), _tool_result(3), _final(4)]
    )

    assert label_trajectory(trajectory, correct=True) == ()


def test_unknown_correctness_skips_correctness_dependent_rules() -> None:
    trajectory = _trajectory(
        [_tool_call(0), _tool_result(1, ok=False), _final(2, answer="7")],
        final_answer="7",
    )

    assert label_trajectory(trajectory, correct=None) == ()


def test_unknown_correctness_still_fires_event_and_termination_rules() -> None:
    trajectory = _trajectory(
        [_event(0, EventKind.MODEL_OUTPUT, {"raw": "x", "generated_tokens": 1})],
        final_answer=None,
        termination=TerminationReason.MAX_STEPS,
    )

    assert label_trajectory(trajectory, correct=None) == (
        FailureLabel.NO_FINAL,
        FailureLabel.BUDGET_EXHAUSTED,
    )


def test_labels_are_returned_in_enum_definition_order() -> None:
    trajectory = _trajectory(
        [_tool_call(0), _tool_result(1, ok=False, error_code="timeout")],
        final_answer=None,
        termination=TerminationReason.MAX_STEPS,
    )

    labels = label_trajectory(trajectory, correct=False)

    assert labels == tuple(label for label in FailureLabel if label in labels)
    assert FailureLabel.NO_FINAL in labels
    assert FailureLabel.TOOL_TIMEOUT in labels
    assert FailureLabel.BUDGET_EXHAUSTED in labels


def test_attribution_direction_covers_every_label() -> None:
    assert set(ATTRIBUTION_DIRECTION) == set(FailureLabel)


def test_wrong_reasoning_is_the_only_model_attributed_label() -> None:
    model_labels = [
        label for label, direction in ATTRIBUTION_DIRECTION.items() if direction is AttributionDirection.MODEL
    ]

    assert model_labels == [FailureLabel.WRONG_REASONING]
