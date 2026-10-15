"""Error taxonomy (report section 6) against payloads the harness really writes."""

import pytest

from adaptive_math.agent.state import EventKind, StepErrorCode
from adaptive_math.recovery.taxonomy import (
    ErrorClass,
    classify_tool_result,
    has_executed_failure,
    is_call_form_error,
    step_error_class,
)


def _payload(*, ok: bool, error_code: str | None, output: str = "") -> dict[str, object]:
    return {
        "name": "python",
        "result": {
            "ok": ok,
            "output": output,
            "error_code": error_code,
            "latency_ms": 3,
            "truncated": False,
            "metadata": {},
        },
    }


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        ("execution_error", ErrorClass.E_EXEC),
        ("output_limit", ErrorClass.E_EXEC),
        ("timeout", ErrorClass.E_TIMEOUT),
        ("unavailable", ErrorClass.E_INFRA),
    ],
)
def test_failed_results_map_to_their_class(error_code: str, expected: ErrorClass) -> None:
    assert classify_tool_result(_payload(ok=False, error_code=error_code)) is expected


def test_a_successful_call_has_no_error_class() -> None:
    assert classify_tool_result(_payload(ok=True, error_code=None, output="42")) is None


def test_a_malformed_call_is_not_an_execution_error() -> None:
    """The code never ran, so the anchor pool must not accept it; it is counted
    separately as a call-form error rather than folded into E_EXEC."""
    payload = _payload(ok=False, error_code="invalid_arguments")

    assert classify_tool_result(payload) is None
    assert is_call_form_error(payload) is True
    assert has_executed_failure(payload) is False


def test_a_coded_success_is_never_an_error() -> None:
    assert has_executed_failure(_payload(ok=True, error_code=None, output="42")) is False


def test_only_execution_errors_qualify_for_the_pool() -> None:
    assert has_executed_failure(_payload(ok=False, error_code="execution_error")) is True
    assert has_executed_failure(_payload(ok=False, error_code="timeout")) is False
    assert has_executed_failure(_payload(ok=False, error_code="unavailable")) is False


def test_a_tool_result_without_a_result_object_is_a_contract_violation() -> None:
    with pytest.raises(TypeError, match="no result object"):
        classify_tool_result({"name": "python"})


def test_step_level_parse_errors_are_classified_and_budget_rejections_are_not() -> None:
    assert (
        step_error_class(EventKind.INVALID_ACTION, StepErrorCode.ACTION_PARSE_ERROR)
        is ErrorClass.E_PARSE
    )
    assert (
        step_error_class(EventKind.INVALID_ACTION, StepErrorCode.TOOL_CALL_BUDGET_EXHAUSTED)
        is None
    )
    assert step_error_class(EventKind.TOOL_RESULT, None) is None
