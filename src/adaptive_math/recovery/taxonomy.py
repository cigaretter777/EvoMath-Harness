"""Error taxonomy for the feedback-recovery diagnostic (report section 6).

The classes are mutually exclusive at the level of a single event, which is
what the pool rules are stated against. E_MATH is a *trajectory* label: a
model can compute correctly and still answer wrongly, so it cannot be read off
one tool result.
"""

from collections.abc import Mapping
from enum import StrEnum

from adaptive_math.agent.state import EventKind, StepErrorCode
from adaptive_math.core.types import JSONValue


class ErrorClass(StrEnum):
    E_PARSE = "E_PARSE"
    E_EXEC = "E_EXEC"
    E_TIMEOUT = "E_TIMEOUT"
    E_INFRA = "E_INFRA"
    E_MATH = "E_MATH"


class StepVerdict(StrEnum):
    OK = "ok"
    FAILED = "failed"
    OUT_OF_BUDGET = "out_of_budget"


# Tool-level codes and where they land. ``invalid_arguments`` and
# ``unknown_tool`` are model-side mistakes in *calling* the tool, not executed
# code, so they are not E_EXEC: the report reserves that class for code that
# reached the interpreter. They are counted separately as call-form errors.
_CALL_FORM_CODES = {"invalid_arguments", "unknown_tool"}
_INFRA_CODES = {"unavailable"}
_EXEC_CODES = {"execution_error", "output_limit"}


def _failed_result(event_payload: Mapping[str, JSONValue]) -> Mapping[str, JSONValue] | None:
    """The failed ToolResult inside a ``tool_result`` payload, else None."""
    result = event_payload.get("result")
    if not isinstance(result, Mapping):
        raise TypeError("tool_result payload has no result object")
    return None if result.get("ok") is True else result


def classify_tool_result(event_payload: Mapping[str, JSONValue]) -> ErrorClass | None:
    """Error class of one ``tool_result`` event, or None when it succeeded.

    The payload is the stored ``{"name":..., "result": ToolResult-as-json}``
    dict, which is what both the live runner and the saved trajectories hold.
    Call-form rejections return None here on purpose; :func:`is_call_form_error`
    reports them separately."""
    result = _failed_result(event_payload)
    if result is None:
        return None
    code = result.get("error_code")
    if code == "timeout":
        return ErrorClass.E_TIMEOUT
    if code in _INFRA_CODES:
        return ErrorClass.E_INFRA
    if code in _EXEC_CODES:
        return ErrorClass.E_EXEC
    if code in _CALL_FORM_CODES:
        return None
    # A failure with no code can only come from a path that forgot to set one;
    # counting it as executed code would inflate the pool, so it stays out.
    return None


def is_call_form_error(event_payload: Mapping[str, JSONValue]) -> bool:
    """True for a rejected call (bad arguments schema, unknown tool): the model
    wrote a malformed call, which is neither executed code nor infrastructure."""
    result = _failed_result(event_payload)
    return result is not None and result.get("error_code") in _CALL_FORM_CODES


def has_executed_failure(event_payload: Mapping[str, JSONValue]) -> bool:
    """E_EXEC and nothing else: the harness's own statement that the code
    reached the interpreter and raised. This is the class the continuation
    pool is built on."""
    return classify_tool_result(event_payload) is ErrorClass.E_EXEC


def step_error_class(
    kind: EventKind, error_code: StepErrorCode | None
) -> ErrorClass | None:
    """Map a *step-level* rejection (parse errors, budget rejections).

    Budget rejections are not errors of the model's code and are deliberately
    not classified as E_EXEC; the report counts them as out-of-budget steps."""
    if kind is not EventKind.INVALID_ACTION:
        return None
    if error_code is StepErrorCode.ACTION_PARSE_ERROR:
        return ErrorClass.E_PARSE
    return None
