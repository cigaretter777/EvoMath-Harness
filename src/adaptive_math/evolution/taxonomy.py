"""Deterministic machine labels for failed trajectories.

``label_trajectory`` is a pure function: it reads a recorded ``Trajectory``
plus correctness signals produced by the private verification boundary and
returns a non-exclusive set of ``FailureLabel`` values in enum definition
order. Labels are facts about observable signals (event error codes,
termination reason, verifier status); modes that require judgement only get
deterministic proxies here — human review labels live in separate annotation
fields, per the evaluation plan.

Attribution direction follows the v2.0 alignment design: ``WRONG_REASONING``
is the only model-capability label (the CAPABILITY_FAILURE equivalent); it is
the fallback when a trajectory failed with no harness-side signal, and such
failures go to the RL hard-example pool, never to the harness patch queue.
"""

from enum import StrEnum

import orjson

from adaptive_math.agent.state import EventKind, StepErrorCode, TerminationReason
from adaptive_math.agent.trace import Trajectory
from adaptive_math.tools.base import ToolErrorCode
from adaptive_math.verifier.service import VerifierStatus


class FailureLabel(StrEnum):
    FORMAT_INVALID = "format_invalid"
    NO_FINAL = "no_final"
    WRONG_REASONING = "wrong_reasoning"
    WRONG_TOOL_CHOICE = "wrong_tool_choice"
    TOOL_CODE_ERROR = "tool_code_error"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_RESULT_IGNORED = "tool_result_ignored"
    PREMATURE_STOP = "premature_stop"
    BUDGET_EXHAUSTED = "budget_exhausted"
    VERIFIER_UNSUPPORTED = "verifier_unsupported"
    VERIFIER_TIMEOUT = "verifier_timeout"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    LOOP_OR_REDUNDANCY = "loop_or_redundancy"


class AttributionDirection(StrEnum):
    HARNESS = "harness"
    MODEL = "model"
    AMBIGUOUS = "ambiguous"


ATTRIBUTION_DIRECTION: dict[FailureLabel, AttributionDirection] = {
    FailureLabel.FORMAT_INVALID: AttributionDirection.HARNESS,
    FailureLabel.NO_FINAL: AttributionDirection.AMBIGUOUS,
    FailureLabel.WRONG_REASONING: AttributionDirection.MODEL,
    FailureLabel.WRONG_TOOL_CHOICE: AttributionDirection.HARNESS,
    FailureLabel.TOOL_CODE_ERROR: AttributionDirection.AMBIGUOUS,
    FailureLabel.TOOL_TIMEOUT: AttributionDirection.HARNESS,
    FailureLabel.TOOL_RESULT_IGNORED: AttributionDirection.HARNESS,
    FailureLabel.PREMATURE_STOP: AttributionDirection.HARNESS,
    FailureLabel.BUDGET_EXHAUSTED: AttributionDirection.HARNESS,
    FailureLabel.VERIFIER_UNSUPPORTED: AttributionDirection.AMBIGUOUS,
    FailureLabel.VERIFIER_TIMEOUT: AttributionDirection.HARNESS,
    FailureLabel.INFRASTRUCTURE_FAILURE: AttributionDirection.AMBIGUOUS,
    FailureLabel.LOOP_OR_REDUNDANCY: AttributionDirection.HARNESS,
}

_TOOL_ERROR_LABEL: dict[ToolErrorCode, FailureLabel] = {
    ToolErrorCode.TIMEOUT: FailureLabel.TOOL_TIMEOUT,
    ToolErrorCode.EXECUTION_ERROR: FailureLabel.TOOL_CODE_ERROR,
    ToolErrorCode.OUTPUT_LIMIT: FailureLabel.TOOL_CODE_ERROR,
    ToolErrorCode.INVALID_ARGUMENTS: FailureLabel.FORMAT_INVALID,
    ToolErrorCode.UNKNOWN_TOOL: FailureLabel.WRONG_TOOL_CHOICE,
    ToolErrorCode.UNAVAILABLE: FailureLabel.INFRASTRUCTURE_FAILURE,
}

_BUDGET_TERMINATIONS = frozenset(
    {
        TerminationReason.MAX_STEPS,
        TerminationReason.MAX_TOOL_CALLS,
        TerminationReason.PYTHON_TIME_BUDGET,
    }
)

_BUDGET_STEP_ERRORS = frozenset(
    {
        StepErrorCode.TOOL_CALL_BUDGET_EXHAUSTED,
        StepErrorCode.PYTHON_TIME_BUDGET_EXHAUSTED,
    }
)


def label_trajectory(
    trajectory: Trajectory,
    *,
    correct: bool | None,
    verifier_status: VerifierStatus | None = None,
    reference_answer: str | None = None,
) -> tuple[FailureLabel, ...]:
    """Label observable failure modes; empty for successful trajectories.

    ``correct=None`` means the verification boundary produced no verdict for
    this task: event- and termination-derived rules still fire, while every
    correctness-dependent rule (PREMATURE_STOP, TOOL_RESULT_IGNORED and the
    WRONG_REASONING fallback) is skipped.
    """
    if correct is True:
        return ()
    labels: set[FailureLabel] = set()

    if verifier_status is VerifierStatus.TIMEOUT:
        labels.add(FailureLabel.VERIFIER_TIMEOUT)
    elif verifier_status is VerifierStatus.INVALID_REFERENCE:
        labels.add(FailureLabel.VERIFIER_UNSUPPORTED)
    elif verifier_status is VerifierStatus.INTERNAL_ERROR:
        labels.add(FailureLabel.INFRASTRUCTURE_FAILURE)
    elif verifier_status is VerifierStatus.INVALID_PREDICTION:
        labels.add(FailureLabel.FORMAT_INVALID)

    if trajectory.termination_reason is TerminationReason.INFRASTRUCTURE_ERROR:
        labels.add(FailureLabel.INFRASTRUCTURE_FAILURE)
    if trajectory.termination_reason in _BUDGET_TERMINATIONS:
        labels.add(FailureLabel.BUDGET_EXHAUSTED)
    if trajectory.final_answer is None:
        labels.add(FailureLabel.NO_FINAL)

    last_call_signature: bytes | None = None
    last_tool_result_ok: bool | None = None
    successful_outputs: list[str] = []
    for event in trajectory.events:
        if event.kind is EventKind.INVALID_ACTION:
            if event.error_code is StepErrorCode.ACTION_PARSE_ERROR:
                labels.add(FailureLabel.FORMAT_INVALID)
            elif event.error_code in _BUDGET_STEP_ERRORS:
                labels.add(FailureLabel.BUDGET_EXHAUSTED)
        elif event.kind is EventKind.TOOL_CALL:
            signature = orjson.dumps(
                [event.payload.get("name"), event.payload.get("arguments")],
                option=orjson.OPT_SORT_KEYS,
            )
            if signature == last_call_signature:
                labels.add(FailureLabel.LOOP_OR_REDUNDANCY)
            last_call_signature = signature
        elif event.kind is EventKind.TOOL_RESULT:
            result = event.payload.get("result")
            if not isinstance(result, dict):
                continue
            code = result.get("error_code")
            if isinstance(code, str):
                try:
                    labels.add(_TOOL_ERROR_LABEL[ToolErrorCode(code)])
                except ValueError:
                    pass  # unknown code strings from older runtimes carry no label
            last_tool_result_ok = result.get("ok") is True
            output = result.get("output")
            if last_tool_result_ok and isinstance(output, str):
                successful_outputs.append(output)

    if (
        correct is False
        and trajectory.termination_reason is TerminationReason.FINAL
        and last_tool_result_ok is False
    ):
        labels.add(FailureLabel.PREMATURE_STOP)

    if correct is False and reference_answer and trajectory.final_answer is not None:
        reference = reference_answer.strip()
        if (
            reference
            and any(reference in output for output in successful_outputs)
            and reference not in trajectory.final_answer
        ):
            labels.add(FailureLabel.TOOL_RESULT_IGNORED)

    if correct is False and not labels:
        labels.add(FailureLabel.WRONG_REASONING)

    return tuple(label for label in FailureLabel if label in labels)
