"""Typed deterministic answer verification service.

``verify_answer`` is the only public entry point. It dispatches on
``ReferenceAnswer.answer_type``, never raises for bad predictions or broken
references, and sets reward to 1.0 only for CORRECT results. Symbolic
(EXPRESSION) comparisons run in an isolated worker process.
"""

import atexit
import hashlib
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.core.types import AnswerType, JSONValue, ReferenceAnswer
from adaptive_math.verifier.normalizer import normalize_surface
from adaptive_math.verifier.numeric import (
    VerifierConfig,
    canonical_form,
    interval_form,
    parse_integer,
    parse_interval,
    parse_number,
    parse_value,
    tolerance_fraction,
)
from adaptive_math.verifier.worker import SymbolicWorker


class VerifierStatus(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    INVALID_PREDICTION = "invalid_prediction"
    INVALID_REFERENCE = "invalid_reference"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


class VerifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: VerifierStatus
    reward: float
    normalized_prediction: str | None
    normalized_reference: str | None
    details: dict[str, JSONValue] = Field(default_factory=dict)


Verifier = Callable[[str, str, VerifierConfig, str], VerifierResult]

_DISPATCH: dict[AnswerType, Verifier] = {}

_expression_worker: SymbolicWorker | None = None


def verify_answer(
    prediction: str,
    reference: ReferenceAnswer,
    config: VerifierConfig | None = None,
    *,
    task_id: str | None = None,
) -> VerifierResult:
    """Verify a prediction against a reference and its acceptable forms.

    The primary reference value is always tried; an acceptable form can only
    upgrade the result to CORRECT. A broken reference is reported as
    INVALID_REFERENCE instead of being masked by a boolean False. task_id
    seeds the symbolic numeric cross-check; it never changes typed verdicts.
    """
    config = config or VerifierConfig()
    seed = task_id or hashlib.sha256(
        f"{prediction}\x00{reference.value}".encode()
    ).hexdigest()[:20]
    verify = _DISPATCH[reference.answer_type]
    best: VerifierResult | None = None
    for form in (reference.value, *reference.acceptable_forms):
        result = verify(prediction, form, config, seed)
        if result.status is VerifierStatus.CORRECT:
            return result
        if result.status is VerifierStatus.INVALID_REFERENCE:
            return result
        if best is None:
            best = result
    assert best is not None  # reference.value is always verified at least once
    return best


def _get_expression_worker() -> SymbolicWorker:
    global _expression_worker
    if _expression_worker is None:
        _expression_worker = SymbolicWorker()
        atexit.register(_expression_worker.close)
    return _expression_worker


def _register(answer_type: AnswerType) -> Callable[[Verifier], Verifier]:
    def decorate(verify: Verifier) -> Verifier:
        _DISPATCH[answer_type] = verify
        return verify

    return decorate


@_register(AnswerType.INTEGER)
def _verify_integer(prediction: str, reference_value: str, config: VerifierConfig, _task_id: str) -> VerifierResult:
    pred_norm = normalize_surface(prediction)
    ref_norm = normalize_surface(reference_value)
    r_ref = parse_integer(ref_norm, config)
    if not r_ref.ok or r_ref.value is None:
        return _invalid_reference(r_ref.error_code)
    r_pred = parse_integer(pred_norm, config)
    if not r_pred.ok or r_pred.value is None:
        if r_pred.error_code == "not_integer":
            return _incorrect(
                pred_norm,
                str(r_ref.value),
                f"prediction is well-formed but not an integer: {pred_norm!r}",
            )
        return _invalid_prediction(r_pred.error_code, str(r_ref.value))
    if r_pred.value == r_ref.value:
        return _correct(str(r_pred.value), str(r_ref.value))
    return _incorrect(str(r_pred.value), str(r_ref.value), "values differ")


@_register(AnswerType.RATIONAL)
def _verify_rational(prediction: str, reference_value: str, config: VerifierConfig, _task_id: str) -> VerifierResult:
    pred_norm = normalize_surface(prediction)
    ref_norm = normalize_surface(reference_value)
    r_ref = parse_number(ref_norm, config)
    if not r_ref.ok or r_ref.value is None:
        return _invalid_reference(r_ref.error_code)
    r_pred = parse_number(pred_norm, config)
    if not r_pred.ok or r_pred.value is None:
        return _invalid_prediction(r_pred.error_code, str(r_ref.value))
    if r_pred.value == r_ref.value:
        return _correct(str(r_pred.value), str(r_ref.value))
    return _incorrect(str(r_pred.value), str(r_ref.value), "values differ")


@_register(AnswerType.REAL)
def _verify_real(prediction: str, reference_value: str, config: VerifierConfig, _task_id: str) -> VerifierResult:
    pred_norm = normalize_surface(prediction)
    ref_norm = normalize_surface(reference_value)
    r_ref = parse_number(ref_norm, config)
    if not r_ref.ok or r_ref.value is None:
        return _invalid_reference(r_ref.error_code)
    r_pred = parse_number(pred_norm, config)
    if not r_pred.ok or r_pred.value is None:
        return _invalid_prediction(r_pred.error_code, str(r_ref.value))
    diff = abs(r_pred.value - r_ref.value)
    limit = tolerance_fraction(config.real_atol) + tolerance_fraction(config.real_rtol) * abs(
        r_ref.value
    )
    details: dict[str, JSONValue] = {
        "atol": config.real_atol,
        "rtol": config.real_rtol,
        "abs_diff": float(diff),
    }
    if diff <= limit:
        return _correct(str(r_pred.value), str(r_ref.value), details)
    return _incorrect(str(r_pred.value), str(r_ref.value), "outside tolerance", details)


@_register(AnswerType.SET)
def _verify_set(prediction: str, reference_value: str, config: VerifierConfig, _task_id: str) -> VerifierResult:
    pred_norm = normalize_surface(prediction)
    ref_norm = normalize_surface(reference_value)
    r_ref = parse_value(ref_norm, config, depth=0)
    if not r_ref.ok or r_ref.value is None:
        return _invalid_reference(r_ref.error_code)
    if not isinstance(r_ref.value, frozenset):
        return _invalid_reference("reference is not a set")
    r_pred = parse_value(pred_norm, config, depth=0)
    if not r_pred.ok or r_pred.value is None:
        return _invalid_prediction(r_pred.error_code, canonical_form(r_ref.value))
    if not isinstance(r_pred.value, frozenset):
        return _incorrect(
            canonical_form(r_pred.value),
            canonical_form(r_ref.value),
            "prediction is not a set",
        )
    if r_pred.value == r_ref.value:
        return _correct(canonical_form(r_pred.value), canonical_form(r_ref.value))
    return _incorrect(canonical_form(r_pred.value), canonical_form(r_ref.value), "sets differ")


@_register(AnswerType.TUPLE)
def _verify_tuple(prediction: str, reference_value: str, config: VerifierConfig, _task_id: str) -> VerifierResult:
    pred_norm = normalize_surface(prediction)
    ref_norm = normalize_surface(reference_value)
    r_ref = parse_value(ref_norm, config, depth=0)
    if not r_ref.ok or r_ref.value is None:
        return _invalid_reference(r_ref.error_code)
    if not isinstance(r_ref.value, tuple):
        return _invalid_reference("reference is not a tuple")
    r_pred = parse_value(pred_norm, config, depth=0)
    if not r_pred.ok or r_pred.value is None:
        return _invalid_prediction(r_pred.error_code, canonical_form(r_ref.value))
    if not isinstance(r_pred.value, tuple):
        return _incorrect(
            canonical_form(r_pred.value),
            canonical_form(r_ref.value),
            "prediction is not a tuple",
        )
    if r_pred.value == r_ref.value:
        return _correct(canonical_form(r_pred.value), canonical_form(r_ref.value))
    return _incorrect(canonical_form(r_pred.value), canonical_form(r_ref.value), "tuples differ")


@_register(AnswerType.INTERVAL)
def _verify_interval(prediction: str, reference_value: str, config: VerifierConfig, _task_id: str) -> VerifierResult:
    pred_norm = normalize_surface(prediction)
    ref_norm = normalize_surface(reference_value)
    r_ref = parse_interval(ref_norm, config)
    if not r_ref.ok or r_ref.value is None:
        return _invalid_reference(r_ref.error_code)
    r_pred = parse_interval(pred_norm, config)
    if not r_pred.ok or r_pred.value is None:
        return _invalid_prediction(r_pred.error_code, interval_form(r_ref.value))
    if r_pred.value == r_ref.value:
        return _correct(interval_form(r_pred.value), interval_form(r_ref.value))
    return _incorrect(interval_form(r_pred.value), interval_form(r_ref.value), "intervals differ")


@_register(AnswerType.EXPRESSION)
def _verify_expression(
    prediction: str, reference_value: str, config: VerifierConfig, task_id: str
) -> VerifierResult:
    del config  # worker limits are infrastructure concerns, not verification semantics
    verdict = _get_expression_worker().compare(prediction, reference_value, task_id)
    return VerifierResult.model_validate(verdict)


def _correct(
    normalized_prediction: str,
    normalized_reference: str,
    details: dict[str, JSONValue] | None = None,
) -> VerifierResult:
    return VerifierResult(
        status=VerifierStatus.CORRECT,
        reward=1.0,
        normalized_prediction=normalized_prediction,
        normalized_reference=normalized_reference,
        details=details or {},
    )


def _incorrect(
    normalized_prediction: str | None,
    normalized_reference: str | None,
    reason: str,
    details: dict[str, JSONValue] | None = None,
) -> VerifierResult:
    merged = {"reason": reason, **(details or {})}
    return VerifierResult(
        status=VerifierStatus.INCORRECT,
        reward=0.0,
        normalized_prediction=normalized_prediction,
        normalized_reference=normalized_reference,
        details=merged,
    )


def _invalid_prediction(error_code: str | None, normalized_reference: str | None) -> VerifierResult:
    return VerifierResult(
        status=VerifierStatus.INVALID_PREDICTION,
        reward=0.0,
        normalized_prediction=None,
        normalized_reference=normalized_reference,
        details={"error_code": error_code or "unknown"},
    )


def no_final_answer_verdict() -> VerifierResult:
    """Verdict for a rollout that terminated without ever producing an answer.

    The eval loop already reports this fact as INVALID_PREDICTION when the
    extractor finds nothing (model_eval._prediction_row). The training loop must
    represent it the same way, because "no verdict" is what makes a terminal state
    skip reward_for_trajectory -- and a skipped reward config pays 0.0, which beats
    an honest wrong answer charged its invalid-action penalty.
    """
    return VerifierResult(
        status=VerifierStatus.INVALID_PREDICTION,
        reward=0.0,
        normalized_prediction=None,
        normalized_reference=None,
        details={"error_code": "no_final_answer"},
    )


def _invalid_reference(error_code: str | None) -> VerifierResult:
    return VerifierResult(
        status=VerifierStatus.INVALID_REFERENCE,
        reward=0.0,
        normalized_prediction=None,
        normalized_reference=None,
        details={"error_code": error_code or "unknown"},
    )
