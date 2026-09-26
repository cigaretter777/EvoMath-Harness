"""Deterministic regression-gate behaviour on synthetic paired outcomes."""

import pytest

from adaptive_math.harness.gate import (
    GatePolicy,
    MetricGuard,
    PairedOutcome,
    evaluate_gate,
)

POLICY = GatePolicy(
    policy_version="gate-v1",
    mode="non_regression",
    max_regression=0.0,
    ci_lower_bound=-0.05,
    mcnemar_alpha=0.05,
)


def _outcomes(pairs: list[tuple[bool, bool]]) -> list[PairedOutcome]:
    return [
        PairedOutcome(task_id=f"task:{index:03d}", champion_correct=champion,
                      candidate_correct=candidate)
        for index, (champion, candidate) in enumerate(pairs)
    ]


def test_identical_arms_pass() -> None:
    pairs = [(True, True)] * 40 + [(False, False)] * 10
    decision = evaluate_gate(_outcomes(pairs), POLICY)
    assert decision.passed
    assert decision.accuracy_delta == 0.0
    assert decision.mcnemar_pvalue == 1.0
    assert decision.task_count == 50


def test_clear_improvement_passes_non_regression() -> None:
    pairs = [(False, True)] * 12 + [(True, True)] * 30 + [(False, False)] * 8
    decision = evaluate_gate(_outcomes(pairs), POLICY)
    assert decision.passed
    assert decision.accuracy_delta == pytest.approx(12 / 50)


def test_significant_regression_fails() -> None:
    pairs = [(True, False)] * 12 + [(True, True)] * 30 + [(False, False)] * 8
    decision = evaluate_gate(_outcomes(pairs), POLICY)
    assert not decision.passed
    assert any("regression" in reason for reason in decision.reasons)
    assert decision.mcnemar_pvalue <= POLICY.mcnemar_alpha


def test_point_floor_fails_even_without_significance() -> None:
    # One net regression: not significant, but max_regression=0 forbids it.
    pairs = [(True, False)] + [(True, True)] * 40 + [(False, False)] * 9
    decision = evaluate_gate(_outcomes(pairs), POLICY)
    assert not decision.passed
    assert any("floor" in reason for reason in decision.reasons)


def test_significant_improvement_mode_requires_significance() -> None:
    policy = GatePolicy(
        policy_version="gate-v2",
        mode="significant_improvement",
        max_regression=0.05,
        ci_lower_bound=-0.10,
        mcnemar_alpha=0.05,
    )
    # Small insignificant improvement must fail.
    small = evaluate_gate(
        _outcomes([(False, True)] + [(True, True)] * 40 + [(False, False)] * 9), policy
    )
    assert not small.passed
    # Large significant improvement must pass.
    large = evaluate_gate(
        _outcomes([(False, True)] * 12 + [(True, True)] * 30 + [(False, False)] * 8), policy
    )
    assert large.passed


def test_metric_guard_blocks_cost_regression() -> None:
    policy = GatePolicy(
        policy_version="gate-v3",
        mode="non_regression",
        max_regression=0.05,
        ci_lower_bound=-0.10,
        mcnemar_alpha=0.05,
        metric_guards=(MetricGuard(metric="invalid_rate", max_delta=0.01),),
    )
    pairs = [(True, True)] * 50
    decision = evaluate_gate(
        _outcomes(pairs), policy,
        champion_metrics={"invalid_rate": 0.02},
        candidate_metrics={"invalid_rate": 0.09},
    )
    assert not decision.passed
    assert any("invalid_rate" in reason for reason in decision.reasons)
    assert decision.metric_deltas["invalid_rate"] == pytest.approx(0.07)


def test_metric_guard_requires_both_arms() -> None:
    policy = GatePolicy(
        policy_version="gate-v3",
        mode="non_regression",
        max_regression=0.05,
        ci_lower_bound=-0.10,
        mcnemar_alpha=0.05,
        metric_guards=(MetricGuard(metric="invalid_rate", max_delta=0.01),),
    )
    with pytest.raises(ValueError, match="invalid_rate"):
        evaluate_gate(_outcomes([(True, True)]), policy,
                      champion_metrics={"invalid_rate": 0.02})


def test_gate_is_deterministic() -> None:
    pairs = [(False, True)] * 5 + [(True, False)] * 3 + [(True, True)] * 42
    first = evaluate_gate(_outcomes(pairs), POLICY)
    second = evaluate_gate(_outcomes(pairs), POLICY)
    assert first == second
    assert first.bootstrap_seed == second.bootstrap_seed


def test_duplicate_task_ids_rejected() -> None:
    rows = _outcomes([(True, True)]) * 2
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_gate(rows, POLICY)


def test_empty_outcomes_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_gate([], POLICY)
