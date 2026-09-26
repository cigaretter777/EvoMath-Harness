"""Deterministic regression gate between a champion and a candidate harness.

The gate consumes *paired* per-task outcomes — the same frozen task set run
under both harnesses with the same decode seeds — and decides whether the
candidate may be promoted. Statistics mirror
``evaluation.model_eval._paired_statistics``: task-level paired bootstrap
95% CI plus an exact McNemar p-value, with the seed and resample count
recorded in the decision so the result is reproducible.
"""

import math
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.core.hashing import sha256_hex

BOOTSTRAP_SEED = 20260918
BOOTSTRAP_RESAMPLES = 10_000


class PairedOutcome(BaseModel):
    """One frozen task evaluated under both harnesses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    champion_correct: bool
    candidate_correct: bool


class MetricGuard(BaseModel):
    """Guard on a paired scalar metric delta (candidate − champion).

    ``metric`` names a summary field such as ``invalid_rate`` or
    ``mean_tool_calls``; the guard fails when the candidate's value exceeds
    the champion's by more than ``max_delta``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    max_delta: float = Field(ge=0)


class GatePolicy(BaseModel):
    """Explicit promotion rule; every threshold is a required field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str
    mode: Literal["non_regression", "significant_improvement"]
    # Point accuracy delta (candidate − champion) must be >= -max_regression.
    max_regression: float = Field(ge=0, le=1)
    # Bootstrap CI lower bound must be >= ci_lower_bound.
    ci_lower_bound: float = Field(ge=-1, le=1)
    # McNemar significance level. In non_regression mode a *significant*
    # regression (p <= alpha and delta < 0) fails the gate; in
    # significant_improvement mode the improvement must reach p <= alpha.
    mcnemar_alpha: float = Field(gt=0, lt=1)
    metric_guards: tuple[MetricGuard, ...] = ()


class GateDecision(BaseModel):
    """Auditable outcome of one gate run; persisted into the registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str
    passed: bool
    reasons: tuple[str, ...]
    task_count: int
    task_ids_sha256: str
    accuracy_delta: float
    paired_bootstrap_ci95: tuple[float, float]
    bootstrap_seed: int
    bootstrap_resamples: int
    mcnemar_pvalue: float
    metric_deltas: dict[str, float]


def evaluate_gate(
    outcomes: list[PairedOutcome],
    policy: GatePolicy,
    *,
    champion_metrics: dict[str, float] | None = None,
    candidate_metrics: dict[str, float] | None = None,
) -> GateDecision:
    """Compare candidate against champion on identical tasks; pure function."""
    if not outcomes:
        raise ValueError("gate requires at least one paired outcome")
    task_ids = [row.task_id for row in outcomes]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("gate outcomes contain duplicate task IDs")

    deltas = [int(row.candidate_correct) - int(row.champion_correct) for row in outcomes]
    n = len(deltas)
    accuracy_delta = sum(deltas) / n

    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(BOOTSTRAP_RESAMPLES))
    ci95 = (samples[249], samples[9749])

    improved = deltas.count(1)
    regressed = deltas.count(-1)
    discordant = improved + regressed
    if discordant == 0:
        pvalue = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
        pvalue = min(1.0, 2 * tail / 2**discordant)

    metric_deltas: dict[str, float] = {}
    reasons: list[str] = []
    champion_metrics = champion_metrics or {}
    candidate_metrics = candidate_metrics or {}
    for guard in policy.metric_guards:
        if guard.metric not in champion_metrics or guard.metric not in candidate_metrics:
            raise ValueError(f"metric guard requires both arms to report {guard.metric!r}")
        delta = candidate_metrics[guard.metric] - champion_metrics[guard.metric]
        metric_deltas[guard.metric] = delta
        if delta > guard.max_delta:
            reasons.append(
                f"metric {guard.metric} regressed by {delta:.4f} (allowed {guard.max_delta:.4f})"
            )

    if accuracy_delta < -policy.max_regression:
        reasons.append(
            f"accuracy delta {accuracy_delta:.4f} below allowed floor {-policy.max_regression:.4f}"
        )
    if ci95[0] < policy.ci_lower_bound:
        reasons.append(
            f"bootstrap CI lower bound {ci95[0]:.4f} below {policy.ci_lower_bound:.4f}"
        )
    if policy.mode == "non_regression":
        if accuracy_delta < 0 and pvalue <= policy.mcnemar_alpha:
            reasons.append(f"statistically significant regression (McNemar p={pvalue:.6g})")
    else:
        if not (accuracy_delta > 0 and pvalue <= policy.mcnemar_alpha):
            reasons.append(
                f"no statistically significant improvement (delta={accuracy_delta:.4f}, "
                f"McNemar p={pvalue:.6g}, alpha={policy.mcnemar_alpha})"
            )

    return GateDecision(
        policy_version=policy.policy_version,
        passed=not reasons,
        reasons=tuple(reasons),
        task_count=n,
        task_ids_sha256=sha256_hex("\n".join(sorted(task_ids)).encode()),
        accuracy_delta=accuracy_delta,
        paired_bootstrap_ci95=ci95,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        mcnemar_pvalue=pvalue,
        metric_deltas=metric_deltas,
    )
