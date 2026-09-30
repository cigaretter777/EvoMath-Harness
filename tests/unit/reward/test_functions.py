import math
from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from adaptive_math.core.types import Budget
from adaptive_math.reward import (
    R0_DEFAULT,
    R1_DEFAULT,
    R2_DEFAULT,
    R3_DEFAULT,
    R4_DEFAULT,
    RewardBreakdown,
    RewardConfig,
    RewardContext,
    compute_reward,
    reward_r0,
    reward_r2,
    reward_r4,
)
from adaptive_math.verifier import VerifierResult, VerifierStatus

CONFIGS = Path(__file__).parents[3] / "configs" / "reward"


def make_verifier(status: VerifierStatus) -> VerifierResult:
    return VerifierResult(
        status=status,
        reward=1.0 if status is VerifierStatus.CORRECT else 0.0,
        normalized_prediction=None,
        normalized_reference=None,
    )


def make_context(
    status: VerifierStatus = VerifierStatus.CORRECT,
    *,
    tool_calls: int = 0,
    tool_successes: int = 0,
    python_seconds: float = 0.0,
    invalid_action_count: int = 0,
    generated_tokens: int = 0,
    max_tool_calls: int = 4,
    max_python_seconds: float = 12.0,
    max_generated_tokens: int = 0,
) -> RewardContext:
    return RewardContext(
        verifier_result=make_verifier(status),
        tool_calls=tool_calls,
        tool_successes=tool_successes,
        python_seconds=python_seconds,
        invalid_action_count=invalid_action_count,
        generated_tokens=generated_tokens,
        max_generated_tokens=max_generated_tokens,
        budget=Budget(
            max_steps=6,
            max_tool_calls=max_tool_calls,
            max_python_seconds=max_python_seconds,
            max_observation_chars=8000,
        ),
    )


@pytest.fixture
def correct_context() -> RewardContext:
    return make_context(tool_calls=2, max_tool_calls=4)


@pytest.fixture
def wrong_context() -> RewardContext:
    return make_context(status=VerifierStatus.INCORRECT, tool_calls=2, max_tool_calls=4)


def test_r0_is_pure_correctness() -> None:
    assert reward_r0(make_context()).total == 1.0
    assert reward_r0(make_context(status=VerifierStatus.INCORRECT)).total == 0.0


def test_r1_subtracts_capped_invalid_penalty() -> None:
    assert reward_r0(make_context(invalid_action_count=5)).total == 1.0  # r0 ignores it
    assert (
        compute_reward(make_context(invalid_action_count=2), R1_DEFAULT).total
        == pytest.approx(0.8)
    )
    assert (
        compute_reward(make_context(invalid_action_count=5), R1_DEFAULT).total
        == pytest.approx(0.7)
    )


def test_r2_exact_formula() -> None:
    result = compute_reward(
        make_context(tool_calls=2, python_seconds=6.0, max_tool_calls=4, max_python_seconds=12.0),
        R2_DEFAULT,
    )
    # 1 * (1 - 0.15*2/4 - 0.10*6/12) = 1 * (1 - 0.075 - 0.05) = 0.875
    assert result.total == pytest.approx(0.875)
    assert result.components["tool_cost"] == pytest.approx(0.075)
    assert result.components["python_cost"] == pytest.approx(0.05)


def test_r2_cost_is_gated_by_correctness(
    correct_context: RewardContext, wrong_context: RewardContext
) -> None:
    correct = reward_r2(correct_context)
    wrong = reward_r2(wrong_context)
    assert correct.total < 1.0
    assert wrong.components["tool_cost"] == 0.0


def test_r3_adds_token_cost_to_r2() -> None:
    r2 = compute_reward(
        make_context(tool_calls=2, max_tool_calls=4), R2_DEFAULT
    ).total
    r3 = compute_reward(
        make_context(tool_calls=2, generated_tokens=500, max_tool_calls=4, max_generated_tokens=1000),
        R3_DEFAULT,
    ).total
    # R3 = R2 - 1 * 0.05 * 500/1000 = R2 - 0.025
    assert r3 == pytest.approx(r2 - 0.025)


def test_token_cost_is_gated_by_correctness() -> None:
    wrong = compute_reward(
        make_context(status=VerifierStatus.INCORRECT, generated_tokens=500, max_generated_tokens=1000),
        R3_DEFAULT,
    )
    assert wrong.total == 0.0
    assert wrong.components["token_cost"] == 0.0


def test_zero_budget_denominators_zero_the_cost_component() -> None:
    result = compute_reward(
        make_context(tool_calls=3, python_seconds=5.0, max_tool_calls=0, max_python_seconds=0.0),
        R2_DEFAULT,
    )
    assert result.components["tool_cost"] == 0.0
    assert result.components["python_cost"] == 0.0
    assert result.total == 1.0


def test_total_is_clipped_to_config_bounds() -> None:
    config = RewardConfig(
        version="r1-test",
        variant="r1",
        tool_weight=0.0,
        python_weight=0.0,
        invalid_weight=1.0,
        invalid_cap=3,
        token_weight=0.0,
        tool_success_bonus=0.0,
        clip_min=-0.5,
        clip_max=1.0,
    )
    # correct - 1.0 * min(3, 3) = -2.0, clipped to -0.5
    assert compute_reward(make_context(invalid_action_count=3), config).total == -0.5


def test_incorrect_valid_trajectories_receive_zero_under_r0_and_r2() -> None:
    context = make_context(status=VerifierStatus.INCORRECT, tool_calls=2)
    assert reward_r0(context).total == 0.0
    assert reward_r2(context).total == 0.0


def test_r4_pays_a_flat_bonus_once_for_a_successful_tool_use() -> None:
    assert reward_r4(make_context(tool_successes=1)).total == pytest.approx(1.1)
    # The bonus pays once, not per success: three successes score as one.
    assert reward_r4(make_context(tool_successes=3)).total == pytest.approx(1.1)
    assert reward_r4(make_context()).total == pytest.approx(1.0)


def test_r4_success_bonus_is_not_gated_by_correctness() -> None:
    # The whole point of R4: a successful tool call pays even when the answer
    # is wrong, because that is the gradient GRPO needs to learn tool use
    # before correctness appears (2026-09-29).
    wrong = reward_r4(make_context(status=VerifierStatus.INCORRECT, tool_successes=1))
    assert wrong.components["correct"] == 0.0
    assert wrong.components["tool_success_bonus"] == pytest.approx(0.10)
    assert wrong.total == pytest.approx(0.10)


def test_r0_through_r3_ignore_tool_successes() -> None:
    for config in (R0_DEFAULT, R1_DEFAULT, R2_DEFAULT, R3_DEFAULT):
        assert (
            compute_reward(make_context(), config).total
            == compute_reward(make_context(tool_successes=2), config).total
        )


def test_reward_is_deterministic(
    correct_context: RewardContext, wrong_context: RewardContext
) -> None:
    for context in (correct_context, wrong_context):
        for config in (R0_DEFAULT, R1_DEFAULT, R2_DEFAULT, R3_DEFAULT, R4_DEFAULT):
            assert compute_reward(context, config) == compute_reward(context, config)


def test_increasing_tool_cost_never_increases_correct_r2_reward() -> None:
    base = compute_reward(make_context(tool_calls=0, max_tool_calls=4), R2_DEFAULT).total
    more = compute_reward(make_context(tool_calls=2, max_tool_calls=4), R2_DEFAULT).total
    even_more = compute_reward(make_context(tool_calls=4, max_tool_calls=4), R2_DEFAULT).total
    assert base >= more >= even_more


def test_reward_breakdown_round_trips() -> None:
    breakdown = compute_reward(make_context(), R2_DEFAULT)
    assert RewardBreakdown.model_validate_json(breakdown.model_dump_json()) == breakdown
    assert breakdown.reward_version == "r2-v1"
    assert breakdown.config_hash == R2_DEFAULT.config_hash()


def test_config_hash_is_stable_and_sensitive() -> None:
    assert R2_DEFAULT.config_hash() == R2_DEFAULT.config_hash()
    assert R2_DEFAULT.config_hash() != R0_DEFAULT.config_hash()


@pytest.mark.parametrize(
    ("filename", "preset"),
    [
        ("r0.yaml", R0_DEFAULT),
        ("r1.yaml", R1_DEFAULT),
        ("r2.yaml", R2_DEFAULT),
        ("r3.yaml", R3_DEFAULT),
        ("r4.yaml", R4_DEFAULT),
    ],
)
def test_yaml_configs_match_code_presets(filename: str, preset: RewardConfig) -> None:
    loaded = yaml.safe_load((CONFIGS / filename).read_text())
    assert RewardConfig.model_validate(loaded) == preset


@given(
    st.booleans(),
    st.integers(min_value=0, max_value=20),
    st.integers(min_value=0, max_value=20),
    st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    st.integers(min_value=0, max_value=20),
    st.integers(min_value=0, max_value=10000),
    st.integers(min_value=0, max_value=20),
)
def test_total_is_finite_and_bounded(
    correct: bool,
    tool_calls: int,
    tool_successes: int,
    python_seconds: float,
    invalid_actions: int,
    generated_tokens: int,
    max_generated_tokens: int,
) -> None:
    context = make_context(
        status=VerifierStatus.CORRECT if correct else VerifierStatus.INCORRECT,
        tool_calls=tool_calls,
        tool_successes=tool_successes,
        python_seconds=python_seconds,
        invalid_action_count=invalid_actions,
        generated_tokens=generated_tokens,
        max_generated_tokens=max_generated_tokens,
    )
    for config in (R0_DEFAULT, R1_DEFAULT, R2_DEFAULT, R3_DEFAULT, R4_DEFAULT):
        total = compute_reward(context, config).total
        assert math.isfinite(total)
        assert config.clip_min <= total <= config.clip_max
