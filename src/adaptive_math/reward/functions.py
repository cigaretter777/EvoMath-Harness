"""Pure, versioned reward functions (R0-R4).

R0 = correct
R1 = correct - invalid_weight * min(invalid_count, invalid_cap)
R2 = correct * (1 - tool_weight * tool_calls/max_tool_calls
                    - python_weight * python_seconds/max_python_seconds)
     - invalid_weight * min(invalid_count, invalid_cap)
R3 = R2 - correct * token_weight * generated_tokens/max_generated_tokens
R4 = correct + tool_success_bonus * [at least one tool call executed
     successfully]
     - invalid_weight * min(invalid_count, invalid_cap)

Cost components apply only to correct trajectories; a zero budget
denominator zeroes its component; the total is clipped to [clip_min,
clip_max]. The variant names are weight presets for one shared formula.
The R4 success bonus is deliberately not gated by correctness: it is the
only positive tool-use signal in the family, and it has to exist while
correctness is still sparse (2026-09-29: under R0-R3 a group of four
rollouts scores identically on tool behaviour, so GRPO has zero
within-group advantage to teach tool use with).
"""

from adaptive_math.reward.types import RewardBreakdown, RewardConfig, RewardContext
from adaptive_math.verifier import VerifierStatus

# Weight presets must stay in sync with configs/reward/*.yaml (pinned by
# tests/unit/reward/test_functions.py::test_yaml_configs_match_code_presets).
R0_DEFAULT = RewardConfig(
    version="r0-v1",
    variant="r0",
    tool_weight=0.0,
    python_weight=0.0,
    invalid_weight=0.0,
    invalid_cap=3,
    token_weight=0.0,
    tool_success_bonus=0.0,
    clip_min=-1.0,
    clip_max=1.0,
)
R1_DEFAULT = RewardConfig(
    version="r1-v1",
    variant="r1",
    tool_weight=0.0,
    python_weight=0.0,
    invalid_weight=0.10,
    invalid_cap=3,
    token_weight=0.0,
    tool_success_bonus=0.0,
    clip_min=-1.0,
    clip_max=1.0,
)
R2_DEFAULT = RewardConfig(
    version="r2-v1",
    variant="r2",
    tool_weight=0.15,
    python_weight=0.10,
    invalid_weight=0.10,
    invalid_cap=3,
    token_weight=0.0,
    tool_success_bonus=0.0,
    clip_min=-1.0,
    clip_max=1.0,
)
R3_DEFAULT = RewardConfig(
    version="r3-v1",
    variant="r3",
    tool_weight=0.15,
    python_weight=0.10,
    invalid_weight=0.10,
    invalid_cap=3,
    token_weight=0.05,
    tool_success_bonus=0.0,
    clip_min=-1.0,
    clip_max=1.0,
)
# First member with a positive tool-use term. No tool/python costs: the run
# exists to make tool use pay, not to economize it. clip_max sits above
# correct + bonus (1.1) so the bonus survives clipping.
R4_DEFAULT = RewardConfig(
    version="r4-v1",
    variant="r4",
    tool_weight=0.0,
    python_weight=0.0,
    invalid_weight=0.10,
    invalid_cap=3,
    token_weight=0.0,
    tool_success_bonus=0.10,
    clip_min=-1.0,
    clip_max=1.2,
)


def compute_reward(context: RewardContext, config: RewardConfig) -> RewardBreakdown:
    correct = 1.0 if context.verifier_result.status is VerifierStatus.CORRECT else 0.0
    invalid_penalty = config.invalid_weight * min(
        context.invalid_action_count, config.invalid_cap
    )
    tool_cost = _cost(context.tool_calls, context.budget.max_tool_calls, config.tool_weight, correct)
    python_cost = _cost(
        context.python_seconds,
        context.budget.max_python_seconds,
        config.python_weight,
        correct,
    )
    token_cost = _cost(
        context.generated_tokens,
        context.max_generated_tokens,
        config.token_weight,
        correct,
    )
    success_bonus = config.tool_success_bonus * (1.0 if context.tool_successes > 0 else 0.0)
    total = (
        correct * (1.0 - tool_cost - python_cost)
        - invalid_penalty
        - correct * token_cost
        + success_bonus
    )
    total = min(max(total, config.clip_min), config.clip_max)
    return RewardBreakdown(
        total=total,
        components={
            "correct": correct,
            "tool_cost": tool_cost,
            "python_cost": python_cost,
            "token_cost": token_cost,
            "invalid_penalty": invalid_penalty,
            "tool_success_bonus": success_bonus,
        },
        reward_version=config.version,
        config_hash=config.config_hash(),
    )


def reward_r0(context: RewardContext, config: RewardConfig | None = None) -> RewardBreakdown:
    return compute_reward(context, config or R0_DEFAULT)


def reward_r1(context: RewardContext, config: RewardConfig | None = None) -> RewardBreakdown:
    return compute_reward(context, config or R1_DEFAULT)


def reward_r2(context: RewardContext, config: RewardConfig | None = None) -> RewardBreakdown:
    return compute_reward(context, config or R2_DEFAULT)


def reward_r3(context: RewardContext, config: RewardConfig | None = None) -> RewardBreakdown:
    return compute_reward(context, config or R3_DEFAULT)


def reward_r4(context: RewardContext, config: RewardConfig | None = None) -> RewardBreakdown:
    return compute_reward(context, config or R4_DEFAULT)


def _cost(used: float, limit: float, weight: float, correct: float) -> float:
    if correct == 0.0 or limit <= 0 or weight == 0.0:
        return 0.0
    return weight * (used / limit)
