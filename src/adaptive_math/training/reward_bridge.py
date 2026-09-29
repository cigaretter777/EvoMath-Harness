"""Shared reward normalization primitives for synchronous GRPO rollout groups."""

import math

from pydantic import BaseModel, ConfigDict

from adaptive_math.agent.state import EventKind
from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.types import Budget
from adaptive_math.reward import (
    RewardBreakdown,
    RewardConfig,
    RewardContext,
    compute_reward,
)
from adaptive_math.verifier.service import VerifierResult


class GroupAdvantages(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    advantages: tuple[float, ...]
    effective: bool
    mean_reward: float
    std_reward: float


def _tool_successes(trajectory: Trajectory) -> int:
    """Count executed tool calls whose recorded result reports ``ok``.

    The event payload shape is the one the environment writes
    (``{"name": ..., "result": {...}}``); failed or rejected calls contribute
    nothing. This is the only positive tool-use signal in the reward family
    (R4's bonus), so it is derived from the trajectory, never from usage
    counters that increment before execution succeeds.
    """
    successes = 0
    for event in trajectory.events:
        if event.kind is not EventKind.TOOL_RESULT:
            continue
        result = event.payload.get("result")
        if isinstance(result, dict) and result.get("ok") is True:
            successes += 1
    return successes


def reward_for_trajectory(
    trajectory: Trajectory,
    verdict: VerifierResult,
    budget: Budget,
    *,
    max_generated_tokens: int,
    config: RewardConfig,
) -> RewardBreakdown:
    """Bridge a terminal rollout to the sole production reward implementation."""
    return compute_reward(
        RewardContext(
            verifier_result=verdict,
            tool_calls=trajectory.usage.tool_calls,
            tool_successes=_tool_successes(trajectory),
            python_seconds=trajectory.usage.python_seconds,
            invalid_action_count=trajectory.usage.invalid_actions,
            generated_tokens=trajectory.usage.generated_tokens,
            max_generated_tokens=max_generated_tokens,
            budget=budget,
        ),
        config,
    )


def group_advantages(rewards: list[float], *, epsilon: float = 1e-6) -> GroupAdvantages:
    """Compute GRPO group-relative advantages without inventing signal for ties."""
    if not rewards:
        raise ValueError("rewards must not be empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not all(math.isfinite(reward) for reward in rewards):
        raise ValueError("rewards must be finite")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std == 0:
        return GroupAdvantages(
            advantages=tuple(0.0 for _ in rewards), effective=False, mean_reward=mean, std_reward=std
        )
    return GroupAdvantages(
        advantages=tuple((reward - mean) / (std + epsilon) for reward in rewards),
        effective=True,
        mean_reward=mean,
        std_reward=std,
    )
