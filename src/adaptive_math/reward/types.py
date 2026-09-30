"""Reward domain types: context, config and breakdown.

Reward functions are pure: they consume VerifierResult plus usage counters
and never re-parse the answer or touch the model.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptive_math.core.hashing import sha256_hex
from adaptive_math.core.types import Budget
from adaptive_math.verifier import VerifierResult


class RewardConfig(BaseModel):
    """Explicit reward ablation parameters; all fields are required so that
    no weight is ever an undocumented hidden default."""

    model_config = ConfigDict(extra="forbid")

    version: str
    variant: Literal["r0", "r1", "r2", "r3", "r4"]
    tool_weight: float = Field(ge=0, le=1)
    python_weight: float = Field(ge=0, le=1)
    invalid_weight: float = Field(ge=0, le=1)
    invalid_cap: int = Field(ge=0)
    token_weight: float = Field(ge=0, le=1)
    # Positive term: paid once when at least one tool call executed
    # successfully (the only member of the family with a tool reward). Zero in
    # R0-R3, whose totals are unchanged by the field's presence.
    tool_success_bonus: float = Field(ge=0, le=1)
    clip_min: float
    clip_max: float

    @model_validator(mode="after")
    def _clip_bounds_are_ordered(self) -> "RewardConfig":
        if self.clip_min > self.clip_max:
            raise ValueError("clip_min must not exceed clip_max")
        return self

    def config_hash(self) -> str:
        return sha256_hex(self.model_dump_json().encode("utf-8"))


class RewardContext(BaseModel):
    """Everything a reward function may read about one trajectory."""

    model_config = ConfigDict(extra="forbid")

    verifier_result: VerifierResult
    tool_calls: int = Field(ge=0)
    tool_successes: int = Field(ge=0)
    python_seconds: float = Field(ge=0)
    invalid_action_count: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    max_generated_tokens: int = Field(ge=0)
    budget: Budget


class RewardBreakdown(BaseModel):
    """Versioned, traceable reward output for one trajectory."""

    model_config = ConfigDict(extra="forbid")

    total: float
    components: dict[str, float]
    reward_version: str
    config_hash: str
