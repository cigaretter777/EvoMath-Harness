"""Content-addressed harness specification.

A *harness* is everything around the policy model that can change agent
behaviour without touching model weights: prompt version, runtime version,
enabled tools, per-episode budget, decoding parameters and the reward
configuration used to score its trajectories.

``HarnessSpec`` bundles those parts into one immutable, hashable object so
that a harness can be patched, compared, promoted and rolled back by hash —
the same versioning discipline already used for ``RewardConfig``
(``configs/reward/*.yaml`` pinned to code presets) and for golden traces
(``TraceEnvelope.content_hash``).
"""

import re

import orjson
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from adaptive_math.agent.model_client import GenerationConfig
from adaptive_math.core.hashing import sha256_hex
from adaptive_math.core.types import Budget
from adaptive_math.reward.types import RewardConfig

HARNESS_SCHEMA_VERSION = "harness-v1"

# Known tool names in src/adaptive_math/tools/. Kept as data (not StrEnum) so
# that adding a tool does not change the schema, only the allowed set.
KNOWN_TOOL_NAMES: tuple[str, ...] = ("python", "sympy")

_VERSION = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class HarnessSpec(BaseModel):
    """Immutable, content-addressed description of one harness version.

    All fields are required so that no behavioural knob is ever an
    undocumented hidden default (same rule as ``RewardConfig``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = HARNESS_SCHEMA_VERSION
    harness_version: str
    prompt_version: str
    runtime_version: str
    tools: tuple[str, ...]
    budget: Budget
    generation: GenerationConfig
    reward: RewardConfig

    @field_validator("harness_version", "prompt_version", "runtime_version")
    @classmethod
    def _version_shape(cls, value: str) -> str:
        if _VERSION.fullmatch(value) is None:
            raise ValueError(f"invalid version string: {value!r}")
        return value

    @field_validator("tools")
    @classmethod
    def _tools_are_known_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("duplicate tool names are not allowed")
        unknown = sorted(set(value) - set(KNOWN_TOOL_NAMES))
        if unknown:
            raise ValueError(f"unknown tool names: {unknown}")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _budget_matches_tools(self) -> "HarnessSpec":
        if not self.tools and self.budget.max_tool_calls > 0:
            raise ValueError("tool budget requires at least one enabled tool")
        if "python" not in self.tools and self.budget.max_python_seconds > 0:
            raise ValueError("python_seconds budget requires the python tool")
        return self

    def spec_hash(self) -> str:
        """SHA-256 over canonical JSON; the harness's content address."""
        encoded = orjson.dumps(self.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
        return sha256_hex(encoded)

    def runtime_tag(self) -> str:
        """Value to store in ``Trajectory.runtime_version`` for this harness."""
        return f"{self.runtime_version}+{self.prompt_version}+{self.harness_version}"
