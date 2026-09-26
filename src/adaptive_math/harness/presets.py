"""Code presets for known harness versions.

Presets must stay in sync with ``configs/harness/*.yaml`` (pinned by
``tests/unit/harness/test_spec.py::test_yaml_configs_match_code_presets``),
the same bidirectional lock used for ``configs/reward/*.yaml``.
"""

from adaptive_math.agent.loop import RUNTIME_VERSION
from adaptive_math.agent.model_client import GenerationConfig
from adaptive_math.agent.prompts import PROMPT_VERSION
from adaptive_math.core.types import Budget
from adaptive_math.harness.spec import HarnessSpec
from adaptive_math.reward.functions import R2_DEFAULT

# The harness currently implied by configs/agent/default.yaml plus the
# runtime/prompt constants and the R2 reward preset.
CURRENT_PRODUCTION = HarnessSpec(
    harness_version="h-v1",
    prompt_version=PROMPT_VERSION,
    runtime_version=RUNTIME_VERSION,
    tools=("python", "sympy"),
    budget=Budget(max_steps=6, max_tool_calls=4, max_python_seconds=12.0,
                  max_observation_chars=8000),
    generation=GenerationConfig(max_new_tokens=1024, temperature=0.0, seed=None),
    reward=R2_DEFAULT,
)
