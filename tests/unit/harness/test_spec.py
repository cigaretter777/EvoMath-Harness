"""HarnessSpec validation, hashing and YAML preset pinning."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from adaptive_math.core.types import Budget
from adaptive_math.harness.presets import CURRENT_PRODUCTION
from adaptive_math.harness.spec import HarnessSpec

CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "harness"


def _spec(**overrides: object) -> HarnessSpec:
    # model_copy(update=...) bypasses validators; re-validate through the
    # schema so validator behaviour is actually exercised.
    payload = CURRENT_PRODUCTION.model_dump(mode="json")
    payload.update(overrides)
    return HarnessSpec.model_validate(payload)


def test_yaml_configs_match_code_presets() -> None:
    loaded = yaml.safe_load((CONFIGS / "champion_v1.yaml").read_text())
    assert HarnessSpec.model_validate(loaded) == CURRENT_PRODUCTION


def test_spec_hash_is_deterministic_and_content_addressed() -> None:
    assert CURRENT_PRODUCTION.spec_hash() == _spec().spec_hash()
    assert len(CURRENT_PRODUCTION.spec_hash()) == 64
    patched = _spec(harness_version="h-v2")
    assert patched.spec_hash() != CURRENT_PRODUCTION.spec_hash()


def test_spec_is_immutable() -> None:
    with pytest.raises(ValidationError):
        CURRENT_PRODUCTION.harness_version = "h-v2"  # type: ignore[misc]


def test_unknown_tool_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown tool"):
        _spec(tools=("python", "calculator"))


def test_duplicate_tools_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        _spec(tools=("python", "python"))


def test_tool_budget_requires_tool() -> None:
    with pytest.raises(ValidationError, match="tool budget"):
        _spec(tools=(), budget=Budget(max_steps=2, max_tool_calls=1,
                                      max_python_seconds=0.0, max_observation_chars=100))


def test_python_seconds_requires_python_tool() -> None:
    with pytest.raises(ValidationError, match="python tool"):
        _spec(tools=("sympy",), budget=Budget(max_steps=2, max_tool_calls=1,
                                              max_python_seconds=1.0, max_observation_chars=100))


def test_direct_harness_is_valid() -> None:
    direct = _spec(
        harness_version="h-direct-v1",
        tools=(),
        budget=Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0.0,
                      max_observation_chars=4000),
    )
    assert direct.tools == ()


def test_runtime_tag_combines_versions() -> None:
    assert CURRENT_PRODUCTION.runtime_tag() == "runtime-v1+agent-v1+h-v1"


def test_extra_fields_rejected() -> None:
    payload = CURRENT_PRODUCTION.model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        HarnessSpec.model_validate(payload)
