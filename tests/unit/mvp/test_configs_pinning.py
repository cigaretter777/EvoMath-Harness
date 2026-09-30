"""The MVP configs are mirrors of the released ones, and stay mirrors.

Why these exist: ``configs/mvp/{sft,grpo}.yaml`` exist so the four MVP entry
points have a default that *is* the released configuration rather than a
lookalike. "Field-for-field copy" is the property that makes that true, and a
copy is exactly the thing that drifts -- one side gets a tuned value, the other
keeps the old one, and the MVP quietly stops reproducing the released numbers.

So each mirror is compared with its source as *parsed data* (not as text: the
mirror carries a provenance header and the source does not), and then run
through the entry point's own validator, so a mirror that is byte-equal but no
longer loadable cannot pass either.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import yaml

ROOT = Path(__file__).resolve().parents[3]
MVP_SFT = ROOT / "configs" / "mvp" / "sft.yaml"
MVP_GRPO = ROOT / "configs" / "mvp" / "grpo.yaml"
RELEASED_SFT = ROOT / "configs" / "sft" / "qwen3_1_7b_dp_v1.yaml"
RELEASED_GRPO = ROOT / "configs" / "grpo" / "qwen3_1_7b_r0.yaml"

# The released SFT run's manifest pin, and the Adapter it produced. Both are
# facts about what "the released weights" means; the mirror must not disagree.
SFT_MANIFEST_SHA256 = "f9cc4c4a41ea7f63e95c9ebe7f6de26ff225fbd6c6d8b144fcb745c78350433b"
MERGED_SFT_MODEL = "/root/autodl-tmp/Adaptive-Solver-main-git/artifacts/models/qwen3_1_7b_sft_dp_v1_merged"


def load_script(path: Path, name: str) -> ModuleType:
    """Load a script for direct testing, registering it before exec."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_sft_mirror_is_field_for_field_the_released_config() -> None:
    assert yaml.safe_load(MVP_SFT.read_text()) == yaml.safe_load(RELEASED_SFT.read_text())


def test_the_grpo_mirror_is_field_for_field_the_released_config() -> None:
    assert yaml.safe_load(MVP_GRPO.read_text()) == yaml.safe_load(RELEASED_GRPO.read_text())


def test_the_sft_mirror_loads_and_pins_the_released_manifest() -> None:
    run_sft = load_script(ROOT / "scripts" / "train" / "run_sft.py", "run_sft_config_pins")
    config = run_sft.load_config(MVP_SFT)
    assert config.data_manifest_sha256 == SFT_MANIFEST_SHA256


def test_the_grpo_mirror_loads_and_trains_on_the_merged_sft_model() -> None:
    run_grpo = load_script(ROOT / "scripts" / "train" / "run_grpo.py", "run_grpo_config_pins")
    config = run_grpo.load_config(MVP_GRPO)
    assert f"actor_rollout_ref.model.path={MERGED_SFT_MODEL}" in config.upstream_overrides
    # The LoRA shape the export step reads back out of this file (train_grpo's
    # --export-adapter), pinned here so the two cannot disagree about it.
    overrides = dict(item.split("=", 1) for item in config.upstream_overrides)
    assert overrides["actor_rollout_ref.model.lora_rank"] == "16"
    assert overrides["actor_rollout_ref.model.lora_alpha"] == "32"
