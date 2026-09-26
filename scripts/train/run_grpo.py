"""Run synchronous Adaptive-Math GRPO through the pinned verl-agent backend.

The entrypoint validates immutable inputs before importing CUDA/Ray, applies
the auditable environment registration patch to the verified upstream checkout,
and then delegates policy updates to ``recipe.hgpo.main_hgpo.run_ppo``.
"""

import argparse
import importlib
import inspect
import json
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from adaptive_math.core.hashing import sha256_hex
from adaptive_math.training.upstreams import validate_manifest

PLACEHOLDER_HASH = "0" * 64
REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "task_pool_path",
        "task_pool_sha256",
        "budget_path",
        "reward_path",
        "output_dir",
        "upstream_config_name",
        "upstream_overrides",
    }
)


class ConfigError(SystemExit):
    pass


@dataclass(frozen=True)
class GRPOConfig:
    schema_version: str
    task_pool_path: Path
    task_pool_sha256: str
    budget_path: Path
    reward_path: Path
    output_dir: Path
    upstream_config_name: str
    upstream_overrides: tuple[str, ...]


def load_config(path: Path) -> GRPOConfig:
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read GRPO config {path}: {exc}") from exc
    if not isinstance(raw, dict) or not REQUIRED_KEYS.issubset(raw):
        missing = sorted(REQUIRED_KEYS.difference(raw if isinstance(raw, dict) else {}))
        raise ConfigError(f"GRPO config missing required keys: {', '.join(missing)}")
    if raw["schema_version"] != "adaptive-math-grpo-v1":
        raise ConfigError("unsupported GRPO schema_version")
    digest = raw["task_pool_sha256"]
    overrides = raw["upstream_overrides"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ConfigError("task_pool_sha256 must be a lowercase SHA-256")
    if digest == PLACEHOLDER_HASH:
        raise ConfigError("task_pool_sha256 cannot be the all-zero placeholder")
    if not isinstance(overrides, list) or not all(isinstance(item, str) for item in overrides):
        raise ConfigError("upstream_overrides must be a list of Hydra override strings")
    return GRPOConfig(
        schema_version=str(raw["schema_version"]),
        task_pool_path=Path(str(raw["task_pool_path"])),
        task_pool_sha256=digest,
        budget_path=Path(str(raw["budget_path"])),
        reward_path=Path(str(raw["reward_path"])),
        output_dir=Path(str(raw["output_dir"])),
        upstream_config_name=str(raw["upstream_config_name"]),
        upstream_overrides=tuple(overrides),
    )


def with_output_dir(config: GRPOConfig, output_dir: Path) -> GRPOConfig:
    if not str(output_dir):
        raise ConfigError("output_dir must not be empty")
    return replace(config, output_dir=output_dir)


def validate_inputs(config: GRPOConfig) -> None:
    for path in (config.task_pool_path, config.budget_path, config.reward_path):
        if not path.is_file():
            raise ConfigError(f"required GRPO input does not exist: {path}")
    actual = sha256_hex(config.task_pool_path.read_bytes())
    if actual != config.task_pool_sha256:
        raise ConfigError(f"task pool hash mismatch: expected {config.task_pool_sha256}, got {actual}")


def dry_run_report(config: GRPOConfig, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    return {
        "ok": True,
        "environment": "adaptive_math",
        "verl_agent_sha": manifest["upstreams"]["verl-agent"]["resolved_sha"],
        "task_pool_path": str(config.task_pool_path),
        "upstream_config_name": config.upstream_config_name,
    }


def run_training(config: GRPOConfig, manifest_path: Path) -> None:
    validate_inputs(config)
    from adaptive_math.training.verl_agent_adapter import apply_verl_agent_environment_patch

    try:
        backend = importlib.import_module("recipe.hgpo.main_hgpo")
        env_manager = importlib.import_module("recipe.hgpo.env_manager")
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ConfigError("pinned verl-agent cloud dependencies are not installed") from exc
    if not callable(getattr(backend, "run_ppo", None)):
        raise ConfigError("pinned verl-agent does not expose recipe.hgpo.main_hgpo.run_ppo")
    _verify_checkout(Path(inspect.getfile(backend)), manifest_path)
    patch = apply_verl_agent_environment_patch(Path(inspect.getfile(env_manager)))
    config_dir = Path(inspect.getfile(backend)).parent / "config"
    if not config_dir.is_dir():
        raise ConfigError(f"pinned verl-agent config directory is missing: {config_dir}")
    # The upstream hgpo_trainer config declares a relative searchpath
    # (file://verl/trainer/config) that only resolves when Hydra runs from the
    # verl-agent repo root. Absolutize it here: Hydra resolves file://
    # searchpath entries against the process CWD, not the config dir.
    verl_trainer_config = config_dir.parents[2] / "verl" / "trainer" / "config"
    overrides = [
        f"hydra.searchpath=[file://{verl_trainer_config}]",
    ] + list(config.upstream_overrides) + [
        "env.env_name=adaptive_math",
        f"+env.task_pool_path={config.task_pool_path.resolve()}",
        f"+env.budget_path={config.budget_path.resolve()}",
        f"+env.reward_path={config.reward_path.resolve()}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        upstream = compose(config_name=config.upstream_config_name, overrides=overrides)
    _write_run_metadata(config, manifest_path, OmegaConf.to_container(upstream, resolve=True), patch.changed)
    backend.run_ppo(upstream)


def _write_run_metadata(
    config: GRPOConfig, manifest_path: Path, resolved_upstream: Any, patch_changed: bool
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "adaptive_math": {
                    "schema_version": config.schema_version,
                    "task_pool_path": str(config.task_pool_path),
                    "task_pool_sha256": config.task_pool_sha256,
                    "budget_path": str(config.budget_path),
                    "reward_path": str(config.reward_path),
                    "output_dir": str(config.output_dir),
                    "upstream_config_name": config.upstream_config_name,
                    "upstream_overrides": list(config.upstream_overrides),
                },
                "upstream": resolved_upstream,
            },
            sort_keys=True,
        )
    )
    manifest = json.loads(manifest_path.read_text())
    (config.output_dir / "backend_adapter.json").write_text(
        json.dumps(
            {
                "verl_agent_sha": manifest["upstreams"]["verl-agent"]["resolved_sha"],
                "environment_registration_patch_changed": patch_changed,
                "pid": os.getpid(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _verify_checkout(backend_file: Path, manifest_path: Path) -> None:
    """Require an editable checkout at exactly the SHA we promised to use."""
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    expected = manifest["upstreams"]["verl-agent"]["resolved_sha"]
    checkout = backend_file.parents[2]
    try:
        actual = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigError(f"verl-agent must be an editable git checkout at {checkout}") from exc
    if actual != expected:
        raise ConfigError(f"verl-agent SHA mismatch: expected {expected}, got {actual}")
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("third_party/manifest.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.output_dir is not None:
        config = with_output_dir(config, args.output_dir)
    if args.dry_run:
        print(json.dumps(dry_run_report(config, args.manifest), sort_keys=True))
        return 0
    run_training(config, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
