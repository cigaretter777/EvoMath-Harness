"""The MVP wrappers forward, and refuse, exactly as designed.

Why these exist: ``train_sft.py`` and ``train_grpo.py`` are thin on purpose --
they add a default config and one optional post-step, and everything else must
reach the entry point underneath unchanged. A wrapper that quietly rewrites or
swallows a flag is worse than no wrapper, because the command in the README and
the command that ran would no longer be the same command.

The dry-runs here are the real thing through ``subprocess``: they are the only
way to pin that the whole chain (wrapper -> entry point -> resolved config)
still runs with no GPU stack present, which is what makes the MVP's config
workflow testable at all.

``prepare_data.py`` is here too: its checks are the mechanical guard against the
2026-09-26 misalignment, and the fixture tests pin the *plumbing* (hash
convention, refusal to replace a different list, which file each check blames)
that the local real-data test cannot pin on a clean checkout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from adaptive_math.core.hashing import sha256_hex

ROOT = Path(__file__).resolve().parents[3]
MVP = ROOT / "scripts" / "mvp"
TRAIN_SFT = MVP / "train_sft.py"
TRAIN_GRPO = MVP / "train_grpo.py"
PREPARE_DATA = MVP / "prepare_data.py"

FROZEN_IDS = ROOT / "artifacts" / "eval" / "thesis_e0_base_direct_b1" / "task_ids.txt"
EVAL_PARQUET = ROOT / "data" / "processed" / "v1" / "frozen_eval.parquet"
HAVE_FROZEN_SET = FROZEN_IDS.is_file() and EVAL_PARQUET.is_file()
skip_without_frozen_set = pytest.mark.skipif(
    not HAVE_FROZEN_SET,
    reason="the frozen parquet and its task list are local artifacts, not tracked",
)

# The pinned backend commit run_grpo.py refuses to launch anything else on.
VERL_AGENT_SHA = "20bd331bdbc9026a5668e11362178e10ab7400c8"


def load_script(path: Path, name: str) -> ModuleType:
    """Load a script for direct testing, registering it before exec."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_sft():
    return load_script(TRAIN_SFT, "mvp_train_sft")


@pytest.fixture(scope="module")
def train_grpo():
    return load_script(TRAIN_GRPO, "mvp_train_grpo")


@pytest.fixture(scope="module")
def prepare_data():
    return load_script(PREPARE_DATA, "mvp_prepare_data")


def run_script(path: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), *argv], capture_output=True, text=True, check=False
    )


# --- train_sft ------------------------------------------------------------------


def test_train_sft_forwards_every_other_flag_untouched(train_sft, monkeypatch) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(train_sft.run_sft, "main", lambda argv: seen.append(argv) or 0)

    rc = train_sft.main(["--data", "d.parquet", "--set", "epochs=1"])

    assert rc == 0
    assert seen == [
        ["--config", str(train_sft.DEFAULT_CONFIG), "--data", "d.parquet", "--set", "epochs=1"]
    ]


def test_train_sft_dry_run_never_merges(train_sft, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(train_sft.run_sft, "main", lambda argv: 0)

    def explode(*args: object) -> Path:
        raise AssertionError("a dry-run must not merge")

    monkeypatch.setattr(train_sft, "merge_adapter", explode)

    rc = train_sft.main(["--merge-to", str(tmp_path / "merged"), "--dry-run"])

    assert rc == 0


def test_train_sft_stops_merging_when_training_failed(train_sft, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(train_sft.run_sft, "main", lambda argv: 2)

    def explode(*args: object) -> Path:
        raise AssertionError("a failed run has no adapter to merge")

    monkeypatch.setattr(train_sft, "merge_adapter", explode)

    assert train_sft.main(["--merge-to", str(tmp_path / "merged")]) == 2


def test_merge_refuses_an_unfinished_run_before_importing_torch(
    train_sft, tmp_path
) -> None:
    """The adapter's COMPLETE marker is checked first, so this fails without a GPU stack.

    Same shape as the real mirror config, pointed at a run directory that has no
    adapter: the point is that the failure names the marker, not a missing torch.
    """
    config = yaml.safe_load((ROOT / "configs" / "mvp" / "sft.yaml").read_text())
    config["output_dir"] = str(tmp_path / "run")
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(FileNotFoundError, match="COMPLETE"):
        train_sft.merge_adapter(config_path, tmp_path / "merged")


def test_train_sft_dry_run_resolves_the_mirror_config() -> None:
    completed = run_script(TRAIN_SFT, "--dry-run")

    assert completed.returncode == 0, completed.stderr
    resolved = json.loads(completed.stdout)["config"]
    mirror = yaml.safe_load((ROOT / "configs" / "mvp" / "sft.yaml").read_text())
    for key, value in mirror.items():
        assert resolved[key] == value, key


# --- train_grpo -----------------------------------------------------------------


def test_train_grpo_forwards_the_config_and_never_exports_on_dry_run(
    train_grpo, monkeypatch, tmp_path
) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(
        train_grpo.run_grpo_direct, "main", lambda argv: seen.append(argv) or 0
    )

    def explode(*args: object) -> Path:
        raise AssertionError("a dry-run must not export an adapter")

    monkeypatch.setattr(train_grpo, "export_adapter", explode)

    rc = train_grpo.main(["--export-adapter", str(tmp_path / "adapter"), "--dry-run"])

    assert rc == 0
    # The wrapper's own flags are consumed, not re-forwarded: downstream sees the
    # mirror config and a dry-run, and nothing else.
    assert seen == [["--config", str(train_grpo.DEFAULT_CONFIG), "--dry-run"]]


def test_train_grpo_picks_the_newest_checkpoint_by_step_number(train_grpo, tmp_path) -> None:
    """Parsed, not sorted as text: ``global_step_100`` precedes ``global_step_50``."""
    for step in (5, 50, 100):
        actor = tmp_path / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True)
        (actor / "model_world_size_1_rank_0.pt").touch()

    assert train_grpo.find_latest_checkpoint(tmp_path) == (
        tmp_path / "global_step_100" / "actor" / "model_world_size_1_rank_0.pt"
    )


def test_train_grpo_fails_loudly_when_no_checkpoint_was_saved(train_grpo, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="nothing to export"):
        train_grpo.find_latest_checkpoint(tmp_path)


def test_export_adapter_anchors_a_relative_output_dir_at_the_repo_root(
    train_grpo, tmp_path, monkeypatch
) -> None:
    """The config writes ``output_dir`` repo-relative; the export must look there.

    Anchored at the repository, not the caller's cwd -- otherwise the export
    silently searches a path that only exists if the launcher happened to run
    from the repo root.
    """
    config = yaml.safe_load((ROOT / "configs" / "mvp" / "grpo.yaml").read_text())
    config["output_dir"] = "runs/r0"
    config_path = tmp_path / "grpo.yaml"
    config_path.write_text(yaml.safe_dump(config))
    actor = tmp_path / "runs" / "r0" / "checkpoints" / "global_step_50" / "actor"
    actor.mkdir(parents=True)
    (actor / "model_world_size_1_rank_0.pt").touch()
    monkeypatch.setattr(train_grpo, "REPO", tmp_path)

    class Recorder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, argv: list[str], check: bool) -> object:
            assert check is False
            self.calls.append(argv)
            return type("Completed", (), {"returncode": 0})()

    recorder = Recorder()
    monkeypatch.setattr(train_grpo, "subprocess", recorder)

    out = train_grpo.export_adapter(config_path, tmp_path / "adapter")

    assert out == tmp_path / "adapter"
    command = recorder.calls[0]
    assert command[command.index("--checkpoint") + 1] == str(
        actor / "model_world_size_1_rank_0.pt"
    )
    assert command[command.index("--rank") + 1] == "16"
    assert command[command.index("--alpha") + 1] == "32"


def test_train_grpo_reads_the_lora_shape_out_of_the_config(train_grpo) -> None:
    config = train_grpo.run_grpo.load_config(train_grpo.DEFAULT_CONFIG)

    assert train_grpo.lora_rank_alpha(config) == (16, 32)


def test_train_grpo_dry_run_names_the_pinned_backend() -> None:
    completed = run_script(TRAIN_GRPO, "--dry-run")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["verl_agent_sha"] == VERL_AGENT_SHA
    assert payload["upstream_config_name"] == "hgpo_trainer"


# --- prepare_data ---------------------------------------------------------------


def write_fixture_pool(tmp_path: Path, ids: list[str]) -> tuple[Path, Path, str]:
    """A tiny parquet plus the manifest shape prepare_data checks it against."""
    pd = pytest.importorskip("pandas")
    parquet = tmp_path / "frozen_eval.parquet"
    pd.DataFrame({"task_id": ids}).to_parquet(parquet, index=False)
    file_hash = sha256_hex(parquet.read_bytes())
    manifest = {
        "splits": {
            "frozen_eval": {
                "name": "frozen_eval",
                "file": parquet.name,
                "count": len(ids),
                "task_ids_hash": sha256_hex("\n".join(ids).encode()),
                "file_hash": file_hash,
            }
        }
    }
    manifest_path = tmp_path / "v1.json"
    manifest_path.write_text(json.dumps(manifest))
    return parquet, manifest_path, sha256_hex("\n".join(ids).encode())


@pytest.fixture
def fixture_pool(tmp_path, prepare_data, monkeypatch):
    """A three-task pool, with the frozen count and pin shrunk to match it."""
    ids = ["omni_math:aaa", "omni_math:bbb", "omni_math:ccc"]
    parquet, manifest_path, ids_hash = write_fixture_pool(tmp_path, ids)
    monkeypatch.setattr(prepare_data, "FROZEN_TASK_COUNT", len(ids))
    monkeypatch.setattr(prepare_data.paired_protocols, "EXPECTED_TASK_IDS_SHA256", ids_hash)
    return ids, parquet, manifest_path


def test_prepare_data_writes_the_regenerated_list(fixture_pool, prepare_data, tmp_path, capsys) -> None:
    ids, parquet, manifest_path = fixture_pool
    out = tmp_path / "task_ids.txt"

    rc = prepare_data.main(
        ["--parquet", str(parquet), "--manifest", str(manifest_path), "--out", str(out), "--eval-only"]
    )

    assert rc == 0
    assert out.read_text() == "\n".join(ids) + "\n"
    assert "PREPARE OK" in capsys.readouterr().out


def test_prepare_data_is_idempotent(fixture_pool, prepare_data, tmp_path, capsys) -> None:
    _, parquet, manifest_path = fixture_pool
    out = tmp_path / "task_ids.txt"
    argv = ["--parquet", str(parquet), "--manifest", str(manifest_path), "--out", str(out), "--eval-only"]

    assert prepare_data.main(argv) == 0
    assert prepare_data.main(argv) == 0

    assert "unchanged" in capsys.readouterr().out


def test_prepare_data_refuses_to_overwrite_a_different_list(
    fixture_pool, prepare_data, tmp_path, capsys
) -> None:
    _, parquet, manifest_path = fixture_pool
    out = tmp_path / "task_ids.txt"
    out.write_text("omni_math:somebody-elses-list\n")
    argv = ["--parquet", str(parquet), "--manifest", str(manifest_path), "--out", str(out), "--eval-only"]

    assert prepare_data.main(argv) == 1
    assert "different content" in capsys.readouterr().err
    # Deliberate, when asked for: the regenerated list is the one under the pin.
    assert prepare_data.main([*argv, "--force"]) == 0


def test_prepare_data_fails_when_the_pool_drifted(fixture_pool, prepare_data, tmp_path, capsys) -> None:
    ids, parquet, manifest_path = fixture_pool
    pd = pytest.importorskip("pandas")
    pd.DataFrame({"task_id": list(reversed(ids))}).to_parquet(parquet, index=False)
    out = tmp_path / "task_ids.txt"

    rc = prepare_data.main(
        ["--parquet", str(parquet), "--manifest", str(manifest_path), "--out", str(out), "--eval-only"]
    )

    assert rc == 1
    assert "not the one the released arms were scored on" in capsys.readouterr().err


def test_prepare_data_fails_when_the_frozen_pin_does_not_match(
    fixture_pool, prepare_data, tmp_path, capsys, monkeypatch
) -> None:
    _, parquet, manifest_path = fixture_pool
    out = tmp_path / "task_ids.txt"
    argv = ["--parquet", str(parquet), "--manifest", str(manifest_path), "--out", str(out), "--eval-only"]

    assert prepare_data.main(argv) == 0  # the fixture pin agrees with the fixture pool

    # ... and now it does not: any edit to the frozen list lands here.
    monkeypatch.setattr(prepare_data.paired_protocols, "EXPECTED_TASK_IDS_SHA256", "0" * 64)
    assert prepare_data.main(argv) == 1

    assert "drifted" in capsys.readouterr().err


def test_prepare_data_eval_only_skips_the_sft_side(fixture_pool, prepare_data, tmp_path) -> None:
    _, parquet, manifest_path = fixture_pool
    rc = prepare_data.main(
        [
            "--parquet", str(parquet),
            "--manifest", str(manifest_path),
            "--out", str(tmp_path / "task_ids.txt"),
            "--sft-split-manifest", str(tmp_path / "missing.json"),
            "--eval-only",
        ]
    )

    assert rc == 0


def test_prepare_data_checks_the_sft_manifest_against_the_config_pin(
    fixture_pool, prepare_data, tmp_path, capsys
) -> None:
    pd = pytest.importorskip("pandas")
    _, parquet, manifest_path = fixture_pool
    splits = {}
    for name in ("train", "validation"):
        path = tmp_path / f"{name}.parquet"
        pd.DataFrame({"text": [name]}).to_parquet(path, index=False)
        splits[name] = {"count": 1, "file": str(path), "sha256": sha256_hex(path.read_bytes())}
    split_manifest = tmp_path / "sft_dp_v1_split.json"
    split_manifest.write_text(json.dumps({"total": 2, **splits}))
    config = yaml.safe_load((ROOT / "configs" / "mvp" / "sft.yaml").read_text())
    config["data_manifest_sha256"] = sha256_hex(split_manifest.read_bytes())
    config["output_dir"] = str(tmp_path / "run")
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(yaml.safe_dump(config))
    argv = [
        "--parquet", str(parquet),
        "--manifest", str(manifest_path),
        "--out", str(tmp_path / "task_ids.txt"),
        "--sft-split-manifest", str(split_manifest),
        "--sft-config", str(config_path),
    ]

    assert prepare_data.main(argv) == 0
    assert "sft data: train 1 / validation 1 rows" in capsys.readouterr().out

    # A config that pins a different manifest is a config that trains on
    # different data than it claims: refused, with both hashes named.
    config["data_manifest_sha256"] = "1" * 64
    config_path.write_text(yaml.safe_dump(config))
    assert prepare_data.main(argv) == 1
    assert "describe different datasets" in capsys.readouterr().err


@skip_without_frozen_set
def test_prepare_data_verifies_the_real_frozen_set(prepare_data, capsys) -> None:
    """The live check, against the artifacts on this host (and the real pin)."""
    rc = prepare_data.main(["--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "PREPARE OK" in out
    assert "id-hash 1fc257f26fbf80ab" in out
    assert "sft data: train 2649 / validation 295 rows" in out
