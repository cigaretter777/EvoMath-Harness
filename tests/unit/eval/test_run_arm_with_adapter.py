"""Tests for the adapter-driving wrapper.

Why these exist: the wrapper is the one new piece of the campaign that sits on the
GPU path. It reuses the frozen runner's parser and ``run`` on purpose -- the arm
that is gated and the arm that runs must be the same argv through the same code --
so what is left to verify is narrow and worth pinning exactly:

1. **The adapter reaches the loader.** A wrapper that parses ``--adapter``,
   records it in the identity, and then runs the arm without it would produce a
   perfectly gated base arm wearing the name of a fine-tuned one. That is the
   worst failure this file can catch, and it is caught here by recording the call
   the loader actually receives.
2. **The loader is restored.** The patch is a mutation of a class the rest of the
   process imports. Left behind, it would silently attach this arm's adapter to the
   next arm run in the same interpreter.
3. **The pre-flight runs before the model.** Cheap checks first: the sidecar
   identity must exist, on disk, before ``run`` is called.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "scripts" / "campaign-20260927" / "run_arm_with_adapter.py"
FROZEN_IDS = ROOT / "artifacts" / "eval" / "thesis_e0_base_direct_b1" / "task_ids.txt"
EVAL_PARQUET = ROOT / "data" / "processed" / "v1" / "frozen_eval.parquet"
SFT_ADAPTER = ROOT / "artifacts" / "sft" / "qwen3_1_7b_sft_dp_v1" / "adapter"
SFT_DIRECT_MANIFEST = (
    ROOT / "artifacts" / "eval" / "thesis_e0_base_direct_b1" / "eval_manifest.json"
)
BASE_SNAPSHOT = Path(
    "/root/autodl-tmp/hf-cache/models--Qwen--Qwen3-1.7B/snapshots/"
    "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)
SANDBOX_URL = "http://localhost:8080"

HAVE_LOCAL_ARTIFACTS = (
    EVAL_PARQUET.is_file()
    and FROZEN_IDS.is_file()
    and SFT_ADAPTER.is_dir()
    and BASE_SNAPSHOT.is_dir()
)
skip_without_local_artifacts = pytest.mark.skipif(
    not HAVE_LOCAL_ARTIFACTS,
    reason="the frozen parquet, its task list, the base snapshot and the adapter are local",
)


@pytest.fixture(scope="module")
def wrapper():
    """The wrapper, imported once.

    Importing it executes the two ``load_script`` calls at its top, so this also
    asserts that the wrapper really does build itself out of the emitter and the
    runner rather than carrying its own copy of either.
    """
    spec = spec_from_file_location("run_arm_with_adapter_under_test", WRAPPER)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def sandbox_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The emitter records this at emit time, and the stored arm's spelling is the
    one that pairs with it. Set for every test here so no test can pass on the
    runner's fallback spelling by accident."""
    monkeypatch.setenv("ADAPTIVE_MATH_SANDBOX_URL", SANDBOX_URL)


def run_argv(out_dir: Path, *, adapter: Path | None = SFT_ADAPTER, kind: str | None = "sft"):
    argv = [
        "--mode", "all-tools",
        "--shard-id", "0",
        "--shard-count", "3",
        "--data", str(EVAL_PARQUET),
        "--task-ids-file", str(FROZEN_IDS),
        "--model", str(BASE_SNAPSHOT),
        "--agent-config", str(ROOT / "configs" / "agent" / "default.yaml"),
        "--reward-config", str(ROOT / "configs" / "reward" / "r0.yaml"),
        "--output-dir", str(out_dir),
    ]
    if adapter is not None:
        argv += ["--adapter", str(adapter)]
    if kind is not None:
        argv += ["--adapter-kind", kind]
    return argv


class RecordingClient:
    """Stands in for the loader, recording what it was asked to load.

    Deliberately a whole class rather than a patched function: the wrapper reads
    the original out of the class ``__dict__`` and puts it back there, so the test
    has to exercise a real classmethod to be testing the code that runs.
    """

    calls: ClassVar[list[tuple[str, str | None]]] = []

    @classmethod
    def from_pretrained(
        cls, model_id: str, *, device: str = "auto", dtype: str = "auto", adapter: str | None = None
    ) -> str:
        RecordingClient.calls.append((model_id, adapter))
        return "client"


@skip_without_local_artifacts
def test_a_run_without_an_adapter_is_a_usage_error(wrapper, tmp_path: Path) -> None:
    """The base arm is the plain runner's job. Accepting it here would create a
    second spelling of an arm that already exists, with an extra layer in front of
    it that nobody asked for."""
    with pytest.raises(SystemExit) as excinfo:
        wrapper.main(run_argv(tmp_path / "run", adapter=None, kind=None))
    assert excinfo.value.code == 2


@skip_without_local_artifacts
def test_an_adapter_without_a_kind_is_refused_by_the_pre_flight(
    wrapper, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The kind decides whether provenance is required, so an unstated kind is not
    a detail -- it would let an RL adapter through without its provenance check."""
    out_dir = tmp_path / "run"
    assert wrapper.main(run_argv(out_dir, adapter=SFT_ADAPTER, kind=None)) == 1
    assert "--adapter-kind" in capsys.readouterr().err
    assert not out_dir.exists()


@skip_without_local_artifacts
def test_a_dry_run_writes_the_sidecar_and_starts_nothing(
    wrapper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-flight in one command. The sidecar lands beside the run directory
    -- inside it, the runner would refuse to start -- and the run directory is
    never created, because a dry run must leave nothing behind but the identity."""
    out_dir = tmp_path / "run"
    argv = [*run_argv(out_dir), "--dry-run"]

    def explode(args: argparse.Namespace) -> None:
        raise AssertionError("a dry run must not reach the runner")

    monkeypatch.setattr(wrapper.rule_baseline, "run", explode)
    assert wrapper.main(argv) == 0

    sidecar = out_dir.with_suffix(".identity.json")
    identity = json.loads(sidecar.read_text())
    assert identity["adapter_sha256"] == json.loads(
        SFT_DIRECT_MANIFEST.read_text()
    )["adapter_sha256"]
    assert identity["model"] == str(BASE_SNAPSHOT)
    assert not out_dir.exists()


@skip_without_local_artifacts
def test_the_adapter_reaches_the_loader_and_the_loader_is_restored(
    wrapper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this file exists for, and its most expensive cousin.

    Reached: the loader is called with ``adapter=`` -- asserted on the call the
    loader *receives*, not on the flag the wrapper parsed, because the two are
    only the same thing if the patch is wired correctly.
    Restored: the class is left exactly as it was found, by identity, so the next
    arm in this interpreter cannot inherit this arm's weights.
    """
    RecordingClient.calls = []
    monkeypatch.setattr(wrapper.rule_baseline, "TransformersModelClient", RecordingClient)
    before = RecordingClient.__dict__["from_pretrained"]

    async def fake_run(args: argparse.Namespace) -> None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        RecordingClient.from_pretrained(str(args.model))

    monkeypatch.setattr(wrapper.rule_baseline, "run", fake_run)
    assert wrapper.main(run_argv(tmp_path / "run")) == 0

    assert RecordingClient.calls == [(str(BASE_SNAPSHOT), str(SFT_ADAPTER))]
    assert RecordingClient.__dict__["from_pretrained"] is before


@skip_without_local_artifacts
def test_the_identity_is_written_before_the_run_and_copied_in_after(
    wrapper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordering, asserted from inside the run.

    The sidecar must already be on disk when ``run`` is entered -- that is the
    whole claim of a pre-flight -- and the copy into the artifact directory may
    only happen afterwards, because the runner refuses to start unless that
    directory is empty.
    """
    out_dir = tmp_path / "run"
    sidecar = out_dir.with_suffix(".identity.json")
    seen: dict[str, object] = {}

    async def fake_run(args: argparse.Namespace) -> None:
        seen["sidecar_existed"] = sidecar.is_file()
        seen["run_dir_when_entered"] = Path(args.output_dir).exists()
        seen["mode"] = args.mode
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        seen["copied_in_during_run"] = (Path(args.output_dir) / "identity.json").exists()

    monkeypatch.setattr(wrapper.rule_baseline, "run", fake_run)
    assert wrapper.main(run_argv(out_dir)) == 0

    assert seen["sidecar_existed"] is True
    assert seen["run_dir_when_entered"] is False
    assert seen["copied_in_during_run"] is False
    assert seen["mode"] == "all-tools"
    assert (out_dir / "identity.json").read_text() == sidecar.read_text()


@skip_without_local_artifacts
def test_the_wrapper_borrows_the_runners_parser_and_run(wrapper) -> None:
    """Identity of the objects, not equality of their behaviour.

    The wrapper's argv is parsed by the emitter's parser, which is the runner's
    parser plus the two adapter flags, and the run is the runner's own coroutine.
    A copy of either would be a second implementation of "the same configuration",
    and a pre-flight is only worth running if it gates the code that runs.
    """
    runner = wrapper.rule_baseline
    assert wrapper.emitter.runner is runner
    assert wrapper.build_parser is wrapper.emitter.build_parser
    assert wrapper.emit is wrapper.emitter.emit
    assert runner.run.__module__ == "rule_baseline"
