"""Tests for the agent-arm identity emitter.

Why these exist: this file is what stands between a GPU booking and the
2026-09-26 failure repeating -- two arms that cannot be paired, discovered after
the hours are spent. Three properties carry that weight, and each is here
because it can break silently:

1. **The runner is frozen.** ``rule_baseline.py`` hashes its own committed blob
   into every manifest as ``router_source_sha256``, a blocking field, and the
   stored arm pins that hash. One edit to that file -- even a new flag -- and
   every future agent arm records a different hash and stops being pairable with
   the stored one. ``test_the_agent_runner_is_still_the_frozen_blob`` is the
   alarm; nothing else in the repository would notice.
2. **The emitter is not a second implementation.** It imports the runner's
   loader, router hash and git hash. A copy would drift, and a drift here is
   invisible until the gate blocks a correct run.
3. **The emitter touches nothing expensive.** It is a pre-flight, so it must be
   provably model-free and sandbox-free: one that loads 3 GB of weights before
   telling you the config is wrong is not a pre-flight.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from adaptive_math.core.hashing import sha256_hex

ROOT = Path(__file__).resolve().parents[3]
EMITTER = ROOT / "scripts" / "eval" / "emit_arm_identity.py"
GATE = ROOT / "scripts" / "eval" / "verify_arm_identity.py"
RUNNER = ROOT / "scripts" / "campaign-20260926" / "rule_baseline.py"

FROZEN_IDS = ROOT / "artifacts" / "eval" / "thesis_e0_base_direct_b1" / "task_ids.txt"
EVAL_PARQUET = ROOT / "data" / "processed" / "v1" / "frozen_eval.parquet"
BASE_SNAPSHOT = Path(
    "/root/autodl-tmp/hf-cache/models--Qwen--Qwen3-1.7B/snapshots/"
    "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)
MERGED_SFT = ROOT / "artifacts" / "models" / "qwen3_1_7b_sft_dp_v1_merged"
SFT_ADAPTER = ROOT / "artifacts" / "sft" / "qwen3_1_7b_sft_dp_v1" / "adapter"
R0_ADAPTER = ROOT / "artifacts" / "runs" / "grpo_qwen3_1_7b_r0" / "r0_adapter"
# The direct arms the agent arms must be two views of: same weights, same bytes.
SFT_DIRECT_MANIFEST = (
    ROOT / "artifacts" / "eval" / "thesis_e0_base_direct_b1" / "eval_manifest.json"
)
R0_DIRECT_MANIFEST = ROOT / "artifacts" / "eval" / "r0_omnimath_200" / "eval_manifest.json"
STORED_MANIFEST = (
    ROOT / "artifacts" / "rollout_health" / "thesis_e0_base_tool.shard0" / "manifest.json"
)
SANDBOX_URL = "http://localhost:8080"

HAVE_FROZEN_SET = FROZEN_IDS.is_file() and EVAL_PARQUET.is_file() and BASE_SNAPSHOT.is_dir()
skip_without_frozen_set = pytest.mark.skipif(
    not HAVE_FROZEN_SET,
    reason="the frozen parquet, its task list and the base snapshot are local artifacts",
)
skip_without_stored_arm = pytest.mark.skipif(
    not (STORED_MANIFEST.is_file() and HAVE_FROZEN_SET),
    reason="the stored base+tool arm is a local rollout artifact, not tracked",
)
skip_without_adapters = pytest.mark.skipif(
    not (SFT_ADAPTER.is_dir() and R0_ADAPTER.is_dir() and HAVE_FROZEN_SET),
    reason="the SFT and r0 adapters are local training artifacts, not tracked",
)


def load_script(path: Path, name: str):
    """Load a script for direct testing, registering it before exec."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def sandbox_url(value: str) -> Iterator[None]:
    """The emitter reads this at emit time; a leaked value would corrupt the
    next test's identity, and ``monkeypatch`` is function-scoped where the
    expensive fixture is not."""
    key = "ADAPTIVE_MATH_SANDBOX_URL"
    previous = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@pytest.fixture(scope="module")
def emitter():
    return load_script(EMITTER, "emit_arm_identity_under_test")


@pytest.fixture(scope="module")
def gate():
    return load_script(GATE, "verify_arm_identity_for_emitter_tests")


def run_argv(
    model: Path,
    out: Path,
    tmp_path: Path,
    *,
    adapter: Path | None = None,
    kind: str | None = None,
) -> list[str]:
    """The run's argv, with absolute paths, as the emitter requires."""
    argv = [
        "--mode", "all-tools",
        "--shard-id", "0",
        "--shard-count", "3",
        "--data", str(EVAL_PARQUET),
        "--task-ids-file", str(FROZEN_IDS),
        "--model", str(model),
        "--agent-config", str(ROOT / "configs" / "agent" / "default.yaml"),
        "--reward-config", str(ROOT / "configs" / "reward" / "r0.yaml"),
        "--output-dir", str(tmp_path / "run"),
        "--out", str(out),
    ]
    if adapter is not None:
        argv += ["--adapter", str(adapter), "--adapter-kind", str(kind)]
    return argv


def weights_notes(advisory: list[str]) -> list[str]:
    """Advice about weights, which is how the gate marks a field that is expected
    to differ between a pair rather than a mismatch."""
    return [note for note in advisory if "(weights:" in note]


@skip_without_stored_arm
def test_the_agent_runner_is_still_the_frozen_blob() -> None:
    """The pairability invariant, and nothing else checks it.

    ``router_source_sha256`` is ``git hash-object`` of the runner itself, so the
    stored arm is pairable only while that file is byte-identical to the blob it
    recorded. Any edit -- a refactor, a new flag, a comment -- breaks the pair,
    and this is the only place that says so out loud.
    """
    stored = json.loads(STORED_MANIFEST.read_text())
    current = subprocess.run(
        ["git", "hash-object", str(RUNNER)], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert current == stored["router_source_sha256"], (
        "scripts/campaign-20260926/rule_baseline.py changed since it produced the "
        "stored base+tool arm. Its blob hash is recorded in that arm's manifest as "
        "router_source_sha256, a blocking identity field, so new agent arms can no "
        "longer be paired with it. Either revert the file or re-run the base arm."
    )


def test_the_emitter_borrows_the_runners_functions(emitter) -> None:
    """Identity of the function objects, not equality of their behaviour: a copy
    would pass an equality check and still drift.

    Compared against the module the emitter itself registered, so this also pins
    that its loader registers -- an unregistered re-execution would produce a
    second, unrelated ``rule_baseline`` whose functions are equal but not the
    same, which is exactly the drift this test exists to catch.
    """
    runner = sys.modules["rule_baseline"]
    assert emitter.load_tasks is runner.load_tasks
    assert emitter.route is runner.route
    assert emitter.git_sha is runner.git_sha
    assert emitter.router_source_sha256 is runner.router_source_sha256


def test_a_relative_recorded_path_is_refused(emitter) -> None:
    """The gate compares path strings, so a relative spelling blocks a pair that
    an absolute one passes. Refused here, where the fix is one re-invocation."""
    with pytest.raises(ValueError, match="must be absolute"):
        emitter.recorded_path("data/processed/v1/frozen_eval.parquet", "--data")
    assert emitter.recorded_path("/abs/frozen_eval.parquet", "--data") == "/abs/frozen_eval.parquet"


def test_a_task_set_that_is_not_the_frozen_one_is_refused(emitter) -> None:
    """Three ways an arm stops being comparable without anyone noticing: a short
    list, a duplicate, and a different list of the right size."""
    frozen = emitter.read_ids(FROZEN_IDS) if FROZEN_IDS.is_file() else [f"t{i}" for i in range(200)]
    with pytest.raises(ValueError, match="not the frozen list"):
        emitter.check_frozen(frozen[:-1], frozen)
    with pytest.raises(ValueError, match="duplicate task ids"):
        emitter.check_frozen([*frozen[:-1], frozen[0]], frozen)
    with pytest.raises(ValueError, match="not the frozen list"):
        emitter.check_frozen([f"other{i}" for i in range(200)], frozen)


@skip_without_frozen_set
def test_the_frozen_list_hashes_to_the_constant(emitter) -> None:
    """The constant the re-scoring scripts share; one definition, checked here so
    a change to the list is a test failure rather than a quiet denominator shift."""
    ids = emitter.read_ids(FROZEN_IDS)
    assert len(ids) == emitter.FROZEN_TASK_COUNT
    assert sha256_hex("\n".join(ids).encode()) == emitter.EXPECTED_TASK_IDS_SHA256


@skip_without_frozen_set
def test_a_shard_id_without_a_count_is_refused(emitter, tmp_path: Path) -> None:
    """Parity with the runner, which refuses the same argv."""
    rc = emitter.main(
        [
            "--mode", "all-tools",
            "--shard-id", "0",
            "--model", str(BASE_SNAPSHOT),
            "--output-dir", str(tmp_path / "run"),
        ]
    )
    assert rc == 1


@pytest.fixture(scope="module")
def emitted(emitter, tmp_path_factory) -> tuple[dict, list[str]]:
    """Run the emitter once, with the model loader and the sandbox rigged to fail.

    The frozen parquet is loaded and the runner imported behind this fixture, so
    it is module-scoped. The stubs make the second and third properties above
    unfalsifiable: if either were reached, the exception escapes ``main`` and the
    test fails loudly rather than passing on a technicality.
    """
    tmp_path = tmp_path_factory.mktemp("emit")
    out = tmp_path / "identity.json"
    reached: list[str] = []
    runner = load_script(RUNNER, "rule_baseline_for_emit_run")

    class Boom:
        def __init__(self, *args: object, **kwargs: object) -> None:
            reached.append("expensive")
            raise AssertionError("the emitter must not load the model or the sandbox")

    original_model = runner.TransformersModelClient
    original_sandbox = runner.SandboxFusionClient
    runner.TransformersModelClient = Boom  # type: ignore[assignment]
    runner.SandboxFusionClient = Boom  # type: ignore[assignment]
    try:
        with sandbox_url(SANDBOX_URL):
            rc = emitter.main(run_argv(BASE_SNAPSHOT, out, tmp_path))
    finally:
        runner.TransformersModelClient = original_model  # type: ignore[assignment]
        runner.SandboxFusionClient = original_sandbox  # type: ignore[assignment]

    assert rc == 0
    return json.loads(out.read_text()), reached


@skip_without_stored_arm
def test_the_emitted_identity_passes_the_gate_against_the_stored_arm(gate, emitted) -> None:
    """The end-to-end pre-flight: the same gate the GPU run must pass, executed
    on a CPU before anything is booked."""
    identity, _ = emitted
    stored = json.loads(STORED_MANIFEST.read_text())
    blocking, _ = gate.compare_identities(stored, identity)
    assert blocking == []


@skip_without_frozen_set
def test_the_emitter_never_reaches_the_model_or_the_sandbox(emitted) -> None:
    """Proved by construction in the fixture: both classes were replaced with
    raising stubs and the run still exited 0."""
    _, reached = emitted
    assert reached == []


@skip_without_adapters
def test_the_sft_arm_gates_clean_and_names_its_weights(emitter, gate, tmp_path: Path) -> None:
    """Arm B: the base snapshot plus the SFT adapter, which is the pair for the
    stored base+tool arm.

    The ``model`` line is identical to the stored arm's, so the *only* thing that
    separates this arm from it is the adapter -- the merge the direct ``sft_direct``
    arm never had is absent from the build path here too, and the contrast carries
    the channel and nothing else. The adapter must still be reported: weights are
    exempt from comparison, never from being stated.
    """
    out = tmp_path / "sft.json"
    with sandbox_url(SANDBOX_URL):
        rc = emitter.main(run_argv(BASE_SNAPSHOT, out, tmp_path, adapter=SFT_ADAPTER, kind="sft"))
    assert rc == 0
    candidate = json.loads(out.read_text())
    stored = json.loads(STORED_MANIFEST.read_text())

    blocking, advisory = gate.compare_identities(stored, candidate)
    assert blocking == []
    assert candidate["model"] == str(BASE_SNAPSHOT)
    notes = weights_notes(advisory)
    assert len(notes) == len(advisory) == 4
    assert {note.split(":")[0] for note in notes} == {
        "adapter",
        "adapter_path",
        "adapter_sha256",
        "adapter_kind",
    }
    assert any(candidate["adapter_sha256"] in note for note in notes)


@skip_without_adapters
def test_the_r0_arm_gates_clean_against_the_same_stored_arm(emitter, gate, tmp_path: Path) -> None:
    """Arm C: merged SFT plus the r0 adapter, the pair for ``r0_direct``.

    One weights note more than arm B -- ``model`` differs, because this arm's base
    is the merged checkpoint the r0 adapter was trained on. All five are weights
    notes, which is the whole reason the gate reports those fields instead of
    comparing them: this arm could not otherwise be paired with the stored one.
    """
    out = tmp_path / "r0.json"
    with sandbox_url(SANDBOX_URL):
        rc = emitter.main(
            run_argv(MERGED_SFT, out, tmp_path, adapter=R0_ADAPTER, kind="rl")
        )
    assert rc == 0
    candidate = json.loads(out.read_text())
    stored = json.loads(STORED_MANIFEST.read_text())

    blocking, advisory = gate.compare_identities(stored, candidate)
    assert blocking == []
    assert candidate["model"] == str(MERGED_SFT)
    notes = weights_notes(advisory)
    assert len(notes) == 5
    assert len([note for note in notes if note.startswith("model:")]) == 1
    assert len(advisory) == 5


@skip_without_adapters
def test_the_recorded_adapter_is_the_one_the_direct_arm_ran(emitter) -> None:
    """P4, mechanically: the agent arm's adapter and the direct arm's adapter must
    be the same bytes, or the pair measures a different model than it claims.

    The hash comes from the emitter's own code path -- the one the run's identity
    is built with -- and is compared against what ``run_model_eval.py`` recorded
    for the direct arms. Nothing else in the repository connects the two, and a
    re-trained adapter at either path would otherwise pass every other test here.
    """
    sft = emitter.adapter_identity(SFT_ADAPTER, "sft")
    r0 = emitter.adapter_identity(R0_ADAPTER, "rl")
    assert sft["adapter_sha256"] == json.loads(SFT_DIRECT_MANIFEST.read_text())["adapter_sha256"]
    assert r0["adapter_sha256"] == json.loads(R0_DIRECT_MANIFEST.read_text())["adapter_sha256"]
    assert sft["adapter_kind"] == "sft"
    assert r0["adapter_kind"] == "rl"
    assert r0["adapter_sha256"] != sft["adapter_sha256"]


def test_a_relative_adapter_path_is_refused(emitter) -> None:
    """Same trap as the other recorded paths, with a sharper consequence: this
    one names the weights, so a relative spelling is a weak claim about what ran."""
    with pytest.raises(ValueError, match="must be absolute"):
        emitter.adapter_identity(Path("artifacts/sft/qwen3_1_7b_sft_dp_v1/adapter"), "sft")


def test_an_rl_adapter_without_provenance_is_refused(emitter, tmp_path: Path) -> None:
    """The direct path requires it, so this must too: an RL adapter whose training
    provenance is missing is not evidence, and the agent arm is where it would go
    unnoticed -- the run itself never reads that file."""
    fake = tmp_path / "adapter"
    fake.mkdir()
    for name in ("COMPLETE", "adapter_config.json", "adapter_model.safetensors"):
        (fake / name).write_text("")
    emitter.adapter_identity(fake, "sft")  # fine for sft
    with pytest.raises(FileNotFoundError, match="rl_provenance.json"):
        emitter.adapter_identity(fake, "rl")
    with pytest.raises(ValueError, match="adapter-kind"):
        emitter.adapter_identity(fake, "grpo")
