"""Contract tests for the arm-identity gate.

Why these exist: the gate decides whether a GPU run may start, so its own
failure modes are expensive in both directions. A false pass spends GPU hours
pairing arms that cannot be compared (the 2026-09-26 incident); a false fail
blocks an experiment that was fine. The asymmetry between "candidate records
more than the baseline" and "candidate omits what the baseline recorded" is
deliberate and is pinned here.
"""

from __future__ import annotations

import json
import subprocess
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / "scripts" / "eval" / "verify_arm_identity.py"


def load_gate():
    spec = spec_from_file_location("verify_arm_identity", GATE)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "mode": "all-tools",
        "model": "/models/qwen3-1.7b/snapshots/70d244cc",
        "generation": {"max_new_tokens": 1024, "temperature": 0.0},
        "task_count": 200,
        "shard_id": 0,
    }
    base.update(overrides)
    return base


def test_identical_identity_is_aligned() -> None:
    gate = load_gate()
    blocking, advisory = gate.compare_identities(_baseline(), _baseline())
    assert blocking == []
    assert advisory == []


def test_differing_identity_field_blocks() -> None:
    gate = load_gate()
    blocking, _ = gate.compare_identities(_baseline(), _baseline(model="/models/other"))
    assert any("model" in problem for problem in blocking)


def test_differing_decoding_config_blocks() -> None:
    """The 09-24 E2 arm differed from its pair only in max_new_tokens; the gate
    must catch a config difference nested inside a field, not just at top level."""
    gate = load_gate()
    candidate = _baseline(generation={"max_new_tokens": 2048, "temperature": 0.0})
    blocking, _ = gate.compare_identities(_baseline(), candidate)
    assert any("generation" in problem for problem in blocking)


def test_candidate_recording_extra_fields_is_advisory_not_blocking() -> None:
    """A new arm that records hashes the 09-26 arm never wrote down is
    strengthening the gate; refusing to run because of it would be backwards."""
    gate = load_gate()
    candidate = _baseline(prompt_source_sha256="a" * 64, verifier_source_sha256="b" * 64)
    blocking, advisory = gate.compare_identities(_baseline(), candidate)
    assert blocking == []
    assert len(advisory) == 2


def test_candidate_omitting_a_baseline_field_blocks() -> None:
    """The reverse direction is a regression in the evidence, so it blocks."""
    gate = load_gate()
    candidate = _baseline()
    del candidate["model"]
    blocking, _ = gate.compare_identities(_baseline(), candidate)
    assert any("model" in problem for problem in blocking)


def test_bookkeeping_fields_are_allowed_to_differ() -> None:
    gate = load_gate()
    candidate = _baseline(shard_id=2, task_count=200, adapter_sha256="c" * 64)
    blocking, advisory = gate.compare_identities(_baseline(), candidate)
    assert blocking == []
    assert advisory == []


def test_source_drift_is_empty_for_a_commit_against_itself() -> None:
    gate = load_gate()
    assert gate.source_drift(ROOT, "HEAD") == ""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return result.stdout


def _make_repo_with_drift(tmp_path: Path) -> tuple[Path, str]:
    """A self-contained repo whose evaluation path drifted after its first
    commit.

    Deliberately not built on this repository's own history: CI checks out with
    the default shallow depth, so any test naming a past commit would fail
    there for a reason that has nothing to do with the gate.
    """
    repo = tmp_path / "repo"
    target = repo / "src" / "adaptive_math" / "agent" / "loop.py"
    target.parent.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "gate@test.invalid")
    _git(repo, "config", "user.name", "gate")
    target.write_text("behaviour = 'old'\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "one")
    baseline_sha = _git(repo, "rev-parse", "HEAD").strip()
    target.write_text("behaviour = 'new'\n")
    _git(repo, "commit", "-qam", "two")
    return repo, baseline_sha


def _write_pair(tmp_path: Path, sha: str) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    payload = json.dumps({"mode": "all-tools", "git_sha": sha})
    baseline.write_text(payload)
    candidate.write_text(payload)
    return baseline, candidate


def test_main_fails_when_drift_is_unjustified(tmp_path: Path) -> None:
    gate = load_gate()
    repo, sha = _make_repo_with_drift(tmp_path)
    baseline, candidate = _write_pair(tmp_path, sha)

    rc = gate.main(
        ["--baseline", str(baseline), "--candidate", str(candidate), "--repo", str(repo)]
    )
    assert rc == 1


def test_main_passes_when_drift_is_justified(tmp_path: Path) -> None:
    gate = load_gate()
    repo, sha = _make_repo_with_drift(tmp_path)
    baseline, candidate = _write_pair(tmp_path, sha)
    justification = tmp_path / "justification.md"
    justification.write_text("Audited: the loop.py change is a rename.\n")

    rc = gate.main(
        [
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--repo",
            str(repo),
            "--justify",
            str(justification),
        ]
    )
    assert rc == 0


def test_main_fails_when_baseline_revision_is_absent(tmp_path: Path) -> None:
    """A shallow clone cannot prove alignment, so the gate must fail closed --
    neither crashing with a traceback nor waving the run through."""
    gate = load_gate()
    repo, _ = _make_repo_with_drift(tmp_path)
    baseline, candidate = _write_pair(tmp_path, "0" * 40)

    rc = gate.main(
        ["--baseline", str(baseline), "--candidate", str(candidate), "--repo", str(repo)]
    )
    assert rc == 1


def test_main_reports_usage_error_for_missing_input(tmp_path: Path) -> None:
    gate = load_gate()
    rc = gate.main(
        ["--baseline", str(tmp_path / "nope.json"), "--candidate", str(tmp_path / "nope2.json")]
    )
    assert rc == 2
