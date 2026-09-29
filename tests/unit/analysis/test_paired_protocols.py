"""Tests for the per-protocol paired comparisons.

Why these exist: a paired test is only paired if both sides are scored on the
same tasks, and the two ways that quietly stops being true are the ones this
file pins down. An arm that covered a subset of the frozen set must be compared
on the overlap *and* have the excluded tasks reported, never silently narrowed
to a flattering denominator; and two arms whose task sets cross must stop the
script rather than be reduced to whatever they happen to share. The McNemar
values themselves come from the repo's own paired test, so they are checked here
only for the direction and exactness the claim depends on.
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from adaptive_math.core.hashing import sha256_hex

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "analysis" / "paired_protocols.py"


def load_script():
    spec = spec_from_file_location("paired_protocols", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def paired():
    return load_script()


def _arm(task_ids: list[str], correct: dict[str, list[str]]) -> dict:
    return {
        "n": len(task_ids),
        "scored_task_ids": sorted(task_ids),
        "correct_task_ids": correct,
    }


def test_a_pair_is_scored_on_the_tasks_both_arms_covered(paired):
    row = paired.paired_row("strict", {"t1"}, {"t2"}, ["t1", "t2", "t3"])
    assert row["n"] == 3
    assert row["base_correct"] == 1
    assert row["arm_correct"] == 1
    assert row["accuracy_delta"] == 0.0
    assert (row["improved"], row["regressed"]) == (1, 1)
    # Exact McNemar over two discordant pairs: 2 * (C(2,0) + C(2,1)) / 2**2,
    # capped at 1 -- a wash is a wash, communicated as p = 1 rather than > 1.
    assert row["mcnemar_pvalue"] == 1.0


def test_a_one_sided_pair_reproduces_the_claim_arithmetic(paired):
    # Three tasks the base arm never read, the arm read all three: the exact
    # McNemar two-sided p is 2 / 2**3 = 0.25, which is the arithmetic behind the
    # published 0.0002 for 13/13 (2 / 2**13).
    row = paired.paired_row("strict", set(), {"t1", "t2", "t3"}, ["t1", "t2", "t3"])
    assert row["accuracy_delta"] == 1.0
    assert (row["improved"], row["regressed"]) == (3, 0)
    assert row["mcnemar_pvalue"] == 0.25

    thirteen = paired.paired_row(
        "strict", set(), {f"t{i}" for i in range(13)}, [f"t{i}" for i in range(13)]
    )
    assert thirteen["mcnemar_pvalue"] == pytest.approx(0.0002, abs=5e-5)


def test_a_narrower_arm_is_compared_on_the_overlap_and_says_what_it_dropped(paired):
    base = _arm(
        ["t1", "t2", "t3", "t4"],
        {"strict": ["t1", "t3"], "coerce": ["t1", "t3"], "lenient": ["t1", "t3", "t4"]},
    )
    arm = _arm(["t1", "t2"], {"strict": ["t1"], "coerce": ["t1", "t2"], "lenient": ["t1", "t2"]})
    data = paired.compare(base, arm, "base", "narrow")
    assert data["excluded_tasks"] == 2
    # The two tasks only the base arm was scored on held one base-correct answer
    # at strict and one more that only the lenient rung reads: both are
    # reported, so a comparison on the overlap cannot be mistaken for a
    # comparison on the whole frozen set.
    assert data["excluded_base_correct"] == {"strict": 1, "coerce": 1, "lenient": 2}
    assert data["protocols"]["strict"]["n"] == 2
    assert data["protocols"]["strict"]["base_correct"] == 1


def test_crossing_task_sets_are_fatal(paired):
    base = _arm(
        ["t1", "t2"],
        {"strict": [], "coerce": [], "lenient": []},
    )
    arm = _arm(["t2", "t3"], {"strict": [], "coerce": [], "lenient": []})
    with pytest.raises(RuntimeError, match="crossing task sets"):
        paired.compare(base, arm, "base", "arm")


def test_disjoint_task_sets_are_fatal(paired):
    base = _arm(["t1"], {"strict": [], "coerce": [], "lenient": []})
    arm = _arm(["t2"], {"strict": [], "coerce": [], "lenient": []})
    with pytest.raises(RuntimeError, match="share no scored tasks"):
        paired.compare(base, arm, "base", "arm")


def test_the_artifact_and_the_task_list_must_agree(paired, tmp_path, monkeypatch):
    # A rescore artifact scored against a different task set cannot have its
    # task ids paired against this one, however similar the arm names look.
    # The frozen list is a local eval artifact, absent in CI, so this test
    # brings its own list and pins the expected hash to it; the real constant
    # is checked in test_the_constant_is_the_frozen_task_set, where the frozen
    # list exists.
    task_ids = tmp_path / "task_ids.txt"
    task_ids.write_text("t1\nt2\n")
    monkeypatch.setattr(paired, "EXPECTED_TASK_IDS_SHA256", sha256_hex(b"t1\nt2"))
    artifact = tmp_path / "rescore.json"
    artifact.write_text(json.dumps({"task_ids_sha256": "0" * 64, "arms": {}}))
    with pytest.raises(RuntimeError, match="was scored against task set"):
        paired.main(
            [
                "--from-json",
                str(artifact),
                "--task-ids",
                str(task_ids),
                "--base",
                "base_direct",
                "--against",
                "sft_direct",
                "--out",
                str(tmp_path / "out.json"),
            ]
        )


@pytest.mark.skipif(
    not (ROOT / "artifacts/eval/thesis_e0_base_direct_b1/task_ids.txt").exists(),
    reason="the frozen task list is a local eval artifact, not tracked",
)
def test_the_constant_is_the_frozen_task_set(paired):
    ids = paired.DEFAULT_TASK_IDS.read_text().splitlines()
    assert sha256_hex("\n".join(ids).encode()) == paired.EXPECTED_TASK_IDS_SHA256
