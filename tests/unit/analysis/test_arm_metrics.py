"""Tests for the six-arm metric table.

Why these exist: the table's job is to be checkable. Three properties carry it
-- the metrics are computed from the stored records with fixed definitions, a
composed arm counts only the tasks its routing file says it reused, and a
mismatch against the run's own recorded counts is fatal rather than a footnote.
All three are pinned here on synthetic arms, because the real arms are large and
would only tell us that a number moved, not which definition moved it.
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "analysis" / "arm_metrics.py"


def load_script():
    spec = spec_from_file_location("arm_metrics", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def metrics():
    return load_script()


def _trajectory(
    task_id: str, *, status: str, tool_calls: int, steps: int, invalid: int, final: str | None
) -> dict:
    return {
        "task_id": task_id,
        "verdict": {"status": status},
        "trajectory": {
            "usage": {
                "steps": steps,
                "tool_calls": tool_calls,
                "invalid_actions": invalid,
                "generated_tokens": 100,
                "final_answer": final,
            },
            "final_answer": final,
            "termination_reason": "final" if final else "max_steps",
        },
    }


def _prediction(task_id: str, *, verifier: str, extract: str, output_tokens: int = 500) -> dict:
    return {
        "task_id": task_id,
        "verifier_status": verifier,
        "extract_status": extract,
        "output_tokens": output_tokens,
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_agent_arm_metrics_come_from_the_usage_blocks(metrics, tmp_path):
    traj = _write(
        tmp_path / "trajectories.jsonl",
        [
            _trajectory("t1", status="correct", tool_calls=2, steps=4, invalid=1, final="7"),
            _trajectory("t2", status="timeout", tool_calls=0, steps=6, invalid=6, final=None),
        ],
    )
    arm = metrics.compute_arm(
        metrics.ArmSpec(
            "agent",
            "agent",
            expected_n=2,
            expected_correct=1,
            expected_verifier_valid=1,
            expected_tool_calls=2,
            expected_invalid_actions=7,
            expected_generated_tokens=200,
            trajectories=traj,
        )
    )
    assert arm["n"] == 2
    assert arm["accuracy"] == 0.5
    assert arm["final_action_rate"] == 0.5
    # The second trajectory ended without a final answer, so no verdict came of
    # it even though the verifier status parses as a status.
    assert arm["verifier_addressable_rate"] == 0.5
    assert arm["tool_calls_per_task"] == 1.0
    assert arm["steps_mean"] == 5.0
    assert arm["invalid_actions_per_task"] == 3.5
    # How the loop stopped, which the step count alone cannot say.
    assert arm["termination_reasons"] == {"final": 1, "max_steps": 1}


def test_direct_arm_has_no_tool_channel_to_measure(metrics, tmp_path):
    preds = _write(
        tmp_path / "predictions.jsonl",
        [
            _prediction("t1", verifier="correct", extract="ok", output_tokens=600),
            _prediction("t2", verifier="timeout", extract="ok", output_tokens=400),
            _prediction("t3", verifier="invalid_prediction", extract="missing"),
        ],
    )
    arm = metrics.compute_arm(
        metrics.ArmSpec(
            "direct",
            "direct",
            expected_n=3,
            expected_correct=1,
            expected_verifier_valid=1,
            expected_tool_calls=0,
            expected_mean_output_tokens=500.0,
            predictions=preds,
        )
    )
    assert arm["accuracy"] == pytest.approx(1 / 3)
    # t2 extracted an answer and then hit a verifier-side status: it counts as a
    # legal final action but not as a verifier-addressable answer.
    assert arm["final_action_rate"] == pytest.approx(2 / 3)
    assert arm["verifier_addressable_rate"] == pytest.approx(1 / 3)
    assert arm["tool_calls_total"] == 0
    assert arm["steps_mean"] == 1.0
    # Not measured, rather than measured as zero.
    assert arm["invalid_actions_per_task"] is None


def test_composed_arm_counts_only_the_tasks_the_routing_file_names(metrics, tmp_path):
    traj = _write(
        tmp_path / "trajectories.jsonl",
        [_trajectory("t1", status="correct", tool_calls=1, steps=3, invalid=0, final="7")],
    )
    pool = _write(
        tmp_path / "base_predictions.jsonl",
        [
            _prediction("t2", verifier="invalid_prediction", extract="missing"),
            _prediction("t3", verifier="invalid_prediction", extract="missing"),
        ],
    )
    routing = _write(
        tmp_path / "routing.jsonl",
        [{"task_id": "t1", "channel": "sympy"}, {"task_id": "t2", "channel": "direct"}],
    )
    arm = metrics.compute_arm(
        metrics.ArmSpec(
            "composed",
            "agent",
            expected_n=2,
            expected_correct=1,
            expected_verifier_valid=1,
            expected_rolled_out=1,
            expected_direct_reused=1,
            expected_tool_calls=1,
            trajectories=traj,
            direct_pool=pool,
            routing=routing,
        )
    )
    # t3 sits in the pool but was rolled out, so adding it would push the arm
    # past the task set it was actually run on.
    assert (arm["n"], arm["rolled_out"], arm["direct_reused"]) == (2, 1, 1)
    assert arm["steps_mean"] == pytest.approx(2.0)
    assert arm["final_action_rate"] == 0.5


def test_a_pool_that_lost_a_routed_task_is_fatal(metrics, tmp_path):
    traj = _write(
        tmp_path / "trajectories.jsonl",
        [_trajectory("t1", status="correct", tool_calls=0, steps=1, invalid=0, final="7")],
    )
    pool = _write(tmp_path / "pool.jsonl", [_prediction("t2", verifier="correct", extract="ok")])
    routing = _write(
        tmp_path / "routing.jsonl",
        [
            {"task_id": "t1", "channel": "python"},
            {"task_id": "t9", "channel": "direct"},
        ],
    )
    with pytest.raises(RuntimeError, match="routing names 1 reused tasks"):
        metrics.compute_arm(
            metrics.ArmSpec(
                "missing-row",
                "agent",
                expected_n=2,
                expected_correct=1,
                expected_verifier_valid=1,
                trajectories=traj,
                direct_pool=pool,
                routing=routing,
            )
        )


def test_a_table_that_contradicts_the_run_is_fatal(metrics, tmp_path):
    traj = _write(
        tmp_path / "trajectories.jsonl",
        [_trajectory("t1", status="correct", tool_calls=0, steps=1, invalid=0, final="7")],
    )
    for field, value in (
        ("expected_correct", 5),
        ("expected_tool_calls", 9),
        ("expected_verifier_valid", 0),
        ("expected_n", 3),
    ):
        kwargs = {
            "expected_n": 1,
            "expected_correct": 1,
            "expected_verifier_valid": 1,
            field: value,
        }
        with pytest.raises(RuntimeError, match="contradict the run"):
            metrics.compute_arm(metrics.ArmSpec("wrong", "agent", trajectories=traj, **kwargs))


def test_percentile_matches_the_campaign_convention(metrics):
    # Nearest-rank on a sorted sample, no interpolation: for 1..200 the p95 is
    # sample[int(200 * 0.95)] = 191, not the 190.05 an interpolating percentile
    # would give. Pinned because the campaign script rounds the same way.
    sample = [float(i) for i in range(1, 201)]
    assert metrics.percentile(sample, 0.95) == 191.0
    with pytest.raises(ValueError):
        metrics.percentile([], 0.5)
