"""Tests for the dual-loop tool-use columns.

Why these exist: the table's three numbers are the deliverable, so the counting
rules must be pinned somewhere smaller than a 200-task run. Two properties
carry it -- the counts come from the recorded events with the fixed definitions
(executed ``tool_call`` events, ``payload.result.ok`` for success), and a count
that disagrees with the run's own ``summary.json`` is refused rather than
reported.
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "analysis" / "tool_use_stats.py"


def load_script():
    spec = spec_from_file_location("tool_use_stats", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stats():
    return load_script()


def _tool_call(sequence: int, name: str = "python") -> dict:
    return {"sequence": sequence, "kind": "tool_call", "payload": {"name": name}}


def _tool_result(sequence: int, *, ok: bool) -> dict:
    return {
        "sequence": sequence,
        "kind": "tool_result",
        "payload": {"name": "python", "result": {"ok": ok, "output": "1" if ok else ""}},
    }


def _trajectory(task_id: str, *, status: str, events: list[dict]) -> dict:
    return {
        "task_id": task_id,
        "verdict": {"status": status},
        "trajectory": {"events": [{"sequence": 0, "kind": "model_output", "payload": {}}, *events]},
    }


def _write_arm(tmp_path: Path, name: str, rows: list[dict], summary: dict | None) -> Path:
    arm = tmp_path / name
    arm.mkdir()
    arm.joinpath("trajectories.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    if summary is not None:
        arm.joinpath("summary.json").write_text(json.dumps(summary))
    return arm


def test_stats_count_executed_calls_and_successes_from_events(stats, tmp_path):
    arm = _write_arm(
        tmp_path,
        "arm",
        [
            _trajectory(
                "t1",
                status="correct",
                events=[_tool_call(1), _tool_result(2, ok=True), _tool_call(3), _tool_result(4, ok=False)],
            ),
            _trajectory("t2", status="incorrect", events=[]),
            _trajectory("t3", status="incorrect", events=[_tool_call(1), _tool_result(2, ok=True)]),
        ],
        summary={"tool_calls_total": 3, "correct_count": 1},
    )

    row = stats.tool_stats(arm)

    assert row["tasks"] == 3
    assert row["executed_tool_calls"] == 3
    assert row["tasks_with_executed_tool_call"] == 2
    assert row["tool_call_rate"] == pytest.approx(2 / 3)
    assert row["calls_per_task"] == pytest.approx(1.0)
    assert row["tool_results"] == 3
    assert row["tool_successes"] == 2
    assert row["tool_success_rate"] == pytest.approx(2 / 3)
    assert row["correct"] == 1
    assert row["accuracy"] == pytest.approx(1 / 3)
    assert row["checked_against_summary"] is True


def test_an_arm_without_results_reports_no_rate_not_a_zero(stats, tmp_path):
    """``n/a`` is a measured absence; a 0.0% would claim the tools were tried."""
    arm = _write_arm(
        tmp_path,
        "quiet",
        [_trajectory("t1", status="incorrect", events=[])],
        summary={"tool_calls_total": 0, "correct_count": 0},
    )

    row = stats.tool_stats(arm)

    assert row["tool_results"] == 0
    assert row["tool_success_rate"] is None


def test_a_count_that_disagrees_with_the_summary_is_refused(stats, tmp_path):
    arm = _write_arm(
        tmp_path,
        "drift",
        [_trajectory("t1", status="correct", events=[_tool_call(1), _tool_result(2, ok=True)])],
        summary={"tool_calls_total": 2, "correct_count": 1},
    )

    with pytest.raises(stats.StatsError, match="summary.json records 2"):
        stats.tool_stats(arm)


def test_an_arm_without_a_summary_is_marked_unchecked(stats, tmp_path):
    """A hand-made slice must not read as checked-against-the-run."""
    arm = _write_arm(
        tmp_path,
        "slice",
        [_trajectory("t1", status="correct", events=[_tool_call(1), _tool_result(2, ok=True)])],
        summary=None,
    )

    row = stats.tool_stats(arm)

    assert row["checked_against_summary"] is False
    assert row["executed_tool_calls"] == 1


def test_the_markdown_row_renders_an_absent_rate_as_na(stats):
    table = stats.render_markdown(
        {
            "quiet": {
                "tasks": 2,
                "correct": 1,
                "accuracy": 0.5,
                "tasks_with_executed_tool_call": 0,
                "tool_call_rate": 0.0,
                "executed_tool_calls": 0,
                "calls_per_task": 0.0,
                "tool_results": 0,
                "tool_successes": 0,
                "tool_success_rate": None,
                "checked_against_summary": True,
            }
        }
    )

    assert "| `quiet` | 2 | 0.0% | 0.000 | n/a | 50.0% (1/2) |" in table


def test_main_names_the_missing_trajectories_instead_of_crashing(stats, tmp_path, capsys):
    rc = stats.main(["--arm", f"gone={tmp_path / 'nowhere'}"])

    assert rc == 1
    assert "TOOL-STATS FAIL" in capsys.readouterr().out
