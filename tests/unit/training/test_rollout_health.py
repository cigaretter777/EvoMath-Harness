import importlib.util
from pathlib import Path

import pytest

from adaptive_math.training.rollout_health import (
    row_to_labeled_task,
    summarize_group_rewards,
)


def _load_rollout_health_script():
    repo_root = Path(__file__).parents[3]
    spec = importlib.util.spec_from_file_location(
        "run_rollout_health", repo_root / "scripts" / "eval" / "run_rollout_health.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_row_to_labeled_task_builds_hidden_reference() -> None:
    row = {
        "task_id": "demo:1",
        "problem": "What is 2+2?",
        "answer_type": "integer",
        "dataset": "demo",
        "split": "rl_dev",
        "source_hash": "abc",
        "pipeline_version": "test-v1",
        "metadata": '{"source_index":"1"}',
        "reference_value": "4",
        "reference_acceptable_forms": ["4.0"],
    }

    labeled = row_to_labeled_task(row)

    assert labeled.task.problem == "What is 2+2?"
    assert labeled.task.metadata == {"source_index": "1"}
    assert labeled.reference.value == "4"
    assert labeled.reference.acceptable_forms == ("4.0",)


def test_summarize_group_rewards_distinguishes_effective_groups() -> None:
    groups = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    summary = summarize_group_rewards(groups)

    assert summary["group_count"] == 3
    assert summary["mixed_group_count"] == 1
    assert summary["all_zero_group_count"] == 1
    assert summary["all_one_group_count"] == 1
    assert summary["effective_group_count"] == 1


def test_select_task_rows_is_deterministic_without_replacement() -> None:
    from adaptive_math.training.rollout_health import select_task_rows

    rows = [
        {"task_id": f"task-{i}"}
        for i in range(20)
    ]

    first = select_task_rows(rows, count=10, seed=42)
    second = select_task_rows(rows, count=10, seed=42)

    first_ids = [row["task_id"] for row in first]
    second_ids = [row["task_id"] for row in second]

    assert first_ids == second_ids
    assert len(first_ids) == 10
    assert len(set(first_ids)) == 10


def test_parser_tolerance_flag_parses_to_rules_tuple() -> None:
    script = _load_rollout_health_script()

    args = script.build_parser().parse_args(
        ["--parser-tolerance", "unclosed_think,bare_final_scalar"]
    )

    assert args.parser_tolerance == ("unclosed_think", "bare_final_scalar")


def test_parser_tolerance_defaults_to_strict() -> None:
    script = _load_rollout_health_script()

    assert script.build_parser().parse_args([]).parser_tolerance is None


def test_parser_tolerance_rejects_unknown_rule() -> None:
    script = _load_rollout_health_script()

    with pytest.raises(SystemExit):
        script.build_parser().parse_args(["--parser-tolerance", "nope"])


def test_normalize_acceptable_forms_accepts_numpy_arrays() -> None:
    import numpy as np

    from adaptive_math.training.rollout_health import _normalize_acceptable_forms

    assert _normalize_acceptable_forms(
        np.asarray([], dtype=object)
    ) == ()

    assert _normalize_acceptable_forms(
        np.asarray(["4.0", "4"], dtype=object)
    ) == ("4.0", "4")
