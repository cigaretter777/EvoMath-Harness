"""Tests for the MVP evaluation orchestrator.

Why these exist: ``evaluate.py`` decides *which weights* each of the three arms
loads and *which tasks* they are scored on. Both are identity claims, and both
have already gone wrong once in this repository's history -- the 2026-09-26
campaign ran its agent arms on the RL training pool while the arms they were to
be paired with had scored the frozen Omni-MATH set (zero task overlap, ~3.5 GPU
hours bought nothing).

So the defaults are pinned to the configs that record the released runs, the
task set is asserted to be passed explicitly on every arm (the runner's own
default is the wrong pool), and the metrics path is checked against a synthetic
run before it is trusted on a real one. The last test re-derives the released
numbers from the stored runs -- the strongest statement available on CPU, and
the one that says the MVP's collation agrees with the campaign's.
"""

from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EVALUATE = ROOT / "scripts" / "mvp" / "evaluate.py"

MVP_SFT_CONFIG = ROOT / "configs" / "mvp" / "sft.yaml"
MVP_GRPO_CONFIG = ROOT / "configs" / "mvp" / "grpo.yaml"
FROZEN_IDS = ROOT / "artifacts" / "eval" / "thesis_e0_base_direct_b1" / "task_ids.txt"
EVAL_PARQUET = ROOT / "data" / "processed" / "v1" / "frozen_eval.parquet"
BASE_SNAPSHOT = Path(
    "/root/autodl-tmp/hf-cache/models--Qwen--Qwen3-1.7B/snapshots/"
    "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)
STORED_RUNS = {
    "base": ROOT / "artifacts" / "rollout_health" / "thesis_e0_base_tool",
    "sft": ROOT / "artifacts" / "rollout_health" / "thesis_e0_sft_tool",
    "grpo": ROOT / "artifacts" / "rollout_health" / "thesis_e0_r0_tool",
}
# The released rows, as every report of that campaign states them.
RELEASED_ROWS = {"base": 13, "sft": 28, "grpo": 27}

HAVE_FROZEN_SET = FROZEN_IDS.is_file() and EVAL_PARQUET.is_file()
skip_without_frozen_set = pytest.mark.skipif(
    not HAVE_FROZEN_SET, reason="the frozen parquet and its task list are local artifacts"
)
skip_without_stored_runs = pytest.mark.skipif(
    not all((run / "summary.json").is_file() for run in STORED_RUNS.values()),
    reason="the released runs are local rollout artifacts, not tracked",
)


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
def evaluate():
    return load_script(EVALUATE, "mvp_evaluate")


def parse(evaluate, *argv: str):
    return evaluate.build_parser().parse_args(list(argv))


def test_the_three_arms_load_the_released_weights(evaluate) -> None:
    args = parse(evaluate, "--out-root", "/tmp/does-not-matter")

    assert evaluate.arm_weights("base", args) == (args.base_model, None, None)
    assert evaluate.arm_weights("sft", args) == (args.base_model, args.sft_adapter, "sft")
    assert evaluate.arm_weights("grpo", args) == (args.grpo_model, args.rl_adapter, "rl")


def test_the_defaults_are_the_paths_the_configs_record(evaluate) -> None:
    """The GRPO arm's model is the merged SFT checkpoint the R0 config trained on.

    If these two ever disagree, the arm is not the released arm -- and nothing
    else in the repository compares them. The config spells the path absolutely
    (it is the file the training host ran), so the comparison is on the
    repository-relative tail: the same merged checkpoint, positioned the same
    way, in whatever checkout this is.
    """
    args = parse(evaluate, "--out-root", "/tmp/does-not-matter")
    grpo = yaml.safe_load(MVP_GRPO_CONFIG.read_text())
    actor = next(
        item.split("=", 1)[1]
        for item in grpo["upstream_overrides"]
        if item.startswith("actor_rollout_ref.model.path=")
    )
    relative = Path(args.grpo_model).relative_to(ROOT)

    assert Path(actor).is_absolute()
    assert Path(actor).parts[-len(relative.parts):] == relative.parts

    sft = yaml.safe_load(MVP_SFT_CONFIG.read_text())
    assert Path(args.base_model).name == sft["model_revision"]
    assert Path(args.sft_adapter).name == "adapter"
    assert Path(args.sft_adapter).parent == (ROOT / sft["output_dir"]).resolve()


def test_every_arm_pins_the_task_set_explicitly(evaluate) -> None:
    """``--data``/``--task-ids-file`` on every arm: the wrong pool is unreachable.

    The runner's ``--pool`` default is the RL training pool; passing it here is
    bookkeeping (the emitter records it), while the parquet + task list are what
    actually select the 200 frozen tasks.
    """
    args = parse(evaluate, "--out-root", "/tmp/out")
    for arm in evaluate.ARMS:
        _, command = evaluate.arm_command(arm, args, evaluate.run_dir_for(arm, args))
        assert "--data" in command
        assert command[command.index("--data") + 1] == str(args.data)
        assert command[command.index("--task-ids-file") + 1] == str(args.task_ids)
        assert command[command.index("--mode") + 1] == "all-tools"


def test_only_the_adapter_arms_go_through_the_adapter_wrapper(evaluate) -> None:
    args = parse(evaluate, "--out-root", "/tmp/out")

    base_script, base_command = evaluate.arm_command(
        "base", args, evaluate.run_dir_for("base", args)
    )
    sft_script, sft_command = evaluate.arm_command("sft", args, evaluate.run_dir_for("sft", args))

    assert base_script == evaluate.RULE_BASELINE
    assert "--adapter" not in base_command
    assert sft_script == evaluate.RUN_ARM
    assert sft_command[sft_command.index("--adapter-kind") + 1] == "sft"


def test_shard_flags_appear_only_when_sharding(evaluate) -> None:
    single = parse(evaluate, "--out-root", "/tmp/out")
    sharded = parse(evaluate, "--out-root", "/tmp/out", "--shard-id", "2", "--shard-count", "3")

    _, command = evaluate.arm_command("base", single, evaluate.run_dir_for("base", single))
    assert "--shard-count" not in command

    assert evaluate.run_dir_for("base", sharded).name == "base.shard2"
    _, sharded_command = evaluate.arm_command(
        "base", sharded, evaluate.run_dir_for("base", sharded)
    )
    assert sharded_command[sharded_command.index("--shard-count") + 1] == "3"


def write_run_dir(tmp_path: Path, statuses: list[str]) -> Path:
    """A synthetic finished run: one trajectory per status, summary in the runner's shape.

    Per record: one tool call, two invalid actions, 100 generated tokens, three
    steps, and a final answer unless the verifier could not address the answer
    at all -- which is what ``reference_invalid`` means and why it is the status
    that separates the two legality columns.
    """
    run_dir = tmp_path / "arm"
    run_dir.mkdir()
    with (run_dir / "trajectories.jsonl").open("w") as handle:
        for index, status in enumerate(statuses):
            handle.write(
                json.dumps(
                    {
                        "task_id": f"omni_math:{index:04d}",
                        "trajectory": {
                            "usage": {
                                "tool_calls": 1,
                                "invalid_actions": 2,
                                "generated_tokens": 100,
                                "steps": 3,
                            },
                            "final_answer": (
                                None if status == "reference_invalid" else {"kind": "verdict"}
                            ),
                            "termination_reason": "final_answer",
                        },
                        "verdict": {"status": status},
                    }
                )
                + "\n"
            )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "mode": "all-tools",
                "task_count": len(statuses),
                "rolled_out": len(statuses),
                "direct_reused": 0,
                "correct_count": statuses.count("correct"),
                "tool_calls_total": len(statuses),
                "invalid_actions_total": 2 * len(statuses),
                "generated_tokens_total": 100 * len(statuses),
                "routing_distribution": {"direct": 76, "python": 66, "sympy": 58},
            }
        )
    )
    return run_dir


def test_derived_verifier_valid_counts_correct_and_incorrect(evaluate, tmp_path) -> None:
    run_dir = write_run_dir(tmp_path, ["correct", "incorrect", "reference_invalid"])

    assert evaluate.derived_verifier_valid(run_dir / "trajectories.jsonl") == 2


def test_compute_metrics_checks_the_run_against_its_own_summary(evaluate, tmp_path) -> None:
    run_dir = write_run_dir(tmp_path, ["correct", "reference_invalid"])

    metrics = evaluate.compute_metrics("sft", run_dir)

    assert metrics["n"] == 2
    assert metrics["correct"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["tool_calls_total"] == 2
    # The reference-invalid row closed without a final answer, so it counts
    # against legality but not against correctness.
    assert metrics["final_action_rate"] == 0.5
    assert metrics["verifier_addressable_rate"] == 0.5

    # A summary that disagrees with the run is a bug: the table must refuse it.
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["correct_count"] = 2
    (run_dir / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="correct"):
        evaluate.compute_metrics("sft", run_dir)


def test_metrics_only_reports_a_missing_run(evaluate, tmp_path, capsys) -> None:
    task_ids = tmp_path / "task_ids.txt"
    task_ids.write_text("omni_math:00000000000000000000\n")

    rc = evaluate.main(
        [
            "--metrics-only",
            "--out-root", str(tmp_path),
            "--task-ids", str(task_ids),
            "--arms", "sft",
        ]
    )

    assert rc == 1
    assert "no summary.json" in capsys.readouterr().err


def test_a_missing_task_list_stops_the_run_before_any_arm(evaluate, tmp_path, capsys) -> None:
    """The list every identity is emitted against, checked before the first arm."""
    rc = evaluate.main(
        ["--dry-run", "--out-root", str(tmp_path), "--task-ids", str(tmp_path / "absent.txt")]
    )

    assert rc == 1
    assert "prepare_data.py" in capsys.readouterr().err


@skip_without_stored_runs
def test_the_metrics_path_reproduces_the_released_numbers(evaluate, tmp_path, capsys) -> None:
    """The released three rows, re-derived on CPU from the stored runs.

    ``--metrics-only`` over symlinked run directories exercises everything the
    collation does short of a fresh rollout: read the summary, check it against
    the trajectories, emit the four metrics.
    """
    for arm, run in STORED_RUNS.items():
        (tmp_path / arm).symlink_to(run)

    rc = evaluate.main(
        ["--metrics-only", "--out-root", str(tmp_path), "--arms", "base", "sft", "grpo"]
    )

    assert rc == 0
    payload = json.loads((tmp_path / "four-metrics.json").read_text())
    rows = {arm: data["metrics"] for arm, data in payload["arms"].items()}
    for arm, correct in RELEASED_ROWS.items():
        assert rows[arm]["n"] == 200
        assert rows[arm]["correct"] == correct, arm
    assert rows["base"]["tool_calls_total"] == 7
    assert rows["sft"]["tool_calls_total"] == 0
    assert payload["task_ids_sha256"].startswith("1fc257f2")
    assert "| arm | mode |" in capsys.readouterr().out


@skip_without_frozen_set
def test_dry_run_emits_an_identity_per_arm(evaluate, tmp_path, capsys, monkeypatch) -> None:
    """The whole CPU pre-flight: task list, paths, adapter hashes, then stop.

    The adapter arms are spawned (their own dry-run is what emits), so the
    subprocesses need the release's local artifacts to exist; without them the
    runner's pre-flight would rightly fail and this test is not meaningful.
    """
    for adapter in (evaluate.DEFAULT_SFT_ADAPTER, evaluate.DEFAULT_RL_ADAPTER):
        if not adapter.is_dir():
            pytest.skip("the released adapters are local training artifacts")

    monkeypatch.setenv("ADAPTIVE_MATH_SANDBOX_URL", "http://localhost:8080")
    rc = evaluate.main(["--dry-run", "--out-root", str(tmp_path)])

    assert rc == 0, capsys.readouterr().err
    for arm, expected_kind in (("base", None), ("sft", "sft"), ("grpo", "rl")):
        identity = json.loads((tmp_path / f"{arm}.identity.json").read_text())
        assert identity["task_count"] == 200
        assert identity["mode"] == "all-tools"
        assert identity["status"] == "dry-run"
        assert identity.get("adapter_kind") == expected_kind
