"""Collate the four MVP metrics for every arm that already exists on disk.

Why this exists
---------------
The MVP design asks for four headline metrics per arm -- accuracy, final-action
legality, tool-call rate, mean trajectory steps -- across Base / SFT / GRPO.
Every number for the arms that were actually run is already stored; what was
missing is one place where they are computed with the same definitions and
checked against the summaries the runs themselves wrote. That check is the
point: each arm's recorded ``summary.json`` is the authority, and this script
refuses to print a table whose counts disagree with it.

Two modes, stated because the numbers are not directly comparable across them:

* ``agent`` arms ran the AgentLoop: steps and tool calls come from the stored
  trajectories' usage blocks, and the final action was legal when the loop
  closed with a final answer -- the loop only records one if the strict parser
  accepted it.
* ``direct`` arms are single-pass generations: one step, no tool channel, so
  their tool-call rate and step count are zero and one *by construction*, not
  by measurement. Their legality column is the extractor's verdict instead.
  They are still in the table because accuracy is comparable, and marking the
  difference is more honest than dropping the rows.

Two legality notions are kept apart because they are not the same number and
the stored summaries only pin the second one:

* ``final_action_rate`` -- the protocol-side question: did a legal final action
  come out at all (agent: loop closed; direct: extractor returned a value).
* ``verifier_addressable`` -- the stored ``valid_answer_rate``: the answer was
  judged correct or incorrect, so the verifier got something to compare. Rows
  that extracted cleanly but hit a verifier-side status (reference invalid,
  timeout, internal error) count here and not in the first column.

The fixed-rule arm is composed: it routed 76 tasks to a direct answer, reused
those stored rows verbatim, and rolled out the other 124, so its step count is
a mixture and its reused subset comes from ``routing.jsonl``.

Usage:
    python scripts/analysis/arm_metrics.py --out artifacts/results/mvp-20260929/arm-metrics.json
"""

# No `from __future__ import annotations` here on purpose: the ArmSpec dataclass
# below is loaded by tests through importlib with the repo's usual
# spec_from_file_location + exec_module pattern, which does not register the
# module in sys.modules, and dataclasses resolves string annotations through
# that registry. Real annotations keep the script loadable by that convention.

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ARTS = REPO / "artifacts"

DIRECT_DIR = ARTS / "eval/thesis_e0_base_direct_b1"
BASE_TOOL_DIR = ARTS / "rollout_health/thesis_e0_base_tool"
RULE_DIR = ARTS / "rollout_health/thesis_e0_rule_strategy"
SFT_TOOL_DIR = ARTS / "rollout_health/thesis_e0_sft_tool"
R0_TOOL_DIR = ARTS / "rollout_health/thesis_e0_r0_tool"
R0_DIR = ARTS / "eval/r0_omnimath_200"
R2_DIR = ARTS / "eval/r2_omnimath_200"

# The routing file marks a task "direct" when the fixed rule answered it
# straight instead of rolling out an agent trajectory.
REUSED_CHANNEL = "direct"
VALID_STATUSES = frozenset({"correct", "incorrect"})


@dataclass(frozen=True)
class ArmSpec:
    """One row of the table, and where its numbers come from."""

    label: str
    mode: str  # "agent" | "direct"
    # The counts the arm's own summary.json recorded. A table that disagrees
    # with the run that produced it is a bug in this script, not a finding, so
    # the three primary counts are required: an arm cannot silently skip them.
    expected_n: int
    expected_correct: int
    expected_verifier_valid: int
    expected_rolled_out: int | None = None
    expected_direct_reused: int | None = None
    expected_tool_calls: int | None = None
    expected_invalid_actions: int | None = None
    expected_generated_tokens: int | None = None
    expected_mean_output_tokens: float | None = None
    trajectories: Path | None = None
    predictions: Path | None = None
    direct_pool: Path | None = None  # stored rows the arm reused verbatim
    routing: Path | None = None  # which of those rows were actually reused


ARMS: tuple[ArmSpec, ...] = (
    # counts: artifacts/eval/thesis_e0_base_direct_b1/summary.json
    ArmSpec(
        "base_direct",
        "direct",
        expected_n=200,
        expected_correct=0,
        expected_verifier_valid=0,
        expected_rolled_out=0,
        expected_direct_reused=0,
        expected_tool_calls=0,
        expected_mean_output_tokens=966.145,
        predictions=DIRECT_DIR / "base_predictions.jsonl",
    ),
    # counts: artifacts/rollout_health/thesis_e0_base_tool/summary.json
    ArmSpec(
        "base_agent",
        "agent",
        expected_n=200,
        expected_correct=13,
        expected_verifier_valid=22,
        expected_rolled_out=200,
        expected_direct_reused=0,
        expected_tool_calls=7,
        expected_invalid_actions=1118,
        expected_generated_tokens=768485,
        trajectories=BASE_TOOL_DIR / "trajectories.jsonl",
    ),
    # counts: artifacts/rollout_health/thesis_e0_rule_strategy/summary.json
    ArmSpec(
        "rule_strategy",
        "agent",
        expected_n=200,
        expected_correct=10,
        expected_verifier_valid=13,
        expected_rolled_out=124,
        expected_direct_reused=76,
        expected_tool_calls=16,
        expected_invalid_actions=689,
        expected_generated_tokens=486683,
        trajectories=RULE_DIR / "trajectories.jsonl",
        direct_pool=DIRECT_DIR / "base_predictions.jsonl",
        routing=RULE_DIR / "routing.jsonl",
    ),
    # counts: artifacts/eval/thesis_e0_base_direct_b1/summary.json, arm "sft"
    ArmSpec(
        "sft_direct",
        "direct",
        expected_n=200,
        expected_correct=27,
        expected_verifier_valid=133,
        expected_rolled_out=0,
        expected_direct_reused=0,
        expected_tool_calls=0,
        expected_mean_output_tokens=679.595,
        predictions=DIRECT_DIR / "sft_predictions.jsonl",
    ),
    # counts: artifacts/eval/r0_omnimath_200/summary.json, arm "sft"
    ArmSpec(
        "r0_grpo_direct",
        "direct",
        expected_n=200,
        expected_correct=22,
        expected_verifier_valid=145,
        expected_rolled_out=0,
        expected_direct_reused=0,
        expected_tool_calls=0,
        expected_mean_output_tokens=599.335,
        predictions=R0_DIR / "sft_predictions.jsonl",
    ),
    # counts: artifacts/eval/r2_omnimath_200/summary.json, arm "sft"
    ArmSpec(
        "r2_grpo_direct",
        "direct",
        expected_n=200,
        expected_correct=25,
        expected_verifier_valid=131,
        expected_rolled_out=0,
        expected_direct_reused=0,
        expected_tool_calls=0,
        expected_mean_output_tokens=690.015,
        predictions=R2_DIR / "sft_predictions.jsonl",
    ),
    # counts: artifacts/rollout_health/thesis_e0_sft_tool/summary.json -- the
    # merged summary the run itself wrote (the sum of its three shard
    # summaries). verifier-valid: the agent arms' summaries carry no such field
    # (same gap as base_agent's 22), so the value comes from the re-scoring pass
    # in artifacts/results/campaign-2026-09-27/tolerance-rescore.json, which
    # reads the trajectories under the same rule: status in {correct, incorrect}.
    ArmSpec(
        "sft_agent",
        "agent",
        expected_n=200,
        expected_correct=28,
        expected_verifier_valid=156,
        expected_rolled_out=200,
        expected_direct_reused=0,
        expected_tool_calls=0,
        expected_invalid_actions=274,
        expected_generated_tokens=228689,
        trajectories=SFT_TOOL_DIR / "trajectories.jsonl",
    ),
    # counts: artifacts/rollout_health/thesis_e0_r0_tool/summary.json;
    # verifier-valid from the same rescore artifact.
    ArmSpec(
        "r0_agent",
        "agent",
        expected_n=200,
        expected_correct=27,
        expected_verifier_valid=186,
        expected_rolled_out=200,
        expected_direct_reused=0,
        expected_tool_calls=0,
        expected_invalid_actions=115,
        expected_generated_tokens=158808,
        trajectories=R0_TOOL_DIR / "trajectories.jsonl",
    ),
)


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile, the convention analyze_e0_baselines.py uses."""
    if not sorted_values:
        raise ValueError("percentile of an empty sample")
    index = min(len(sorted_values) - 1, int(len(sorted_values) * q))
    return sorted_values[index]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _check(label: str, name: str, computed: Any, expected: Any) -> None:
    if expected is None:
        return
    if computed != expected:
        raise RuntimeError(
            f"{label}: computed {name}={computed}, the stored summary records {expected}; "
            "the table would contradict the run"
        )


def _reused_rows(spec: ArmSpec) -> list[dict[str, Any]]:
    """The stored direct rows this arm reused, as named by its routing file.

    Filtering matters: the pool holds a row for every task, but the arm only
    reused the ones it routed to a direct answer, and counting the whole pool
    would inflate the arm's denominator past the 200 tasks it was run on.
    """
    if spec.direct_pool is None or spec.routing is None:
        raise ValueError(f"{spec.label}: a reused pool needs both direct_pool and routing")
    reused_ids = {
        row["task_id"] for row in _read_jsonl(spec.routing) if row["channel"] == REUSED_CHANNEL
    }
    rows = [row for row in _read_jsonl(spec.direct_pool) if row["task_id"] in reused_ids]
    if len(rows) != len(reused_ids):
        raise RuntimeError(
            f"{spec.label}: routing names {len(reused_ids)} reused tasks but the pool has "
            f"{len(rows)} of them"
        )
    return rows


def compute_arm(spec: ArmSpec) -> dict[str, Any]:
    """Compute the four metrics for one arm, then check them against the run."""
    correct = 0
    verifier_valid = 0
    tool_calls = 0
    invalid_actions = 0
    generated_tokens = 0
    final_actions = 0
    steps: list[float] = []
    rolled_out = 0

    terminations: Counter[str] = Counter()
    if spec.trajectories is not None:
        for record in _read_jsonl(spec.trajectories):
            rolled_out += 1
            status = (record["verdict"] or {}).get("status")
            correct += int(status == "correct")
            verifier_valid += int(status in VALID_STATUSES)
            usage = record["trajectory"]["usage"]
            tool_calls += int(usage["tool_calls"])
            invalid_actions += int(usage["invalid_actions"])
            generated_tokens += int(usage["generated_tokens"])
            final_actions += int(bool(record["trajectory"]["final_answer"]))
            terminations[str(record["trajectory"]["termination_reason"])] += 1
            steps.append(float(usage["steps"]))

    reused = _reused_rows(spec) if spec.direct_pool is not None else []
    direct_rows = _read_jsonl(spec.predictions) if spec.predictions is not None else []
    output_tokens = 0
    for row in reused + direct_rows:
        status = row["verifier_status"]
        correct += int(status == "correct")
        verifier_valid += int(status in VALID_STATUSES)
        final_actions += int(row["extract_status"] == "ok")
        output_tokens += int(row["output_tokens"])
        steps.append(1.0)

    n = rolled_out + len(reused) + len(direct_rows)
    if n == 0:
        raise ValueError(f"{spec.label}: no records found")

    _check(spec.label, "tasks", n, spec.expected_n)
    _check(spec.label, "correct", correct, spec.expected_correct)
    _check(spec.label, "verifier-valid answers", verifier_valid, spec.expected_verifier_valid)
    _check(spec.label, "rolled-out tasks", rolled_out, spec.expected_rolled_out)
    _check(spec.label, "reused tasks", len(reused), spec.expected_direct_reused)
    _check(spec.label, "tool calls", tool_calls, spec.expected_tool_calls)
    _check(spec.label, "invalid actions", invalid_actions, spec.expected_invalid_actions)
    _check(spec.label, "generated tokens", generated_tokens, spec.expected_generated_tokens)
    if spec.expected_mean_output_tokens is not None:
        _check(
            spec.label,
            "mean output tokens",
            round(output_tokens / len(direct_rows), 3),
            spec.expected_mean_output_tokens,
        )

    ordered_steps = sorted(steps)
    return {
        "mode": spec.mode,
        "n": n,
        "rolled_out": rolled_out,
        "direct_reused": len(reused),
        "correct": correct,
        "accuracy": correct / n,
        "final_action_rate": final_actions / n,
        "verifier_addressable_rate": verifier_valid / n,
        "tool_calls_total": tool_calls,
        "tool_calls_per_task": tool_calls / n,
        "steps_mean": sum(steps) / n,
        "steps_p95": percentile(ordered_steps, 0.95),
        # A direct arm has no action channel to violate, so this is None rather
        # than 0: absence of measurement, not a measured absence.
        "invalid_actions_per_task": invalid_actions / n if spec.mode == "agent" else None,
        # Why the loop stopped, which is what separates "answered" from "ran out
        # of steps": the step count alone cannot tell those apart.
        "termination_reasons": dict(sorted(terminations.items())) or None,
    }


def _format_table(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        (
            "| arm | mode | n | correct | accuracy | final action | verifier-valid | tool calls | "
            "calls/task | steps mean | steps p95 | invalid/task |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, data in results.items():
        invalid = data["invalid_actions_per_task"]
        lines.append(
            f"| {label} | {data['mode']} | {data['n']} | {data['correct']} | "
            f"{data['accuracy']:.1%} | {data['final_action_rate']:.1%} | "
            f"{data['verifier_addressable_rate']:.1%} | {data['tool_calls_total']} | "
            f"{data['tool_calls_per_task']:.3f} | {data['steps_mean']:.2f} | "
            f"{data['steps_p95']:.0f} | {'n/a' if invalid is None else f'{invalid:.2f}'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    args = parser.parse_args(argv)

    results = {spec.label: compute_arm(spec) for spec in ARMS}
    print(_format_table(results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"arms": results}, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
