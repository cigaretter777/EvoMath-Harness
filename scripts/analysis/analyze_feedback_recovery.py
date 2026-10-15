"""Analysis for the feedback-recovery diagnostic (report section 8).

Reads the run directory written by scripts/eval/run_feedback_recovery.py and
produces analysis/{tables.md,summary.json,paired_effects.json} plus the audit
sample in audit/labels.jsonl.

Pre-specified, in this order and not exchanged afterwards:
  primary contrast   F - M
  secondary contrast P - F
  unit of analysis   the anchor (task); each resample keeps all three arms and
                     all four continuations of the anchors it draws
  missing data       infrastructure-incomplete branches are missing, never
                     wrong; the main analysis uses complete anchor groups only
                     and reports the missing-as-correct / missing-as-wrong
                     sensitivity range
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from adaptive_math.recovery.arms import HINT_TEXT, MARKER_TEXT
from adaptive_math.recovery.seeds import derive_seed
from adaptive_math.recovery.taxonomy import ErrorClass

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = derive_seed("run_order", "paired_bootstrap")
ARMS = ("F", "P", "M")
PRIMARY = ("F", "M")
SECONDARY = ("P", "F")
AUDIT_B_ERROR_TRAJECTORIES = 20
AUDIT_C_BRANCHES = 30


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _dedupe_attempts(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Count each branch once (sections 7.2 and 10.6).

    A sample whose first attempt hit an infrastructure fault was retried under
    the same seeds, so the log holds two records for one branch. The usable
    attempt wins; if every attempt was infrastructure, the branch stays in as
    *missing* -- never as a wrong answer, and never twice. Retrying after a
    non-infrastructure failure is a protocol violation, so it is reported
    rather than silently resolved."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["logical_sample_id"]), []).append(record)
    kept: list[dict[str, Any]] = []
    retried = 0
    all_missing = 0
    violations: list[str] = []
    for sample_id, attempts in grouped.items():
        attempts.sort(key=lambda record: int(record.get("attempt", 0)))
        if len(attempts) > 1:
            retried += 1
            if not attempts[0].get("infra_failure"):
                violations.append(f"{sample_id}: retried after a non-infrastructure failure")
        usable = [attempt for attempt in attempts if not attempt.get("infra_failure")]
        if usable:
            kept.append(usable[0])
        else:
            all_missing += 1
            kept.append(attempts[0])
    kept.sort(key=lambda record: int(record.get("execution_order", 0)))
    return kept, {
        "records_before_dedupe": len(records),
        "logical_samples": len(kept),
        "retried_samples": retried,
        "samples_missing_after_retry": all_missing,
        "protocol_violations": violations,
    }


def _normalize_code(code: str) -> str:
    """The only normalisation applied before comparing code: line endings and
    surrounding whitespace. Anything else (formatting, renames) would turn an
    objective "ran the same code again" metric into an opinion."""
    return code.replace("\r\n", "\n").replace("\r", "\n").strip()


def _anchor_original_code(prefix_events: Sequence[Mapping[str, Any]]) -> str | None:
    for event in reversed(prefix_events):
        if event.get("kind") == "tool_call":
            arguments = event.get("payload", {}).get("arguments", {})
            code = arguments.get("code") if isinstance(arguments, Mapping) else None
            return str(code) if code is not None else None
    return None


def _suffix_python_calls(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        event
        for event in record.get("suffix_events", [])
        if event.get("kind") == "tool_call" and event.get("payload", {}).get("name") == "python"
    ]


def _first_suffix_result(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for event in record.get("suffix_events", []):
        if event.get("kind") == "tool_result":
            return cast("Mapping[str, Any]", event)
    return None


def _inference_ms_estimate(events: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Rough generation wall-clock per branch, from event timestamp deltas.

    Each model turn is stamped when it finished, so the gap to the previous
    event is that turn's generation time plus the tool call before it. The
    first suffix turn has no previous suffix event (its clock starts at the
    anchor), so it is excluded and the count of measured turns is returned
    with the total -- an estimate with a stated denominator, not a profiler."""
    total = 0
    measured = 0
    previous: int | None = None
    for event in events:
        stamp = event.get("monotonic_ms")
        if (
            event.get("kind") == "model_output"
            and isinstance(stamp, int)
            and previous is not None
        ):
            total += max(0, stamp - previous)
            measured += 1
        if isinstance(stamp, int):
            previous = stamp
    return total, measured


def _branch_metrics(record: Mapping[str, Any], original_code: str | None) -> dict[str, Any]:
    events = record.get("suffix_events", [])
    calls = _suffix_python_calls(record)
    first = calls[0] if calls else None
    first_code = (
        str(first["payload"].get("arguments", {}).get("code"))
        if first is not None
        else None
    )
    result = _first_suffix_result(record)
    result_payload = result.get("payload", {}).get("result", {}) if result else {}
    repeated = (
        first_code is not None
        and original_code is not None
        and _normalize_code(first_code) == _normalize_code(original_code)
    )
    code_changed = (
        first_code is not None
        and original_code is not None
        and not repeated
    )
    calls_again = first is not None
    next_ok = bool(result_payload.get("ok") is True) if result is not None else None
    error_code = result_payload.get("error_code") if result is not None else None
    if next_ok is None or next_ok:
        next_class = None
    elif error_code == "execution_error":
        next_class = ErrorClass.E_EXEC.value
    elif error_code == "timeout":
        next_class = ErrorClass.E_TIMEOUT.value
    elif error_code == "unavailable":
        next_class = ErrorClass.E_INFRA.value
    else:
        next_class = "call_form"
    invalid_actions = sum(1 for event in events if event.get("kind") == "invalid_action")
    inference_ms, inference_turns = _inference_ms_estimate(events)
    return {
        "called_again": calls_again,
        "next_call_executed": next_ok,
        "next_call_error_class": next_class,
        "repeated_original_code": repeated,
        "repeated_same_error_class": bool(repeated and next_class == ErrorClass.E_EXEC.value),
        # Section 8.2's "original error resolved" needs execution results *and*
        # manual inspection: a rewrite that merely runs is not a repair. This
        # flag is the automated half only, named as a proxy so it is never
        # reported as the human-verified metric.
        "next_call_executed_and_code_changed_PROXY_only": bool(
            calls_again and next_ok is True and code_changed
        ),
        "invalid_actions": invalid_actions,
        "direct_final_without_calling": not calls_again
        and record.get("termination_reason") == "final",
        "budget_terminated": record.get("termination_reason")
        in {"max_steps", "python_time_budget"},
        "suffix_tool_calls": record.get("suffix_usage", {}).get("tool_calls", 0),
        "suffix_generated_tokens": record.get("suffix_usage", {}).get("generated_tokens", 0),
        "suffix_python_seconds": record.get("suffix_usage", {}).get("python_seconds", 0.0),
        "suffix_inference_ms_estimate": inference_ms,
        "suffix_inference_ms_measured_turns": inference_turns,
    }


def _arm_means(
    records: Sequence[Mapping[str, Any]], *, missing_as: float | None
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """Per anchor and arm: the mean of its usable continuations, plus how many
    went into each mean.

    ``missing_as=None`` drops infrastructure-incomplete branches (the main
    analysis), otherwise the given value stands in for them (sensitivity).
    The counts are what let the caller demand four usable samples per arm
    instead of quietly averaging over fewer."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in records:
        anchor = str(record["anchor_id"])
        arm = str(record["arm"])
        if record.get("infra_failure"):
            if missing_as is None:
                continue
            grouped.setdefault(anchor, {}).setdefault(arm, []).append(float(missing_as))
        else:
            grouped.setdefault(anchor, {}).setdefault(arm, []).append(
                1.0 if record.get("correct") else 0.0
            )
    means: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for anchor, arms in grouped.items():
        means[anchor] = {
            arm: statistics.fmean(values) for arm, values in arms.items() if values
        }
        counts[anchor] = {arm: len(values) for arm, values in arms.items() if values}
    return means, counts


def _complete_anchors(
    means: Mapping[str, Mapping[str, float]],
    counts: Mapping[str, Mapping[str, int]],
    expected: int,
) -> tuple[list[str], list[str]]:
    """Anchors usable for the primary analysis: every arm present with exactly
    the planned number of usable continuations. Section 8.1 excludes anchors
    with fewer; anchors that only lost branches to missing data stay in the
    sensitivity runs."""
    complete = [
        anchor
        for anchor in sorted(means)
        if set(means[anchor]) == set(ARMS)
        and set(counts.get(anchor, {})) == set(ARMS)
        and all(counts[anchor][arm] == expected for arm in ARMS)
    ]
    return complete, [anchor for anchor in sorted(means) if anchor not in complete]


def _paired_bootstrap(
    means: Mapping[str, Mapping[str, float]],
    anchors: Sequence[str],
    contrast: tuple[str, str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    left, right = contrast
    observed = statistics.fmean([means[a][left] - means[a][right] for a in anchors])
    rng = random.Random(seed)
    n = len(anchors)
    samples: list[float] = []
    for _ in range(draws):
        pick = [anchors[rng.randrange(n)] for _ in range(n)]
        samples.append(statistics.fmean([means[a][left] - means[a][right] for a in pick]))
    samples.sort()
    lower = samples[int(0.025 * draws)]
    upper = samples[int(0.975 * draws) - 1]
    values = {means[a][left] - means[a][right] for a in anchors}
    degenerate = len(values) == 1 and values.pop() == 0.0
    return {
        "contrast": f"{left} - {right}",
        "anchors": n,
        "difference": observed,
        "ci95": [lower, upper],
        "draws": draws,
        "seed": seed,
        "degenerate_zero_width": bool(degenerate),
        "note": (
            "all anchors show an identical zero difference; the interval carries "
            "no information and is flagged rather than read as equivalence"
            if degenerate
            else None
        ),
    }


def _auxiliary(
    records: Sequence[Mapping[str, Any]],
    originals: Mapping[str, str | None],
) -> dict[str, Any]:
    """Section 8.2's auxiliary metrics, each with an explicit denominator."""
    valid = [r for r in records if not r.get("infra_failure")]
    per_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rows = [r for r in valid if r["arm"] == arm]
        metrics = [_branch_metrics(r, originals.get(str(r["anchor_id"]))) for r in rows]
        called = [m for m in metrics if m["called_again"]]
        correct_rows = [r for r in rows if r.get("correct")]
        per_arm[arm] = {
            "branches": len(rows),
            "correct": sum(1 for r in rows if r.get("correct")),
            "accuracy": (sum(1 for r in rows if r.get("correct")) / len(rows)) if rows else None,
            "called_again_rate": (
                len(called) / len(rows) if rows else None
            ),
            "next_call_success_all_branches": (
                sum(1 for m in metrics if m["next_call_executed"]) / len(rows) if rows else None
            ),
            "next_call_success_conditional": (
                sum(1 for m in called if m["next_call_executed"]) / len(called) if called else None
            ),
            "repeated_original_code_rate": (
                sum(1 for m in metrics if m["repeated_original_code"]) / len(rows) if rows else None
            ),
            "repeated_same_error_class_rate": (
                sum(1 for m in metrics if m["repeated_same_error_class"]) / len(rows)
                if rows
                else None
            ),
            "next_call_executed_and_code_changed_PROXY_only": (
                sum(1 for m in metrics if m["next_call_executed_and_code_changed_PROXY_only"])
                / len(rows)
                if rows
                else None
            ),
            "direct_final_without_calling_rate": (
                sum(1 for m in metrics if m["direct_final_without_calling"]) / len(rows)
                if rows
                else None
            ),
            "budget_terminated_rate": (
                sum(1 for m in metrics if m["budget_terminated"]) / len(rows) if rows else None
            ),
            "invalid_actions_total": sum(m["invalid_actions"] for m in metrics),
            "suffix_tool_calls_mean": (
                statistics.fmean([m["suffix_tool_calls"] for m in metrics]) if metrics else None
            ),
            "suffix_generated_tokens_mean": (
                statistics.fmean([m["suffix_generated_tokens"] for m in metrics])
                if metrics
                else None
            ),
            "suffix_python_seconds_mean": (
                statistics.fmean([m["suffix_python_seconds"] for m in metrics])
                if metrics
                else None
            ),
            "suffix_inference_ms_estimate_mean": (
                statistics.fmean([m["suffix_inference_ms_estimate"] for m in metrics])
                if metrics
                else None
            ),
            "suffix_inference_ms_measured_turns": sum(
                m["suffix_inference_ms_measured_turns"] for m in metrics
            ),
            "correct_only_suffix_tokens_mean": (
                statistics.fmean(
                    [
                        m["suffix_generated_tokens"]
                        for r, m in zip(rows, metrics, strict=True)
                        if r.get("correct")
                    ]
                )
                if correct_rows
                else None
            ),
        }
    return per_arm


def _select_audit_sample(
    stage_b: Sequence[Mapping[str, Any]], continuations: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pre-fixed audit selection (section 8.3).

    B: a seeded random sample of the error trajectories -- section 8.3 asks for
    random, and file order is only incidentally arbitrary. C: round-robin over
    the three arms from a seeded shuffle, so arms are equally represented and no
    outcome is favoured, plus every branch the classifier put in a rare class
    (call form, timeout) up to a small cap, because "cover the different error
    types" cannot be met by a plain random draw when those classes are rare.

    The rule and its seed are written into the audit file itself, so the sample
    can be reproduced and its ex-ante nature checked."""
    rng = random.Random(derive_seed("state_selection", "audit_sample"))
    selected: list[dict[str, Any]] = []
    errors = [
        record
        for record in stage_b
        if record.get("error_counts", {}).get(ErrorClass.E_EXEC.value, 0) > 0
    ]
    b_sample = (
        rng.sample(errors, AUDIT_B_ERROR_TRAJECTORIES)
        if len(errors) > AUDIT_B_ERROR_TRAJECTORIES
        else list(errors)
    )
    for record in sorted(b_sample, key=lambda item: str(item["trace_id"])):
        selected.append(
            {
                "kind": "stage_b_error_trajectory",
                "trace_id": record["trace_id"],
                "task_id": record["task_id"],
                "sample_index": record["sample_index"],
                "error_counts": record["error_counts"],
                "correct": record["correct"],
                "selection": "random_from_errors",
                "human_label": None,
                "automated_label": "has_exec_error",
                "reviewer": "executing agent (not independent)",
            }
        )

    originals = {str(record["anchor_id"]): None for record in continuations}
    rare: list[Mapping[str, Any]] = []
    by_arm: dict[str, list[Mapping[str, Any]]] = {arm: [] for arm in ARMS}
    for record in continuations:
        metrics = _branch_metrics(record, originals.get(str(record["anchor_id"])))
        if metrics["next_call_error_class"] in {"call_form", ErrorClass.E_TIMEOUT.value}:
            rare.append(record)
        by_arm.setdefault(str(record["arm"]), []).append(record)
    for arm in ARMS:
        rng.shuffle(by_arm[arm])

    picked: list[Mapping[str, Any]] = []
    picked_ids: set[str] = set()

    def _add(record: Mapping[str, Any], reason: str) -> None:
        logical_id = str(record["logical_sample_id"])
        if logical_id in picked_ids or len(picked) >= AUDIT_C_BRANCHES:
            return
        picked_ids.add(logical_id)
        picked.append({**record, "_selection": reason})

    for record in rare[: AUDIT_C_BRANCHES // 6]:
        _add(record, "rare_error_class")
    index = 0
    while len(picked) < AUDIT_C_BRANCHES and any(by_arm[arm] for arm in ARMS):
        for arm in ARMS:
            if index < len(by_arm[arm]):
                _add(by_arm[arm][index], "arm_round_robin_from_seeded_shuffle")
            if len(picked) >= AUDIT_C_BRANCHES:
                break
        index += 1

    for record in picked:
        rationales = _branch_metrics(record, originals.get(str(record["anchor_id"])))
        selected.append(
            {
                "kind": "stage_c_branch",
                "logical_sample_id": record["logical_sample_id"],
                "attempt_id": record["attempt_id"],
                "arm": record["arm"],
                "anchor_id": record["anchor_id"],
                "task_id": record["task_id"],
                "correct": record["correct"],
                "termination_reason": record["termination_reason"],
                "observation_before": record["observation_before"],
                "observation_after": record["observation_after"],
                "selection": record["_selection"],
                "human_label": None,
                "automated_label": (
                    "call_form_error"
                    if rationales["next_call_error_class"] == "call_form"
                    else "needs_review"
                ),
                "reviewer": "executing agent (not independent)",
            }
        )
    rule = {
        "seed": derive_seed("state_selection", "audit_sample"),
        "stage_b": (
            f"seeded random sample of up to {AUDIT_B_ERROR_TRAJECTORIES} trajectories "
            "whose automated error inventory contains E_EXEC"
        ),
        "stage_c": (
            "up to 5 branches the classifier flagged as rare classes (call form / "
            f"timeout), then round-robin over arms from a seeded shuffle, up to "
            f"{AUDIT_C_BRANCHES} branches"
        ),
        "not_used_for_selection": [
            "the branch's correctness",
            "whether the branch recovered",
            "the arm's aggregate accuracy",
        ],
        "reviewer": "executing agent; no independent reviewer available",
        "fixed_before_stage_c_finished": True,
    }
    return selected, rule


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-continuations-per-arm", type=int, default=4)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    stage_a = _read_json(run_dir / "stage_a" / "summary.json")
    stage_b_trajectories = _read_jsonl(run_dir / "stage_b" / "trajectories.jsonl")
    stage_b_summary = _read_json(run_dir / "stage_b" / "summary.json")
    selected = _read_jsonl(run_dir / "anchors" / "selected.jsonl")
    continuations_raw = _read_jsonl(run_dir / "stage_c" / "continuations.jsonl")
    continuations, attempt_dedup = _dedupe_attempts(continuations_raw)

    # The anchor's failing call is the prefix's last tool_call; the branch log
    # keeps the prefix only as the text the model saw, so recover the code from
    # the B trajectory that produced the anchor.
    originals: dict[str, str | None] = {
        str(record["anchor_id"]): None for record in continuations
    }
    by_trace = {record["trace_id"]: record for record in stage_b_trajectories}
    for record in selected:
        source = by_trace.get(str(record["source_trajectory_id"]))
        if source is None:
            continue
        prefix = source["trajectory"]["events"][: int(record["prefix_event_count"])]
        originals[str(record["anchor_id"])] = _anchor_original_code(prefix)

    missing = sum(1 for record in continuations if record.get("infra_failure"))
    means, counts = _arm_means(continuations, missing_as=None)
    complete, incomplete = _complete_anchors(
        means, counts, args.expected_continuations_per_arm
    )
    sensitivity: dict[str, dict[str, dict[str, float]]] = {}
    sensitivity_counts: dict[str, dict[str, dict[str, int]]] = {}
    for name, substitute in (("missing_as_correct", 1.0), ("missing_as_wrong", 0.0)):
        scenario_means, scenario_counts = _arm_means(continuations, missing_as=substitute)
        sensitivity[name] = scenario_means
        sensitivity_counts[name] = scenario_counts

    paired: list[dict[str, Any]] = []
    for contrast in (PRIMARY, SECONDARY):
        paired.append(_paired_bootstrap(means, complete, contrast))
    sensitivity_pairs: list[dict[str, Any]] = []
    for name, arm_means in sensitivity.items():
        # Sensitivity keeps anchors that lost branches to infrastructure, but
        # still demands the full plan: a group missing an entire arm says
        # nothing about how the missing branches would have gone.
        anchors = [
            anchor
            for anchor in sorted(arm_means)
            if set(arm_means[anchor]) == set(ARMS)
            and all(
                sensitivity_counts[name][anchor][arm] == args.expected_continuations_per_arm
                for arm in ARMS
            )
        ]
        for contrast in (PRIMARY, SECONDARY):
            if anchors:
                row = _paired_bootstrap(arm_means, anchors, contrast)
                row["scenario"] = name
                sensitivity_pairs.append(row)

    arm_accuracy = {
        arm: (
            statistics.fmean([means[a][arm] for a in complete if arm in means[a]])
            if complete
            else None
        )
        for arm in ARMS
    }
    auxiliary = _auxiliary(continuations, originals)
    audit, audit_rule = _select_audit_sample(stage_b_trajectories, continuations)
    (run_dir / "audit").mkdir(parents=True, exist_ok=True)
    with (run_dir / "audit" / "labels.jsonl").open("w") as handle:
        handle.write(
            json.dumps({"kind": "selection_rule", **audit_rule}, ensure_ascii=False) + "\n"
        )
        for row in audit:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    small_sample = len(complete) < 20
    effects = {
        "unit_of_analysis": "anchor (task)",
        "arms": list(ARMS),
        "primary_contrast": "F - M",
        "secondary_contrast": "P - F",
        "complete_anchor_groups": len(complete),
        "incomplete_anchor_groups": incomplete,
        "missing_branches": missing,
        "arm_accuracy_over_complete_groups": arm_accuracy,
        "paired": paired,
        "sensitivity": sensitivity_pairs,
        "small_sample_note": (
            "fewer than 20 anchors: intervals are descriptive only (section 8.1)"
            if small_sample
            else None
        ),
    }
    (analysis_dir / "paired_effects.json").write_text(
        json.dumps(effects, ensure_ascii=False, indent=2) + "\n"
    )

    # DEVIATIONS D11: stage_b/summary.json's tasks_with_any_exec_error counts
    # every task instead of the tasks that saw an E_EXEC. The buggy field is
    # inside the frozen runner, so the corrected value is recomputed here from
    # the raw trajectories, which are unaffected.
    tasks_with_exec_error = {
        str(record["task_id"])
        for record in stage_b_trajectories
        if record.get("error_counts", {}).get(ErrorClass.E_EXEC.value, 0) > 0
    }
    summary = {
        "run_dir": str(run_dir),
        "analysis_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "stage_b_recomputed": {
            "tasks_with_any_exec_error_corrected": len(tasks_with_exec_error),
            "stage_b_summary_field_is_buggy": (
                "stage_b/summary.json.tasks_with_any_exec_error counts all tasks "
                "with a rollout; use this corrected value (DEVIATIONS D11)"
            ),
        },
        "stage_a_gate": stage_a.get("gate", {}),
        "stage_a_verifier_smoke": stage_a.get("verifier_smoke"),
        "stage_b": stage_b_summary,
        "anchors_selected": len(selected),
        "pool_verdict": stage_b_summary.get("pool_verdict"),
        "continuations": len(continuations),
        "attempt_dedupe": attempt_dedup,
        "continuations_by_arm": {
            arm: sum(1 for record in continuations if record["arm"] == arm) for arm in ARMS
        },
        "infrastructure_missing": missing,
        "auxiliary": auxiliary,
        "effects": {
            "complete_anchor_groups": len(complete),
            "arm_accuracy": arm_accuracy,
            "primary": paired[0],
            "secondary": paired[1],
        },
        "audit": {
            "records": len(audit),
            "rule": audit_rule,
            "rule_written_to": str(run_dir / "audit" / "labels.jsonl"),
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "code_normalization_rule": (
            "line endings and surrounding whitespace only "
            "(\\r\\n/\\r -> \\n, then str.strip); stated so the objective "
            '"same code again" metric stays separable from the human judgement '
            "of similar-but-edited code"
        ),
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )

    tables = [
        "# 反馈恢复诊断：结果表（自动生成）",
        "",
        (
            "生成脚本：`scripts/analysis/analyze_feedback_recovery.py`；"
            f"配对 bootstrap 次数={BOOTSTRAP_DRAWS}，seed={BOOTSTRAP_SEED}。"
        ),
        "",
        "## A 门槛",
        "",
        _table(
            ["检查", "值", "阈值", "通过"],
            [
                [key, value.get("value"), value.get("threshold"), value.get("pass")]
                for key, value in summary["stage_a_gate"].get("checks", {}).items()
            ],
        ),
        "",
        "## B 自然运行（§6）",
        "",
        _table(
            ["指标", "值"],
            [[key, value] for key, value in sorted(stage_b_summary.get("natural", {}).items())],
        ),
        "",
        "## C 锚点后正确率（仅完整锚点组）",
        "",
        (
            f"完整锚点组={len(complete)}，不完整={len(incomplete)}，基础设施缺失分支={missing}；"
            f"重试样本={attempt_dedup['retried_samples']}，"
            f"重试后仍缺失={attempt_dedup['samples_missing_after_retry']}，"
            f"协议违规={len(attempt_dedup['protocol_violations'])}。"
        ),
        "",
        _table(["臂", "锚点后正确率"], [[arm, arm_accuracy[arm]] for arm in ARMS]),
        "",
        "## 配对差异（锚点级配对 bootstrap）",
        "",
        _table(
            ["对比", "差值", "95% 区间下", "95% 区间上", "锚点数", "退化"],
            [
                [
                    row["contrast"],
                    row["difference"],
                    row["ci95"][0],
                    row["ci95"][1],
                    row["anchors"],
                    row["degenerate_zero_width"],
                ]
                for row in paired
            ],
        ),
        "",
        "主对比 F−M，次对比 P−F，均在结果出来前指定，不按结果交换主次。",
        "",
        "## 缺失敏感性",
        "",
        _table(
            ["场景", "对比", "差值", "95% 区间下", "95% 区间上"],
            [
                [row["scenario"], row["contrast"], row["difference"], row["ci95"][0], row["ci95"][1]]
                for row in sensitivity_pairs
            ],
        ),
        "",
        "## 辅助指标（§8.2，分母见列名）",
        "",
        _table(
            ["指标"] + [arm for arm in ARMS],
            [
                [key] + [auxiliary[arm].get(key) for arm in ARMS]
                for key in sorted(
                    {
                        key
                        for arm in ARMS
                        for key in auxiliary[arm]
                    }
                )
            ],
        ),
        "",
        "## 干预文本（冻结）",
        "",
        f"- P 提示：`{HINT_TEXT}`",
        f"- M 标记：`{MARKER_TEXT}`",
    ]
    (analysis_dir / "tables.md").write_text("\n".join(tables) + "\n")

    print(json.dumps({k: v for k, v in summary.items() if k != "auxiliary"}, ensure_ascii=False, indent=2))
    print(f"\nwrote {analysis_dir}/tables.md, summary.json, paired_effects.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
