#!/usr/bin/env python3
"""Read-only post-hoc audit for midterm_uniform_reward_v1.

This script does not modify training targets, evaluations, or the report.
It re-derives, from the on-disk rollout audit, rollout dumps, train logs and
evaluation journals:

1. training metrics after removing post-terminal inactive rows;
2. the reconciliation between terminal reward, episode/reward/mean and
   critic/rewards/mean;
3. per-trajectory trainer-input verification;
4. the R2 cost-signal / normalized-advantage decomposition;
5. the advantage impact of the two R1 verifier false negatives;
6. common-197 paired tool-time sensitivity;
7. McNemar sign conventions and exact tests;
8. timing-source provenance and exact-McNemar MDE.

Outputs are JSON files under
artifacts/experiments/midterm_uniform_reward_v1/audit/posthoc_20261023/.
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import math
import os
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "artifacts" / "experiments" / "midterm_uniform_reward_v1"
OUT = EXP / "audit" / "posthoc_20261023"
GROUPS = ("r0", "r1", "r2")
EVAL_GROUPS = ("sft", "r0", "r1", "r2")
TIMEOUT_TASKS = {
    "math:26fd91ac7be838ec3f8c",
    "math:2892076449fc22ef1afd",
    "math:2c584c5199712c434c39",
}
STEP_RE = re.compile(r"step:(\d+) - ")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SLOT_RE = re.compile(r"#(\d+)$")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_training(group: str):
    """Return resets, per-trajectory rows, and per-trajectory summary."""
    by_traj = collections.defaultdict(list)
    resets = []
    path = EXP / "training" / group / "rollout_audit.jsonl"
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            event = rec.get("event")
            if event == "reset":
                resets.append(rec)
            elif event == "step":
                by_traj[(rec["global_update"], rec["group_id"], rec["sample_index"])].append(rec)

    traj = {}
    for key, rows in by_traj.items():
        rows.sort(key=lambda r: r["step_index"])
        done = [r for r in rows if r.get("done")]
        terminal = done[0] if done else rows[-1]
        T = terminal["step_index"]
        active = [r for r in rows if r["step_index"] <= T]
        inactive = [r for r in rows if r["step_index"] > T]
        traj[key] = {
            "rows": rows,
            "active": active,
            "inactive": inactive,
            "terminal": terminal,
            "turns": T,
            "had_done": bool(done),
        }
    return resets, traj


def load_train_log(group: str):
    metrics = {}
    path = EXP / "training" / group / "train.log"
    with path.open(errors="ignore") as fh:
        for line in fh:
            match = STEP_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            line = ANSI_RE.sub("", line)
            rec = {}
            for key in (
                "critic/rewards/mean",
                "critic/advantages/mean",
                "critic/advantages/max",
                "critic/advantages/min",
                "episode/reward/mean",
                "episode/success_rate",
            ):
                found = re.search(re.escape(key) + r":(-?[0-9.]+)", line)
                if found:
                    rec[key] = float(found.group(1))
            metrics[step] = rec
    return metrics


def load_eval(group: str):
    """Latest valid attempt per task_id (same rule as the analysis)."""
    best = {}
    path = EXP / "evaluation" / "formal_unified_tool_eval_20261023_v2" / f"group_{group}.jsonl"
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if not rec.get("valid"):
                continue
            task = rec["task_id"]
            attempt = rec.get("attempt", 1)
            if task not in best or attempt > best[task].get("attempt", 1):
                best[task] = rec
    return best


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------

def exact_mcnemar_p(b01: int, b10: int) -> float:
    """Exact two-sided McNemar p, matching scipy.stats.binomtest."""
    m = b01 + b10
    if m == 0:
        return 1.0
    k = min(b01, b10)
    p_obs = math.comb(m, k) / 2**m
    total = sum(
        math.comb(m, i) / 2**m
        for i in range(m + 1)
        if math.comb(m, i) / 2**m <= p_obs + 1e-15
    )
    return min(1.0, total)


def holm_adjust(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [None] * len(pvalues)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def percentile(sorted_values, q: float) -> float:
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci(values, indices, seed=20261022, reps=10000):
    """Paired percentile bootstrap using a caller-supplied index matrix."""
    observed = sum(values) / len(values)
    draws = []
    for idx in indices:
        draws.append(sum(values[i] for i in idx) / len(idx))
    draws.sort()
    return observed, percentile(draws, 0.025), percentile(draws, 0.975)


def make_indices(n: int, reps: int = 10000, seed: int = 20261022):
    import random

    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(reps)]


def binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0 if k == 0 else 0.0
    if p >= 1:
        return 1.0 if k == n else 0.0
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def exact_mcnemar_power(n: int, pi_d: float, delta: float, alpha: float = 0.05) -> float:
    p10 = (pi_d + delta) / 2
    p01 = (pi_d - delta) / 2
    if p01 < 0:
        return 0.0
    power = 0.0
    for m in range(n + 1):
        pm = binom_pmf(m, n, pi_d)
        if pm < 1e-12:
            continue
        conditional = 0.0
        for b10 in range(m + 1):
            b01 = m - b10
            if exact_mcnemar_p(b01, b10) <= alpha:
                conditional += binom_pmf(b10, m, p10 / pi_d)
        power += pm * conditional
    return power


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# audit sections
# ---------------------------------------------------------------------------

def audit_active_rows(traj_by_group):
    result = {}
    for group, traj in traj_by_group.items():
        blocks = collections.defaultdict(
            lambda: {"trajectories": 0, "active_rows": 0, "invalid": 0, "tool_calls": 0.0}
        )
        rec = {
            "trajectories": len(traj),
            "rows_total": 0,
            "active_rows": 0,
            "inactive_rows": 0,
            "invalid_active": 0,
            "invalid_inactive": 0,
            "invalid_active_duplicate_rows": 0,
            "tool_calls_active": 0.0,
            "tool_calls_inactive": 0.0,
            "terminal_reward_sum": 0.0,
            "statuses": collections.Counter(),
            "termination_reasons": collections.Counter(),
            "turns_per_trajectory": [],
            "missing_done": 0,
            "trajectories_with_inactive_invalid": 0,
            "inactive_invalid_on_invalid_terminal": 0,
        }
        for key, item in traj.items():
            active = item["active"]
            inactive = item["inactive"]
            terminal = item["terminal"]
            rec["rows_total"] += len(item["rows"])
            rec["active_rows"] += len(active)
            rec["inactive_rows"] += len(inactive)
            rec["invalid_active"] += sum(1 for r in active if not r.get("is_action_valid"))
            rec["invalid_inactive"] += sum(1 for r in inactive if not r.get("is_action_valid"))
            rec["tool_calls_active"] += sum(float(r.get("tool_calling") or 0.0) for r in active)
            rec["tool_calls_inactive"] += sum(float(r.get("tool_calling") or 0.0) for r in inactive)
            rec["terminal_reward_sum"] += float(terminal["reward"])
            rec["statuses"][terminal.get("verifier_status")] += 1
            rec["termination_reasons"][terminal.get("termination_reason")] += 1
            rec["turns_per_trajectory"].append(len(active))
            if not item["had_done"]:
                rec["missing_done"] += 1
            if any(not r.get("is_action_valid") for r in inactive):
                rec["trajectories_with_inactive_invalid"] += 1
                if not terminal.get("is_action_valid"):
                    rec["inactive_invalid_on_invalid_terminal"] += 1

            block = "1-10" if key[0] < 10 else ("41-50" if key[0] >= 40 else None)
            if block:
                bucket = blocks[block]
                bucket["trajectories"] += 1
                bucket["active_rows"] += len(active)
                bucket["invalid"] += sum(1 for r in active if not r.get("is_action_valid"))
                bucket["tool_calls"] += sum(float(r.get("tool_calling") or 0.0) for r in active)

        rec["invalid_rate_active"] = rec["invalid_active"] / rec["active_rows"]
        rec["invalid_rate_all_rows"] = rec["invalid_active"] / rec["rows_total"]
        rec["tool_calls_all_rows"] = rec["tool_calls_active"] + rec["tool_calls_inactive"]
        rec["tool_calls_per_trajectory_active"] = rec["tool_calls_active"] / rec["trajectories"]
        rec["tool_calls_per_trajectory_all_rows"] = rec["tool_calls_all_rows"] / rec["trajectories"]
        rec["actions_per_trajectory_mean"] = sum(rec["turns_per_trajectory"]) / len(
            rec["turns_per_trajectory"]
        )
        rec["terminal_reward_mean"] = rec["terminal_reward_sum"] / rec["trajectories"]
        rec["blocks"] = {}
        for name, bucket in blocks.items():
            rec["blocks"][name] = {
                **bucket,
                "invalid_rate": bucket["invalid"] / bucket["active_rows"],
                "tool_calls_per_trajectory": bucket["tool_calls"] / bucket["trajectories"],
            }
        rec["statuses"] = dict(rec["statuses"])
        rec["termination_reasons"] = dict(rec["termination_reasons"])
        rec.pop("turns_per_trajectory")
        result[group] = rec
    return result


def audit_reward_reconciliation(traj_by_group, logs):
    result = {}
    for group, traj in traj_by_group.items():
        per_update = collections.defaultdict(list)
        weighted_numerator = collections.defaultdict(float)
        weighted_denominator = collections.defaultdict(float)
        for key, item in traj.items():
            gu = key[0]
            reward = float(item["terminal"]["reward"])
            turns = item["turns"]
            per_update[gu].append(reward)
            weighted_numerator[gu] += reward * turns
            weighted_denominator[gu] += turns

        per_trajectory_mean = (
            sum(sum(values) for values in per_update.values())
            / sum(len(values) for values in per_update.values())
        )
        weighted_per_update = [
            weighted_numerator[gu] / weighted_denominator[gu] for gu in sorted(per_update)
        ]
        length_weighted_mean = sum(weighted_per_update) / len(weighted_per_update)
        logged = logs[group]
        logged_critic = sum(logged[s]["critic/rewards/mean"] for s in logged) / len(logged)
        logged_episode = sum(logged[s]["episode/reward/mean"] for s in logged) / len(logged)
        logged_success = sum(logged[s]["episode/success_rate"] for s in logged) / len(logged)

        result[group] = {
            "terminal_reward_per_trajectory_mean": per_trajectory_mean,
            "length_weighted_per_update_mean": length_weighted_mean,
            "logged_critic_rewards_mean": logged_critic,
            "logged_episode_reward_mean": logged_episode,
            "logged_episode_success_rate": logged_success,
            "diff_length_weighted_vs_logged_critic": length_weighted_mean - logged_critic,
            "diff_per_trajectory_vs_logged_episode": per_trajectory_mean - logged_episode,
            "interpretation": (
                "critic/rewards/mean is the unweighted mean of the 50 per-update "
                "length-weighted terminal rewards; each trajectory contributes its "
                "terminal reward once per active turn row. episode/reward/mean is "
                "the per-trajectory mean of the same terminal reward. For R0 the "
                "reward is binary, so 0.298125 vs 0.28264 is purely a weighting "
                "difference, not a reward inconsistency."
            ),
        }
    return result


def audit_trainer_inputs(traj_by_group):
    result = {}
    for group, traj in traj_by_group.items():
        updates = collections.defaultdict(list)
        for key, item in traj.items():
            updates[key[0]].append((key, item))
        files = sorted(
            glob.glob(str(EXP / "training" / group / "rollout_dump" / "*.jsonl")),
            key=lambda p: int(os.path.basename(p).split(".")[0]),
        )
        mismatches = []
        total_lines = 0
        for path in files:
            update = int(os.path.basename(path).split(".")[0]) - 1
            with open(path) as handle:
                dump_scores = [json.loads(line)["score"] for line in handle]
            total_lines += len(dump_scores)
            expected = []
            items = sorted(
                updates.get(update, []),
                key=lambda pair: (int(SLOT_RE.search(pair[0][1]).group(1)), pair[0][2]),
            )
            for _key, item in items:
                expected += [float(item["terminal"]["reward"])] * item["turns"]
            ok = len(expected) == len(dump_scores) and all(
                abs(a - b) < 1e-6 for a, b in zip(expected, dump_scores)
            )
            if not ok:
                mismatches.append(
                    {
                        "update": update + 1,
                        "dump_lines": len(dump_scores),
                        "expected_lines": len(expected),
                    }
                )

        formula_mismatch = 0
        status_mismatch = 0
        missing_done = 0
        for item in traj.values():
            terminal = item["terminal"]
            components = terminal["reward_components"]
            if group == "r0":
                expected_reward = components["correct"]
            elif group == "r1":
                expected_reward = components["correct"] - components["invalid_penalty"]
            else:
                expected_reward = (
                    components["correct"] * (1 - components["tool_cost"] - components["time_cost"])
                    - components["invalid_penalty"]
                )
            expected_reward = max(-1.0, min(1.0, expected_reward))
            if abs(expected_reward - float(terminal["reward"])) > 1e-6:
                formula_mismatch += 1
            if (terminal.get("verifier_status") == "correct") != (
                components["correct"] == 1.0
            ):
                status_mismatch += 1
            if not item["had_done"]:
                missing_done += 1

        result[group] = {
            "trajectories": len(traj),
            "rollout_dump_files": len(files),
            "rollout_dump_lines": total_lines,
            "expected_active_rows": sum(item["turns"] for item in traj.values()),
            "update_score_mismatches": mismatches,
            "reward_formula_mismatches": formula_mismatch,
            "verifier_status_vs_correct_component_mismatches": status_mismatch,
            "trajectories_without_done": missing_done,
            "note": (
                "Dump lines are ordered by (group slot, sample_index) and match "
                "terminal reward repeated once per active turn within 1e-6 for "
                "all updates. This verifies the trainer input rows and explains "
                "the turn-row weighting in critic/rewards/mean."
            ),
        }
    return result


def audit_r2_cost(traj_by_group):
    traj = traj_by_group["r2"]
    groups = collections.defaultdict(list)
    for key, item in traj.items():
        groups[(key[0], key[1])].append((key[2], item["terminal"]))
    composition = collections.Counter()
    nonzero = {"all_correct": 0, "all_wrong": 0, "mixed": 0}
    for samples in groups.values():
        rewards = [float(t["reward"]) for _, t in samples]
        correct = sum(1 for _, t in samples if t["reward_components"]["correct"] == 1.0)
        kind = "all_correct" if correct == 4 else ("all_wrong" if correct == 0 else "mixed")
        composition[kind] += 1
        if len(set(rewards)) > 1:
            nonzero[kind] += 1

    all_correct = [
        samples
        for samples in groups.values()
        if all(t["reward_components"]["correct"] == 1.0 for _, t in samples)
    ]
    group_rows = []
    for samples in all_correct:
        rewards = [float(t["reward"]) for _, t in samples]
        mean = sum(rewards) / len(rewards)
        std = statistics.stdev(rewards)
        advantages = [(reward - mean) / (std + 1e-6) for reward in rewards]
        tool_cost = [t["reward_components"]["tool_cost"] for _, t in samples]
        time_cost = [t["reward_components"]["time_cost"] for _, t in samples]
        count_only = [1.0 - value for value in tool_cost]
        time_only = [1.0 - value for value in time_cost]

        def max_abs_adv(values):
            mu = sum(values) / len(values)
            sd = statistics.stdev(values)
            return max(abs((value - mu) / (sd + 1e-6)) for value in values)

        group_rows.append(
            {
                "rewards": rewards,
                "mean": mean,
                "std": std,
                "advantages": advantages,
                "max_abs_advantage": max(abs(a) for a in advantages),
                "tool_cost": tool_cost,
                "time_cost": time_cost,
                "count_only_max_abs_advantage": max_abs_adv(count_only),
                "time_only_max_abs_advantage": max_abs_adv(time_only),
                "tool_cost_spread": max(tool_cost) - min(tool_cost),
                "time_cost_spread": max(time_cost) - min(time_cost),
                "time_spread_seconds": (max(time_cost) - min(time_cost)) * 600.0,
            }
        )

    all_advantages = [a for row in group_rows for a in row["advantages"]]
    spreads = sorted(row["time_spread_seconds"] for row in group_rows)
    result = {
        "composition": dict(composition),
        "nonzero_std_groups": nonzero,
        "all_correct_groups": len(all_correct),
        "all_correct_groups_with_tool_cost_spread": sum(
            1 for row in group_rows if row["tool_cost_spread"] > 1e-12
        ),
        "all_correct_groups_with_time_cost_spread": sum(
            1 for row in group_rows if row["time_cost_spread"] > 1e-12
        ),
        "all_correct_std": {
            "min": min(row["std"] for row in group_rows),
            "median": statistics.median(row["std"] for row in group_rows),
            "max": max(row["std"] for row in group_rows),
        },
        "all_correct_max_abs_advantage": {
            "min_per_group": min(row["max_abs_advantage"] for row in group_rows),
            "median_per_group": statistics.median(
                row["max_abs_advantage"] for row in group_rows
            ),
            "max_per_group": max(row["max_abs_advantage"] for row in group_rows),
        },
        "all_correct_advantage_counts": {
            "samples": len(all_advantages),
            "abs_gt_0_5": sum(1 for a in all_advantages if abs(a) > 0.5),
            "abs_gt_1_0": sum(1 for a in all_advantages if abs(a) > 1.0),
        },
        "counterfactual": {
            "count_only_median_max_abs_advantage": statistics.median(
                row["count_only_max_abs_advantage"] for row in group_rows
            ),
            "time_only_median_max_abs_advantage": statistics.median(
                row["time_only_max_abs_advantage"] for row in group_rows
            ),
            "actual_median_max_abs_advantage": statistics.median(
                row["max_abs_advantage"] for row in group_rows
            ),
        },
        "time_spread_seconds": {
            "min": spreads[0],
            "median": statistics.median(spreads),
            "p75": statistics.quantiles(spreads, n=4)[2],
            "max": spreads[-1],
            "groups_gt_0_5s": sum(1 for value in spreads if value > 0.5),
        },
        "top_groups_by_advantage": sorted(
            group_rows, key=lambda row: -row["max_abs_advantage"]
        )[:5],
        "interpretation": (
            "Raw reward spreads around 1e-3 are divided by a group std of the "
            "same order, so GRPO advantages are O(1). In 32 of 33 all-correct "
            "groups the only variation is time_cost; the remaining group has a "
            "0-call versus 1-call difference. Whether the time variation is "
            "genuine efficiency or timing noise is not identifiable without "
            "repeated executions of identical code."
        ),
    }
    return result


def audit_false_negatives(traj_by_group):
    traj = traj_by_group["r1"]
    affected = [
        (3, "g3:math:163a181d4fc954c57e62#5", 2),
        (39, "g39:math:530dbff74491f8eca29f#1", 2),
    ]
    result = []
    for gu, gid, sample in affected:
        samples = {}
        for (key_gu, key_gid, key_sample), item in traj.items():
            if key_gu == gu and key_gid == gid:
                samples[key_sample] = item["terminal"]
        if sample not in samples:
            result.append({"global_update": gu, "group_id": gid, "missing": True})
            continue
        ordered = sorted(samples)
        before = [float(samples[s]["reward"]) for s in ordered]
        after = list(before)
        idx = ordered.index(sample)
        penalty = samples[sample]["reward_components"].get("invalid_penalty", 0.0)
        after[idx] = max(-1.0, min(1.0, 1.0 - penalty))

        def adv(values):
            mean = sum(values) / len(values)
            std = statistics.stdev(values)
            return mean, std, [(value - mean) / (std + 1e-6) for value in values]

        mean_b, std_b, adv_b = adv(before)
        mean_a, std_a, adv_a = adv(after)
        result.append(
            {
                "global_update": gu,
                "group_id": gid,
                "sample_index": sample,
                "terminal_status": samples[sample].get("verifier_status"),
                "rewards_before": before,
                "rewards_after": after,
                "mean_before": mean_b,
                "std_before": std_b,
                "advantages_before": adv_b,
                "mean_after": mean_a,
                "std_after": std_a,
                "advantages_after": adv_a,
                "other_samples_advantage_delta": [
                    adv_a[i] - adv_b[i] for i in range(len(ordered)) if i != idx
                ],
            }
        )
    return result


def audit_time_sensitivity(eval_data):
    ids = set.intersection(*(set(data) for data in eval_data.values()))
    common = sorted(ids)
    kept = sorted(ids - TIMEOUT_TASKS)

    def seconds(group, task):
        return float(eval_data[group][task]["usage"].get("tool_seconds", 0.0))

    def paired_diff(a, b, tasks, indices):
        values = [seconds(a, task) - seconds(b, task) for task in tasks]
        return bootstrap_ci(values, indices)

    indices_200 = make_indices(len(common), seed=20261022)
    indices_197 = make_indices(len(kept), seed=20261022)

    result = {
        "common_task_ids": len(common),
        "kept_task_ids": len(kept),
        "excluded_timeout_tasks": sorted(TIMEOUT_TASKS),
        "excluded_task_tool_seconds": {
            task: {group: seconds(group, task) for group in EVAL_GROUPS}
            for task in sorted(TIMEOUT_TASKS)
        },
        "group_means": {
            group: {
                "mean_200": sum(seconds(group, task) for task in common) / len(common),
                "mean_197": sum(seconds(group, task) for task in kept) / len(kept),
                "sum_197": sum(seconds(group, task) for task in kept),
            }
            for group in EVAL_GROUPS
        },
        "paired_197": {},
        "paired_200_reference": {},
    }
    for treated, baseline in (("r2", "r1"), ("r1", "r0"), ("r0", "sft"), ("r2", "sft")):
        observed, low, high = paired_diff(treated, baseline, kept, indices_197)
        result["paired_197"][f"{treated}-{baseline}"] = {
            "delta_seconds_per_task": observed,
            "ci95_low": low,
            "ci95_high": high,
        }
    for treated, baseline in (("r2", "r1"), ("r1", "r0"), ("r0", "sft")):
        observed, low, high = paired_diff(treated, baseline, common, indices_200)
        result["paired_200_reference"][f"{treated}-{baseline}"] = {
            "delta_seconds_per_task": observed,
            "ci95_low": low,
            "ci95_high": high,
        }
    result["interpretation"] = (
        "The three tasks hit by R1's 30-second timeouts are removed from all "
        "four groups. On the common 197 tasks R2-R1 is +0.0005 s/task "
        "[-0.0065, +0.0082], i.e. no evidence that R2 is cheaper; the raw "
        "200-task -0.4608 s/task is entirely the R1 timeout artifact."
    )
    return result


def audit_mcnemar(eval_data):
    ids = sorted(set.intersection(*(set(data) for data in eval_data.values())))

    def correct(group, task):
        return bool(eval_data[group][task]["correct"])

    comparisons = []
    for treated, baseline in (("r0", "sft"), ("r1", "r0"), ("r2", "r1")):
        b01 = sum(1 for task in ids if not correct(baseline, task) and correct(treated, task))
        b10 = sum(1 for task in ids if correct(baseline, task) and not correct(treated, task))
        p = exact_mcnemar_p(b01, b10)
        comparisons.append(
            {
                "comparison": f"{treated}-{baseline}",
                "treated": treated,
                "baseline": baseline,
                "n_pairs": len(ids),
                "b01_baseline_wrong_treated_correct": b01,
                "b10_baseline_correct_treated_wrong": b10,
                "delta_correct_rate": (b01 - b10) / len(ids),
                "odds_ratio_improved_over_regressed": b01 / b10 if b10 else None,
                "mcnemar_exact_two_sided_p": p,
            }
        )
    adjusted = holm_adjust([row["mcnemar_exact_two_sided_p"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_p"] = value
    return {
        "comparisons": comparisons,
        "note": (
            "The table and p-values in report appendix D and section 6.2's "
            "reading bullets use b01=baseline wrong/treated correct and "
            "b10=baseline correct/treated wrong. Section 2.8 and the formula "
            "line at the top of section 6.2 invert these labels; the numbers "
            "themselves are correct."
        ),
    }


def audit_timing(eval_data, traj_by_group):
    tally = {}
    for group in EVAL_GROUPS:
        counts = collections.Counter()
        timeouts = []
        for task, rec in eval_data[group].items():
            for call in rec.get("tool_calls", []):
                key = (
                    call.get("name"),
                    call.get("timing_source"),
                    bool(call.get("timing_missing", False)),
                    call.get("error_code"),
                )
                counts[str(key)] += 1
                if call.get("error_code") == "timeout":
                    timeouts.append(
                        {
                            "task_id": task,
                            "name": call.get("name"),
                            "tool_seconds": call.get("tool_seconds"),
                            "latency_ms": call.get("latency_ms"),
                            "timeout_limit": call.get("timeout_limit"),
                            "timing_source": call.get("timing_source"),
                            "timing_missing": call.get("timing_missing"),
                            "infrastructure_error": call.get("infrastructure_error"),
                            "python_seconds": rec["usage"].get("python_seconds"),
                        }
                    )
        tally[group] = {"tool_call_timing_counts": dict(counts), "timeouts": timeouts}

    training_tools = {}
    for group, traj in traj_by_group.items():
        counts = collections.Counter()
        for item in traj.values():
            for row in item["active"]:
                if not row.get("tool_calling"):
                    continue
                match = re.search(r'"name"\s*:\s*"([^"]+)"', row.get("action_text", ""))
                counts[match.group(1) if match else "unparsed"] += 1
        training_tools[group] = dict(counts)

    return {
        "evaluation": tally,
        "training_active_tool_names": training_tools,
        "notes": [
            (
                "All evaluation calls are python; normal calls use "
                "sandbox_execution_time with timing_missing=false."
            ),
            (
                "The three R1 timeouts have timing_source=client_round_trip, "
                "timing_missing=true, python_seconds=0, and tool_seconds≈31.03 "
                "(requested timeout 30 s plus round-trip overhead). They are "
                "charged fallback values, not measured execution time."
            ),
            (
                "Training is predominantly python; a small number of sympy calls "
                "use worker_process wall time. R2's time term therefore mixes "
                "sandbox execution time and worker-process wall time, not one "
                "uniform measurement boundary."
            ),
        ],
    }


def audit_mde():
    rows = {}
    for pi_d in (0.04, 0.05, 0.055, 0.06):
        powers = {
            delta: exact_mcnemar_power(200, pi_d, delta)
            for delta in (0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06)
        }
        mde80 = next((delta for delta, power in powers.items() if power >= 0.8), None)
        mde90 = next((delta for delta, power in powers.items() if power >= 0.9), None)
        rows[str(pi_d)] = {
            "power_by_delta": powers,
            "mde80_grid": mde80,
            "mde90_grid": mde90,
        }
    return {
        "n_pairs": 200,
        "alpha": 0.05,
        "two_sided_exact_mcnemar": True,
        "rows": rows,
        "note": (
            "CI width is not the minimum detectable effect. With 10-11 "
            "discordant pairs out of 200 (pi_d≈0.05-0.055), the exact McNemar "
            "test reaches 80% power only for true differences around 4.5-5.0 "
            "percentage points; the observed CI widths of about 6-7 pp reflect "
            "the realized split, not a 6-7 pp detection floor."
        ),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    traj_by_group = {}
    resets_by_group = {}
    logs = {}
    for group in GROUPS:
        resets, traj = load_training(group)
        resets_by_group[group] = resets
        traj_by_group[group] = traj
        logs[group] = load_train_log(group)

    eval_data = {group: load_eval(group) for group in EVAL_GROUPS}

    outputs = {
        "active_row_recompute.json": audit_active_rows(traj_by_group),
        "reward_reconciliation.json": audit_reward_reconciliation(traj_by_group, logs),
        "trainer_input_verification.json": audit_trainer_inputs(traj_by_group),
        "r2_cost_advantage.json": audit_r2_cost(traj_by_group),
        "false_negative_impact.json": audit_false_negatives(traj_by_group),
        "time_sensitivity_common197.json": audit_time_sensitivity(eval_data),
        "mcnemar_sign_audit.json": audit_mcnemar(eval_data),
        "timing_measurement_audit.json": audit_timing(eval_data, traj_by_group),
        "mcnemar_mde.json": audit_mde(),
    }

    source_files = []
    for group in GROUPS:
        source_files.extend(
            [
                EXP / "training" / group / "rollout_audit.jsonl",
                EXP / "training" / group / "train.log",
            ]
        )
    for group in EVAL_GROUPS:
        source_files.append(
            EXP
            / "evaluation"
            / "formal_unified_tool_eval_20261023_v2"
            / f"group_{group}.jsonl"
        )

    provenance = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repo_root": str(ROOT),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_files
        },
        "reset_counts": {group: len(resets) for group, resets in resets_by_group.items()},
        "reset_unique_pairs": {
            group: len({(rec["prompt_slot_id"], rec["sample_index"]) for rec in resets})
            for group, resets in resets_by_group.items()
        },
    }
    outputs["provenance.json"] = provenance

    for name, payload in outputs.items():
        with (OUT / name).open("w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=False)
            fh.write("\n")
    print(f"wrote {len(outputs)} files to {OUT}")


if __name__ == "__main__":
    main()
