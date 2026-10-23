#!/usr/bin/env python
"""Read-only check of the GRPO advantage standard-deviation convention.

For every training update, rebuild per-group GRPO advantages from the trainer
dump scores using both the sample standard deviation (G-1, torch.std default)
and the population standard deviation (G), then compare the batch max/min with
the values logged in train.log.  The check is read-only and writes one JSON
under the posthoc audit directory.
"""

from __future__ import annotations

import collections
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "artifacts" / "experiments" / "midterm_uniform_reward_v1"
OUT = EXP / "audit" / "posthoc_20261023" / "advantage_std_check.json"

STEP_RE = re.compile(r"step:(\d+) - ")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SLOT_RE = re.compile(r"#(\d+)$")
GROUPS = ("r0", "r1", "r2")


def load_trajectories(group: str):
    traj = {}
    path = EXP / "training" / group / "rollout_audit.jsonl"
    for line in path.open():
        rec = json.loads(line)
        key = (rec["global_update"], rec["group_id"], rec["sample_index"])
        if rec.get("event") == "reset":
            traj[key] = {"rows": [], "done": None}
        elif rec.get("event") == "step":
            item = traj[key]
            item["rows"].append(rec)
            if item["done"] is None and rec.get("done"):
                item["done"] = rec
    return traj


def load_log(group: str):
    metrics = {}
    path = EXP / "training" / group / "train.log"
    for line in path.open(errors="ignore"):
        match = STEP_RE.search(line)
        if not match:
            continue
        line = ANSI_RE.sub("", line)
        step = int(match.group(1))
        rec = {}
        for key in ("critic/advantages/max", "critic/advantages/min"):
            found = re.search(re.escape(key) + r":(-?[0-9.]+)", line)
            if found:
                rec[key] = float(found.group(1))
        metrics[step] = rec
    return metrics


def advantage_check(group: str):
    traj = load_trajectories(group)
    logs = load_log(group)

    updates = collections.defaultdict(list)
    for key, item in traj.items():
        terminal = item["done"] or item["rows"][-1]
        updates[key[0]].append((key, terminal["step_index"]))

    exact_g1 = 0
    exact_g = 0
    errors_g1 = []
    errors_g = []
    checked = 0

    for update, items in sorted(updates.items()):
        dump_path = EXP / "training" / group / "rollout_dump" / f"{update + 1}.jsonl"
        dump_scores = [json.loads(line)["score"] for line in dump_path.open()]
        items_sorted = sorted(
            items,
            key=lambda pair: (int(SLOT_RE.search(pair[0][1]).group(1)), pair[0][2]),
        )

        # The dump is ordered by (group slot, sample_index); each sample has
        # one line per active turn.
        index = 0
        score_by_sample = {}
        for key, turns in items_sorted:
            rows = dump_scores[index : index + turns]
            index += turns
            score_by_sample[key] = sum(rows) / len(rows)

        groups = collections.defaultdict(list)
        for key, _turns in items_sorted:
            groups[key[1]].append((key[2], score_by_sample[key]))

        adv_g1 = []
        adv_g = []
        for samples in groups.values():
            rewards = [reward for _sample, reward in samples]
            mean = sum(rewards) / len(rewards)
            sd_g1 = statistics.stdev(rewards)  # sample std, G-1
            sd_g = statistics.pstdev(rewards)  # population std, G
            adv_g1.extend((reward - mean) / (sd_g1 + 1e-6) for reward in rewards)
            adv_g.extend((reward - mean) / (sd_g + 1e-6) for reward in rewards)

        log = logs.get(update + 1)
        if not log or "critic/advantages/max" not in log:
            continue
        checked += 1

        err_g1 = max(
            abs(max(adv_g1) - log["critic/advantages/max"]),
            abs(min(adv_g1) - log["critic/advantages/min"]),
        )
        err_g = max(
            abs(max(adv_g) - log["critic/advantages/max"]),
            abs(min(adv_g) - log["critic/advantages/min"]),
        )
        errors_g1.append(err_g1)
        errors_g.append(err_g)
        if (
            round(max(adv_g1), 3) == log["critic/advantages/max"]
            and round(min(adv_g1), 3) == log["critic/advantages/min"]
        ):
            exact_g1 += 1
        if (
            round(max(adv_g), 3) == log["critic/advantages/max"]
            and round(min(adv_g), 3) == log["critic/advantages/min"]
        ):
            exact_g += 1

    return {
        "updates_checked": checked,
        "g_minus_1_exact_after_log_rounding_3dp": exact_g1,
        "g_exact_after_log_rounding_3dp": exact_g,
        "max_abs_error_g_minus_1": max(errors_g1),
        "max_abs_error_g": max(errors_g),
    }


def main():
    result = {
        "method": (
            "Rebuild per-group GRPO advantages from rollout_dump scores using "
            "statistics.stdev (G-1) and statistics.pstdev (G); compare batch "
            "max/min with train.log values rounded to 3 decimals."
        ),
        "groups": {group: advantage_check(group) for group in GROUPS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
