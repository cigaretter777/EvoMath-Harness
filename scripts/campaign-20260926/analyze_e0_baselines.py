"""Analyze the thesis E0/E1 baseline campaign (2026-09-26).

Reads the three queue-slot outputs plus the stored E2 champion arm and
produces: per-arm correctness/cost statistics, the four pre-registered paired
comparisons (McNemar exact + paired bootstrap), and the byte-identity
cross-check between the regenerated SFT arm and the E2 champion arm.

CPU-only. Every number it prints is derived from artifacts on disk.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from adaptive_math.evaluation.model_eval import _paired_statistics

REPO = Path(__file__).resolve().parents[2]
ARTS = REPO / "artifacts"

DIRECT = ARTS / "eval/thesis_e0_base_direct_b1"
BASE_TOOL = ARTS / "rollout_health/thesis_e0_base_tool"
RULE = ARTS / "rollout_health/thesis_e0_rule_strategy"
CHAMPION = ARTS / "eval/harness_e2_champion_cap1024/sft_predictions.jsonl"


def load_predictions(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["task_id"]] = row
    return rows


def load_trajectories(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def verdict_label(record: dict) -> str:
    verdict = record["verdict"]
    if verdict is None:
        return "invalid_prediction"
    return verdict["status"]


def correct_count(rows: dict[str, dict]) -> int:
    return sum(1 for r in rows.values() if r["verifier_status"] == "correct")


def summarize_direct(name: str, rows: dict[str, dict]) -> dict:
    statuses = Counter(r["verifier_status"] for r in rows.values())
    lat = sorted(r["generation_latency_ms"] for r in rows.values())
    toks = sorted(r["output_tokens"] for r in rows.values())
    return {
        "arm": name,
        "n": len(rows),
        "correct": statuses["correct"],
        "incorrect": statuses["incorrect"],
        "invalid_prediction": statuses["invalid_prediction"],
        "latency_ms": {
            "mean": round(sum(lat) / len(lat)),
            "p95": lat[int(len(lat) * 0.95)],
        },
        "output_tokens": {
            "mean": round(sum(toks) / len(toks)),
            "p95": toks[int(len(toks) * 0.95)],
        },
    }


def summarize_agent(name: str, records: list[dict]) -> dict:
    labels = Counter(verdict_label(r) for r in records)
    tool_calls = sorted(r["trajectory"]["usage"]["tool_calls"] for r in records)
    invalids = sorted(r["trajectory"]["usage"]["invalid_actions"] for r in records)
    tokens = sorted(r["trajectory"]["usage"]["generated_tokens"] for r in records)
    lat = sorted(r["rollout_seconds"] for r in records)
    n = len(records)

    def pct(sorted_vals, q):
        return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]

    return {
        "arm": name,
        "n": n,
        "correct": labels["correct"],
        "incorrect": labels["incorrect"],
        "invalid_prediction": labels["invalid_prediction"],
        "tool_calls": {
            "mean": round(sum(tool_calls) / n, 3),
            "median": pct(tool_calls, 0.5),
            "p95": pct(tool_calls, 0.95),
            "hit_cap_4": sum(1 for t in tool_calls if t >= 4),
            "zero": sum(1 for t in tool_calls if t == 0),
        },
        "invalid_actions": {
            "mean": round(sum(invalids) / n, 3),
            "total": sum(invalids),
        },
        "generated_tokens": {
            "mean": round(sum(tokens) / n),
            "p95": pct(tokens, 0.95),
        },
        "rollout_seconds": {
            "mean": round(sum(lat) / n),
            "p95": pct(lat, 0.95),
        },
    }


def paired(name: str, arm_a: dict[str, bool], arm_b: dict[str, bool]) -> dict:
    common = sorted(set(arm_a) & set(arm_b))
    comparison = [
        {"sft_correct": arm_a[t], "base_correct": arm_b[t]} for t in common
    ]
    stats = _paired_statistics(comparison)
    stats["n_paired"] = len(common)
    stats["comparison"] = name
    # sft/base key names are the helper's hardcoded schema; relabel for the
    # report so readers see which arms were compared.
    stats["delta_improved"] = sum(
        1 for c in comparison if int(bool(c["sft_correct"])) - int(bool(c["base_correct"])) > 0
    )
    stats["delta_regressed"] = sum(
        1 for c in comparison if int(bool(c["sft_correct"])) - int(bool(c["base_correct"])) < 0
    )
    return stats


def byte_identity_check() -> dict:
    new_sft = load_predictions(DIRECT / "sft_predictions.jsonl")
    old_sft = load_predictions(CHAMPION)
    common = sorted(set(new_sft) & set(old_sft))
    identical = sum(
        1 for t in common if new_sft[t]["raw_output"] == old_sft[t]["raw_output"]
    )
    return {
        "n_shared": len(common),
        "byte_identical": identical,
        "pass": identical == len(common) == 200,
    }


def main() -> int:
    missing = [
        str(p)
        for p in (
            DIRECT / "base_predictions.jsonl",
            DIRECT / "sft_predictions.jsonl",
            BASE_TOOL / "trajectories.jsonl",
            BASE_TOOL / "COMPLETE",
            RULE / "trajectories.jsonl",
            RULE / "COMPLETE",
        )
        if not p.exists()
    ]
    if missing:
        print("MISSING artifacts:", *missing, sep="\n  ")
        return 1

    base_direct = load_predictions(DIRECT / "base_predictions.jsonl")
    sft_direct = load_predictions(DIRECT / "sft_predictions.jsonl")
    base_tool = load_trajectories(BASE_TOOL / "trajectories.jsonl")
    rule = load_trajectories(RULE / "trajectories.jsonl")
    with (RULE / "routing.jsonl").open() as handle:
        routing = {json.loads(l)["task_id"]: json.loads(l)["channel"] for l in handle}

    identity = byte_identity_check()

    # Direct arms keep their stored verifier_status (computed at eval time by
    # the same code revision that generated them).
    direct_base_correct = {
        t: r["verifier_status"] == "correct" for t, r in base_direct.items()
    }
    direct_sft_correct = {
        t: r["verifier_status"] == "correct" for t, r in sft_direct.items()
    }
    tool_correct = {r["task_id"]: verdict_label(r) == "correct" for r in base_tool}
    rule_correct = {r["task_id"]: verdict_label(r) == "correct" for r in rule}
    # Fixed-rule direct channel: reuse the base-direct arm's verdicts.
    rule_full_correct = dict(rule_correct)
    for task_id, channel in routing.items():
        if channel == "direct":
            rule_full_correct[task_id] = direct_base_correct.get(task_id, False)

    per_arm = [
        summarize_direct("E0a base-direct", base_direct),
        summarize_direct("E1 sft-direct", sft_direct),
        summarize_agent("E0b base+tool", base_tool),
        summarize_agent("E0c fixed-rule (rolled out)", rule),
    ]
    rule_full = {
        "rule_correct": sum(rule_full_correct.values()),
        "rule_n": len(rule_full_correct),
        "rule_rolled_out_correct": sum(rule_correct.values()),
        "rule_direct_reused": sum(
            1 for c in routing.values() if c == "direct"
        ),
    }

    pairings = [
        paired("E0b base+tool vs E0a base-direct", tool_correct, direct_base_correct),
        paired("E0c fixed-rule vs E0a base-direct", rule_full_correct, direct_base_correct),
        paired("E0c fixed-rule vs E0b base+tool", rule_full_correct, tool_correct),
        paired("E1 sft-direct vs E0a base-direct", direct_sft_correct, direct_base_correct),
    ]

    report = {
        "identity_check": identity,
        "per_arm": per_arm,
        "rule_arm": rule_full,
        "pairings": pairings,
    }
    out_dir = ARTS / "results/campaign-20260926"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )

    print("=== identity check (sft arm vs E2 champion) ===")
    print(json.dumps(identity, indent=2))
    print("\n=== per-arm ===")
    for arm in per_arm:
        print(json.dumps(arm, ensure_ascii=False, indent=2))
    print("\n=== fixed-rule composite ===")
    print(json.dumps(rule_full, indent=2))
    print("\n=== paired comparisons ===")
    for p in pairings:
        print(json.dumps(p, indent=2))
    print(f"\nwrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
