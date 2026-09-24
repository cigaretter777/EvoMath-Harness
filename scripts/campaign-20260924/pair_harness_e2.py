"""Paired statistics for the E2 harness campaign and the C1 batch confound.

E2: champion cap=1024 vs candidate cap=2048 on the frozen Omni-MATH 200, same
checkpoint, same task order, greedy, batch 1. Reuses the production estimator
(evaluation.model_eval._paired_statistics) so the interval and the exact McNemar
p-value are computed by the same code that produced the four-way comparison.

C1: the 2026-09-14 SFT baseline was generated at batch_size=8 and the 2026-09-19
RL arms at batch_size=1. H1 reproduces the baseline's model identity at batch 1,
so H1 vs the stored baseline isolates the batching (and commit) difference. Raw
string equality is reported alongside verdict agreement: greedy decoding should
be batch invariant, and if it is not, that is a property of the stack worth
knowing before any paired claim rests on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_math.evaluation.model_eval import _paired_statistics  # noqa: E402


def _rows(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["task_id"]] = row
    return out


def _correct(row: dict) -> bool:
    return row.get("verifier_status") == "correct"


def _truncation_share(rows: dict[str, dict], cap: int) -> dict:
    tokens = [int(r["output_tokens"]) for r in rows.values()]
    at_cap = sum(1 for t in tokens if t >= cap - 4)
    return {
        "mean_output_tokens": round(sum(tokens) / len(tokens), 1),
        "median_output_tokens": sorted(tokens)[len(tokens) // 2],
        "at_cap": at_cap,
        "at_cap_rate": round(at_cap / len(tokens), 4),
    }


def _latency(rows: dict[str, dict]) -> dict:
    lat = sorted(float(r["generation_latency_ms"]) for r in rows.values())
    n = len(lat)
    return {
        "p50_ms": round(lat[n // 2], 1),
        "p95_ms": round(lat[int(n * 0.95)], 1),
        "mean_ms": round(sum(lat) / n, 1),
        "total_minutes": round(sum(lat) / 1000 / 60, 1),
    }


def e2(champion: Path, candidate: Path, *, cap_champion: int, cap_candidate: int) -> dict:
    a = _rows(champion)
    b = _rows(candidate)
    assert set(a) == set(b), "arms must cover the same task ids"

    comparison = [
        {
            "task_id": tid,
            # _paired_statistics reads these two keys: "base" is the reference
            # arm (champion) and "sft" the comparison arm (candidate).
            "base_correct": _correct(a[tid]),
            "sft_correct": _correct(b[tid]),
            "base_status": a[tid]["verifier_status"],
            "sft_status": b[tid]["verifier_status"],
        }
        for tid in sorted(a)
    ]
    stats = _paired_statistics(comparison)
    paired = dict(Counter(
        "improved" if c["sft_correct"] and not c["base_correct"]
        else "regressed" if c["base_correct"] and not c["sft_correct"]
        else "unchanged"
        for c in comparison
    ))

    status_shift = Counter(
        f"{a[tid]['verifier_status']}->{b[tid]['verifier_status']}"
        for tid in sorted(a)
        if a[tid]["verifier_status"] != b[tid]["verifier_status"]
    )

    return {
        "task_count": len(comparison),
        "correct_champion": sum(1 for c in comparison if c["base_correct"]),
        "correct_candidate": sum(1 for c in comparison if c["sft_correct"]),
        "delta_correct": sum(1 for c in comparison if c["sft_correct"])
        - sum(1 for c in comparison if c["base_correct"]),
        "paired": paired,
        "paired_statistics": stats,
        "verifier_status_shifts": dict(status_shift.most_common()),
        "champion": {
            "cap": cap_champion,
            "statuses": dict(Counter(r["verifier_status"] for r in a.values())),
            "truncation": _truncation_share(a, cap_champion),
            "latency": _latency(a),
        },
        "candidate": {
            "cap": cap_candidate,
            "statuses": dict(Counter(r["verifier_status"] for r in b.values())),
            "truncation": _truncation_share(b, cap_candidate),
            "latency": _latency(b),
        },
        "changed_raw_outputs": sum(
            1 for tid in a if a[tid]["raw_output"] != b[tid]["raw_output"]
        ),
    }


def c1(baseline_batch8: Path, h1_batch1: Path) -> dict:
    a = _rows(baseline_batch8)
    b = _rows(h1_batch1)
    shared = sorted(set(a) & set(b))
    verdict_agree = sum(1 for t in shared if a[t]["verifier_status"] == b[t]["verifier_status"])
    correct_agree = sum(1 for t in shared if _correct(a[t]) == _correct(b[t]))
    raw_identical = sum(1 for t in shared if a[t]["raw_output"] == b[t]["raw_output"])
    discordant = [
        {
            "task_id": t,
            "baseline_status": a[t]["verifier_status"],
            "h1_status": b[t]["verifier_status"],
            "baseline_tokens": a[t]["output_tokens"],
            "h1_tokens": b[t]["output_tokens"],
            "raw_identical": a[t]["raw_output"] == b[t]["raw_output"],
        }
        for t in shared
        if a[t]["verifier_status"] != b[t]["verifier_status"]
    ]
    return {
        "shared_task_count": len(shared),
        "verdict_agreement": verdict_agree,
        "verdict_agreement_rate": round(verdict_agree / len(shared), 4),
        "correct_agreement": correct_agree,
        "correct_agreement_rate": round(correct_agree / len(shared), 4),
        "raw_output_identical": raw_identical,
        "raw_output_identical_rate": round(raw_identical / len(shared), 4),
        "baseline_correct": sum(1 for t in shared if _correct(a[t])),
        "h1_correct": sum(1 for t in shared if _correct(b[t])),
        "discordant_verdicts": discordant[:20],
        "discordant_count": len(discordant),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-batch8", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "e2_cap_patch": e2(
            args.champion, args.candidate, cap_champion=1024, cap_candidate=2048
        ),
        "c1_batch_confound": c1(args.baseline_batch8, args.champion),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
