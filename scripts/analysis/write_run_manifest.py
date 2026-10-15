"""Merge the per-stage manifests into the run-level manifest (section 11).

The stage runner writes ``manifest.<stage>.json`` next to its own artifacts;
section 11 asks for a single ``manifest.json`` describing the whole run. This
merges them without inventing anything: stage entries are copied verbatim, and
the run-level fields are derived from those entries plus ``protocol.lock.json``
and the per-stage summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STAGES = ("a", "b", "c")


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    stages: dict[str, Any] = {}
    for stage in STAGES:
        stages[stage] = _read(run_dir / f"manifest.{stage}.json")
    lock = _read(run_dir / "protocol.lock.json")

    stage_summary: dict[str, Any] = {}
    for stage, name in (("a", "stage_a"), ("b", "stage_b"), ("c", "stage_c")):
        summary = _read(run_dir / name / "summary.json")
        if summary is not None:
            stage_summary[stage] = summary

    missing_stages = [stage for stage, payload in stages.items() if payload is None]
    manifest: dict[str, Any] = {
        "run_id": next(
            (payload.get("run_id") for payload in stages.values() if payload), None
        ),
        "written_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": {
            "path": (lock or {}).get("frozen", {}).get("protocol", {}).get("path"),
            "sha256": (lock or {}).get("frozen", {}).get("protocol", {}).get("sha256"),
            "spec_hash": (lock or {}).get("spec_hash"),
        },
        "model": {
            "path": next(
                (payload.get("model") for payload in stages.values() if payload), None
            ),
            "version": next(
                (payload.get("model_version") for payload in stages.values() if payload),
                None,
            ),
            "chat_template": next(
                (payload.get("chat_template") for payload in stages.values() if payload),
                None,
            ),
        },
        "budget": next(
            (payload.get("budget") for payload in stages.values() if payload), None
        ),
        "generation": next(
            (payload.get("generation") for payload in stages.values() if payload), None
        ),
        "data": {
            stage: payload.get("data")
            for stage, payload in stages.items()
            if payload is not None
        },
        "sandbox_probe": next(
            (payload.get("sandbox_probe") for payload in stages.values() if payload),
            None,
        ),
        "stages": stages,
        "stage_summaries": stage_summary,
        "stages_missing": missing_stages,
        "status_derivation": (
            "STATUS.json is authoritative; this manifest records what each stage "
            "reported, including stages that were never run"
        ),
        "artifacts": {
            name: {"sha256": _sha256(run_dir / name)}
            for name in (
                "protocol.lock.json",
                "run_plan.md",
                "DEVIATIONS.md",
                "stage_a/summary.json",
                "stage_b/trajectories.jsonl",
                "anchors/candidates.jsonl",
                "anchors/selected.jsonl",
                "stage_c/continuations.jsonl",
            )
            if (run_dir / name).is_file()
        },
    }
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out} (missing stages: {missing_stages or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
