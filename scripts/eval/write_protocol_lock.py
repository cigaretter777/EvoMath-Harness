"""Freeze the protocol after Stage A passes (protocol section 5).

Collects the hashes that make the later stages checkable -- the protocol
document, the agent config, the two data pools, the code that runs them, the
model fingerprint and the seed scheme -- plus the Stage A gate evidence, and
writes ``protocol.lock.json`` into the run directory. The lock carries a
``spec_hash`` over everything except its own timestamp, so a later divergence
is visible instead of arguable: stages B and C are then launched with
``--protocol-hash <spec_hash[:12]>`` and their manifests record it.

This does not freeze by itself. Freezing is the decision to stop changing these
files; the lock only makes the decision auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CODE_GLOBS = (
    "src/adaptive_math/recovery/*.py",
    "src/adaptive_math/agent/*.py",
    "src/adaptive_math/tools/sandboxfusion.py",
    "src/adaptive_math/verifier/*.py",
    "scripts/eval/run_feedback_recovery.py",
    "scripts/eval/check_seed_reproducibility.py",
    "scripts/data/build_feedback_recovery_set.py",
    "scripts/analysis/analyze_feedback_recovery.py",
    "configs/agent/feedback_recovery.yaml",
    "tests/unit/recovery/*.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for pattern in CODE_GLOBS:
        for path in sorted(REPO.glob(pattern)):
            if path.is_file():
                hashes[str(path.relative_to(REPO))] = _sha256_file(path)
    return hashes


def _git(*args: str) -> str:
    try:
        done = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
        )
    except OSError:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.a.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}; the lock needs the stage A manifest")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return payload


def _spec_hash(spec: dict[str, Any]) -> str:
    canonical = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--protocol",
        default="/root/autodl-tmp/BudgetConstrainedReasoningMathAgent/2026-10-14-feedback-recovery-validation.md",
    )
    parser.add_argument(
        "--data", nargs="+", default=["data/processed/diagnostics/feedback_recovery/fr_a10.parquet",
                                       "data/processed/diagnostics/feedback_recovery/fr_b40.parquet"]
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = _manifest(run_dir)
    protocol_path = Path(args.protocol)

    from adaptive_math.recovery.seeds import MAIN_SEED

    spec: dict[str, Any] = {
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
            "sections_frozen": ["5 gate", "6 B stage", "7 C arms", "8 analysis"],
        },
        "model": {
            "path": manifest["model"],
            "version": manifest["model_version"],
            "files": manifest["model_fingerprint"]["files"],
            "fingerprint_sha256": manifest["model_fingerprint"].get("digest")
            or _spec_hash(manifest["model_fingerprint"]),
            "chat_template": manifest["chat_template"],
        },
        "budget": manifest["budget"],
        "generation": {
            **manifest["generation"],
            "parser_tolerance": manifest["parser_tolerance"],
        },
        "seeds": {
            "main_seed": MAIN_SEED,
            "construction": "sha256('seed-v1:{main_seed}' : purpose : parts) via hashlib",
            "generation_key": "(stage, task_id, sample_id, arm, turn)",
            "purposes": ["task_selection", "state_selection", "run_order", "generation"],
        },
        "data": {
            str(path): (
                {"sha256": _sha256_file(REPO / path), "exists": True}
                if (REPO / path).is_file()
                else {"sha256": None, "exists": False}
            )
            for path in args.data
        },
        "manifest_file": {
            "path": "data/manifests/diagnostic_feedback_recovery.json",
            "sha256": _sha256_file(REPO / "data/manifests/diagnostic_feedback_recovery.json"),
        },
        "code": {
            "branch": _git("branch", "--show-current"),
            "head_commit": _git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(_git("status", "--porcelain")),
            "files": _code_hashes(),
        },
        "stage_a": {
            "gate": json.loads((run_dir / "stage_a" / "summary.json").read_text())["gate"],
            "seed_check": json.loads((run_dir / "stage_a" / "seed_check.json").read_text()),
            "manual_read": {
                "trajectories": 3,
                "indices": [1, 15, 16],
                "file": "stage_a/trajectories.jsonl",
                "findings": [
                    (
                        "s1 of task 5aa83a29045b: repeated 'domain' kwarg -> SyntaxError; "
                        "stderr returned in full; model abandoned and answered 800 (ref 21)"
                    ),
                    (
                        "s1 of task 19121547cc3c: final answer truncated at the 2048 ceiling "
                        "(finish_reason=length, action_parse_error), then answered 0 correctly"
                    ),
                    (
                        "s0 of task 32441bfcdbec: NameError not fixed; same code resent to "
                        "another tool -> invalid_arguments; answered 33 (ref 14)"
                    ),
                ],
                "reviewer": "executing agent; no independent reviewer available",
            },
            "sandbox_probe": json.loads((run_dir / "sandbox_probe.json").read_text()),
            "attempt0_failure_kept_at": "stage_a_attempt0_model_error/",
        },
    }
    lock: dict[str, Any] = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spec_hash": _spec_hash(spec),
        "frozen": spec,
        "note": (
            "stages B and C must run with --protocol-hash <spec_hash[:12]>; any change "
            "to a file listed here invalidates them and needs a new run version"
        ),
    }
    out = run_dir / "protocol.lock.json"
    out.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    print(f"spec_hash = {lock['spec_hash']}")
    print(f"short     = {lock['spec_hash'][:12]}")
    print(f"wrote     = {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
