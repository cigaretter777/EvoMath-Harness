#!/usr/bin/env python3
"""MVP data prep: rebuild the frozen 200-task list and verify every data input.

Why this exists
---------------
The 2026-09-26 campaign spent ~3.5 GPU hours running agent arms on the *RL
training* pool (`openr1_math_220k` ids) while the direct arms under evaluation
scored `frozen_eval.parquet` (`omni_math` ids). The two 200-task sets share
zero task_ids, so nothing could be paired -- an identity assumption nobody
verified, and the trap is one default away: the runner's ``--pool`` default is
still the RL pool. This script is the CPU-side answer for the MVP: it
regenerates the 200 ids from the frozen parquet and refuses to let them drift
from the hash every stored arm and every analysis script already pins
(``paired_protocols.EXPECTED_TASK_IDS_SHA256``).

What it verifies, in order:
  1. ``frozen_eval.parquet`` against the *tracked* manifest ``data/manifests/v1.json``
     (file bytes hash, row count, task-ids hash) -- the parquet itself is a
     build artifact and is not in git, so the manifest is the authority.
  2. the regenerated 200 ids (first 200 parquet rows, in file order) hash to
     ``1fc257f2...``, i.e. the exact list the released arms were scored on.
  3. the SFT inputs: ``data/manifests/sft_dp_v1_split.json``'s own file hash
     equals the ``data_manifest_sha256`` pin in ``configs/mvp/sft.yaml``, and
     each split parquet matches the hash recorded inside it. Skipped with
     ``--eval-only``.

The RL task pool is deliberately not re-checked here: ``run_grpo.py`` verifies
``artifacts/task_pools/rl_r0_200.jsonl`` against the pin in the GRPO config
before it starts training, and duplicating that would give two places to keep
in sync. ``--build`` delegates to ``scripts/data/build_dataset.py`` for the
case where the parquet does not exist yet (fresh clone).

Usage:
    uv run python scripts/mvp/prepare_data.py                 # verify + write
    uv run python scripts/mvp/prepare_data.py --eval-only      # skip SFT inputs
    uv run python scripts/mvp/prepare_data.py --build --registry configs/data/sources.yaml ...
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = REPO / "data" / "processed" / "v1" / "frozen_eval.parquet"
DEFAULT_MANIFEST = REPO / "data" / "manifests" / "v1.json"
DEFAULT_TASK_IDS = REPO / "artifacts" / "eval" / "thesis_e0_base_direct_b1" / "task_ids.txt"
DEFAULT_SFT_SPLIT_MANIFEST = REPO / "data" / "manifests" / "sft_dp_v1_split.json"
DEFAULT_SFT_CONFIG = REPO / "configs" / "mvp" / "sft.yaml"
BUILD_DATASET = REPO / "scripts" / "data" / "build_dataset.py"
RUN_SFT = REPO / "scripts" / "train" / "run_sft.py"
PAIRED_PROTOCOLS = REPO / "scripts" / "analysis" / "paired_protocols.py"

# The frozen eval subset: the first FROZEN_TASK_COUNT rows of frozen_eval.parquet,
# in file order. Same count emit_arm_identity.py freezes over.
FROZEN_TASK_COUNT = 200


class PrepareError(RuntimeError):
    """A check failed; the caller prints it and exits 1."""


def load_script(name: str, path: Path) -> ModuleType:
    """Import a sibling script, registering it before executing it.

    Registration is not optional: a module executed without a ``sys.modules``
    entry breaks ``from __future__ import annotations`` resolution at class
    creation, which is how this repository's scripts fail when loaded any other
    way.
    """
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


run_sft = load_script("run_sft", RUN_SFT)
paired_protocols = load_script("paired_protocols", PAIRED_PROTOCOLS)


def ids_sha256(ids: list[str]) -> str:
    """The repository's task-list hash: ids joined by newlines, no trailing newline.

    Documented next to EXPECTED_TASK_IDS_SHA256 in paired_protocols.py; used
    both by that module and by the stored arms' summaries.
    """
    from adaptive_math.core.hashing import sha256_hex

    return sha256_hex("\n".join(ids).encode("utf-8"))


def read_task_ids(parquet: Path) -> list[str]:
    """All task_ids of a frozen parquet, in file order."""
    import pandas as pd

    if not parquet.is_file():
        raise PrepareError(
            f"{parquet} is missing: it is a build artifact, not tracked in git "
            "(run --build, or point --parquet at an existing copy)"
        )
    try:
        frame = pd.read_parquet(parquet)
    except Exception as error:
        raise PrepareError(f"{parquet} is not readable as parquet: {error}") from error
    if "task_id" not in frame.columns:
        raise PrepareError(f"{parquet} has no task_id column: {list(frame.columns)}")
    return [str(task_id) for task_id in frame["task_id"]]


def check_frozen_pool(parquet: Path, manifest_path: Path) -> list[str]:
    """frozen_eval.parquet against the tracked manifest. Returns all task_ids."""
    from adaptive_math.core.hashing import sha256_hex

    if not manifest_path.is_file():
        raise PrepareError(f"{manifest_path} is missing (it is tracked in git)")
    manifest = json.loads(manifest_path.read_text())
    try:
        split = manifest["splits"]["frozen_eval"]
    except KeyError as error:
        raise PrepareError(f"{manifest_path} has no splits.frozen_eval") from error

    actual_file_hash = sha256_hex(parquet.read_bytes())
    if actual_file_hash != split["file_hash"]:
        raise PrepareError(
            f"{parquet} has hash {actual_file_hash}, but {manifest_path} records "
            f"{split['file_hash']}: the frozen eval parquet is not the one the "
            "released arms were scored on"
        )

    ids = read_task_ids(parquet)
    if len(ids) != split["count"]:
        raise PrepareError(f"{parquet} has {len(ids)} rows, manifest says {split['count']}")
    actual_ids_hash = ids_sha256(ids)
    if actual_ids_hash != split["task_ids_hash"]:
        raise PrepareError(
            f"{parquet} task_ids hash to {actual_ids_hash}, but {manifest_path} "
            f"records {split['task_ids_hash']}"
        )
    return ids


def check_task_list(ids: list[str]) -> list[str]:
    """The first FROZEN_TASK_COUNT ids must hash to the pinned constant."""
    if len(ids) < FROZEN_TASK_COUNT:
        raise PrepareError(f"frozen pool has {len(ids)} rows, need {FROZEN_TASK_COUNT}")
    frozen = ids[:FROZEN_TASK_COUNT]
    if len(set(frozen)) != len(frozen):
        raise PrepareError(f"first {FROZEN_TASK_COUNT} task_ids contain duplicates")
    actual = ids_sha256(frozen)
    expected = paired_protocols.EXPECTED_TASK_IDS_SHA256
    if actual != expected:
        raise PrepareError(
            f"the first {FROZEN_TASK_COUNT} task_ids hash to {actual}, but every "
            f"released arm was scored on {expected}: the frozen eval set has "
            "drifted (contents or row order). Do not run arms until this is "
            "understood; pairing against the stored arms would be invalid."
        )
    return frozen


def write_task_list(out: Path, frozen: list[str], *, force: bool) -> bool:
    """Write the task list; refuse to silently replace a different one.

    Byte comparison first, so the common re-run is a no-op. A file with
    *different* content is exactly the drift this script exists to catch, so
    replacing it is a deliberate act (``--force``) rather than a side effect.
    Returns True if the file was written.
    """
    text = "\n".join(frozen) + "\n"
    if out.is_file():
        existing = out.read_bytes()
        if existing == text.encode("utf-8"):
            return False
        lines = [line.strip() for line in out.read_text().splitlines() if line.strip()]
        if not force:
            raise PrepareError(
                f"{out} exists with different content than the regenerated list "
                f"(existing {len(lines)} ids hashing to {ids_sha256(lines)}, "
                f"regenerated {ids_sha256(frozen)}): pass --force to replace it"
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return True


def check_sft_data(split_manifest_path: Path, config_path: Path) -> dict[str, object]:
    """Link configs/mvp/sft.yaml's data pin to the split manifest and its parquets."""
    from adaptive_math.core.hashing import sha256_hex

    if not split_manifest_path.is_file():
        raise PrepareError(f"{split_manifest_path} is missing (it is tracked in git)")
    raw = split_manifest_path.read_bytes()
    actual_manifest_hash = sha256_hex(raw)
    config = run_sft.load_config(config_path)
    pin = config.data_manifest_sha256
    if actual_manifest_hash != pin:
        raise PrepareError(
            f"{split_manifest_path} has hash {actual_manifest_hash}, but {config_path} "
            f"pins data_manifest_sha256={pin}: the config and the split manifest "
            "describe different datasets, so this config would not reproduce the "
            "released adapter"
        )

    manifest = json.loads(raw)
    for name in ("train", "validation"):
        entry = manifest.get(name)
        if not isinstance(entry, dict):
            raise PrepareError(f"{split_manifest_path} has no {name} split")
        path = Path(entry["file"])
        if not path.is_absolute():
            path = REPO / path
        if not path.is_file():
            raise PrepareError(f"{split_manifest_path} lists {path}, which is missing")
        actual = sha256_hex(path.read_bytes())
        if actual != entry["sha256"]:
            raise PrepareError(
                f"{path} has hash {actual}, but {split_manifest_path} records {entry['sha256']}"
            )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_TASK_IDS,
        help="where the 200-task list is written (default: the path the runner "
        "and the identity emitter already default to)",
    )
    parser.add_argument("--sft-split-manifest", type=Path, default=DEFAULT_SFT_SPLIT_MANIFEST)
    parser.add_argument("--sft-config", type=Path, default=DEFAULT_SFT_CONFIG)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="skip the SFT training-data checks (evaluate-only workflow)",
    )
    parser.add_argument("--force", action="store_true", help="replace a differing task list")
    parser.add_argument("--dry-run", action="store_true", help="check, but write nothing")
    parser.add_argument(
        "--build",
        action="store_true",
        help="first delegate to scripts/data/build_dataset.py (all remaining args are its own)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, forwarded = build_parser().parse_known_args(argv)

    if not args.build and forwarded:
        # Unknown arguments are only forwarded to the builder; anywhere else a
        # typo would silently leave a default in place and still print
        # PREPARE OK, which is exactly the kind of green light this script
        # exists to refuse.
        print(f"PREPARE FAIL: unrecognised arguments {forwarded}", file=sys.stderr)
        return 1

    if args.build:
        build_dataset = load_script("build_dataset", BUILD_DATASET)
        rc = build_dataset.main(forwarded)
        if rc != 0:
            return rc

    try:
        ids = check_frozen_pool(args.parquet, args.manifest)
        frozen = check_task_list(ids)
        wrote = False
        if not args.dry_run:
            wrote = write_task_list(args.out, frozen, force=args.force)
        sft_manifest = None if args.eval_only else check_sft_data(
            args.sft_split_manifest, args.sft_config
        )
    except PrepareError as error:
        print(f"PREPARE FAIL: {error}", file=sys.stderr)
        return 1

    print(
        f"PREPARE OK: {args.parquet.name} {len(ids)} rows and task-ids hash match "
        f"{args.manifest}"
    )
    action = "dry-run, not written" if args.dry_run else ("written" if wrote else "unchanged")
    print(
        f"  task list: {len(frozen)} ids, id-hash {ids_sha256(frozen)[:16]}..., "
        f"{args.out} {action}"
    )
    if sft_manifest is not None:
        print(
            f"  sft data: train {sft_manifest['train']['count']} / "
            f"validation {sft_manifest['validation']['count']} rows, manifest hash "
            f"matches the pin in {args.sft_config}"
        )
    else:
        print("  sft data: skipped (--eval-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
