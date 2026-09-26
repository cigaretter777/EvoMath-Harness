#!/usr/bin/env bash
# Preflight: verify that an agent-arm task set is identical to the stored
# direct-eval arm it must pair against, BEFORE any GPU minute is spent.
#
# Why this exists: the 2026-09-26 baseline campaign first ran its agent arms
# on the RL training pool (openr1_math_220k ids) while the direct arms
# evaluate the frozen eval parquet (omni_math ids). Zero task overlap, so no
# pairing was possible and ~3.5 GPU hours were spent on the wrong set. The
# failure was an identity assumption that was never verified. This gate
# verifies it on CPU in seconds; it must pass before the queue launches any
# agent slot.
#
# Checks (all CPU-only):
#   1. $TASK_IDS exists, is unique, and has the expected count
#   2. its id set equals the stored base-direct predictions' id set
#   3. every id resolves to a row in the parquet the agent arm will read
# Fails hard with a non-zero exit and a specific message on any violation.
set -u

REPO=/root/autodl-tmp/Adaptive-Solver-main-git
PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
TASK_IDS=${1:-$REPO/artifacts/eval/thesis_e0_base_direct_b1/task_ids.txt}
PREDICTIONS=${2:-$REPO/artifacts/eval/thesis_e0_base_direct_b1/base_predictions.jsonl}
PARQUET=${3:-$REPO/data/processed/v1/frozen_eval.parquet}

env -u OMP_NUM_THREADS "$PY" - "$TASK_IDS" "$PREDICTIONS" "$PARQUET" <<'EOF'
import json, sys
from pathlib import Path

import pandas as pd

ids_path = Path(sys.argv[1])
preds_path = Path(sys.argv[2])
parquet_path = Path(sys.argv[3])

if not ids_path.is_file():
    sys.exit(f"GATE FAIL: {ids_path} missing")
if not preds_path.is_file():
    sys.exit(f"GATE FAIL: {preds_path} missing")
if not parquet_path.is_file():
    sys.exit(f"GATE FAIL: {parquet_path} missing")

ids = [l.strip() for l in ids_path.read_text().splitlines() if l.strip()]
if len(set(ids)) != len(ids):
    sys.exit(f"GATE FAIL: duplicate task_ids in {ids_path}")
if not ids:
    sys.exit(f"GATE FAIL: {ids_path} is empty")

preds = [json.loads(l)["task_id"] for l in preds_path.open()]
if set(ids) != set(preds):
    sys.exit(
        f"GATE FAIL: task set mismatch with stored direct arm "
        f"({len(ids)} ids vs {len(preds)} predictions, "
        f"overlap {len(set(ids) & set(preds))})"
    )
if len(preds) != len(set(preds)):
    sys.exit(f"GATE FAIL: duplicate task_ids in {preds_path}")

frame = pd.read_parquet(parquet_path)
present = set(frame["task_id"].tolist())
missing = [i for i in ids if i not in present]
if missing:
    sys.exit(
        f"GATE FAIL: {len(missing)} task_ids absent from {parquet_path}, "
        f"e.g. {missing[:3]}"
    )

print(f"GATE OK: {len(ids)} task_ids aligned across ids-file, "
      f"stored direct arm, and {parquet_path}")
EOF
