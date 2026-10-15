#!/usr/bin/env bash
# Pre-A capability check: does a candidate policy actually emit parse-valid
# tool calls under the diagnostics runner protocol?
#
# Motivation (2026-10-14): round-1 ran the dp_v1 policy and measured 0 executed
# tool calls in 1600 episodes. The training data shows why -- sft_dp_v1 records
# carry behavior="direct" and an EMPTY tool schema list, while sft_tir_v1
# records carry behavior="python" and the python schema. A feedback-recovery
# study needs a policy with tool competence, so the candidate must be verified
# empirically instead of trusting a directory name (report section 3.1).
#
# This runs the SAME runner stage A will use, on a couple of tasks per model,
# with the sandbox down if necessary: parse validity does not depend on the
# tool actually executing. Executions that do happen while the sandbox is
# unreachable degrade to "unavailable" and are labelled as such in the report.
#
#   setsid nohup bash scripts/diagnostics/preflight_model_capability.sh \
#       artifacts/diagnostics/feedback_recovery/<run_id> >/dev/null 2>&1 &

set -u

REPO=/root/autodl-tmp/Adaptive-Solver-main-git
cd "$REPO" || exit 1

PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python   # torch runtime (.venv has no torch)
TIR="$REPO/artifacts/models/qwen3_1_7b_sft_tir_merged"
DPV1="$REPO/artifacts/models/qwen3_1_7b_sft_dp_v1_merged"

RUN_DIR=${1:?usage: preflight_model_capability.sh <run_dir>}
OUT=$RUN_DIR/preflight
mkdir -p "$OUT"
LOG=$OUT/preflight.log

DATA=data/processed/diagnostics/r1/diagnostic_r1_pilot20.parquet
TASK_COUNT=2
GROUP_SIZE=2
TEMPERATURE=0.7
MAX_NEW_TOKENS=2048
TOLERANCE=unclosed_think,bare_final_scalar

export HF_HOME=/root/autodl-tmp/hf-cache
export WANDB_MODE=disabled
export ADAPTIVE_MATH_SANDBOX_URL=${ADAPTIVE_MATH_SANDBOX_URL:-http://localhost:8080}

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

run_model() {
    local tag=$1 model=$2
    if [ -d "$OUT/$tag" ] && [ -n "$(ls -A "$OUT/$tag" 2>/dev/null)" ]; then
        log "skip $tag: output dir not empty"
        return 0
    fi
    log "start $tag ($model)"
    PYTHONPATH=src "$PY" scripts/eval/run_rollout_health.py \
        --data "$DATA" \
        --model "$model" \
        --agent-config configs/agent/default.yaml \
        --reward-config configs/reward/r0.yaml \
        --task-count "$TASK_COUNT" \
        --group-size "$GROUP_SIZE" \
        --selection-seed 20261014 \
        --temperature "$TEMPERATURE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --parser-tolerance "$TOLERANCE" \
        --output-dir "$OUT/$tag" \
        >> "$OUT/$tag.log" 2>&1
    local rc=$?
    log "$tag exit rc=$rc"
    return $rc
}

log "preflight start; HEAD $(git rev-parse --short HEAD)"
rc=0
run_model tir_merged "$TIR" || rc=1
run_model dp_v1_merged "$DPV1" || rc=1
log "preflight done rc=$rc"

exit $rc
