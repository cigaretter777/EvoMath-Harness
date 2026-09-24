#!/usr/bin/env bash
# R2 rollout-health diagnostic watchdog (2026-09-19): waits for the GPU to be
# attached, then launches the tool_call=0 diagnosis run; relaunches on death;
# exits when COMPLETE appears. Run detached: setsid nohup ... &
set -u
REPO=/root/autodl-tmp/Adaptive-Solver-main-git
PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
MERGED=$REPO/artifacts/models/qwen3_1_7b_sft_dp_v1_merged
ADAPTER=$REPO/artifacts/runs/grpo_qwen3_1_7b_r2/r2_adapter
OUT=$REPO/artifacts/rollout_health/r2_diag_10x4_seed42
LOG=$REPO/artifacts/runs/r2_rollout_health_diag.log

export HF_ENDPOINT=https://hf-mirror.com
export ADAPTIVE_MATH_SANDBOX_URL=http://127.0.0.1:8080
unset OMP_NUM_THREADS

launch() {
    setsid nohup "$PY" "$REPO/scripts/eval/run_rollout_health.py" \
        --data "$REPO/data/processed/v1/rl_dev.parquet" \
        --model "$MERGED" \
        --adapter "$ADAPTER" \
        --agent-config "$REPO/configs/agent/default.yaml" \
        --reward-config "$REPO/configs/reward/r2.yaml" \
        --task-count 10 \
        --group-size 4 \
        --selection-seed 42 \
        --output-dir "$OUT" \
        >> "$LOG" 2>&1 </dev/null &
}

while :; do
    if [ -f "$OUT/COMPLETE" ]; then
        echo "$(date -u +%H:%M:%SZ) watchdog: COMPLETE marker found, exiting" >> "$LOG"
        exit 0
    fi
    # The vendor CLI under /usr/bin is 0 bytes on this host and exits 0 under bash,
    # so the bare status check this replaces reported a card that is not attached.
    gpu_json="$(bash "$REPO/scripts/cloud/gpu_probe.sh" 2>/dev/null)" && gpu_ok=yes || gpu_ok=no
    if [ "$gpu_ok" != yes ]; then
        echo "$(date -u +%H:%M:%SZ) watchdog: no usable GPU ($gpu_json)" >> "$LOG"
        sleep 120
        continue
    fi
    if ! pgrep -f "[r]un_rollout_health.py.*r2_diag_10x4" >/dev/null; then
        echo "$(date -u +%H:%M:%SZ) watchdog: GPU up, rollout-health not running, launching" >> "$LOG"
        launch
    fi
    sleep 120
done
