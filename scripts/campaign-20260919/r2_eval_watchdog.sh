#!/usr/bin/env bash
# R2 eval watchdog (2026-09-19): relaunches the r2_omnimath_200 eval if it dies,
# exits when the COMPLETE marker appears. Run detached: setsid nohup ... &
set -u
REPO=/root/autodl-tmp/Adaptive-Solver-main-git
PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
MERGED=$REPO/artifacts/models/qwen3_1_7b_sft_dp_v1_merged
OUT=$REPO/artifacts/eval/r2_omnimath_200
LOG=$REPO/artifacts/runs/r2_eval_resume.log

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
export ADAPTIVE_MATH_SANDBOX_URL=http://127.0.0.1:8080
unset OMP_NUM_THREADS

launch() {
    setsid nohup "$PY" "$REPO/scripts/eval/run_model_eval.py" \
        --sft-parquet "$REPO/data/processed/sft_dp_v1/train.parquet" \
        --sft-manifest "$REPO/data/manifests/sft_dp_v1_split.json" \
        --adapter "$REPO/artifacts/runs/grpo_qwen3_1_7b_r2/r2_adapter" \
        --output-dir "$OUT" \
        --limit 200 --rl-adapter --adapter-only \
        --model-id "$MERGED" \
        >> "$LOG" 2>&1 </dev/null &
}

while :; do
    if [ -f "$OUT/COMPLETE" ]; then
        echo "$(date -u +%H:%M:%SZ) watchdog: COMPLETE marker found, exiting" >> "$LOG"
        exit 0
    fi
    if ! pgrep -f "[r]un_model_eval.py.*r2_omnimath_200" >/dev/null; then
        echo "$(date -u +%H:%M:%SZ) watchdog: eval not running, (re)launching" >> "$LOG"
        launch
    fi
    sleep 120
done
