#!/usr/bin/env bash
# Overnight campaign chain (2026-09-19): formal R0 -> R2 short -> evals.
set -u
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
export ADAPTIVE_MATH_SANDBOX_URL=http://127.0.0.1:8080
unset OMP_NUM_THREADS
REPO=/root/autodl-tmp/Adaptive-Solver-main-git
PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
LOG="$REPO/artifacts/runs/overnight_chain.log"
MERGED="$REPO/artifacts/models/qwen3_1_7b_sft_dp_v1_merged"
FAIL=0

step() { echo; echo "===== [$1] $(date -u +%H:%M:%SZ) ====="; }

run_grpo() {
    "$PY" "$REPO/artifacts/runs/run_grpo_direct.py" --config "$1"
}

last_checkpoint_pt() { # $1 = run dir
    local iter_file="$1/checkpoints/latest_checkpointed_iteration.txt"
    local iter
    iter=$(cat "$iter_file" 2>/dev/null | tr -d ' \n')
    [ -z "$iter" ] && iter=$(ls "$1/checkpoints" 2>/dev/null | grep -o "global_step_[0-9]*" | sort -t_ -k3 -n | tail -1 | grep -o "[0-9]*")
    [ -z "$iter" ] && return 1
    echo "$1/checkpoints/global_step_$iter/actor/model_world_size_1_rank_0.pt"
}

export_adapter() { # $1 = run dir, $2 = adapter out dir
    local pt
    pt=$(last_checkpoint_pt "$1") || { echo "no checkpoint in $1"; return 1; }
    echo "exporting adapter from $pt"
    "$PY" "$REPO/artifacts/runs/export_verl_lora.py" \
        --checkpoint "$pt" --out "$2" --base-model "$MERGED"
}

eval_adapter_only() { # $1 = adapter dir, $2 = output dir
    "$PY" "$REPO/scripts/eval/run_model_eval.py" \
        --sft-parquet "$REPO/data/processed/sft_dp_v1/train.parquet" \
        --sft-manifest "$REPO/data/manifests/sft_dp_v1_split.json" \
        --adapter "$1" --output-dir "$2" --limit 200 \
        --rl-adapter --adapter-only \
        --model-id "$MERGED"
}

{
    step "R0 formal (50 steps)"
    if run_grpo "$REPO/configs/grpo/qwen3_1_7b_r0.yaml"; then
        step "export R0 adapter"
        export_adapter "$REPO/artifacts/runs/grpo_qwen3_1_7b_r0" \
            "$REPO/artifacts/runs/grpo_qwen3_1_7b_r0/r0_adapter" || FAIL=1
    else
        echo "R0 TRAINING FAILED"; FAIL=1
    fi

    if [ "$FAIL" -eq 0 ]; then
        step "R2 short (12 steps)"
        if run_grpo "$REPO/configs/grpo/qwen3_1_7b_r2.yaml"; then
            step "export R2 adapter"
            export_adapter "$REPO/artifacts/runs/grpo_qwen3_1_7b_r2" \
                "$REPO/artifacts/runs/grpo_qwen3_1_7b_r2/r2_adapter" || FAIL=1
        else
            echo "R2 TRAINING FAILED"; FAIL=1
        fi
    fi

    if [ "$FAIL" -eq 0 ]; then
        step "R0 adapter-only eval (200 tasks)"
        eval_adapter_only "$REPO/artifacts/runs/grpo_qwen3_1_7b_r0/r0_adapter" \
            "$REPO/artifacts/eval/r0_omnimath_200" || { echo "R0 EVAL FAILED"; FAIL=1; }
    fi

    if [ "$FAIL" -eq 0 ]; then
        step "R2 adapter-only eval (200 tasks)"
        eval_adapter_only "$REPO/artifacts/runs/grpo_qwen3_1_7b_r2/r2_adapter" \
            "$REPO/artifacts/eval/r2_omnimath_200" || { echo "R2 EVAL FAILED"; FAIL=1; }
    fi

    step "chain end (FAIL=$FAIL)"
} 2>&1 | tee "$LOG"
exit "$FAIL"
