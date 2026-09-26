#!/usr/bin/env bash
# Prepared R3 launch path for this box (2026-09-25). Nothing here runs until a
# GPU is attached and a human starts it: this script exists so that "开卡了" is one
# command instead of a re-derivation of the environment at 3am.
#
# Every export below is a fact measured on this container, not a preference:
#   HF_ENDPOINT     huggingface.co is unreachable directly; only the mirror works.
#   PYTHON          the project .venv has no torch; GPU work must use this env.
#   WANDB_MODE      no API key here; verl's default tries to open a session.
#   MIN_DISK_GB     the 500GB default is written for the cloud H100 host. This is a
#                   250GB volume, so the runbook's single-GPU override applies.
#   OMP_NUM_THREADS ships as an invalid 0; every entry point must unset it.
#
# Deliberately does NOT set ADAPTIVE_MATH_ALLOW_DIRTY_RUN. A run whose git_sha does
# not exist cannot be paired against another run, which is the only reason the
# 2026-09-19 four-arm comparison was believable at all.
set -euo pipefail

REPO=/root/autodl-tmp/Adaptive-Solver-main-git
cd "$REPO"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export WANDB_MODE=disabled
export ADAPTIVE_MATH_MIN_DISK_GB=100
export ADAPTIVE_MATH_MIN_RAM_GB=64
export PYTHON=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
export ADAPTIVE_MATH_SANDBOX_URL=http://127.0.0.1:8080
export ADAPTIVE_MATH_TASK_MANIFEST="${ADAPTIVE_MATH_TASK_MANIFEST:-artifacts/task_pools/rl_r0_200.jsonl}"
export ADAPTIVE_MATH_GRPO_CONFIG="${ADAPTIVE_MATH_GRPO_CONFIG:-configs/grpo/qwen3_1_7b_r2.yaml}"
unset OMP_NUM_THREADS

# expandable_segments trips vLLM's CuMem assertion on this stack; it is set in the
# container image, so clear it rather than discover it after 20 minutes of loading.
unset PYTORCH_CUDA_ALLOC_CONF

if [[ -n $(git status --porcelain) ]]; then
    echo "tree is dirty: commit first, or the run's git_sha will describe a state that never existed" >&2
    exit 1
fi

echo "=== sandbox + GPU gate (no GPU minutes spent) ==="
scripts/cloud/preflight.sh --dry-run
echo "=== resolved config (dry run) ==="
"$PYTHON" scripts/train/run_grpo_direct.py --config "$ADAPTIVE_MATH_GRPO_CONFIG" --dry-run

if [[ "${1:-}" != "--go" ]]; then
    echo
    echo "preflight and config validated; nothing launched. Re-run with --go when the card is up."
    exit 0
fi

scripts/cloud/launch_grpo.sh
