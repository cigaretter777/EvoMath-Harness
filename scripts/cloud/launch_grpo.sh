#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true; shift; fi
: "${ADAPTIVE_MATH_GRPO_CONFIG:?set GRPO YAML path}"
: "${ADAPTIVE_MATH_SANDBOX_URL:?set SandboxFusion URL}"
if [[ $# -ne 0 ]]; then echo "usage: $0 [--dry-run]" >&2; exit 2; fi
if [[ -n $(git status --porcelain) && "${ADAPTIVE_MATH_ALLOW_DIRTY_RUN:-0}" != "1" ]]; then echo "refusing dirty tree" >&2; exit 1; fi
run_id="grpo-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
run_dir="${ADAPTIVE_MATH_RUNS_DIR:-artifacts/runs}/${run_id}"
echo "run_id=${run_id} config=${ADAPTIVE_MATH_GRPO_CONFIG} output=${run_dir} dry_run=${dry_run}"
# Gate before any GPU minute is spent: an agent rollout with a dead sandbox trains
# against a tool that always answers UNAVAILABLE.
if "$dry_run"; then scripts/cloud/preflight.sh --dry-run; exit 0; fi
scripts/cloud/preflight.sh
mkdir -p "$run_dir"; git rev-parse HEAD >"$run_dir/git_sha"
# Engine choice. torchrun hangs this container at the register-center handshake, so
# the default is the in-process launcher that both completed GRPO runs actually used;
# ADAPTIVE_MATH_USE_TORCHRUN=1 keeps the multi-process path for the cloud image.
if [[ "${ADAPTIVE_MATH_USE_TORCHRUN:-0}" == "1" ]]; then
    engine=(torchrun --nproc_per_node="${ADAPTIVE_MATH_NPROC_PER_NODE:-1}" scripts/train/run_grpo.py)
else
    engine=("${PYTHON:-python}" scripts/train/run_grpo_direct.py)
fi
"${engine[@]}" --config "$ADAPTIVE_MATH_GRPO_CONFIG" --output-dir "$run_dir" 2>&1 | tee "$run_dir/console.log"
