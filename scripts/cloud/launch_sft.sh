#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true; shift; fi
: "${ADAPTIVE_MATH_SFT_CONFIG:?set SFT YAML path}"
: "${ADAPTIVE_MATH_SFT_DATA:?set SFT parquet path}"
: "${ADAPTIVE_MATH_SFT_MANIFEST:?set SFT manifest path}"
if [[ $# -ne 0 ]]; then echo "usage: $0 [--dry-run]" >&2; exit 2; fi
if [[ -n $(git status --porcelain) && "${ADAPTIVE_MATH_ALLOW_DIRTY_RUN:-0}" != "1" ]]; then echo "refusing dirty tree" >&2; exit 1; fi
run_id="sft-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
run_dir="${ADAPTIVE_MATH_RUNS_DIR:-artifacts/runs}/${run_id}"
echo "run_id=${run_id} config=${ADAPTIVE_MATH_SFT_CONFIG} output=${run_dir} dry_run=${dry_run}"
# SFT trains on frozen trajectories and never calls a tool, so the sandbox probe is
# off by default here; inputs, GPU and backend contract are still gated.
: "${ADAPTIVE_MATH_REQUIRE_SANDBOX:=0}"
export ADAPTIVE_MATH_REQUIRE_SANDBOX
if "$dry_run"; then scripts/cloud/preflight.sh --dry-run; exit 0; fi
scripts/cloud/preflight.sh
mkdir -p "$run_dir"; git rev-parse HEAD >"$run_dir/git_sha"
accelerate launch scripts/train/run_sft.py --config "$ADAPTIVE_MATH_SFT_CONFIG" --data "$ADAPTIVE_MATH_SFT_DATA" --data-manifest "$ADAPTIVE_MATH_SFT_MANIFEST" --set "output_dir=\"${run_dir}\"" 2>&1 | tee "$run_dir/console.log"
