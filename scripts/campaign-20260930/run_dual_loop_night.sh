#!/usr/bin/env bash
# The dual-loop cold start, end to end: TIR-SFT -> merge -> R4 GRPO -> adapter
# export -> the three-arm evaluation. One detached process so an SSH drop
# cannot strand the night between two stages.
#
# Stages and their hard gates:
#   1. wait for a GPU (device-node probe, the same gate the campaigns use);
#   2. SFT     -- scripts/mvp/train_sft.py, configs/mvp/sft_tir.yaml, then the
#                 merge the GRPO stage trains on. Gate: the adapter's COMPLETE
#                 marker *and* the merged directory exist;
#   3. GRPO    -- scripts/mvp/train_grpo.py, configs/mvp/grpo_tir_r4.yaml, then
#                 the LoRA export. Gate: the exported adapter directory has a
#                 config file.
#   4. eval    -- scripts/campaign-20260930/run_dual_loop_eval.sh, which owns
#                 its own sandbox/GPU waits, shards, merges and checks.
#
# The sandbox tunnel is checked before GRPO (a dead sandbox makes every tool
# call fail, which under the R4 reward is a *wrong training signal*, not just a
# slower run) and then watched every 5 minutes for the rest of the night: drops
# and recoveries are timestamped into the log so a bad window can be attributed
# after the fact instead of guessed at.
#
# Exit codes mirror the eval launcher: 0 complete, 1 failed, 3 complete-with-a-
# flag. A failure names its stage in FAILED and stops; finished stages are never
# re-run (their COMPLETE markers are honoured).
#
# Launch, detached so SSH drops cannot kill it:
#   cd /root/autodl-tmp/Adaptive-Solver-main-git
#   setsid nohup bash scripts/campaign-20260930/run_dual_loop_night.sh >/dev/null 2>&1 &

set -u

REPO=/root/autodl-tmp/Adaptive-Solver-main-git
cd "$REPO" || exit 1

PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python   # torch/peft runtime
VENV_PY=$REPO/.venv/bin/python                            # CPU-only steps

LOGDIR=$REPO/artifacts/runs/dual_loop_night_20260930
mkdir -p "$LOGDIR"
QLOG=$LOGDIR/night.log
STATUS=$LOGDIR/status.json
FAILED=$LOGDIR/FAILED
WATCH=$LOGDIR/sandbox_watch.log

SFT_CONFIG=configs/mvp/sft_tir.yaml
SFT_DATA=data/processed/sft_tir_v1/train.parquet
SFT_SPLIT_MANIFEST=data/manifests/sft_tir_v1_split.json
SFT_ADAPTER_DIR=$REPO/artifacts/sft/qwen3_1_7b_sft_tir/adapter
MERGED_MODEL=$REPO/artifacts/models/qwen3_1_7b_sft_tir_merged
GRPO_CONFIG=configs/mvp/grpo_tir_r4.yaml
GRPO_ADAPTER=$REPO/artifacts/runs/grpo_qwen3_1_7b_tir_r4/r4_adapter
EVAL_LAUNCHER=scripts/campaign-20260930/run_dual_loop_eval.sh

SANDBOX_URL=${ADAPTIVE_MATH_SANDBOX_URL:-http://localhost:8080}
POLL_SECONDS=${ADAPTIVE_MATH_QUEUE_POLL:-60}
WATCH_SECONDS=${ADAPTIVE_MATH_WATCH_SECONDS:-300}

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export WANDB_MODE=disabled
export ADAPTIVE_MATH_SANDBOX_URL=$SANDBOX_URL
unset OMP_NUM_THREADS
unset PYTORCH_CUDA_ALLOC_CONF

log() { echo "$(date -u +%FT%TZ) $*" >> "$QLOG"; }

status() {
    printf '{"ts":"%s","step":"%s","state":"%s","detail":"%s"}\n' \
        "$(date -u +%FT%TZ)" "$1" "$2" "$3" > "$STATUS"
}

fail() {
    log "FAILED: $*"
    status "${1%% *}" failed "$*"
    printf '%s\n' "$*" > "$FAILED"
    exit 1
}

gpu_ok() { bash "$REPO/scripts/cloud/gpu_probe.sh" >/dev/null 2>&1; }

sandbox_ok() {
    "$VENV_PY" - <<'PY' >/dev/null 2>&1
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("preflight", "scripts/cloud/preflight.py")
m = importlib.util.module_from_spec(spec); sys.modules["preflight"] = m
spec.loader.exec_module(m)
os.environ.setdefault("ADAPTIVE_MATH_SANDBOX_URL", "http://localhost:8080")
ev = m._sandbox_evidence(True)
sys.exit(0 if ev.get("reachable") and ev.get("probe") == "pong+42" else 1)
PY
}

wait_gpu() {
    until gpu_ok; do
        log "no GPU yet, waiting"
        status gpu waiting "no nvidia device"
        sleep "$POLL_SECONDS"
    done
    log "GPU present"
}

# The sandbox in the background: one line per up/down *transition*, so the log
# stays readable and a bad window is a timestamped range.
sandbox_watchdog() {
    local last="unknown" now
    while :; do
        if sandbox_ok; then now=up; else now=down; fi
        if [ "$now" != "$last" ]; then
            echo "$(date -u +%FT%TZ) sandbox $now ($SANDBOX_URL)" >> "$WATCH"
            log "sandbox watchdog: $now"
            last=$now
        fi
        sleep "$WATCH_SECONDS"
    done
}

assert_inputs_present() {
    local path
    for path in "$SFT_CONFIG" "$SFT_DATA" "$SFT_SPLIT_MANIFEST" "$GRPO_CONFIG" "$EVAL_LAUNCHER"; do
        [ -e "$path" ] || fail "missing input $path (is the TIR build + freeze done?)"
    done
    log "inputs present: $SFT_CONFIG, $SFT_DATA, $GRPO_CONFIG"
}

stage_sft() {
    if [ -f "$SFT_ADAPTER_DIR/COMPLETE" ]; then
        log "SFT already complete ($SFT_ADAPTER_DIR/COMPLETE); skipping"
    else
        status sft running "train_sft.py (TIR cold start)"
        "$PY" scripts/mvp/train_sft.py --config "$SFT_CONFIG" \
            --data "$SFT_DATA" --data-manifest "$SFT_SPLIT_MANIFEST" \
            --merge-to "$MERGED_MODEL" >> "$LOGDIR/sft.log" 2>&1 \
            || fail "sft: train_sft.py failed (rc=$?)"
        log "SFT finished"
    fi
    [ -f "$SFT_ADAPTER_DIR/COMPLETE" ] || fail "sft: no COMPLETE marker in $SFT_ADAPTER_DIR"
    [ -d "$MERGED_MODEL" ] || fail "sft: merged model missing at $MERGED_MODEL"
    status sft complete "adapter + merged model on disk"
}

stage_grpo() {
    if [ -f "$GRPO_ADAPTER/config.json" ]; then
        log "GRPO export already complete ($GRPO_ADAPTER); skipping"
        return 0
    fi
    until sandbox_ok; do
        log "sandbox not live before GRPO, waiting at $SANDBOX_URL"
        status grpo waiting "sandbox"
        sleep "$POLL_SECONDS"
    done
    status grpo running "train_grpo.py (R4 continuation)"
    "$PY" scripts/mvp/train_grpo.py --config "$GRPO_CONFIG" \
        --export-adapter "$GRPO_ADAPTER" >> "$LOGDIR/grpo.log" 2>&1 \
        || fail "grpo: train_grpo.py failed (rc=$?)"
    [ -f "$GRPO_ADAPTER/config.json" ] || fail "grpo: exported adapter has no config.json"
    status grpo complete "r4 adapter exported"
    log "GRPO finished"
}

stage_eval() {
    status eval running "run_dual_loop_eval.sh"
    bash "$EVAL_LAUNCHER" >> "$QLOG" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then
        fail "eval: run_dual_loop_eval.sh rc=$rc (see $LOGDIR/../dual_loop_eval_20260930/)"
    fi
    return "$rc"
}

assert_inputs_present
log "=== dual-loop night chain: SFT -> GRPO -> eval; $SANDBOX_URL ==="
status launch waiting "gpu"
sandbox_watchdog &
WATCHDOG=$!
trap 'kill "$WATCHDOG" 2>/dev/null' EXIT

wait_gpu
stage_sft
stage_grpo
stage_eval
rc=$?

if [ "$rc" -eq 3 ]; then
    log "=== night chain done; eval complete with a flag ==="
    exit 3
fi
log "=== night chain done; all stages complete ==="
exit 0
