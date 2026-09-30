#!/usr/bin/env bash
# The dual-loop deliverable's evaluation: scripts/mvp/evaluate.py, three
# *new-weight* arms (base / tool-SFT / tool-GRPO) x three shards, one
# evaluate.py process per shard, arms serial inside each shard.
#
# Why this is not scripts/campaign-20260929/run_mvp_eval.sh: that launcher was a
# *reproduction* run -- its gates compared each fresh arm against the stored
# same-arm manifest of identical weights. These arms are new (the TIR SFT
# cold start and its R4 GRPO continuation), so there is no stored manifest to
# gate against; the comparable checks are internal to this run:
#   * sft  vs base must differ on >=50% of first turns (else the SFT adapter
#     never loaded);
#   * grpo vs sft  must differ likewise (else the R4 adapter never loaded);
#   * base is the one arm of identical weights and protocol to a stored arm, so
#     its agreement with the stored base run is re-measured and reported as a
#     continuity reading (flagged, never fatal -- the 20260929 amendment
#     allows base source drift).
#
# Discipline copied from scripts/campaign-20260929/run_mvp_eval.sh:
#   * GPU gate on /dev nodes, the canonical sandbox probe from preflight.py;
#   * a per-shard CPU pre-flight (evaluate.py --dry-run) that refuses a missing
#     adapter, a non-frozen task list or a stale output dir before any weights
#     move;
#   * merge then --metrics-only; partial dirs are never resumed into;
#   * tool-use columns (rate / success rate) off the merged trajectories, held
#     to each run's own summary.json.
# Deliberately single-attempt: an incomplete shard fails loudly and is left in
# place for a by-hand rerun.
#
# Exit codes: 0 = complete, all checks passed; 1 = failed (see FAILED);
# 3 = run complete but a check flagged (see checks.log / status.json).
#
# Launch, detached so SSH drops cannot kill it:
#   cd /root/autodl-tmp/Adaptive-Solver-main-git
#   setsid nohup bash scripts/campaign-20260930/run_dual_loop_eval.sh >/dev/null 2>&1 &

set -u

REPO=/root/autodl-tmp/Adaptive-Solver-main-git
cd "$REPO" || exit 1

PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python   # torch/peft runtime
VENV_PY=$REPO/.venv/bin/python                            # CPU-only steps

LOGDIR=$REPO/artifacts/runs/dual_loop_eval_20260930
mkdir -p "$LOGDIR"
QLOG=$LOGDIR/launcher.log
CHECKS=$LOGDIR/checks.log
STATUS=$LOGDIR/status.json
FAILED=$LOGDIR/FAILED
VOID=$LOGDIR/VOID
OUT=$REPO/artifacts/mvp/dual_loop_eval
RESULTS=$REPO/artifacts/results/dual-loop-20260930
STORED_BASE=$REPO/artifacts/rollout_health/thesis_e0_base_tool
FROZEN_RUNNER=scripts/campaign-20260926/rule_baseline.py
FROZEN_BLOB=e591c886f6ec674413f6e9dc7313bd4e46bd4084
SHARDS=3
ARMS="base sft grpo"

# The new arms' weights: the TIR cold start's adapter, its merge, and the R4
# GRPO adapter exported off that merge. Paths are the ones the entry points
# write; a missing one fails the pre-flight, not 200 rollouts later.
SFT_ADAPTER=$REPO/artifacts/sft/qwen3_1_7b_sft_tir/adapter
GRPO_MODEL=$REPO/artifacts/models/qwen3_1_7b_sft_tir_merged
RL_ADAPTER=$REPO/artifacts/runs/grpo_qwen3_1_7b_tir_r4/r4_adapter
ARM_FLAGS="--sft-adapter $SFT_ADAPTER --grpo-model $GRPO_MODEL --rl-adapter $RL_ADAPTER"

SANDBOX_URL=${ADAPTIVE_MATH_SANDBOX_URL:-http://localhost:8080}
POLL_SECONDS=${ADAPTIVE_MATH_QUEUE_POLL:-60}
STALL_SECONDS=${ADAPTIVE_MATH_QUEUE_STALL:-1800}

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

probe_sandbox() {
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
}

wait_sandbox() {
    until probe_sandbox; do
        log "sandbox not live, waiting at $SANDBOX_URL"
        status sandbox waiting "$SANDBOX_URL"
        sleep "$POLL_SECONDS"
    done
}

assert_frozen_and_clean() {
    local blob sha dirty
    blob=$(git hash-object "$FROZEN_RUNNER")
    if [ "$blob" != "$FROZEN_BLOB" ]; then
        fail "P5: $FROZEN_RUNNER hashes $blob, stored arms pinned $FROZEN_BLOB"
    fi
    sha=$(git rev-parse HEAD)
    dirty=$(git status --porcelain --untracked-files=no)
    log "HEAD $sha; frozen runner blob $blob; tracked tree ${dirty:+DIRTY: $dirty}${dirty:-clean}"
    [ -z "$dirty" ] || log "note: tracked files changed relative to HEAD"
}

assert_new_weights_present() {
    local path
    for path in "$SFT_ADAPTER" "$GRPO_MODEL" "$RL_ADAPTER"; do
        [ -e "$path" ] || fail "missing new-arm weights: $path (did the night chain finish?)"
    done
}

# No pre-existing run dirs: the runner refuses a non-empty output dir, and a
# merged dir from an earlier attempt would otherwise be silently kept.
assert_fresh_out_dirs() {
    local name i dir
    for name in $ARMS; do
        for i in $(seq 0 $((SHARDS - 1))); do
            dir="$OUT/$name.shard$i"
            if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
                fail "stale run dir $dir -- move it aside before launching"
            fi
        done
        if [ -d "$OUT/$name" ]; then
            fail "stale merged dir $OUT/$name -- move it aside before launching"
        fi
    done
}

# Per-shard CPU pre-flight with the runtime interpreter: emits each arm's
# identity sidecar and refuses a missing adapter, a task list that is not the
# frozen 200, or a bad path -- before any weights move.
preflight_arms() {
    local i
    for i in $(seq 0 $((SHARDS - 1))); do
        # shellcheck disable=SC2086
        "$PY" "$REPO/scripts/mvp/evaluate.py" --dry-run $ARM_FLAGS \
            --shard-count "$SHARDS" --shard-id "$i" --out-root "$OUT" >> "$QLOG" 2>&1 \
            || fail "pre-flight (identity emit) failed for shard $i"
    done
    log "pre-flight OK: $SHARDS x $ARMS identity sidecars emitted, weights present"
}

launch_shards() {
    local i pid children="" last_growth rc newest
    for i in $(seq 0 $((SHARDS - 1))); do
        # shellcheck disable=SC2086
        "$PY" "$REPO/scripts/mvp/evaluate.py" --shard-count "$SHARDS" $ARM_FLAGS \
            --shard-id "$i" --out-root "$OUT" >> "$LOGDIR/shard$i.log" 2>&1 &
        pid=$!
        children="$children $pid"
        log "shard $i -> pid $pid (log $LOGDIR/shard$i.log)"
    done
    children="${children# }"
    status run running "shards=$SHARDS"
    last_growth=$(date +%s)
    while :; do
        sleep "$POLL_SECONDS"
        local alive=yes p
        for p in $children; do
            if kill -0 "$p" 2>/dev/null; then alive=no; fi
        done
        if [ "$alive" = yes ]; then
            rc=0
            # shellcheck disable=SC2086
            wait $children 2>/dev/null || rc=$?
            log "shard processes exited (wait rc=$rc; COMPLETE files are the truth)"
            return 0
        fi
        newest=$(find "$OUT" -name 'trajectories.jsonl' -newermt "-${STALL_SECONDS} seconds" 2>/dev/null | head -1)
        if [ -z "$newest" ] && [ "$last_growth" -lt "$(($(date +%s) - STALL_SECONDS))" ]; then
            log "no journal growth in ${STALL_SECONDS}s, killing shard processes ($children)"
            # shellcheck disable=SC2086
            kill -TERM $children 2>/dev/null; sleep 10
            # shellcheck disable=SC2086
            kill -KILL $children 2>/dev/null
            # shellcheck disable=SC2086
            wait $children 2>/dev/null
            return 1
        fi
        [ -n "$newest" ] && last_growth=$(date +%s)
    done
}

require_complete() {
    local name i missing=""
    for name in $ARMS; do
        for i in $(seq 0 $((SHARDS - 1))); do
            [ -f "$OUT/$name.shard$i/COMPLETE" ] || missing="$missing $name.shard$i"
        done
    done
    [ -z "$missing" ] || fail "incomplete shards:$missing (left in place)"
}

merge_arms() {
    local name i args
    for name in $ARMS; do
        args=""
        for i in $(seq 0 $((SHARDS - 1))); do args="$args $OUT/$name.shard$i"; done
        # shellcheck disable=SC2086
        "$VENV_PY" scripts/campaign-20260926/merge_rule_shards.py \
            --output-dir "$OUT/$name" --shards $args >> "$QLOG" 2>&1 \
            || fail "$name: merge failed"
        if [ ! -f "$OUT/$name/identity.json" ]; then
            cp "$OUT/$name.shard0/identity.json" "$OUT/$name/identity.json"
        fi
        log "$name merged: $(tr -d '\n ' < "$OUT/$name/summary.json")"
    done
}

metrics_only() {
    # shellcheck disable=SC2086
    "$VENV_PY" scripts/mvp/evaluate.py --metrics-only $ARM_FLAGS \
        --out-root "$OUT" >> "$QLOG" 2>&1 \
        || fail "metrics-only failed"
}

tool_use_table() {
    "$VENV_PY" scripts/analysis/tool_use_stats.py \
        --arm "base=$OUT/base" --arm "sft=$OUT/sft" --arm "grpo=$OUT/grpo" \
        --out "$RESULTS/tool-use.json" >> "$CHECKS" 2>&1 \
        || fail "tool-use table failed (counts did not match a run's summary)"
    tail -n 8 "$CHECKS" >> "$QLOG"
    cp "$OUT/four-metrics.json" "$RESULTS/four-metrics.json" 2>/dev/null || true
}

# $1=label $2=candidate dir $3=reference dir $4=reference label -> rc
# rc 0 = different enough; rc 1 = >=50% byte-identical first turns (the alarm).
first_turn() {
    local rc=0
    "$VENV_PY" scripts/campaign-20260927/first_turn_divergence.py \
        --reference "$3" --candidate "$2" \
        --reference-label "$4" --candidate-label "$1" \
        >> "$CHECKS" 2>&1 || rc=$?
    printf '\n' >> "$CHECKS"
    tail -n 12 "$CHECKS" >> "$QLOG"
    return $rc
}

must_differ() {  # $1=candidate label $2=candidate dir $3=reference label $4=reference dir
    local rc
    if first_turn "$1" "$2" "$4" "$3"; then
        log "$1 vs $3: different (the adapter loaded)"
    else
        rc=$?
        if [ "$rc" -eq 1 ]; then
            status "$1" void "first-turn: $1 agrees with $3 at >=50% -- its adapter did not load"
            printf '%s void: first-turn alarm\n' "$1" >> "$VOID"
            fail "$1: VOID (first-turn alarm vs $3)"
        fi
        fail "$1: first-turn check vs $3 could not evaluate (rc=$rc)"
    fi
}

assert_frozen_and_clean
assert_new_weights_present
assert_fresh_out_dirs
mkdir -p "$RESULTS"
log "=== dual-loop eval launch: $ARMS, $SHARDS shards each, new weights; $SANDBOX_URL ==="
status launch waiting "gpu + sandbox"
wait_gpu
wait_sandbox
preflight_arms
status launch running "launching $SHARDS shard processes"
launch_shards || fail "shards stalled or were killed"
require_complete
status run complete "merging and scoring"
merge_arms
metrics_only

flagged=0

# The two internal adapter-loading checks this run can make.
log "adapter-load check: sft vs base (must differ)"
must_differ "sft" "$OUT/sft" "base" "$OUT/base"
log "adapter-load check: grpo vs sft (must differ)"
must_differ "grpo" "$OUT/grpo" "sft" "$OUT/sft"

# Continuity reading: base has identical weights and protocol to the stored
# base arm, so agreement is expected here. Flagged, never fatal (the 20260929
# amendment permits base source drift).
if first_turn "fresh base (continuity)" "$OUT/base" "$STORED_BASE" "stored base"; then
    log "CONTINUITY: fresh base below 50% first-turn agreement with the stored base run"
    flagged=1
else
    rc=$?
    if [ "$rc" -eq 1 ]; then
        log "CONTINUITY: fresh base reproduces the stored base arm (>=50% byte-identical first turns)"
    else
        log "CONTINUITY: fresh base vs stored base could not be compared (rc=$rc)"
        flagged=1
    fi
fi

tool_use_table

{
    printf '{"ts":"%s","head":"%s","shards":%s,"arms":"%s","out":"%s","flagged":%s}\n' \
        "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$SHARDS" "$ARMS" "$OUT" "$flagged"
} > "$LOGDIR/SUMMARY.json"

for name in $ARMS; do
    log "$name summary: $(tr -d '\n ' < "$OUT/$name/summary.json")"
    for i in $(seq 0 $((SHARDS - 1))); do
        log "  $name.shard$i: $(wc -l < "$OUT/$name.shard$i/trajectories.jsonl") trajectories"
    done
done

if [ "$flagged" -eq 0 ]; then
    status done complete "gates + metrics + checks passed"
    log "=== done ==="
    exit 0
fi
status done flagged "run complete; a check flagged -- see checks.log"
log "=== done with flags ==="
exit 3
