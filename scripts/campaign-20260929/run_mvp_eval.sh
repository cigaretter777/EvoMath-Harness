#!/usr/bin/env bash
# The MVP evaluation, end to end through its own entry point:
# scripts/mvp/evaluate.py, three arms (base / sft / grpo) x three shards, one
# evaluate.py process per shard, arms serial inside each shard.
#
# Pre-registered before its first GPU minute:
# docs/results/campaign-2026-09-27/preregistration.md §8c. The settings are
# §8b's argv verbatim -- `evaluate.py --dry-run` prints exactly what each arm
# will run.
#
# Discipline copied from scripts/campaign-20260927/run_bc_arms.sh:
#   * GPU gate on /dev nodes (not nvidia-smi), the canonical sandbox probe from
#     preflight.py (never a second curl);
#   * the nine identity gates (arm x shard, same-arm stored baselines) on CPU
#     before any weights move, and again against each run's real manifest;
#   * §8b's first-turn rule on the two adapter arms, and the reproduction
#     reading (fresh arm vs its stored same-weight run) on all three;
#   * merge then --metrics-only; partial dirs are never resumed into.
# Deliberately single-attempt (this is a supervised ~3 h run): an incomplete
# shard fails loudly and is left in place for a by-hand rerun.
#
# Exit codes: 0 = complete, all checks passed; 1 = failed (see FAILED/VOID);
# 3 = run complete but a check flagged (see checks.log / status.json).
#
# Launch, detached so SSH drops cannot kill it:
#   cd /root/autodl-tmp/Adaptive-Solver-main-git
#   setsid nohup bash scripts/campaign-20260929/run_mvp_eval.sh >/dev/null 2>&1 &

set -u

REPO=/root/autodl-tmp/Adaptive-Solver-main-git
cd "$REPO" || exit 1

PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python   # torch/peft runtime
VENV_PY=$REPO/.venv/bin/python                            # CPU-only steps

LOGDIR=$REPO/artifacts/runs/mvp_eval_20260929
mkdir -p "$LOGDIR"
QLOG=$LOGDIR/launcher.log
CHECKS=$LOGDIR/checks.log
STATUS=$LOGDIR/status.json
FAILED=$LOGDIR/FAILED
VOID=$LOGDIR/VOID
OUT=$REPO/artifacts/mvp/eval
STORED=$REPO/artifacts/rollout_health
JUSTIFY=docs/results/campaign-2026-09-27/source-drift-justification.md
FROZEN_RUNNER=scripts/campaign-20260926/rule_baseline.py
FROZEN_BLOB=e591c886f6ec674413f6e9dc7313bd4e46bd4084
SHARDS=3
ARMS="base sft grpo"

SANDBOX_URL=${ADAPTIVE_MATH_SANDBOX_URL:-http://localhost:8080}
POLL_SECONDS=${ADAPTIVE_MATH_QUEUE_POLL:-60}
STALL_SECONDS=${ADAPTIVE_MATH_QUEUE_STALL:-1800}

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export WANDB_MODE=disabled
export ADAPTIVE_MATH_SANDBOX_URL=$SANDBOX_URL
unset OMP_NUM_THREADS
# expandable_segments trips the CUDA allocator assertion in this image.
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

# The canonical probe: ping *and* a real execution, through the same function
# the launcher gate uses (preflight.py::_sandbox_evidence). Not a second curl.
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

# The stored same-arm baselines the MVP arms pair with.
stored_manifest() {  # $1=arm $2=shard
    case "$1" in
        base) printf '%s/thesis_e0_base_tool.shard%s/manifest.json' "$STORED" "$2" ;;
        sft)  printf '%s/thesis_e0_sft_tool.shard%s/manifest.json' "$STORED" "$2" ;;
        grpo) printf '%s/thesis_e0_r0_tool.shard%s/manifest.json' "$STORED" "$2" ;;
    esac
}

stored_arm() {  # $1=arm -> stored merged arm dir
    case "$1" in
        base) printf '%s/thesis_e0_base_tool' "$STORED" ;;
        sft)  printf '%s/thesis_e0_sft_tool' "$STORED" ;;
        grpo) printf '%s/thesis_e0_r0_tool' "$STORED" ;;
    esac
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

# The nine sidecars are emitted with the runtime interpreter -- so the
# pre-flight cannot disagree with the run -- and each is gated against its
# stored same-arm shard manifest. Nothing loads weights before all nine rc=0.
preflight_and_gate() {
    local name i
    for i in $(seq 0 $((SHARDS - 1))); do
        "$PY" "$REPO/scripts/mvp/evaluate.py" --dry-run \
            --shard-count "$SHARDS" --shard-id "$i" --out-root "$OUT" >> "$QLOG" 2>&1 \
            || fail "identity emit failed for shard $i"
    done
    for name in $ARMS; do
        for i in $(seq 0 $((SHARDS - 1))); do
            "$VENV_PY" scripts/eval/verify_arm_identity.py \
                --baseline "$(stored_manifest "$name" "$i")" \
                --candidate "$OUT/$name.shard$i.identity.json" \
                --justify "$JUSTIFY" >> "$QLOG" 2>&1 \
                || fail "$name.shard$i: pre-launch identity gate failed (no run)"
        done
        log "$name: pre-launch identity gates OK on all $SHARDS shards"
    done
}

launch_shards() {
    local i pid children="" last_growth rc newest
    for i in $(seq 0 $((SHARDS - 1))); do
        "$PY" "$REPO/scripts/mvp/evaluate.py" --shard-count "$SHARDS" \
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
        # Carry the run's weights identity onto the merged dir (shard 0's; the
        # shard id is ALLOWED_TO_DIFFER in the gate and is inert for the metrics
        # payload, which is all this file feeds).
        if [ ! -f "$OUT/$name/identity.json" ]; then
            cp "$OUT/$name.shard0/identity.json" "$OUT/$name/identity.json"
        fi
        log "$name merged: $(tr -d '\n ' < "$OUT/$name/summary.json")"
    done
}

metrics_only() {
    "$VENV_PY" scripts/mvp/evaluate.py --metrics-only --out-root "$OUT" >> "$QLOG" 2>&1 \
        || fail "metrics-only failed"
}

gate_real_manifests() {
    local name i
    for name in $ARMS; do
        for i in $(seq 0 $((SHARDS - 1))); do
            "$VENV_PY" scripts/eval/verify_arm_identity.py \
                --baseline "$(stored_manifest "$name" "$i")" \
                --candidate "$OUT/$name.shard$i/manifest.json" \
                --justify "$JUSTIFY" >> "$QLOG" 2>&1 \
                || fail "$name: post-run gate failed on the real manifest of shard $i"
        done
        log "$name: post-run gates OK on all real shard manifests"
    done
}

# $1=label $2=candidate dir $3=reference dir $4=reference label -> rc
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

assert_frozen_and_clean
assert_fresh_out_dirs
log "=== MVP eval launch: $ARMS, $SHARDS shards each, via scripts/mvp/evaluate.py; $SANDBOX_URL ==="
status launch waiting "gpu + sandbox"
wait_gpu
wait_sandbox
preflight_and_gate
status launch running "launching $SHARDS shard processes"
launch_shards || fail "shards stalled or were killed"
require_complete
status run complete "merging and scoring"
merge_arms
metrics_only
gate_real_manifests

flagged=0

# §8b, verbatim: an adapter arm that agrees with the arm it must differ from on
# >= 50% of first outputs never loaded its adapter. That arm is void.
log "§8b first-turn check: fresh sft vs stored base arm (must differ)"
if first_turn "fresh sft" "$OUT/sft" "$(stored_arm base)" "stored base (A)"; then
    :
else
    rc=$?
    if [ "$rc" -eq 1 ]; then
        status sft void "first-turn: sft agrees with base at >=50% -- adapter did not load"
        printf 'sft void: first-turn alarm\n' >> "$VOID"
        fail "sft: VOID (first-turn alarm)"
    fi
    fail "sft: first-turn check could not evaluate (rc=$rc)"
fi

log "§8b first-turn check: fresh grpo vs stored sft arm (must differ)"
if first_turn "fresh grpo" "$OUT/grpo" "$(stored_arm sft)" "stored sft (B)"; then
    :
else
    rc=$?
    if [ "$rc" -eq 1 ]; then
        status grpo void "first-turn: grpo agrees with sft at >=50% -- adapter did not load"
        printf 'grpo void: first-turn alarm\n' >> "$VOID"
        fail "grpo: VOID (first-turn alarm)"
    fi
    fail "grpo: first-turn check could not evaluate (rc=$rc)"
fi

# The reproduction reading (§8c): this run re-runs stored arms of identical
# weights over an unchanged run path, so here the >=50% outcome is the expected
# one and a low agreement is the flag. Same tool, expected outcome inverted --
# its printed "did not load" wording belongs to the two checks above.
for name in $ARMS; do
    if first_turn "fresh $name (re-run)" "$OUT/$name" "$(stored_arm "$name")" "stored $name"; then
        log "REPRO: fresh $name vs stored $name below 50% first-turn agreement"
        [ "$name" = base ] || flagged=1
    else
        rc=$?
        if [ "$rc" -eq 1 ]; then
            log "REPRO: fresh $name reproduces stored $name (>=50% byte-identical first outputs)"
        else
            log "REPRO: fresh $name vs stored $name could not be compared (rc=$rc)"
            [ "$name" = base ] || flagged=1
        fi
    fi
done

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
status done flagged "run complete; reproduction check flagged -- see checks.log"
log "=== done with flags ==="
exit 3
