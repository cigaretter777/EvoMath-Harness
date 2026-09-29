#!/usr/bin/env bash
# Campaign 2026-09-27 §8b: the two adapter arms, B and C, over the frozen 200.
#
#   B = base snapshot + sft adapter        (the weights of sft_direct)
#   C = merged SFT   + r0 adapter          (the weights of r0's direct rows)
#
# It *drives* the frozen runner through scripts/campaign-20260927/
# run_arm_with_adapter.py and never edits scripts/campaign-20260926/
# rule_baseline.py -- whose blob hash this script re-asserts before spending a
# GPU second (P5), since a changed blob silently breaks the pair with the
# stored arm.
#
# Discipline copied from scripts/campaign-20260926/thesis_e0_queue.sh:
# GPU gate on /dev nodes (not nvidia-smi), the canonical sandbox probe from
# preflight.py (never a second curl), per-shard identity gate on CPU before any
# weights move, partial-dir-aside instead of resuming into one, bounded
# attempts, stall detection on journal growth, shard-parallel then merge.
#
# Order is B fully, then its post-run checks, then C: the §8b first-turn check
# gates C on B, so a B whose adapter silently failed to load stops the night
# instead of buying a second arm that rests on a broken comparison.
#
# Idempotent: completed shards and completed arms are skipped, so re-running
# after a container restart picks up where the evidence left off.
set -u

REPO=/root/autodl-tmp/Adaptive-Solver-main-git
cd "$REPO" || exit 1

PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python  # torch/peft runtime
VENV_PY=$REPO/.venv/bin/python                          # CPU-only: gate + checks

LOGDIR=$REPO/artifacts/runs/bc_arms_20260929
mkdir -p "$LOGDIR"
QLOG=$LOGDIR/launcher.log
STATUS=$LOGDIR/status.json
FAILED=$LOGDIR/FAILED

BASE_MODEL=/root/autodl-tmp/hf-cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
SFT_ADAPTER=$REPO/artifacts/sft/qwen3_1_7b_sft_dp_v1/adapter
MERGED_SFT=$REPO/artifacts/models/qwen3_1_7b_sft_dp_v1_merged
R0_ADAPTER=$REPO/artifacts/runs/grpo_qwen3_1_7b_r0/r0_adapter
DATA=$REPO/data/processed/v1/frozen_eval.parquet
TASK_IDS=$REPO/artifacts/eval/thesis_e0_base_direct_b1/task_ids.txt
AGENT_CFG=$REPO/configs/agent/default.yaml
REWARD_CFG=$REPO/configs/reward/r0.yaml
JUSTIFY=docs/results/campaign-2026-09-27/source-drift-justification.md
OUT=$REPO/artifacts/rollout_health
FROZEN_RUNNER=scripts/campaign-20260926/rule_baseline.py
FROZEN_BLOB=e591c886f6ec674413f6e9dc7313bd4e46bd4084
SHARDS=3

SANDBOX_URL=${ADAPTIVE_MATH_SANDBOX_URL:-http://localhost:8080}
POLL_SECONDS=${ADAPTIVE_MATH_QUEUE_POLL:-60}
STALL_SECONDS=${ADAPTIVE_MATH_QUEUE_STALL:-1800}
MAX_ATTEMPTS=${ADAPTIVE_MATH_QUEUE_ATTEMPTS:-4}

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export WANDB_MODE=disabled
export ADAPTIVE_MATH_SANDBOX_URL=$SANDBOX_URL
unset OMP_NUM_THREADS
# expandable_segments trips the CUDA allocator assertion in this image.
unset PYTORCH_CUDA_ALLOC_CONF

log() { echo "$(date -u +%FT%TZ) $*" >> "$QLOG"; }

status() {
    # $1=arm $2=state $3=detail
    printf '{"ts":"%s","arm":"%s","state":"%s","detail":"%s"}\n' \
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
        log "$1: no GPU yet, waiting"
        status "$1" waiting_gpu "no nvidia device"
        sleep "$POLL_SECONDS"
    done
}

wait_sandbox() {
    until probe_sandbox; do
        log "$1: sandbox not live, waiting"
        status "$1" waiting_sandbox "$SANDBOX_URL"
        sleep "$POLL_SECONDS"
    done
}

# P5 and a clean tree, asserted here rather than trusted. The blob hash is the
# blocking field every stored agent arm pinned; if it moved, stop now.
assert_frozen_and_clean() {
    local blob sha dirty
    blob=$(git hash-object "$FROZEN_RUNNER")
    if [ "$blob" != "$FROZEN_BLOB" ]; then
        fail "P5: $FROZEN_RUNNER hashes $blob, stored arms pinned $FROZEN_BLOB"
    fi
    sha=$(git rev-parse HEAD)
    dirty=$(git status --porcelain --untracked-files=no)
    log "HEAD $sha; frozen runner blob $blob; tracked tree ${dirty:+DIRTY: $dirty}${dirty:-clean}"
    if [ -n "$dirty" ]; then
        log "note: tracked files changed relative to HEAD -- the source drift is justified in $JUSTIFY, but the launch should be reproducible from a commit"
    fi
}

# $1=model $2=adapter $3=kind $4=out-dir $5=shard -> ARGV
build_argv() {
    ARGV=("$PY" "$REPO/scripts/campaign-20260927/run_arm_with_adapter.py"
        --mode all-tools --shard-id "$5" --shard-count "$SHARDS"
        --data "$DATA" --task-ids-file "$TASK_IDS"
        --model "$1" --adapter "$2" --adapter-kind "$3"
        --agent-config "$AGENT_CFG" --reward-config "$REWARD_CFG"
        --output-dir "$4")
}

# P2, mechanised per shard, before any weights load: emit the sidecar with the
# runtime interpreter (so the run below cannot disagree with what was gated) and
# gate it against the stored arm's manifest for the same shard.
preflight_shard() {
    local name=$1 out=$2 model=$3 adapter=$4 kind=$5 i=$6
    build_argv "$model" "$adapter" "$kind" "$out.shard$i" "$i"
    if ! "${ARGV[@]}" --dry-run >> "$QLOG" 2>&1; then
        fail "$name: emit failed for shard $i"
    fi
    if ! "$VENV_PY" scripts/eval/verify_arm_identity.py \
        --baseline "$OUT/thesis_e0_base_tool.shard$i/manifest.json" \
        --candidate "$out.shard$i.identity.json" \
        --justify "$JUSTIFY" >> "$QLOG" 2>&1; then
        fail "$name: identity gate failed for shard $i (no run: §6e)"
    fi
    log "$name: shard $i identity gate OK"
}

shards_complete() {
    local out=$1 i
    for i in $(seq 0 $((SHARDS - 1))); do
        [ -f "$out.shard$i/COMPLETE" ] || return 1
    done
    return 0
}

# Run one arm: 3 shards concurrently, merge, no resume (a partial dir is moved
# aside and its shard reruns fresh at the same shard id, §6c).
run_arm() {
    local name=$1 out=$2 model=$3 adapter=$4 kind=$5 attempt=1

    while :; do
        if [ -f "$out/COMPLETE" ] && shards_complete "$out"; then
            log "$name: complete"
            status "$name" complete "$out"
            return 0
        fi
        if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
            fail "$name: exhausted $MAX_ATTEMPTS attempts"
        fi
        wait_gpu "$name"
        wait_sandbox "$name"

        local i children="" dirs=""
        for i in $(seq 0 $((SHARDS - 1))); do
            dirs="$dirs $out.shard$i"
            if [ -f "$out.shard$i/COMPLETE" ]; then
                log "$name: shard $i already complete, skipping"
                continue
            fi
            if [ -d "$out.shard$i" ] && [ -n "$(ls -A "$out.shard$i" 2>/dev/null)" ]; then
                local aside="$out.shard$i.partial-$(date -u +%Y%m%dT%H%M%SZ)"
                mv "$out.shard$i" "$aside"
                log "$name: moved partial shard $i aside -> $aside"
            fi
            preflight_shard "$name" "$out" "$model" "$adapter" "$kind" "$i"
            build_argv "$model" "$adapter" "$kind" "$out.shard$i" "$i"
            "${ARGV[@]}" >> "$QLOG" 2>&1 &
            children="$children $!"
            log "$name: attempt $attempt/$MAX_ATTEMPTS shard $i -> pid $! (dir $out.shard$i)"
        done
        dirs="${dirs# }"
        children="${children# }"

        if [ -z "$children" ]; then
            # Nothing to launch; all shards were already complete.
            :
        else
            status "$name" running "attempt=$attempt shards=$SHARDS"
            local last_growth
            last_growth=$(date +%s)
            while :; do
                sleep "$POLL_SECONDS"
                local alive=yes pid
                for pid in $children; do
                    if kill -0 "$pid" 2>/dev/null; then alive=no; fi
                done
                if [ "$alive" = yes ]; then
                    local rc=0
                    # shellcheck disable=SC2086
                    wait $children 2>/dev/null || rc=$?
                    log "$name: children exited rc=$rc"
                    break
                fi
                local newest
                # shellcheck disable=SC2086
                newest=$(find $dirs -name 'trajectories.jsonl' -newermt "-${STALL_SECONDS} seconds" 2>/dev/null | head -1)
                if [ -z "$newest" ] && [ "$last_growth" -lt "$(($(date +%s) - STALL_SECONDS))" ]; then
                    log "$name: no journal growth in ${STALL_SECONDS}s, killing children ($children)"
                    # shellcheck disable=SC2086
                    kill -TERM $children 2>/dev/null; sleep 10
                    # shellcheck disable=SC2086
                    kill -KILL $children 2>/dev/null
                    # shellcheck disable=SC2086
                    wait $children 2>/dev/null
                    break
                fi
                if [ -n "$newest" ]; then last_growth=$(date +%s); fi
            done
        fi

        if shards_complete "$out"; then
            local args=""
            for i in $(seq 0 $((SHARDS - 1))); do args="$args $out.shard$i"; done
            # shellcheck disable=SC2086
            "$PY" "$REPO/scripts/campaign-20260926/merge_rule_shards.py" \
                --output-dir "$out" --shards $args >> "$QLOG" 2>&1 \
                || fail "$name: merge failed"
            if [ -f "$out/COMPLETE" ]; then
                log "$name: merged and complete"
                status "$name" complete "$out"
                return 0
            fi
        fi
        log "$name: attempt $attempt ended without all shards complete"
        attempt=$((attempt + 1))
    done
}

# P2 against the authoritative artifact: each shard's real manifest, written by
# the unmodified runner, next to the stored arm's manifest for the same shard.
gate_real_manifests() {
    local name=$1 out=$2 i
    for i in $(seq 0 $((SHARDS - 1))); do
        if ! "$VENV_PY" scripts/eval/verify_arm_identity.py \
            --baseline "$OUT/thesis_e0_base_tool.shard$i/manifest.json" \
            --candidate "$out.shard$i/manifest.json" \
            --justify "$JUSTIFY" >> "$QLOG" 2>&1; then
            fail "$name: post-run gate failed on real manifest of shard $i"
        fi
        log "$name: post-run gate OK vs stored shard $i manifest"
    done
}

# §8b's first-turn rule. Returns 0 on pass, 1 on alarm, 2 on cannot-evaluate.
first_turn() {
    local label=$1 candidate=$2 reference=$3 reference_label=$4
    local tmp="$LOGDIR/first_turn_${label}_last.txt" rc=0
    "$VENV_PY" scripts/campaign-20260927/first_turn_divergence.py \
        --reference "$reference" --candidate "$candidate" \
        --reference-label "$reference_label" --candidate-label "$label" \
        > "$tmp" 2>&1 || rc=$?
    cat "$tmp" >> "$QLOG"
    printf '\n' >> "$QLOG"
    cat "$tmp" >> "$LOGDIR/checks.log"
    printf '\n' >> "$LOGDIR/checks.log"
    return $rc
}

assert_frozen_and_clean
log "=== launch: B (base+sft adapter) then C (merged SFT+r0 adapter), $SHARDS shards each, $SANDBOX_URL ==="
status launch running "B then C"

# --- Arm B ------------------------------------------------------------------
B_OUT=$OUT/thesis_e0_sft_tool
run_arm B "$B_OUT" "$BASE_MODEL" "$SFT_ADAPTER" sft
gate_real_manifests B "$B_OUT"
log "B: first-turn check vs stored arm A"
status B checks "first-turn vs A"
if first_turn B "$B_OUT" "$OUT/thesis_e0_base_tool" "A (stored base+tool)"; then
    log "B: first-turn PASS (adapter loaded)"
else
    rc=$?
    if [ "$rc" -eq 1 ]; then
        status B void "adapter did not load; C not launched"
        log "B: VOID -- first-turn alarm. C is not launched: its pre-registered comparison is against B."
        printf 'B void: first-turn alarm\n' >> "$LOGDIR/VOID"
        exit 1
    fi
    fail "B: first-turn check could not evaluate (rc=$rc)"
fi

# --- Arm C ------------------------------------------------------------------
C_OUT=$OUT/thesis_e0_r0_tool
run_arm C "$C_OUT" "$MERGED_SFT" "$R0_ADAPTER" rl
gate_real_manifests C "$C_OUT"
log "C: first-turn check vs B"
status C checks "first-turn vs B"
if first_turn C "$C_OUT" "$B_OUT" "B (base+sft adapter)"; then
    log "C: first-turn PASS (adapter loaded)"
else
    rc=$?
    if [ "$rc" -eq 1 ]; then
        status C void "adapter did not load"
        log "C: VOID -- first-turn alarm."
        printf 'C void: first-turn alarm\n' >> "$LOGDIR/VOID"
        exit 1
    fi
    fail "C: first-turn check could not evaluate (rc=$rc)"
fi

# --- Summary ----------------------------------------------------------------
{
    printf '{"ts":"%s","head":"%s","arms":{"B":"%s","C":"%s"},"shards":%s}\n' \
        "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$B_OUT" "$C_OUT" "$SHARDS"
} > "$LOGDIR/SUMMARY.json"
for d in "$B_OUT" "$C_OUT"; do
    log "$(basename "$d"): $(cat "$d/summary.json" 2>/dev/null | tr -d '\n ')"
    for i in $(seq 0 $((SHARDS - 1))); do
        log "  shard $i: $(wc -l < "$d.shard$i/trajectories.jsonl") trajectories"
    done
done
status done complete "both arms complete; gates and first-turn checks passed"
log "=== done ==="
exit 0
