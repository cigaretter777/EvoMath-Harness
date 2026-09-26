#!/usr/bin/env bash
# Thesis E0/E1 baseline queue (2026-09-26). Serial, single-writer on the GPU.
#
# Slots:
#   base_direct   -- run_model_eval WITHOUT --adapter-only: base + sft arms,
#                    batch 1, greedy, cap 1024, frozen 200. The sft arm must
#                    be byte-identical to the E2 champion arm (same identity,
#                    same decoding): that is the pipeline cross-check.
#   base_tool     -- rule_baseline --mode all-tools: base model + both tools.
#   rule_strategy -- rule_baseline --mode rule: fixed-rule routing into
#                    sympy-only / python-only / direct (reuses slot 1).
#
# Guardrails carried over from night_queue.sh (2026-09-24) and
# campaign_monitor.sh: bounded attempts, stall detection on journal growth
# (not process liveness), status heartbeat, partial-dir-aside instead of
# relaunching into them, sandbox liveness probe before the tool slots.
# The queue is idempotent: re-run it after a container restart and it picks
# up where the evidence left off.
set -u

REPO=/root/autodl-tmp/Adaptive-Solver-main-git
PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
QLOG=$REPO/artifacts/runs/thesis_e0_queue.log
STATUS=$REPO/artifacts/runs/thesis_e0_queue_status.json

MODEL_ID=/root/autodl-tmp/hf-cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
SFT_ADAPTER=$REPO/artifacts/sft/qwen3_1_7b_sft_dp_v1/adapter
SFT_ADAPTER_SHA=2868f83e9c3bfbe4a0eb6a115575ab4d7fe7cf4357028dcf9c4391c2a71fecaa
SFT_PARQUET=$REPO/data/processed/sft_dp_v1/train.parquet
SFT_MANIFEST=$REPO/data/manifests/sft_dp_v1_split.json

DIRECT_OUT=$REPO/artifacts/eval/thesis_e0_base_direct_b1
BASE_TOOL_OUT=$REPO/artifacts/rollout_health/thesis_e0_base_tool
RULE_OUT=$REPO/artifacts/rollout_health/thesis_e0_rule_strategy

SANDBOX_URL=${ADAPTIVE_MATH_SANDBOX_URL:-http://127.0.0.1:8080}
POLL_SECONDS=${ADAPTIVE_MATH_QUEUE_POLL:-60}
STALL_SECONDS=${ADAPTIVE_MATH_QUEUE_STALL:-1800}
MAX_ATTEMPTS=${ADAPTIVE_MATH_QUEUE_ATTEMPTS:-4}

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export WANDB_MODE=disabled
export ADAPTIVE_MATH_SANDBOX_URL=$SANDBOX_URL
unset OMP_NUM_THREADS
# expandable_segments trips vLLM's CuMem assertion; it ships in the image env.
unset PYTORCH_CUDA_ALLOC_CONF

log() { echo "$(date -u +%FT%TZ) $*" >> "$QLOG"; }

status() {
    # $1=slot $2=state $3=detail
    printf '{"ts":"%s","slot":"%s","state":"%s","detail":"%s"}\n' \
        "$(date -u +%FT%TZ)" "$1" "$2" "$3" > "$STATUS"
}

fail() {
    log "FAILED: $*"
    status "${1%% *}" failed "$*"
    printf '%s\n' "$*" > "$REPO/artifacts/runs/thesis_e0_queue_FAILED"
    exit 1
}

gpu_ok() { bash "$REPO/scripts/cloud/gpu_probe.sh" >/dev/null 2>&1; }

probe_sandbox() {
    local ping exec_out
    ping=$(curl -s -m 6 "$SANDBOX_URL/v1/ping" 2>/dev/null)
    exec_out=$(curl -s -m 20 -X POST "$SANDBOX_URL/run_code" -H 'Content-Type: application/json' \
        -d '{"code":"print(6*7)","language":"python"}' 2>/dev/null | grep -o '"stdout":"42' | head -1)
    [ "$ping" = '"pong"' ] && [ -n "$exec_out" ]
}

wait_gpu() {
    local slot=$1
    until gpu_ok; do
        log "$slot: no GPU yet, waiting"
        status "$slot" waiting_gpu "no nvidia device"
        sleep "$POLL_SECONDS"
    done
}

wait_sandbox() {
    local slot=$1
    until probe_sandbox; do
        log "$slot: sandbox not live, waiting"
        status "$slot" waiting_sandbox "$SANDBOX_URL"
        sleep "$POLL_SECONDS"
    done
}

# Slot 1: model_eval base+sft arms. Resumes from its .in_progress journal;
# completion is out/summary.json. (Proven pattern, night_queue.sh 2026-09-24.)
run_eval_slot() {
    local out=$DIRECT_OUT journal="$DIRECT_OUT.in_progress" attempt=1

    while :; do
        if [ -f "$out/summary.json" ]; then
            log "base_direct: complete ($out/summary.json present)"
            status base_direct complete "$out"
            return 0
        fi
        if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
            fail "base_direct exhausted $MAX_ATTEMPTS attempts"
        fi
        wait_gpu base_direct

        log "base_direct: attempt $attempt/$MAX_ATTEMPTS out=$out"
        status base_direct running "attempt=$attempt"

        "$PY" "$REPO/scripts/eval/run_model_eval.py" \
            --eval-parquet "$REPO/data/processed/v1/frozen_eval.parquet" \
            --data-manifest "$REPO/data/manifests/v1.json" \
            --sft-parquet "$SFT_PARQUET" \
            --sft-manifest "$SFT_MANIFEST" \
            --adapter "$SFT_ADAPTER" \
            --expected-adapter-sha256 "$SFT_ADAPTER_SHA" \
            --output-dir "$out" \
            --model-id "$MODEL_ID" \
            --max-new-tokens 1024 \
            --batch-size 1 \
            --limit 200 >> "$QLOG" 2>&1 &
        local child=$!

        local last_growth
        last_growth=$(date +%s)
        while :; do
            sleep "$POLL_SECONDS"
            if ! kill -0 "$child" 2>/dev/null; then
                wait "$child"; local rc=$?
                if [ -f "$out/summary.json" ]; then
                    log "base_direct: child exited rc=$rc with summary.json"
                    break
                fi
                log "base_direct: child exited rc=$rc without summary.json (journal kept for resume)"
                attempt=$((attempt + 1))
                break
            fi
            local newest
            newest=$(find "$journal" -name '*_predictions.jsonl' -newermt "-${STALL_SECONDS} seconds" 2>/dev/null | head -1)
            if [ -z "$newest" ]; then
                log "base_direct: no journal growth in ${STALL_SECONDS}s, killing $child"
                kill -TERM "$child" 2>/dev/null; sleep 10; kill -KILL "$child" 2>/dev/null
                wait "$child" 2>/dev/null
                attempt=$((attempt + 1))
                break
            fi
        done
    done
}

# Slots 2-3: rule_baseline agent rollouts. No resume: a partial dir is moved
# aside and the slot reruns fresh (the 2026-09-19 watchdog lesson). Completion
# is the COMPLETE marker.
#
# shard_count > 0 runs that many concurrent processes over disjoint task
# shards (out.shard0..N), then merges them into $out. The per-task generation
# path is unchanged; only scheduling differs, so sharded and unsharded runs
# produce the same trajectories.
run_rule_slot() {
    local name=$1 out=$2 mode=$3 shards=${4:-0} attempt=1

    while :; do
        if [ -f "$out/COMPLETE" ]; then
            log "$name: complete"
            status "$name" complete "$out"
            return 0
        fi
        if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
            fail "$name exhausted $MAX_ATTEMPTS attempts"
        fi
        wait_gpu "$name"
        wait_sandbox "$name"

        if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
            local aside="$out.partial-$(date -u +%Y%m%dT%H%M%SZ)"
            mv "$out" "$aside"
            log "$name: moved partial dir aside -> $aside"
        fi
        local i
        for i in $(seq 0 $((shards - 1))); do
            if [ -d "$out.shard$i" ] && [ -n "$(ls -A "$out.shard$i" 2>/dev/null)" ]; then
                local aside="$out.shard$i.partial-$(date -u +%Y%m%dT%H%M%SZ)"
                mv "$out.shard$i" "$aside"
                log "$name: moved partial shard $i aside -> $aside"
            fi
        done

        log "$name: attempt $attempt/$MAX_ATTEMPTS mode=$mode shards=$shards out=$out"
        status "$name" running "attempt=$attempt mode=$mode shards=$shards"

        local child="" dirs="" i last_growth
        last_growth=$(date +%s)
        if [ "$shards" -eq 0 ]; then
            dirs="$out"
            "$PY" "$REPO/scripts/campaign-20260926/rule_baseline.py" \
                --mode "$mode" \
                --output-dir "$out" \
                --temperature 0 \
                --max-new-tokens 1024 >> "$QLOG" 2>&1 &
            child="$!"
        else
            for i in $(seq 0 $((shards - 1))); do
                dirs="$dirs $out.shard$i"
                "$PY" "$REPO/scripts/campaign-20260926/rule_baseline.py" \
                    --mode "$mode" \
                    --output-dir "$out.shard$i" \
                    --shard-id "$i" \
                    --shard-count "$shards" \
                    --temperature 0 \
                    --max-new-tokens 1024 >> "$QLOG" 2>&1 &
                child="$child $!"
            done
        fi
        child="${child# }"; dirs="${dirs# }"

        while :; do
            sleep "$POLL_SECONDS"
            local all_dead=yes
            for pid in $child; do
                if kill -0 "$pid" 2>/dev/null; then all_dead=no; fi
            done
            if [ "$all_dead" = yes ]; then
                local rc=0
                wait $child 2>/dev/null || rc=$?
                local all_complete=yes
                for d in $dirs; do
                    [ -f "$d/COMPLETE" ] || all_complete=no
                done
                if [ "$all_complete" = yes ]; then
                    log "$name: children exited rc=$rc with COMPLETE"
                    if [ "$shards" -gt 0 ]; then
                        # shellcheck disable=SC2086
                        "$PY" "$REPO/scripts/campaign-20260926/merge_rule_shards.py" \
                            --output-dir "$out" --shards $dirs >> "$QLOG" 2>&1
                    fi
                    if [ -f "$out/COMPLETE" ]; then
                        log "$name: complete"
                        break
                    fi
                fi
                log "$name: children exited rc=$rc without all COMPLETE"
                attempt=$((attempt + 1))
                break
            fi
            local newest
            # shellcheck disable=SC2086
            newest=$(find $dirs -name 'trajectories.jsonl' -newermt "-${STALL_SECONDS} seconds" 2>/dev/null | head -1)
            if [ -z "$newest" ] && [ "$last_growth" -lt "$(($(date +%s) - STALL_SECONDS))" ]; then
                log "$name: no journal growth in ${STALL_SECONDS}s, killing children ($child)"
                kill -TERM $child 2>/dev/null; sleep 10; kill -KILL $child 2>/dev/null
                wait $child 2>/dev/null
                attempt=$((attempt + 1))
                break
            fi
            if [ -n "$newest" ]; then
                last_growth=$(date +%s)
            fi
        done
    done
}

log "queue start (thesis E0 baselines)"
status init starting "three slots: base_direct, base_tool, rule_strategy"

run_eval_slot
run_rule_slot base_tool "$BASE_TOOL_OUT" all-tools 0
run_rule_slot rule_strategy "$RULE_OUT" rule 3

log "queue done: all three slots complete"
status all done "base_direct + base_tool + rule_strategy complete"
printf '%s\n' "$(date -u +%FT%TZ) all slots complete" > "$REPO/artifacts/runs/thesis_e0_queue_DONE"
