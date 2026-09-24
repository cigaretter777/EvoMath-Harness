#!/usr/bin/env bash
# Overnight campaign queue (2026-09-24). Serial, single-writer on the GPU.
#
# Slots:
#   wait_p1  -- the R2 rollout-health diagnostic must finish first
#   h1       -- champion arm: SFT model, cap 1024, batch 1, frozen 200
#   h2       -- candidate arm: identical config, ONLY max_new_tokens 1024->2048
#
# Guardrails (the 2026-09-19 incident: a watchdog relaunched into a non-empty
# output dir and spun on FileExistsError for four days, undetected):
#   - bounded attempts per slot; on exhaustion write FAILED and exit non-zero
#   - stall detection on the journal's mtime, not just on process liveness
#   - status.json heartbeat so an outside monitor can see the real state
#   - run_model_eval resumes from its .in_progress journal, so a relaunch after
#     a stall or a container restart continues instead of restarting
set -u

REPO=/root/autodl-tmp/Adaptive-Solver-main-git
PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
QLOG=$REPO/artifacts/runs/night_queue_20260924.log
STATUS=$REPO/artifacts/runs/night_queue_status.json
P1_DIR=$REPO/artifacts/rollout_health/r2_diag_10x4_seed42_live

MODEL_ID=/root/autodl-tmp/hf-cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
SFT_ADAPTER=$REPO/artifacts/sft/qwen3_1_7b_sft_dp_v1/adapter
SFT_ADAPTER_SHA=2868f83e9c3bfbe4a0eb6a115575ab4d7fe7cf4357028dcf9c4391c2a71fecaa
SFT_PARQUET=$REPO/data/processed/sft_dp_v1/train.parquet
SFT_MANIFEST=$REPO/data/manifests/sft_dp_v1_split.json

POLL_SECONDS=${ADAPTIVE_MATH_QUEUE_POLL:-60}
STALL_SECONDS=${ADAPTIVE_MATH_QUEUE_STALL:-1800}
MAX_ATTEMPTS=${ADAPTIVE_MATH_QUEUE_ATTEMPTS:-4}
P1_DEADLINE_SECONDS=${ADAPTIVE_MATH_QUEUE_P1_DEADLINE:-7200}

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
unset OMP_NUM_THREADS

log() { echo "$(date -u +%FT%TZ) $*" >> "$QLOG"; }

status() {
    # $1=slot $2=state $3=detail
    printf '{"ts":"%s","slot":"%s","state":"%s","detail":"%s","p1_trajectories":%s}\n' \
        "$(date -u +%FT%TZ)" "$1" "$2" "$3" \
        "$(wc -l < "$P1_DIR/trajectories.jsonl" 2>/dev/null || echo 0)" > "$STATUS"
}

fail() {
    log "FAILED: $*"
    status "${1%% *}" failed "$*"
    printf '%s\n' "$*" > "$REPO/artifacts/runs/night_queue_FAILED"
    exit 1
}

# Run one eval slot to completion. Args: slot_name out_dir max_new_tokens
run_slot() {
    local name=$1 out=$2 cap=$3
    local journal="$out.in_progress"
    local attempt=1

    while :; do
        if [ -f "$out/summary.json" ]; then
            log "$name: complete ($out/summary.json present)"
            status "$name" complete "$out"
            return 0
        fi
        if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
            fail "$name exhausted $MAX_ATTEMPTS attempts"
        fi

        nvidia-smi >/dev/null 2>&1 || { log "$name: no GPU yet, waiting"; sleep "$POLL_SECONDS"; continue; }

        log "$name: attempt $attempt/$MAX_ATTEMPTS cap=$cap out=$out"
        status "$name" running "attempt=$attempt cap=$cap"

        "$PY" "$REPO/scripts/eval/run_model_eval.py" \
            --eval-parquet "$REPO/data/processed/v1/frozen_eval.parquet" \
            --data-manifest "$REPO/data/manifests/v1.json" \
            --sft-parquet "$SFT_PARQUET" \
            --sft-manifest "$SFT_MANIFEST" \
            --adapter "$SFT_ADAPTER" \
            --expected-adapter-sha256 "$SFT_ADAPTER_SHA" \
            --output-dir "$out" \
            --model-id "$MODEL_ID" \
            --max-new-tokens "$cap" \
            --batch-size 1 \
            --limit 200 \
            --adapter-only >> "$QLOG" 2>&1 &
        local child=$!

        local last_growth
        last_growth=$(date +%s)
        while :; do
            sleep "$POLL_SECONDS"
            if ! kill -0 "$child" 2>/dev/null; then
                wait "$child"; local rc=$?
                if [ -f "$out/summary.json" ]; then
                    log "$name: child exited rc=$rc with summary.json"
                    break
                fi
                log "$name: child exited rc=$rc without summary.json (journal kept for resume)"
                attempt=$((attempt + 1))
                break
            fi
            # liveness = the journal growing, not merely a live pid
            local newest
            newest=$(find "$journal" -name '*_predictions.jsonl' -newermt "-${STALL_SECONDS} seconds" 2>/dev/null | head -1)
            if [ -z "$newest" ]; then
                log "$name: no journal growth in ${STALL_SECONDS}s, killing $child"
                kill -TERM "$child" 2>/dev/null; sleep 10; kill -KILL "$child" 2>/dev/null
                wait "$child" 2>/dev/null
                attempt=$((attempt + 1))
                break
            fi
        done
    done
}

log "queue start; waiting for P1 ($P1_DIR/COMPLETE)"
status wait_p1 waiting "rollout-health diagnostic"
waited=0
while [ ! -f "$P1_DIR/COMPLETE" ]; do
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
    if [ "$waited" -gt "$P1_DEADLINE_SECONDS" ]; then
        fail "wait_p1 P1 did not complete within ${P1_DEADLINE_SECONDS}s"
    fi
done
log "P1 complete after ${waited}s"

run_slot h1_champion_cap1024 "$REPO/artifacts/eval/harness_e2_champion_cap1024" 1024
run_slot h2_candidate_cap2048 "$REPO/artifacts/eval/harness_e2_candidate_cap2048" 2048

log "queue done"
status all done "both arms complete"
