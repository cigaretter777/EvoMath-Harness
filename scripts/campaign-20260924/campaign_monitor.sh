#!/usr/bin/env bash
# Persistent campaign monitor (2026-09-24).
#
# Why this exists: the previous night_queue ran its two slots and exited, and
# nothing replaced it, so the GPU sat idle for over four hours after the arms
# completed. A queue that stops is not a monitor. This process loops until every
# slot has produced its COMPLETE evidence, keeps a sandbox liveness record going
# for the whole campaign instead of a fixed number of probes, and launches the
# next slot itself.
#
# Slots, in order:
#   p2_group_structure  rollout-health 25x4, strict parser, live tools
#   h3_tolerant_parser  rollout-health 10x4, tolerant parser from the worktree,
#                       gated on artifacts/runs/h3_ready (patch + green tests)
set -u

MAIN=/root/autodl-tmp/Adaptive-Solver-main-git
WORK=/root/autodl-tmp/Adaptive-Solver-campaign-work
PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
MLOG=$MAIN/artifacts/runs/campaign_monitor.log
STATUS=$MAIN/artifacts/runs/campaign_monitor_status.json
HEALTH=$MAIN/artifacts/runs/sandbox_health_20260924.log
SANDBOX_URL=${ADAPTIVE_MATH_SANDBOX_URL:-http://127.0.0.1:8080}
POLL=${ADAPTIVE_MATH_MONITOR_POLL:-60}

MERGED=$MAIN/artifacts/models/qwen3_1_7b_sft_dp_v1_merged
R2_ADAPTER=$MAIN/artifacts/runs/grpo_qwen3_1_7b_r2/r2_adapter

export HF_ENDPOINT=https://hf-mirror.com
export ADAPTIVE_MATH_SANDBOX_URL=$SANDBOX_URL
export WANDB_MODE=disabled
unset OMP_NUM_THREADS

log() { echo "$(date -u +%FT%TZ) monitor: $*" >> "$MLOG"; }

status() {
    printf '{"ts":"%s","state":"%s","slot":"%s","gpu_procs":%s,"sandbox":"%s"}\n' \
        "$(date -u +%FT%TZ)" "$1" "$2" "$(pgrep -cf '[r]un_rollout_health.py|[r]un_model_eval.py')" "$3" \
        > "$STATUS"
}

probe_sandbox() {
    local ping exec_out
    ping=$(curl -s -m 6 "$SANDBOX_URL/v1/ping" 2>/dev/null)
    exec_out=$(curl -s -m 20 -X POST "$SANDBOX_URL/run_code" -H 'Content-Type: application/json' \
        -d '{"code":"print(6*7)","language":"python"}' 2>/dev/null | grep -o '"stdout":"42' | head -1)
    echo "$(date -u +%FT%TZ) ping=$ping exec_probe=${exec_out:-FAILED}" >> "$HEALTH"
    [ "$ping" = '"pong"' ] && [ -n "$exec_out" ]
}

launch_rollout_health() {
    # $1=output dir $2=script path $3..=extra args
    local out=$1 script=$2; shift 2
    local name; name=$(basename "$out")
    log "launching $name: $script $*"
    setsid nohup "$PY" "$script" \
        --data "$MAIN/data/processed/v1/rl_dev.parquet" \
        --model "$MERGED" \
        --adapter "$R2_ADAPTER" \
        --agent-config "$MAIN/configs/agent/default.yaml" \
        --reward-config "$MAIN/configs/reward/r2.yaml" \
        --selection-seed 42 \
        --output-dir "$out" "$@" \
        >> "$MAIN/artifacts/runs/$name.log" 2>&1 </dev/null &
}

P2_DIR=$MAIN/artifacts/rollout_health/p2_group_structure_25x4_seed42
H3_DIR=$MAIN/artifacts/rollout_health/h3_tolerant_parser_10x4_seed42

log "monitor start"
while :; do
    sandbox_ok=unknown
    if probe_sandbox; then sandbox_ok=live; else sandbox_ok=dead; fi

    p2_done=no; [ -f "$P2_DIR/COMPLETE" ] && p2_done=yes
    h3_done=no; [ -f "$H3_DIR/COMPLETE" ] && h3_done=yes
    running=$(pgrep -cf '[r]un_rollout_health.py')

    if [ "$p2_done" = yes ] && { [ "$h3_done" = yes ] || [ ! -f "$MAIN/artifacts/runs/h3_ready" ]; }; then
        log "all launchable slots complete (p2=$p2_done h3=$h3_done h3_ready=$([ -f "$MAIN/artifacts/runs/h3_ready" ] && echo yes || echo no))"
        status done all "$sandbox_ok"
        echo "$(date -u +%FT%TZ) all slots complete" > "$MAIN/artifacts/runs/campaign_monitor_DONE"
        exit 0
    fi

    if [ "$running" -eq 0 ]; then
        if ! nvidia-smi >/dev/null 2>&1; then
            status waiting_gpu none "$sandbox_ok"
            log "no GPU; waiting"
        elif [ "$p2_done" = no ]; then
            # move a partial run aside instead of relaunching into it (the
            # 2026-09-19 watchdog spun for four days doing exactly that)
            if [ -d "$P2_DIR" ] && [ -n "$(ls -A "$P2_DIR" 2>/dev/null)" ]; then
                mv "$P2_DIR" "$P2_DIR.partial-$(date -u +%Y%m%dT%H%M%SZ)"
                log "moved partial P2 aside"
            fi
            launch_rollout_health "$P2_DIR" "$MAIN/scripts/eval/run_rollout_health.py" \
                --task-count 25 --group-size 4
            status running p2_group_structure "$sandbox_ok"
        elif [ "$h3_done" = no ] && [ -f "$MAIN/artifacts/runs/h3_ready" ]; then
            if [ "$sandbox_ok" != live ]; then
                status waiting_sandbox h3_tolerant_parser "$sandbox_ok"
                log "H3 needs live tools; sandbox is $sandbox_ok"
            else
                if [ -d "$H3_DIR" ] && [ -n "$(ls -A "$H3_DIR" 2>/dev/null)" ]; then
                    mv "$H3_DIR" "$H3_DIR.partial-$(date -u +%Y%m%dT%H%M%SZ)"
                    log "moved partial H3 aside"
                fi
                PYTHONPATH=$WORK/src launch_rollout_health "$H3_DIR" \
                    "$WORK/scripts/eval/run_rollout_health.py" \
                    --task-count 10 --group-size 4 \
                    --parser-tolerance unclosed_think,bare_final_scalar
                status running h3_tolerant_parser "$sandbox_ok"
            fi
        else
            status waiting_h3_patch h3_tolerant_parser "$sandbox_ok"
        fi
    else
        status running "$(pgrep -af '[r]un_rollout_health.py' | grep -o 'output-dir [^ ]*' | head -1)" "$sandbox_ok"
    fi
    sleep "$POLL"
done
