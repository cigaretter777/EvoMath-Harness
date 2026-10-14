#!/usr/bin/env bash
# Round-1 thesis diagnostics -- pilot launcher.
#
# SFT policy x four single-variable budget configs (max_tool_calls 0/1/2/4)
# x 20 stratified tasks x 4 same-question samples, through the existing
# scripts/eval/run_rollout_health.py entry. One process per arm, four
# concurrent on one GPU (~4 x 3.5 GB fp16 weights; if that ever OOMs, run
# two waves via ADAPTIVE_MATH_ROUND1_ARMS="0 1" then "2 4").
#
# Pre-registration: docs/experiments/thesis-round1-prereg.md
#
# Gates before any weights move (all CPU-first):
#   * pilot parquet sha256 matches data/manifests/diagnostic_r1.json;
#   * expected task selection precomputed through the same select_task_rows
#     code path -- after the run each arm's task_pool.jsonl must match it
#     exactly (four-arm same-question pairing);
#   * model dir holds model.safetensors; output dirs are fresh;
#   * then wait for GPU device nodes (/dev) and the canonical sandbox
#     pong+42 probe, as the campaign launchers do.
#
# Post-run: per-arm episode count (expect 20 x 4 = 80), termination-reason
# summary, and the four-way task_pool diff. Exit codes: 0 = complete and
# paired; 1 = failed (see FAILED); 3 = complete but a post-check flagged.
#
# Launch, detached so SSH drops cannot kill it:
#   cd /root/autodl-tmp/Adaptive-Solver-main-git
#   setsid nohup bash scripts/diagnostics/run_round1_pilot.sh >/dev/null 2>&1 &

set -u

REPO=/root/autodl-tmp/Adaptive-Solver-main-git
cd "$REPO" || exit 1

PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python   # torch runtime
VENV_PY=$REPO/.venv/bin/python                            # CPU-only steps

ROUND1=$REPO/artifacts/diagnostics/r1
LOGDIR=$REPO/artifacts/runs/diagnostics_r1_pilot
mkdir -p "$LOGDIR" "$ROUND1"
QLOG=$LOGDIR/launcher.log
CHECKS=$LOGDIR/checks.log
STATUS=$LOGDIR/status.json
FAILED=$LOGDIR/FAILED
PIDS=$LOGDIR/pids
EXPECTED_IDS=$LOGDIR/expected_task_ids_pilot20.txt

ARMS=${ADAPTIVE_MATH_ROUND1_ARMS:-"0 1 2 4"}

PILOT_PARQUET=data/processed/diagnostics/r1/diagnostic_r1_pilot20.parquet
MANIFEST=data/manifests/diagnostic_r1.json
MODEL=$REPO/artifacts/models/qwen3_1_7b_sft_dp_v1_merged
REWARD_CONFIG=configs/reward/r0.yaml
SELECTION_SEED=20261014
GROUP_SIZE=4
TASK_COUNT=20
PER_GEN_TOKENS=1024
TEMPERATURE=0.7
TOLERANCE=unclosed_think,bare_final_scalar

SANDBOX_URL=${ADAPTIVE_MATH_SANDBOX_URL:-http://localhost:8080}
POLL_SECONDS=${ADAPTIVE_MATH_QUEUE_POLL:-60}

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export WANDB_MODE=disabled
export ADAPTIVE_MATH_SANDBOX_URL=$SANDBOX_URL
unset OMP_NUM_THREADS
unset PYTORCH_CUDA_ALLOC_CONF

log()    { echo "$(date -u +%FT%TZ) $*" >> "$QLOG"; }
status() { printf '{"ts":"%s","step":"%s","state":"%s","detail":"%s"}\n' \
             "$(date -u +%FT%TZ)" "$1" "$2" "$3" > "$STATUS"; }
fail()   { log "FAILED: $*"; status "${1%% *}" failed "$*"; printf '%s\n' "$*" > "$FAILED"; exit 1; }
flag()   { log "FLAGGED: $*"; echo "FLAGGED $*" >> "$CHECKS"; }

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

check_inputs() {
    echo "== round1 pilot checks $(date -u +%FT%TZ) ==" >> "$CHECKS"
    "$VENV_PY" - <<'PY' >> "$CHECKS" 2>&1 || fail "C1: pilot parquet sha does not match the manifest"
import hashlib, json
from pathlib import Path
manifest = json.loads(Path("data/manifests/diagnostic_r1.json").read_text())
entry = manifest["pilot"]
path = Path(entry["path"])
sha = hashlib.sha256(path.read_bytes()).hexdigest()
assert sha == entry["sha256"], f"{sha} != {entry['sha256']}"
print(f"C1 ok: pilot parquet sha {sha[:12]}, rows {entry['rows']}")
PY

    PYTHONPATH=src "$VENV_PY" - > "$EXPECTED_IDS" 2>> "$CHECKS" <<'PY' || fail "C2: expected task selection computed nonzero"
import pandas as pd
from adaptive_math.training.rollout_health import select_task_rows
rows = (
    pd.read_parquet("data/processed/diagnostics/r1/diagnostic_r1_pilot20.parquet")
    .to_dict(orient="records")
)
selected = select_task_rows(rows, count=20, seed=20261014)
assert len({row["task_id"] for row in selected}) == 20
print("\n".join(row["task_id"] for row in selected))
PY
    echo "C2 ok: $(wc -l < "$EXPECTED_IDS") expected task ids -> $EXPECTED_IDS" >> "$CHECKS"

    [ -f "$MODEL/model.safetensors" ] || fail "C3: model not found at $MODEL"
    local cap out
    for cap in $ARMS; do
        out=$ROUND1/sft_b${cap}_pilot
        if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
            fail "C3: stale output dir $out -- move it aside before launching"
        fi
    done
    echo "C3 ok: model at $MODEL; output dirs fresh" >> "$CHECKS"
}

launch_arm() {
    local cap=$1
    PYTHONPATH=src "$PY" scripts/eval/run_rollout_health.py \
        --data "$PILOT_PARQUET" \
        --model "$MODEL" \
        --agent-config "configs/agent/diagnostic_b${cap}.yaml" \
        --reward-config "$REWARD_CONFIG" \
        --task-count "$TASK_COUNT" \
        --group-size "$GROUP_SIZE" \
        --selection-seed "$SELECTION_SEED" \
        --temperature "$TEMPERATURE" \
        --max-new-tokens "$PER_GEN_TOKENS" \
        --parser-tolerance "$TOLERANCE" \
        --output-dir "$ROUND1/sft_b${cap}_pilot" \
        >> "$LOGDIR/b${cap}.log" 2>&1
}

post_check() {
    "$VENV_PY" - <<'PY'
import json, os
from pathlib import Path
arms = os.environ.get("ADAPTIVE_MATH_ROUND1_ARMS", "0 1 2 4").split()
expected = Path("artifacts/runs/diagnostics_r1_pilot/expected_task_ids_pilot20.txt").read_text().split()
flags = 0
for cap in arms:
    out = Path(f"artifacts/diagnostics/r1/sft_b{cap}_pilot")
    episodes = [json.loads(line) for line in (out / "trajectories.jsonl").read_text().splitlines() if line.strip()]
    pool = [json.loads(line)["task"]["task_id"] for line in (out / "task_pool.jsonl").read_text().splitlines() if line.strip()]
    term = {}
    for episode in episodes:
        key = episode["trajectory"]["termination_reason"]
        term[key] = term.get(key, 0) + 1
    paired = pool == expected
    complete = len(episodes) == len(expected) * 4
    if not (paired and complete):
        flags += 1
    print(f"b{cap}: episodes={len(episodes)}/{len(expected) * 4} paired={paired} term={term}")
print(f"post-check flags: {flags}")
raise SystemExit(0 if flags == 0 else 3)
PY
}

status start running "round1 pilot launcher"
log "launcher start; arms=$ARMS"

check_inputs

wait_gpu
wait_sandbox
status gpu ready "device + sandbox pong+42"
log "gpu and sandbox ready; launching arms: $ARMS"

: > "$PIDS"
for cap in $ARMS; do
    launch_arm "$cap" &
    echo "$cap $!" >> "$PIDS"
done

for cap in $ARMS; do
    pid=$(awk -v c="$cap" '$1 == c {print $2}' "$PIDS")
    rc=0
    wait "$pid" || rc=$?
    if [ "$rc" -ne 0 ]; then
        fail "arm b$cap exited rc=$rc (see $LOGDIR/b$cap.log)"
    fi
    log "arm b$cap exit 0"
    echo "arm b$cap exit 0" >> "$CHECKS"
done

status checks running "post-run pairing and completeness"
rc=0
post_check >> "$CHECKS" 2>&1 || rc=$?
if [ "$rc" -eq 3 ]; then
    flag "post-check flagged: pairing/completeness (see $CHECKS)"
    status checks flagged "see $CHECKS"
    log "pilot complete but flagged"
    exit 3
fi
if [ "$rc" -ne 0 ]; then
    fail "post-check errored rc=$rc (see $CHECKS)"
fi

status done complete "four arms paired and complete"
log "pilot complete; all checks passed"
exit 0
