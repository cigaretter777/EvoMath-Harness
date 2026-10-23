#!/bin/bash
# Operational resume driver for the 2026-10-23 R2 continuation evaluation.
# The evaluator is frozen; this driver only relaunches it with --resume when
# it exits because a transient sandbox-unavailable trajectory paused the run.
# It never edits the evaluator, the run manifest, or any journal row.
set -u

cd /root/autodl-tmp/BudgetConstrainedReasoningMathAgent
EVAL_DIR=artifacts/experiments/midterm_uniform_reward_v1/evaluation/r2_continuation_20261023
AUDIT=artifacts/experiments/midterm_uniform_reward_v1/audit/continuation_prep_20261023
HIST="$AUDIT/eval_resume_history.log"
PAUSES="$AUDIT/eval_pause_events.jsonl"
MAX_ATTEMPTS=200
unexpected=0
attempt=0

while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt + 1))
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  log="$AUDIT/eval_resume_${attempt}_${ts}.log"
  echo "[driver] attempt=$attempt start=$ts log=$log" | tee -a "$HIST"

  PYTHONPATH=/root/autodl-tmp/BudgetConstrainedReasoningMathAgent/src \
    PYTHONUNBUFFERED=1 \
    ADAPTIVE_MATH_SANDBOX_URL=http://127.0.0.1:8080 \
    /root/autodl-tmp/conda-envs/adaptive-math/bin/python scripts/eval/run_unified_tool_eval.py \
    --output-dir "$EVAL_DIR" --groups sft,r2,r2_75,r2_100 --resume \
    >"$log" 2>&1
  rc=$?
  echo "[driver] attempt=$attempt rc=$rc end=$(date -u +%Y%m%dT%H%M%SZ)" | tee -a "$HIST"

  if [ "$rc" -eq 0 ]; then
    echo "[driver] completed" | tee -a "$HIST"
    exit 0
  fi

  if [ "$rc" -eq 2 ] || [ "$rc" -eq 3 ]; then
    unexpected=0
    for f in "$EVAL_DIR"/group_*_PAUSED.json; do
      [ -f "$f" ] || continue
      printf '{"recorded_at_utc":"%s","attempt":%d,"file":"%s","content":' \
        "$(date -u +%FT%TZ)" "$attempt" "$(basename "$f")" >>"$PAUSES"
      cat "$f" >>"$PAUSES"
      printf '}\n' >>"$PAUSES"
    done
    sleep 3
    continue
  fi

  unexpected=$((unexpected + 1))
  echo "[driver] unexpected exit rc=$rc consecutive=$unexpected" | tee -a "$HIST"
  if [ "$unexpected" -ge 5 ]; then
    echo "[driver] aborting after repeated unexpected exits" | tee -a "$HIST"
    exit "$rc"
  fi
  sleep 10
done

echo "[driver] max attempts reached" | tee -a "$HIST"
exit 1
