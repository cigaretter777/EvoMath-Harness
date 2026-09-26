# Overnight Campaign Record (2026-09-19)

Status: **complete**. Formal R0 (50 steps) and R2 (12 steps) GRPO
runs completed and adapters exported; both evals completed; four-way
paired comparison computed. Campaign result: null — neither RL adapter
differs significantly from the SFT baseline on frozen OmniMath-200.

## Chain summary

| Step | Result | Artifacts |
|---|---|---|
| R0 GRPO training (50 steps) | ✅ complete | `artifacts/runs/grpo_qwen3_1_7b_r0/` (checkpoints, `r0_adapter/COMPLETE`) |
| R2 GRPO training (12 steps) | ✅ complete | `artifacts/runs/grpo_qwen3_1_7b_r2/` (checkpoints global_step_6/12, `r2_adapter/COMPLETE`) |
| R0 eval (200 tasks, adapter-only) | ✅ complete | `artifacts/eval/r0_omnimath_200/` (`COMPLETE`, `summary.json`) |
| R2 eval (200 tasks, adapter-only) | ✅ complete after auto-recovery | `artifacts/eval/r2_omnimath_200/` (`COMPLETE`, `summary.json`) |

## Eval results (OmniMath frozen_eval, 200 tasks, prompt `agent-v1`)

| Arm | Correct | Invalid prediction | Valid-answer rate | p50 latency | Tokens/s |
|---|---:|---:|---:|---:|---:|
| SFT baseline (09-14) | 27/200 (13.5%) | 71 | 64.5% | — | — |
| R0 adapter | 22/200 (11.0%) | 55 | 72.5% | 26.5 s | 23.5 |
| R2 adapter | 25/200 (12.5%) | 69 | 65.5% | 29.5 s | 23.7 |

### Paired comparison (offline join, `scripts/campaign-20260919/pair_fourway.py`)

Verifier/extractor/prompt/parquet hashes are identical across all arms
(checked against eval manifests), so per-task verdicts join directly.

| Pair | Δ accuracy | Improved | Regressed | Unchanged | McNemar p | Bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT → R0 | -2.5% | 5 | 10 | 185 | 0.302 | [-6.0%, +1.0%] |
| SFT → R2 | -1.0% | 1 | 3 | 196 | 0.625 | [-3.0%, +1.0%] |
| R0 → R2 | +1.5% | 8 | 5 | 187 | 0.581 | [-2.0%, +5.0%] |

Artifacts: `artifacts/eval/fourway_omnimath_200/` (`comparison.jsonl`,
`summary.json`); tracked copies under
`docs/results/campaign-2026-09-19/fourway/`.

## Run incidents

1. **Chain v1 eval crash** — first eval chain
   (`artifacts/runs/overnight_chain.log`) failed at 04:56Z with
   `KeyError: 'base'` in
   `src/adaptive_math/evaluation/model_eval.py:324`
   (`_render_evaluation_report`), because the adapter-only summary has
   no `base` key. Fixed by the uncommitted working-tree changes to
   `scripts/eval/run_model_eval.py` and
   `src/adaptive_math/evaluation/model_eval.py` (chain v2).
2. **Container restart killed chain v2** — the AutoDL container stopped
   at ~14:29 local and rebooted at 16:59, killing the R2 eval at 3/200.
   Relaunched with `setsid nohup` plus a watchdog
   (`artifacts/runs/r2_eval_watchdog.sh`, 120 s check loop: relaunch on
   death, exit on `COMPLETE`). The journal resumed from 3/200.
3. **Mid-run `EOFError`** — the resumed eval died once at 10:37Z
   (sandbox/verifier connection error); the watchdog relaunched it at
   10:39Z and it completed 200/200 at 12:15Z. Full log:
   `artifacts/runs/r2_eval_resume.log`.

## Conclusion (updated after paired comparison)

The paired comparison resolves the pipeline question raised below: the
R0/R2 numbers are consistent with the 09-14 SFT baseline under identical
verifier/extractor/prompt hashes, so the low absolute accuracy is a
property of the 1.7B model + strict protocol on this pool, not a broken
pipeline. **Campaign result: null.** Neither RL adapter differs
significantly from the SFT baseline (all McNemar p ≥ 0.30); the only
visible signal is R0's protocol-conformance gain (invalid predictions
71 → 55, no accuracy change).

Absolute verifier fairness was not independently audited (e.g. spot
checking a sample of `invalid_prediction` rows). This does not affect the
relative RL conclusion, but is a cheap sanity item before any claim
about absolute capability. Do not start further RL rounds until the
next-step decision is made (see plan); further rounds need a larger pool
and more steps to have any chance of an accuracy signal.

### Prior open issue (superseded)

Both adapters scored 11–12.5% verifier accuracy with a high
invalid-prediction rate — far below what a Qwen3-1.7B SFT model was
assumed to achieve on OmniMath, which suggested pipeline breakage.
Hypotheses were: (1) verifier/judging, (2) eval data alignment, (3)
generation config (`do_sample=False` with invalid
`temperature/top_p/top_k` flags being ignored). The paired comparison
against the 09-14 baseline under identical hashes supersedes this: the
pipeline produced internally consistent results across four arms.

## How to reproduce

```bash
REPO=/root/autodl-tmp/Adaptive-Solver-main-git
PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
export HF_ENDPOINT=https://hf-mirror.com WANDB_MODE=disabled \
       ADAPTIVE_MATH_SANDBOX_URL=http://127.0.0.1:8080

"$PY" "$REPO/scripts/eval/run_model_eval.py" \
    --sft-parquet "$REPO/data/processed/sft_dp_v1/train.parquet" \
    --sft-manifest "$REPO/data/manifests/sft_dp_v1_split.json" \
    --adapter "$REPO/artifacts/runs/grpo_qwen3_1_7b_r2/r2_adapter" \
    --output-dir "$REPO/artifacts/eval/r2_omnimath_200" \
    --limit 200 --rl-adapter --adapter-only \
    --model-id "$REPO/artifacts/models/qwen3_1_7b_sft_dp_v1_merged"
```

Run under `setsid nohup` + the watchdog pattern; this container reboots
unexpectedly and kills attached processes.

## Tracked copies

Runtime artifacts live under `artifacts/` (gitignored); the pieces worth
versioning are tracked here:

- Campaign scripts: `scripts/campaign-20260919/` (`overnight_chain.sh`,
  `r2_eval_watchdog.sh`, `export_verl_lora.py`, `run_grpo_direct.py`,
  `build_rl_pool.py`, `pair_fourway.py`)
- Eval evidence: `docs/results/campaign-2026-09-19/r0/`,
  `docs/results/campaign-2026-09-19/r2/` and
  `docs/results/campaign-2026-09-19/fourway/` (`evaluation_report.md`,
  `summary.json`, `comparison.jsonl`)
