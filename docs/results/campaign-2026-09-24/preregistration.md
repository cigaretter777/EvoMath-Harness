# Pre-registration — overnight campaign 2026-09-24

Committed **before** any of the results below exist. This file is the decision
rule; the analysis scripts are committed alongside it, so every number in the
morning report is reproducible by re-running a committed script. If a result
contradicts this document, the document wins and the result is reported as a
falsification, not re-interpreted.

Host: AutoDL single RTX 4090 24GB. Checkout pinned at `942b7d5` for all GPU
arms (this commit adds only docs/scripts; `src/adaptive_math` is byte-identical
to `942b7d5`, verified by the extractor/verifier source hashes below).

Shared identity of every GPU arm (from `run_model_eval.py --dry-run`, 2026-09-24):

| field | value |
|---|---|
| base_model | `hf-cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc…` |
| adapter_sha256 | `2868f83e9c3bfbe4a0eb6a115575ab4d7fe7cf4357028dcf9c4391c2a71fecaa` |
| eval_parquet_sha256 | `3b3255316c7e57480982e041d5e7172ca63b6052b15f7a5269c9ede574e05c43` |
| extractor_source_sha256 | `029570bd2957e313a74e17ea4b96cf12fc16c2e5b523b329fe18bf1636707581` |
| verifier_source_sha256 | `b690cc7deb03abacb4444cc5485f0aff4e986b1b6ff0907c070359f712503fa7` |
| data_manifest_sha256 | `908af3c0e65a297df9e5cea9399a70e7c0baabbd6d390abb49b96295834054b6` |
| decoding | `do_sample=False`, `batch_size=1`, `limit=200` (frozen Omni-MATH, sorted by task_id) |

---

## P1 — R2 rollout-health diagnostic (10 tasks × 4 samples, live tools)

**Question.** During training the R2 run recorded `tool_call_count=0` for all 12
steps. Is that (a) the policy never emitting a tool action, (b) emitting
near-miss actions that fail to parse, or (c) emitting valid calls whose
observations it then ignores?

**Measured, pre-specified.** Per trajectory: termination reason, tool events,
`invalid_actions`, generated tokens, reward; per group: mean/std/effective.
Sandbox liveness logged every 60s (`sandbox_health_20260924.log`) so any
mid-run tunnel drop is attributable to a minute rather than silently mixing arms.

**Decision rule.**
- If tool events == 0 in all 40 → the behaviour-prior branch of step 2 is
  triggered: no reward term points at tool use, and SFT (dp_v1, pure DIRECT)
  demonstrated none. Report as *structural*, not as "12 steps was too few".
- If any trajectory contains a raw with `<tool_call` that failed to parse → the
  failure is protocol-side, and it becomes a Harness-patch candidate (outer loop).
- `effective_group_rate` is reported with n=10 and explicitly labelled
  underpowered; it gates *whether more RL is worth GPU time*, it does not by
  itself justify a training run.

## E1 — Harness patch candidate: unclosed `<think>` rescue (offline, no GPU)

**Mechanism (read from `parser.py`, not assumed).** `_ENVELOPE` is applied with
`fullmatch`; its `think` group requires a closing `</think>`. A turn of the form
`<think>\n\n<final>{…}</final>` therefore fails with `INVALID_ENVELOPE` even
though the action itself is well-formed. Observed live in P1: one R2 sample
re-emitted the identical 48-char string 6 times, was rejected 6 times, hit
`max_steps` and scored 0 **while holding the correct answer**.

**Candidate rule.** Strip a leading `<think>` that is never closed, then re-run the
existing `fullmatch`. Nothing else changes. Applied as post-processing to stored
`raw_output` fields; no production code is modified for the measurement.

**Population.** All stored raws: `sft_dp_v1_omnimath_200_evalv2` (base, sft),
`r0_omnimath_200`, `r2_omnimath_200` (200 each), plus P1 trajectories.

**Pre-committed prediction.** The 2026-09-19 invalid audit found 95.4% of
invalids are truncation at the 1024 cap, i.e. they have **no** closing
`</final>` at all and cannot be rescued by this rule. Therefore E1 should
rescue **≤ 5%** of invalid predictions in the eval arms. If it rescues more, my
model of the failure population is wrong and that is reported as the finding.

**Promotion gate (all four, else REJECT).**
1. ≥ 1 stored arm gains a `correct` verdict after rescue;
2. **zero** arms lose a `correct` verdict (rescue must never demote);
3. paired McNemar exact on the rescued-vs-not table, reported with its p-value
   whatever it is — a non-significant result is reported as non-significant;
4. implemented behind tests, in a branch that does not touch the GPU arms.

## E2 — Harness patch candidate: generation cap 1024 → 2048 (paired, GPU)

**Hypothesis.** Most invalid predictions are truncation at the cap; raising the
cap converts them into extractable finals and raises accuracy, at a token and
latency cost. Single field changed; proven by the dry-run manifest diff above
(the two arms differ in `generation_config.max_new_tokens` and nothing else).

**Arms.** `harness_e2_champion_cap1024` (H1) vs `harness_e2_candidate_cap2048`
(H2), same checkpoint, same 200 tasks in the same order, greedy, batch 1.

**Primary endpoint.** Paired `correct` count per task_id, H2 − H1.

**Secondary endpoints.** invalid-prediction count, mean/p95 output tokens,
mean/p95 generation latency, and the truncation-class share of remaining
invalids (records at `output_tokens ≥ cap−4`).

**Pre-committed prediction.** The invalid audit says 186/195 invalids are
cap-truncation. If the hypothesis is right, H2's invalid count drops by at least
40% of H1's truncation-class invalids. If it does not, the truncated generations
were not "almost finished" — they were diverging — and the patch is rejected on
mechanism even if accuracy moves.

**Promotion gate (all four, else REJECT).**
1. Δcorrect point estimate ≥ −2 tasks (non-inferiority floor of 1%);
2. paired bootstrap (10,000 resamples, seed 20260913 for comparability with the
   four-way analysis) 95% CI lower bound ≥ −3 tasks;
3. any claim of *improvement* requires McNemar exact p < 0.05 **and**
   Δcorrect ≥ +5 tasks;
4. cost guard stated explicitly, not hidden: mean output tokens and p95 latency
   will rise by construction. Promote only if gate 3 passes; otherwise report
   the cost/benefit table and mark REJECTED.

**Partial completion (pre-committed).** H2 is the expensive arm (SFT outputs hit
the 1024 cap on 176/200 stored records, so a 2048 cap roughly doubles its
token count). If the night ends mid-arm, the analysis runs on the completed
prefix — both arms cover the same task_ids because generation order is
deterministic (sorted by task_id) — and the report labels it `n=<prefix>` and
**underpowered**. No extrapolation to 200 will be presented as a result.

## C1 — Confound measurement: batch_size 8 vs 1

**Finding that motivates it.** The 2026-09-14 SFT baseline was generated at
`batch_size=8`; the 2026-09-19 R0/R2 arms at `batch_size=1`. Prompt, extractor,
verifier and data hashes match across those runs; the generation batch does not.
The prior session's note that the four arms are "自洽" is therefore only partly
correct.

**Measurement.** H1 (batch 1, cap 1024, same checkpoint and adapter sha as the
stored baseline) vs the stored `sft_dp_v1_omnimath_200_evalv2` arm, task by task.

**Pre-committed interpretation.**
- Agreement ≥ 198/200 verdicts → batching is not a material confound; the
  four-way null result stands as reported.
- Agreement < 198/200 → the four-way comparison carries an uncontrolled
  generation-batch difference and must be relabelled in the docs as such. This
  is a negative finding about our own prior campaign and will be written down
  either way.

## What this campaign will NOT claim

- That GRPO improved mathematical ability (the four-way paired result was null).
- That tool use was "learned" or "failed to be learned" from 12 steps.
- Any accuracy number from an arm whose manifest hashes do not match this table.
- Any significance claim without the exact test and its p-value shown.
