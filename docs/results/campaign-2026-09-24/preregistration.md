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

## Addendum A (2026-09-24, committed before any E1 result was computed)

**Correction to E1's measurement population.** E1 as pre-registered above named
the four direct-eval arms as its population. That is wrong, and the error was
found by reading the code rather than by looking at results:

`src/adaptive_math/evaluation/model_eval.py:204` scores a generation with
`extract(turn.text)` from `adaptive_math.verifier`. It never calls
`parse_action`. The envelope parser (`src/adaptive_math/agent/parser.py`) is
used only on the **agent path**: `agent/loop.py:40`, `training/verl_environment.py:136`
(GRPO rollouts), `training/sft_records.py`, `training/solution_traces.py`.

Consequences, all pre-committed here:

1. E1's population is the **agent-path raws only**: P1's 40 trajectories
   (R2 adapter), the 2026-09-15 smoke's 40 (SFT, pre-RL), and the 4-record
   dead-tools arm. The eval arms are excluded, and any E1 number quoted against
   them would be structurally zero and therefore meaningless.
2. Architectural finding for the Harness design: **the agent protocol parser and
   the eval extractor are two independent code paths.** A parser patch changes
   training rewards and agent rollouts (inner loop); an extractor patch changes
   eval labels (outer loop). Patch eligibility must be stated per loop, and the
   HarnessSpec must record which of the two a patch touches.
3. Disclosure: the E1 mechanism was observed on a 4-trajectory subsample of P1
   *before* this addendum (5 of 14 model_outputs opened `<think>` without closing
   it; one sample re-emitted an identical rejected string 6 times and hit
   `max_steps` while holding the correct answer). The rescue rate on agent-path
   turns is therefore **not** an uninformative prediction — the subsample already
   suggests it is large. What remains pre-committed and untouched is the
   promotion gate: rescue must gain ≥1 correct verdict, must demote zero, and
   the McNemar p-value is reported whatever it says.
4. Additional endpoint added, not replaced: for each rescued trajectory, report
   termination reason before/after (`max_steps` → `final`) and reward
   before/after, because on the agent path this patch changes **reward**, i.e.
   it changes what GRPO would have been trained on — a training-relevant patch,
   not a cosmetic label fix.

## Addendum B (2026-09-24 ~19:00Z, committed before H3 ran and before any E2 result)

### B.1 E1 outcome — both rules REJECTED on gate 1

Measured by `scripts/campaign-20260924/eval_parser_rescue.py` on the agent-path
population (P1 live run 40 trajectories, 2026-09-15 smoke 40, dead-tools arm 4):

| rule | turns rescued to `ok` | identifiable trajectories | correct gained | correct demoted | verdict |
|---|---|---|---|---|---|
| A unclosed think | 4 of 72 (plus 6 advanced to a different error) | 1 | 0 | 0 | **REJECT** |
| B bare final scalar | 6 of 72 | 2 | 0 | 0 | **REJECT** |

Every answer recovered by either rule was judged `incorrect` by the production
verifier (`\frac{3}{2}`, `3024`, `1004`), so gate 1 fails and gate 2 is
trivially satisfied. The pre-registered claim that the rescued sample was
"holding the correct answer" is **withdrawn**: the reference for that task is
`120` and the model answered `130^{\circ}`. On the 2026-09-15 smoke run
(pre-RL SFT) rule A found 1 malformed turn in 40; on the R2 run it found 21
malformed turns in 72 — protocol degradation is a property of the RL arm, not of
the SFT checkpoint.

### B.2 Finding F3 — reward inversion for budget-exhausted trajectories

Verified in code and in data, not inferred:

- `agent/environment.py:142-145` — `evaluate()` returns `None` when
  `final_answer is None`.
- `training/verl_environment.py:139,151` — when the verdict is `None`, reward
  stays `0.0` and `reward_for_trajectory` is never called, so the configured
  invalid-action penalty is skipped entirely.
- `scripts/eval/run_rollout_health.py` — same shape (`if verdict is not None`).
- `evaluation/model_eval.py:209` — the eval loop scores the identical situation
  as an explicit `INVALID_PREDICTION`. The two loops disagree.

Measured on P1 (`analyze_reward_inversion.py`): 6 of 40 trajectories terminated
at `max_steps` with `invalid_actions=6` and `verdict=None`, each scored `0.0`.
Aligning the agent path with the eval path moves reward_sum 16.8 → 15.0
(exactly 6 × −0.3 = invalid_weight 0.1 × cap 3) and **flips 2 samples in group
4 from advantage +0.5773 to −0.8165**: under the shipped plumbing, GRPO was
reinforcing the two protocol-failing samples in that group relative to their
peers. `effective_group_rate` is unchanged (0.6 → 0.6); the defect changes the
*direction* of the signal, not its availability. Control: the 2026-09-15 smoke
run (R0, `invalid_weight=0`) has 0 no-verdict trajectories and 0 flips, so the
inversion requires an invalid penalty and protocol failure together.

**Not recoverable, stated as a gap:** the R2 training run stored no per-step
group rewards (wandb disabled; only checkpoints and `resolved_config.yaml`), so
the incidence of this inversion *during* the 12 training steps cannot be
measured. Any claim about its training-time magnitude would be invented.
Process fix required: log per-step group reward distributions, as §6.1 of the
design already asks for.

**Proposed fix (inner loop, not run tonight):** when a trajectory terminates
without a final answer, score it with an `INVALID_PREDICTION` verdict and still
apply the reward config, so the agent path and the eval path agree. Requires a
failing test first, in a branch that does not touch the E2 checkout.

### B.3 H3 — parser-tolerance diagnostic (registered as a diagnostic, not a candidate)

Rules A and B are rejected for promotion by B.1. H3 is therefore **not** a
promotion attempt; it is the only way to answer the remaining step-2 question,
which the rejected patch happens to unblock: *when the protocol accepts the
output shapes this policy actually emits, does it call tools at all, and does it
use what the tool returns?*

- Design: same 10 tasks, same selection seed 42, same group size 4, same R2
  adapter, same reward config, same live sandbox, same generation config as P1.
  The only difference is a parser that applies rules A and B.
- Isolation: H3 runs from a separate git worktree so the E2 arms keep executing
  against the pinned checkout; the E2 manifests' source hashes cannot change.
- Endpoints, pre-specified: count of `tool_call` events (>0 or ==0); for each
  executed tool call, whether the following model turn references the returned
  observation; termination mix; invalid-action count; reward distribution.
- Decision rule: if `tool_call` events remain 0 with a tolerant parser, the
  zero-tool-use result is a **policy/prior** property (SFT demonstrated no tool
  call, and no reward term points at one) and the behaviour-prior branch of
  step 2 is triggered. If tool calls appear, the zero count in training was a
  **protocol** artifact, and the inner-loop conclusion recorded on 2026-09-19
  must be rewritten.

## What this campaign will NOT claim

- That GRPO improved mathematical ability (the four-way paired result was null).
- That tool use was "learned" or "failed to be learned" from 12 steps.
- Any accuracy number from an arm whose manifest hashes do not match this table.
- Any significance claim without the exact test and its p-value shown.
