# Overnight campaign 2026-09-24 — results

Pre-registration: `preregistration.md` in this directory (committed as `35898ff`
before any result existed; addendum B committed as `004b60c` before H3 was to run).
Every number below is produced by a committed script listed in §8.

Host: AutoDL, single RTX 4090 24GB. Main checkout pinned at `35898ff` for all GPU
arms. Sandbox: SandboxFusion over an SSH remote forward on `127.0.0.1:8080`,
probed every minute (`ping` + a real `print(6*7)` execution).

## 1. Verdicts

| id | question | result | verdict |
|---|---|---|---|
| E2 | does raising the generation cap 1024→2048 improve accuracy? | 27/200 → 27/200, **200/200 tasks unchanged**, McNemar p=1.0, bootstrap CI [0.0, 0.0] | **REJECT** |
| C1 | is the 2026-09-14 batch-8 baseline comparable to the batch-1 RL arms? | raw outputs identical on **3/200**; verdict labels agree on 130/200; correct/incorrect agrees on 185/200 | **confounded — prior four-way comparison must be relabelled** |
| P1 | why was `tool_call_count=0` during R2 training? | the policy *does* emit tool calls; the strict envelope parser rejects them (21/72 turns malformed) | **protocol artifact, not policy absence** |
| E1 | do parser-tolerance rules recover correct answers? | 10 turns rescued, 3 identifiable trajectories, **0 correct recovered**, reward −1.8 in aggregate | **REJECT** |
| F3 | is the R2 reward well-formed at budget exhaustion? | budget-exhausted trajectories score **0.0** while honest wrong answers with invalid actions score **−0.2**; 2 samples in one group flip from advantage **+0.5773** to **−0.8165** under the fix | **defect confirmed, fix not yet run** |

## 2. E2 — generation cap 1024 → 2048 (paired, frozen Omni-MATH 200)

Single-field pairing verified from the two `eval_manifest.json` files: identical
`git_sha` (`35898ff`), `adapter_sha256` (`2868f83e…`), `extractor_source_sha256`
(`029570bd…`), `verifier_source_sha256` (`b690cc7d…`), `eval_parquet_sha256`
(`3b325531…`), `task_ids_sha256` (`1fc257f2…`), `do_sample=false`, `batch_size=1`.
The only differing field is `generation_config.max_new_tokens`.

| metric | champion cap=1024 | candidate cap=2048 |
|---|---|---|
| correct | 27 (13.5%) | 27 (13.5%) |
| incorrect | 102 | 122 |
| invalid_prediction | 71 | 51 |
| valid_answer_rate | 64.5% | 74.5% |
| at-cap generations | 69/200 (34.5%) | 47/200 (23.5%) |
| mean output tokens | 679.6 | 945.5 (+39%) |
| p50 / p95 latency | 23.0s / 34.3s | 23.1s / **68.1s** (+98% p95) |
| generation wall clock | 73.9 min | 103.3 min (+40%) |
| peak GPU memory | 3.63 GB | 3.75 GB |

Paired outcome: `improved 0, regressed 0, unchanged 200`. Not one task changed
its correctness. The only label movement is `invalid_prediction → incorrect` on
exactly 20 tasks: doubling the budget turned 20 unextractable generations into
extractable **wrong** answers.

Gate evaluation (pre-registered §E2):
1. non-inferiority Δ ≥ −2 tasks → Δ = 0, **pass**
2. bootstrap CI lower bound ≥ −3 → CI [0.0, 0.0], **pass**
3. improvement requires McNemar p < 0.05 and Δ ≥ +5 → p = 1.0, Δ = 0, **fail**
4. cost guard → tokens +39%, p95 latency +98%, wall +40%, with zero accuracy gain, **fail**

Mechanism prediction (pre-registered): invalids should fall by ≥40% of the
truncation class. Observed 71 → 51 = −28%, and **none** of the 20 recovered
answers was correct. The prediction fails, so the patch is rejected on mechanism
as well as on cost.

**Interpretation.** The truncated generations were not "one more step from
correct"; they were diverging. This falsifies the hypothesis carried over from
the 2026-09-19 campaign that reducing `TRUNCATED_AT_CAP` was the right first
Evo target. Corollary: **invalid rate is not the accuracy bottleneck.** Both arms
solve the same 27 tasks; the candidate merely relabels 20 failures from
"unextractable" to "wrong".

Internal consistency check that supports the pairing: 131/200 raw outputs are
byte-identical between the arms (exactly the generations that did not hit the
champion's cap), and all 69 that differ are the champion's at-cap set.

## 3. C1 — the batch-size confound in the prior four-way comparison

The 2026-09-14 SFT baseline was generated at `batch_size=8`; the 2026-09-19 R0/R2
arms at `batch_size=1`. H1 reproduces the baseline's model identity (same base
snapshot, same adapter sha) at batch 1, so H1 vs the stored baseline isolates
batching (plus the commit difference `59ed1c2` → `35898ff`).

| measure | value |
|---|---|
| shared tasks | 200 |
| raw outputs byte-identical | **3 (1.5%)** |
| verifier_status agreement | 130/200 (65.0%) |
| correct/incorrect agreement | 185/200 (92.5%) |
| correct count | baseline 26 → H1 27 |
| discordant verifier_status | 70 |

Greedy decoding is **not** batch invariant on this stack: 197/200 generations
differ in text. Most discordance is label shuffling between `invalid_prediction`
and `incorrect`, but 15 tasks (7.5%) disagree on correctness itself.

Pre-registered interpretation: agreement < 198/200 ⇒ the four-way comparison
carries an uncontrolled generation-batch difference. The reported deltas in that
campaign were −5, −2 and +3 tasks — **smaller than the 15-task discordance that
batching alone introduces**. Consequences:

- The null result survives (no significant difference was claimed), but any
  *directional* reading of SFT→R0/R2 differences of a few tasks is not supportable.
- `docs/results/campaign-2026-09-19/` must be relabelled with this confound.
- Rule going forward: **all arms of a paired comparison must share batch size**,
  and the batch size belongs in the identity table, not just in generation config.
- E2 was immune to this by construction (both arms batch 1, same commit), which
  is why its null is exact rather than approximate.

## 4. P1 — R2 rollout-health diagnostic (10 tasks × 4 samples, live tools)

Config: R2 adapter on the merged SFT model, `configs/reward/r2.yaml`,
`configs/agent/default.yaml`, seed 42, temperature 0.7, cap 1024, sandbox live
for the whole run (60s probes all `pong` + `42`). Output:
`artifacts/rollout_health/r2_diag_10x4_seed42_live/`.

| quantity | value |
|---|---|
| trajectories | 40 |
| termination | final 34, **max_steps 6** |
| tool events executed | **0** |
| invalid actions | **38** (across 40 trajectories) |
| model turns | 72 |
| parse outcomes | ok 34, invalid_envelope 23, multiple_actions 8, invalid_schema 6, invalid_json 1 |
| turns with an unclosed `<think>` | **21/72 (29.2%)** |
| trajectories that re-emitted an identical rejected output | 6 |
| mean generated tokens | 629.5 (p50 391, max 2108) |
| reward | mean 0.42, std 0.50; positive 17, zero 22, negative 1 |
| groups | 10, effective **6** (rate 0.6), all-zero groups 0 |

The decisive observation: among the 6 looping trajectories, two contain
well-formed tool calls that never executed:

- a `python` tool call carrying `from sympy import symbols, solve, sqrt ...`,
  re-emitted **5** times in one trajectory
- a `sympy` tool call with `operation=solve, expression=b-168`, re-emitted **3** times

Both were rejected as `invalid_envelope` because the turn opened a think block
and never closed it. `tool_calls_total = 0` therefore does **not** mean the
policy never tried to use a tool; it means the protocol layer discarded every
attempt, and the model then repeated itself until the step budget ran out.

Attribution per the design's failure-attribution rules: this is a
**Harness/protocol** defect, not evidence that a 1.7B model cannot learn tool
use. The 2026-09-19 conclusion "R2 never learned the value of tools" must be
rewritten: the tool channel was never open during that training run either,
since `training/verl_environment.py` uses the same strict `parse_action`.

Contrast with the pre-RL SFT checkpoint (2026-09-15 smoke, 40 trajectories):
1 malformed turn in 40, `invalid_actions_total = 1`. The protocol degradation is
a property of the RL arm, which points at F3 below rather than at SFT.

Group structure: 6/10 groups effective (rate 0.6), zero all-zero groups, mean
reward 0.42. The 0.1 effective-group rate measured on the 2026-09-15 SFT smoke
run does **not** carry over to the R2 policy; quoting it as a general property of
the task pool was wrong and is withdrawn. n=10 groups remains small.

## 5. E1 — parser-tolerance rules, offline counterfactual (REJECT)

Rules evaluated as post-processing on stored agent-path raws (P1 live 40
trajectories, 2026-09-15 smoke 40, dead-tools arm 4). The direct-eval arms are
excluded by construction: `evaluation/model_eval.py` scores with
`verifier.extract` and never calls `parse_action`, so a parser patch cannot move
them. That split is itself an architectural finding — **the agent protocol parser
and the eval extractor are two independent code paths**, and any HarnessSpec must
record which one a patch touches.

| rule | turns rescued to ok | identifiable trajectories | correct gained | correct demoted | reward delta |
|---|---|---|---|---|---|
| A unclosed think | 4/72 (6 more advanced to a different error) | 1 | 0 | 0 | −0.2 |
| B bare final scalar | 6/72 | 2 | 0 | 0 | −0.4 |

Every rescued answer (`\frac{3}{2}`, `3024`, `1004`) was judged `incorrect` by the
production verifier, so gate 1 fails and both rules are **REJECTED**. Because the
invalid penalty had already accrued before the rescued turn, tolerating the
malformed envelope *lowers* reward (0.0 → −0.2) under the current plumbing.

Withdrawn claim: the pre-registration asserted one looping sample was "holding
the correct answer". The reference for that task is `120`; the model answered
`130^{\circ}`. The assertion was wrong and is retracted here rather than left
standing in the addendum.

Identifiability limit, stated rather than papered over: 2 further trajectories
would have executed a rescued tool call, which changes every later turn, so
their outcome is not computable offline. Settling them needs a paired re-run
(H3, §7).

## 6. F3 — reward inversion at budget exhaustion (defect confirmed)

Code path, verified by reading:

- `agent/environment.py:142-145` — `evaluate()` returns `None` when
  `final_answer is None`
- `training/verl_environment.py:139,151` — with a `None` verdict the reward stays
  `0.0` and `reward_for_trajectory` is never called, so the configured
  invalid-action penalty is skipped
- `scripts/eval/run_rollout_health.py` — same shape
- `evaluation/model_eval.py:209` — the eval loop scores the identical situation as
  an explicit `INVALID_PREDICTION`

The two loops disagree about the same fact. Measured on P1:

| | value |
|---|---|
| trajectories with no verdict | 6/40, all `max_steps`, all `invalid_actions=6`, all scored **0.0** |
| honest wrong answer with 2 invalid actions | **−0.2** |
| reward sum, as shipped → aligned with eval | 16.8 → **15.0** (exactly 6 × −0.3 = invalid_weight 0.1 × cap 3) |
| samples whose GRPO advantage flips sign | **2** (group 4): +0.5773 → −0.8165 |
| effective group rate before/after | 0.6 → 0.6 (unchanged) |
| control (2026-09-15 smoke, R0, invalid_weight 0) | 0 no-verdict trajectories, 0 flips |

So under R1/R2 the invalid penalty cannot fire on the trajectories that accumulate
the most invalid actions, because those are exactly the ones that exhaust the
budget. Within a group, looping to exhaustion scores *above* honest wrong
termination, and GRPO's group-relative advantage turns that into a **positive
learning signal for protocol failure**. This is a candidate mechanism for the
observed direction SFT (1 invalid action / 40 trajectories) → R2 (38 / 40), and
for R2's eval invalid count (69) exceeding R0's (55) — R0 has no invalid penalty,
hence no inversion.

The fix changes the direction of the signal, not its availability
(`effective_group_rate` is unchanged), so it is not a "more signal" claim.

**Not recoverable:** the R2 training run stored no per-step group rewards (wandb
disabled; only checkpoints and `resolved_config.yaml`), so the incidence of this
inversion during those 12 steps cannot be measured. No estimate is offered.
Process fix required: persist per-step group reward distributions, which §6.1 of
the design already asks to monitor.

## 7. Not done

- **H3** (rollout-health 10×4 with the tolerant parser, live tools) did not run.
  Its purpose was to answer the one question E1 could not: when the protocol
  accepts what this policy actually emits, do tool calls execute, and does the
  next turn use the returned observation? The parser patch exists as opt-in
  `tolerate=` with 7 failing tests written first, in the local worktree branch
  `campaign-20260924-work`; the implementation was not finished, so **no patch
  code is pushed and nothing in the main checkout changed**.
- Verifier coverage gaps (6 classes from the 2026-09-19 invalid audit) were not
  touched.
- 25-task group-structure measurement was not run; group statistics rest on n=10.

## 8. Reproduction

```bash
# P1 analysis
python scripts/campaign-20260924/analyze_rollout_health.py \
  --run-dir artifacts/rollout_health/r2_diag_10x4_seed42_live \
  --health-log artifacts/runs/sandbox_health_20260924.log

# E1 offline counterfactual
python scripts/campaign-20260924/eval_parser_rescue.py \
  --run-dir artifacts/rollout_health/r2_diag_10x4_seed42_live \
  --run-dir artifacts/rollout_health/smoke_10x4_seed42 \
  --run-dir artifacts/rollout_health/r2_diag_10x4_seed42.deadtools-2traj \
  --max-generated-tokens 6144

# F3 reward inversion
python scripts/campaign-20260924/analyze_reward_inversion.py \
  --run-dir artifacts/rollout_health/r2_diag_10x4_seed42_live \
  --run-dir artifacts/rollout_health/smoke_10x4_seed42

# E2 paired statistics + C1 batch confound
python scripts/campaign-20260924/pair_harness_e2.py \
  --champion artifacts/eval/harness_e2_champion_cap1024/sft_predictions.jsonl \
  --candidate artifacts/eval/harness_e2_candidate_cap2048/sft_predictions.jsonl \
  --baseline-batch8 artifacts/eval/sft_dp_v1_omnimath_200_evalv2/sft_predictions.jsonl
```

GPU accounting for the night: E2 champion 73.9 min + E2 candidate 103.3 min of
generation, P1 ~19 min, then **4h13m idle** (21:18Z → 01:31Z). Cause: the queue
was written to run two slots and exit, and no persistent process replaced it, so
nothing noticed completion and nothing launched the next slot. The monitor in
`scripts/campaign-20260924/campaign_monitor.sh` loops until every slot has
COMPLETE evidence, moves partial run directories aside instead of relaunching
into them, bounds retries, detects stalls from journal growth rather than process
liveness, and keeps the sandbox probe running for the whole campaign. It is
committed but was **not** started, since the campaign was stopped.

## 9. Artifact hashes

`artifacts/` is gitignored, so raw evidence stays on the host. These identify
exactly which files the numbers above came from:

| sha256 | artifact |
|---|---|
| `063f9f57f07084fefe8e4cd3dccba7d732246e0ca1be387f3678da3287b3cba3` | `artifacts/results/campaign-20260924/e2_c1_paired.json` |
| `5cdb4c8a8268f78e4ba25d5a56de1a7aed00575620da328905f52c197fcceb06` | `artifacts/eval/harness_e2_champion_cap1024/sft_predictions.jsonl` |
| `1e329136f1c21da8a92ac0ad4ebd918f591fa36fde102c6209fcf5c99e5f88e0` | `artifacts/eval/harness_e2_candidate_cap2048/sft_predictions.jsonl` |
| `51b0d862ea395d88eef51ef012fb2771319a08cb0fe349876a98cc198f69f6e7` | `artifacts/eval/harness_e2_champion_cap1024/eval_manifest.json` |
| `22cc6d0b77af8fc60d6a50b1253a1a12846d4bb0bccf6c7f907274b18145cb68` | `artifacts/eval/harness_e2_candidate_cap2048/eval_manifest.json` |
| `748ccc7fc02e495c233f163ddffe2b4f10a4633c88e4885241803396f7775d4a` | `artifacts/rollout_health/r2_diag_10x4_seed42_live/analysis_p1.json` |
| `c27a979dd7e67925e29a5626f3103f2ff86cdb0b166248d27327cb7df39246c5` | `artifacts/results/campaign-20260924/e1_parser_rescue.json` |
| `b60b51c79de3c75ea4f529683b7e20caaa1e3120e1993af6170792d90d89c02d` | `artifacts/results/campaign-20260924/f3_reward_inversion.json` |
