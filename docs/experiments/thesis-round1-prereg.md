# Pre-registration: round-1 budget diagnostics (thesis)

Date 2026-10-14. Entry: `scripts/eval/run_rollout_health.py` at pin
`6f580bd` (origin/main). No new experimental system; no training in this
round. Launcher: `scripts/diagnostics/run_round1_pilot.sh`.

## Question

Does a finite tool-call budget constrain this policy's solving ability —
and does the constraint act as a task-difficulty axis (usable extra budget
buys correctness) or as an interaction-efficiency axis (budget is spent,
correctness is not bought)? Round 1 answers the first half of the thesis
chain with a frozen policy, before any RL comparison.

Falsifiable sub-questions:

1. Is the accuracy curve over `max_tool_calls` {0,1,2,4} non-flat?
2. Do trajectories actually use the larger budgets (call-count
   distribution, termination reasons), or are b2/b4 structurally
   non-binding?
3. For the same question, does the same correctness occur at different
   call counts (correct trajectories at different costs; failure-then-
   recovery present)?

## Design

Frozen policy `artifacts/models/qwen3_1_7b_sft_dp_v1_merged` (the SFT arm of
the dual-loop eval). No adapter, no training, no weight change.

Data: `data/manifests/diagnostic_r1.json` — pilot 20 rows (4 difficulty
strata x 5) and formal 100 rows (4 x 25). Built from `frozen_eval` minus the
spent 200-task eval set; zero normalized-text overlap against train /
rl_dev / sft_dev; deterministic rebuild (parquet sha pinned in the
manifest). The diagnostic set is not the training pool and was not selected
by model performance.

Arms (single variable):

| arm | config | max_tool_calls | everything else |
|---|---|---|---|
| b0 | configs/agent/diagnostic_b0.yaml | 0 | steps 6 / python 12.0 s / obs 8000 |
| b1 | configs/agent/diagnostic_b1.yaml | 1 | same |
| b2 | configs/agent/diagnostic_b2.yaml | 2 | same |
| b4 | configs/agent/diagnostic_b4.yaml | 4 | same |

Sampling: per task per arm `--group-size 4` at `--temperature 0.7`,
`--max-new-tokens 1024` (per generation, not a per-task token budget). The
same `--selection-seed 20261014` and the same data file for all four arms;
the runner writes each arm's selected pool to `task_pool.jsonl`, and the
launcher verifies the four pools are identical (the same-question pairing
requirement). The seed controls task selection only — it is not generation
randomness.

Parser protocol: `--parser-tolerance unclosed_think,bare_final_scalar` in
all four arms — the tolerant setting the training environment already uses
(`verl_environment.py`), so evaluation and training share one action
protocol. Recorded in each run's manifest. The runner default is strict
parsing; strict and tolerant are never mixed inside one comparison.

Reward config: `configs/reward/r0.yaml`. Reward is *recorded* only in this
round; cost terms are recomputed offline from trajectory `usage`.

## Pilot gates (20 tasks; pass/fail before the formal 100)

1. All four arms complete 80 episodes each (20 x 4).
2. The four `task_pool.jsonl` files match each other and the precomputed
   expected selection exactly.
3. Termination-reason shares are inspected: the step cap (6), the token cap
   (1024 per generation) and the python-time cap (12 s) are re-examined
   before the formal run if `max_steps` / truncation dominates any arm
   ("以上数值沿用现有默认配置，是否足够，要根据试运行中的截断与终止原因
   判断").
4. Tool error codes are spot-checked so a dead sandbox is not mistaken for
   policy behaviour.

Gate failure -> fix, rerun the pilot. Pilot trajectories are never used as
formal evidence.

## Primary readings (pre-registered)

Read b1/b2/b4 as the informative contrast; b0 is the zero-call condition of
the same agent protocol, NOT a direct-answer baseline:

* accuracy rises with cap and call counts stay at or below cap -> budget has
  task value; "multi-step budget allocation" is a real difficulty axis;
* calls rise with cap, accuracy flat -> inefficient interaction (the
  hypothesis's low-efficiency half, visible end to end);
* curve flat and call counts pinned at one value across caps -> the budget
  is structurally non-binding for this policy; the multi-step-budget
  premise is NOT supported at this stage (a valid negative result — prior
  greedy eval data showed 0/400 episodes with >= 2 executed calls).

Also reported: termination-reason shares, mean generated tokens,
invalid-action counts; aggregated per question, never per trajectory.

## Tables (analyzer: scripts/analysis/analyze_budget_diagnostics.py)

* Table A (budget effect): cap x {accuracy, mean calls, share of 0/1/2/3/4
  calls, termination-reason shares, mean generated tokens}. Tool-call rate
  alone is insufficient by design: prior data has ~100% call rate yet
  exactly one call per task.
* Table B (within-question room): task_id x {samples correct n/4,
  call-count range among correct samples, among wrong samples,
  failure-then-recovery presence}. Supports "same correctness, different
  cost" claims.
* Table C (reward visibility): per-question {answer-reward variance, total
  reward variance, call-cost variance, all-equal flag,
  correct-but-different-cost flag}. Online rewards are R0; cost terms are
  recomputed offline and labelled 重算诊断值 (recomputed diagnostic values)
  — never presented as trainer-side advantages. Real advantages require a
  trainer-side export.

Statistics aggregate by question; four same-question samples are never
treated as four independent questions.

## Explicit non-claims

* Not a direct-answer baseline comparison: b0 keeps the agent protocol and
  tool schemas. `run_model_eval.py` (direct answering) is separate work and
  cannot substitute for tool-interaction experiments.
* Generation seeds are not yet wired into `transformers_client.py`: pilot
  sampling is exploratory. Independent generation seeding + unique
  trajectory identity + input hashes land before any formal, reviewable
  comparison; four repeats at temperature 0.7 are stochastic draws, not
  four benchmark items.
* Round 1 alone cannot show that the RL signal ignores trajectory
  differences — that is the reward-audit / RL-contrast stage (offline
  advantage recomputation must stay labelled, and trainer-side advantages
  need instrumenting).

## Next-step decision tree (execution plan, section 7)

1. Always exactly one call at every cap -> inspect cold-start data and
   exploration space first; do not claim multi-step-budget difficulty.
2. Correct trajectories at different costs with identical rewards -> launch
   the small-scale contrast: correctness GRPO vs fixed-cost-penalty GRPO.
3. Frequent uncorrectable tool errors -> separate tool competence, feedback
   delivery, and post-feedback policy.
4. Many all-equal-reward groups -> analyze success sparsity / reward
   resolution.
5. A simple cost penalty already works -> only then test whether dual
   advantages add anything.

## Version discipline

Pin `6f580bd` (origin/main): configs, data manifest, runner and model path
all verified against it. The user's Mac worktree `065c25b` is not on origin
and is not used here; configurations and results from the two versions are
never mixed. Round-1 files in this working tree are uncommitted by user
instruction (2026-10-14).
