# Pre-registration — thesis baseline campaign 2026-09-26 (E0/E1)

Committed **before** any GPU result exists. Decision rules live here; if a
result contradicts this document, the document wins.

Scope: the first GPU phase of the BC-GRPO thesis plan (2026-09-26): the
no-training baseline evaluation. RL arms (E2-E6) are later phases and are
pre-registered separately when they are designed.

Host: AutoDL, single RTX 4090 24GB. Checkout: branch
`fix/20260926-parser-tolerance-verifier-timeouts`, the commit recorded in each
arm's manifest. Sandbox: SandboxFusion over the SSH forward on
`127.0.0.1:8080`, probed before every tool slot.

## Pool

Frozen Omni-MATH 200: `artifacts/task_pools/rl_r0_200.jsonl` (the
`rl_r0_200` pool, seed 20260919, leakage-checked) = `data/processed/v1/frozen_eval.parquet`
restricted to the same 200 task_ids (task_ids_sha256 `1fc257f2…`). All stored
arms share these task_ids; every new arm here uses the same pool, so any pair
is task-aligned by construction.

## Arms

| id | arm | model | channel | decoding | status |
|---|---|---|---|---|---|
| E0a | base-direct | Qwen3-1.7B (70d244cc) | none (eval extractor path) | greedy, batch 1, cap 1024 | **new this run** |
| E0b | base+tool | same base | agent loop, both tools (sympy+python) | greedy, cap 1024, budget max_steps 6 / max_tool_calls 4 | **new this run** |
| E0c | fixed-rule | same base | agent loop, rule-routed single tool (or none) | greedy, cap 1024, same budget | **new this run** |
| E1 | SFT-direct | SFT dp_v1 adapter (2868f83e) | none (eval extractor path) | greedy, batch 1, cap 1024 | exists (E2 champion, 2026-09-24); regenerated here as the slot-1 sft arm (identity check below) |

Reward config for both agent arms: `configs/reward/r0.yaml` (correctness only;
no tool-cost, no invalid penalty). A no-training baseline must not bake in an
RL reward structure.

## Fixed-rule router (E0c) — v1 rules, first match wins

1. **sympy** if the problem text matches
   `\bsolve\b | \bequations?\b | \broots?\s+of\b | =\s*0 | [a-z]=\s*(?=[^$\s])`
   (the lookahead excludes the pool's answer blanks `n=$ \qquad`);
2. **python** if `problem_type ∈ {Combinatorics, Number Theory}` or the text
   matches `\bdigits?\b | \bsum\b | \bproduct\b | \bremainder\b | \bmod\b | \d{4,} | \^{ | !`;
3. **direct** otherwise.

The direct channel performs no rollout and reuses the E0a base-direct arm —
the fixed-rule strategy is literally "route, then answer directly or with the
one routed tool", so its direct branch *is* the base-direct arm.

The router is naive by design: it is a no-training baseline whose imperfections
are part of what is measured. The dry-run distribution over the 200 tasks,
computed before the GPU run and committed in the manifest
(`routing_distribution`), was direct 50 / python 93 / sympy 57.

## Endpoints, pre-specified

Per arm: `correct` count and `invalid_prediction` count (production verifier,
current extractor), plus for the two agent arms: tool_calls total, mean/median/
p95 tool calls per trajectory, over-budget rate (tool_calls > 4), invalid-action
count, mean/p95 generated tokens, mean/p95 rollout latency, termination mix.

Comparisons (each reported with its exact test and p-value):

- E0b vs E0a (paired McNemar exact + paired bootstrap, seed 20260913): does
  tool access alone move correctness?
- E0c vs E0a: does rule-routed tool access move correctness?
- E0c vs E0b: does restricting the tool set by rule help or hurt?
- E1 vs E0a: SFT's contribution (already measured at batch 1, but re-derived
  from the same run for a single consistent table).

## Internal consistency check (pre-committed)

Slot 1 regenerates both arms of the paired eval (base + sft, batch 1). The sft
arm's **raw outputs must be byte-identical** to
`artifacts/eval/harness_e2_champion_cap1024/sft_predictions.jsonl` (same model
identity, same adapter sha, same decoding, same commit-level code path). If
they differ, the campaign stops and reports a pipeline drift instead of
publishing baseline numbers. Labels may differ: the verifier/extractor were
fixed after 2026-09-24 (relabel audit 2026-09-25: 4 labels per arm moved
invalid→incorrect, zero correct counts changed).

## Addendum A (2026-09-26 ~13:00 CST, committed before the corrected arms ran)

**Pool misalignment found during analysis.** The direct arms (E0a/E1) evaluate
`frozen_eval.parquet` rows whose task_ids are `omni_math:<hash>`; the agent
arms (E0b/E0c) first ran against `artifacts/task_pools/rl_r0_200.jsonl` whose
task_ids are `openr1_math_220k:<hash>`. The two sets share **zero** task_ids
(and zero source_hashes and zero problem texts): they are different 200-task
sets, so the first agent-arm runs cannot be paired with the direct arms. The
"same pool" claim in this document was written without verifying task_ids and
is **withdrawn**.

**Correction.** E0b and E0c are re-run on `frozen_eval.parquet` rows selected
by the stored base-direct arm's task_ids (order preserved). Only the task
source changes; model identity, decoding, budget, tools and reward config are
unchanged. Consequences:

1. The first agent-arm runs are archived as `*_openr1_pool` and are **not**
   part of the paired campaign. They remain useful as training-pool
   diagnostics (the RL pool is drawn from openr1_math_220k).
2. The router's `problem_type` rule is **dormant on Omni-MATH**: the parquet's
   metadata column is empty (`{}` for all 200 rows). Only the text rules fire;
   the routing table below is still deterministic and pre-committed.
3. The dry-run routing distribution over the corrected set replaces the
   distribution quoted above and is recorded in each arm's manifest.

Everything else in this document is unchanged.

## What this campaign will NOT claim

- Anything about RL (no RL arm runs here).
- Any accuracy number from an arm whose identity hashes do not match this
  document.
- Any significance claim without the exact test and p-value shown.
- That the fixed-rule router is optimal or fair; it is one documented naive
  strategy, chosen before seeing any of its outcomes.
