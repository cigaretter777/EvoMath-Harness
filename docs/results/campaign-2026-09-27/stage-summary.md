# Stage summary — the two adapter agent arms (campaign 2026-09-27)

**Date:** 2026-09-29
**Scope:** the §8b amendment of
[`preregistration.md`](preregistration.md) — two new agent arms over the frozen
Omni-MATH-200, rolled out on one RTX 4090. No training, no config change, no
arm added or dropped after the fact.
**Runner:** `scripts/campaign-20260927/run_bc_arms.sh` drives the frozen
`scripts/campaign-20260926/rule_baseline.py` through
`run_arm_with_adapter.py`; the runner's blob hash is asserted before launch
(`e591c886…`, P5) and was byte-identical throughout.
**Wall clock:** 2026-09-28T18:18:04Z → 19:50:18Z = **1 h 32 m** for both arms
(B 51 m, C 41 m), against the 2–3 GPU h the fork was scoped at.

| arm | weights | what it is |
|---|---|---|
| **B → `sft_agent`** | base snapshot `70d244cc` + sft adapter `2868f83e…` | `sft_direct`'s weights, loaded `sft_direct`'s way |
| **C → `r0_agent`** | merged SFT + r0 adapter `21a3f4aa…` | r0's weights, which had no agent row before |

## 1. Identity evidence (P2, P4, P5)

* **Pre-flight, CPU, per shard × per arm: 6/6 `GATE OK — 9 identity fields`,
  exit 0.** Re-emitting the pre-registered shard-0 sidecars at this HEAD
  reproduced them byte for byte (`e69ad876…`, `8193c993…`); shards 1 and 2 were
  emitted and gated the same way before any weights moved.
* **Post-run, against the authoritative artifact: 3/3 + 3/3 `GATE OK` on each
  shard's real `manifest.json`** (the runner writes it; the sidecar is only the
  pre-flight).
* **P4, the loaded weights (§8b's first-turn rule):**

| comparison | result | alarm | reading |
|---|---|---|---|
| B vs stored A, first `model_output` of each shared task | **0/200 identical** | ≥ 50% | the sft adapter loaded |
| C vs B, same | **22/200 = 11.0%** | ≥ 50% | the r0 adapter loaded |

  Both are far from the alarm and sit at the §8b poles: 0/200 is exactly the
  base-vs-sft pole measured on the direct rows, and 11% is within the same
  order as the sft-vs-r0 pole (12/200). The check that produced them
  (`scripts/campaign-20260927/first_turn_divergence.py`) did not exist before
  this run; it was validated on three poles first (identical weights 200/200 →
  alarm; exactly 50.0% → alarm; 49.5% → pass; disjoint sets → refuses to
  evaluate).
* Nothing was retried: no shard died, no partial directory was set aside, and
  the sandbox watchdog logged **no outage** (single line: `sandbox ok (initial)`).

## 2. The eight-arm metric table

Produced by `scripts/analysis/arm_metrics.py` (the two new arms were registered
in it; every one of their counts cross-checks against the `summary.json` the run
itself wrote). New artifact:
`artifacts/results/campaign-2026-09-27/arm-metrics.json`, sha256 `a781206565b102e1…`.
The six stored arms reproduce their pinned rows exactly.

| arm | mode | n | correct | accuracy | final action | verifier-valid | tool calls | calls/task | steps mean | steps p95 | invalid/task |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_direct` | direct | 200 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0.000 | 1.00 | 1 | n/a |
| `base_agent` | agent | 200 | 13 | 6.5% | 13.5% | 11.0% | 7 | 0.035 | 5.76 | 6 | 5.59 |
| `rule_strategy` | agent | 200 | 10 | 5.0% | 8.0% | 6.5% | 16 | 0.080 | 3.98 | 6 | 3.44 |
| `sft_direct` | direct | 200 | 27 | 13.5% | 66.5% | 66.5% | 0 | 0.000 | 1.00 | 1 | n/a |
| `r0_grpo_direct` | direct | 200 | 22 | 11.0% | 74.5% | 72.5% | 0 | 0.000 | 1.00 | 1 | n/a |
| `r2_grpo_direct` | direct | 200 | 25 | 12.5% | 66.0% | 65.5% | 0 | 0.000 | 1.00 | 1 | n/a |
| **`sft_agent`** | agent | 200 | **28** | **14.0%** | **79.5%** | **78.0%** | **0** | 0.000 | 2.17 | 6 | 1.37 |
| **`r0_agent`** | agent | 200 | **27** | **13.5%** | **94.0%** | **93.0%** | **0** | 0.000 | 1.51 | 6 | 0.57 |

Two readings are visible before any test is run, and both are about the
envelope rather than the mathematics: the new arms close with a legal final
answer 79.5% / 94.0% of the time where `base_agent` managed 13.5%, and they use
**fewer** steps (2.17 / 1.51 vs 5.76) — they answer instead of looping.

## 3. The pre-committed tests

Paired, exact McNemar, two-sided α = 0.05, same 200 tasks, paired bootstrap CI
(seed 20260913, 10 000 resamples). Artifacts:
`paired-agent-arms.json` (`a9dcc730693825d2…`), `paired-sft-vs-r0-agent.json`
(`e1b8f673bf09b5f0…`). All comparisons are on the **strict** ruler unless named
otherwise (§5).

* **H1 — confirmed.** `sft_agent` 28/200 vs `base_agent` 13/200: **+7.5 pt,
  95% CI [+3.0, +12.5], 20 up / 5 down, p = 0.0041.** The headline holds.
  (`r0_agent` vs `base_agent` is the same story: +7.0 pt, p = 0.0125.)
* **H2 — half-confirmed, and the failing half is the interesting one.** The
  final-action half is confirmed: 159/200 (79.5%) vs `base_agent`'s 27/200
  (13.5%). The tool-call half is **not**: executed tool calls went from 7 to
  **0**, not above 7. The funnel explains it —
  `protocol-gap.json` (`399d2ebc73b84d42…`), arm names `base_tool`/`rule` as in
  the pinned artifact:

| arm | turns | executed tool | executed final | tool attempts | tasks w/ executable attempt |
|---|---:|---:|---:|---:|---:|
| `base_tool` (= A) | 1152 | 7 | 27 | 422 | 193/200 |
| `sft_agent` (B) | 433 | 0 | 159 | 13 | 1/200 |
| `r0_agent` (C) | 303 | 0 | 188 | 11 | 3/200 |

  SFT carried its format discipline into the loop — as *"close with a legal
  final answer"*, not as *"call the tools"*. Attempts collapsed alongside
  executions (422 → 13), so this is not an envelope failure on the tool side;
  the tool channel went silent.
* **H3 — null, as pre-committed.** `r0_agent` vs `sft_agent` strict: −0.5 pt,
  95% CI [−4.0, +3.0], 6 up / 7 down, **p = 1.0**. Reported as exploratory and
  as nothing.
* **H4 — placed on the ladder.** `sft_agent` under the three rulers: 28 / 28 /
  28 (coerce and lenient gain exactly 0). `base_agent` moves 13 → 35 → 40. So
  **at `lenient`, `base_agent` (40/200) beats `sft_agent` (28/200), −6.0 pt,
  p = 0.0227**, and at `coerce` the difference is not significant (35 vs 28,
  p = 0.248). Read under §5: the strict headline above stands as the headline,
  and the tolerant rows are re-scoring ceilings over recorded text, not scores
  the arms would have earned. The gap between the rulers is itself the finding
  it was in §5 — base's answers were in its transcripts, refused by the
  envelope.

## 4. What this campaign does not claim

§9 stands verbatim. Three additions from these arms:

* **Not** that the accuracy gain comes from tool execution — now the strongest
  form of the point: `sft_agent` executed **zero** tools and still doubled
  strict accuracy over `base_agent` (p = 0.0041). The measured gap is about the
  envelope and about answering at all, not about tools.
* **Not** that SFT learned to use sympy or any tool: the demonstration-yield
  gate failed (32/200), all three arms use tools they were never trained on,
  and B/C executed none.
* **Not** that the agent arms beat their direct rows: `sft_agent` 28 vs
  `sft_direct` 27 and `r0_agent` 27 vs `r0_grpo_direct` 22 are descriptive
  level-comparisons; no significance test for them was pre-committed, and none
  is reported.

## 5. Disclosures

* The agent arms' `summary.json` has no verifier-valid field (the same gap
  `base_agent` has). The value in the table comes from the re-scoring pass
  (`tolerance-rescore.json`, `6b6f591023fda52a…`), which reads the trajectories
  under the same rule: `status in {correct, incorrect}`. This is stated in the
  registry comment in `arm_metrics.py`.
* The re-scoring pass reproduces the pinned six-arm artifact **field for
  field, all six arms**, and adds two; the task-set sha gate (`1fc257f2…`) and
  the rung-nesting check both passed. The known verifier drift is unchanged and
  unchanged in size: r0 4 rows, r2 1 row, non-correct statuses only.
* Naming: the pinned `protocol-gap.json` calls the arms `base_tool` and `rule`;
  this campaign's artifacts call them `base_agent`/`rule_strategy` (metrics,
  paired, rescore) and `sft_agent`/`r0_agent` for the new ones. Same runs,
  different labels; the dirs are in §0 of the pre-registration.
* §8b's example argv is shard 0 of 3; each arm was run on all three shards so
  the pre-registered denominators (200, §4) hold.

## 6. Reproduce

```bash
# arms (GPU): see scripts/campaign-20260927/run_bc_arms.sh; pre-flight per
# shard, partial-dir-aside, bounded attempts, then the two post-run checks.
# everything below is CPU-only and re-reads the stored artifacts:
python scripts/analysis/tolerance_rescore.py --arm base_direct=… --arm … \
  --out artifacts/results/campaign-2026-09-27/tolerance-rescore.json
python scripts/analysis/arm_metrics.py --out artifacts/results/campaign-2026-09-27/arm-metrics.json
python scripts/analysis/paired_protocols.py --from-json …/tolerance-rescore.json \
  --base base_agent --against sft_agent --against r0_agent \
  --out artifacts/results/campaign-2026-09-27/paired-agent-arms.json
python scripts/analysis/paired_protocols.py --from-json …/tolerance-rescore.json \
  --base sft_agent --against r0_agent \
  --out artifacts/results/campaign-2026-09-27/paired-sft-vs-r0-agent.json
python scripts/analysis/protocol_gap.py --arm base_tool=… --arm rule=… \
  --arm sft_agent=… --arm r0_agent=… \
  --out artifacts/results/campaign-2026-09-27/protocol-gap.json
```
