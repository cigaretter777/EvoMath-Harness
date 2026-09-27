# sympy demo yield gate — 2026-09-27

Preregistered in the `sft_dp_v2` plan, Phase 0: **yield ≥ 300 verifier-correct
sympy demonstrations** from the `dp_v1` pool, denominator reported alongside
every count, and — if the bar is not met — stop there, write the negative
result up, and **do not open the GPU**.

**Verdict: FAIL.** The most favorable projection over the whole pool is
**228.2** against a bar of 300. The GPU stage is not opened.

Every number below is produced by committed scripts (`scripts/data/sympy_pool_census.py`,
`scripts/data/sympy_pool_gate.py`, committed as `8a36fe4`) and the raw artifacts
are archived under `artifacts/sympy_yield_gate/` with the sha256s recorded in §7.

## 1. Why the gate exists

The three-arm evaluation cannot say anything interesting about tool use while
`tool_call_count` is identically zero. The chain that produces that zero is:

1. `data/manifests/sft_dp_v1_10k.source-input.json` records `mode = direct`, and
   `scripts/data/build_source_sft.py:87` passes `ToolRegistry([])` in that mode.
   Re-counted directly on `data/processed/sft_dp_v1/train.parquet` (2649 rows,
   sha256 `c0d8945d1832faccc4b104c00e9bff4465d2a1d3593ee95ff7fc29e6cb11da43`):
   all `2649/2649` system prompts end with `Tool schemas: []`, and `0/2649`
   assistant turns contain a `<tool_call>`.
2. At agent time `src/adaptive_math/agent/prompts.py:13` renders the *real* registry, which is an
   out-of-distribution input for the policy. It does the only thing it was
   trained to do: emit `<final>` immediately.
3. GRPO cannot repair this on its own. All four samples within a group call
   tools zero times, so the `tool_weight` term is a constant within the group,
   its within-group advantage is zero, and the gradient through it is
   identically zero. Exploration is deadlocked.

So the fix has to be on the data side: put verifier-correct sympy demonstrations
into the SFT data. This gate asks whether the pool can supply them at all,
before any GPU hour is spent on the training run that would consume them.

## 2. The criterion, verbatim from the preregistration

> 产出 **≥300** 条 sympy 记录,且每题的工具输出经 `verify_answer` 与 gold 一致;
> 同时报告产出率分母(扫描了多少题),不许只报绝对数。
> **不达标就停在这里**,把产出率与失败原因写成结论,不开 GPU。

## 3. Result

Two stages, because the compiler and the verifier fail for different reasons and
conflating them would hide where the yield is lost.

### Stage 1 — census: can a candidate be compiled at all?

All 10000 pool records, deterministic, no execution (`pool-census.json`):

| candidates produced | records | share |
|---|---|---|
| 0 | 8574 | 85.7% |
| 1 | 837 | 8.4% |
| 2 | 256 | 2.6% |
| 3 | 126 | 1.3% |
| 4 | 207 | 2.1% |
| **≥ 1 (candidate-bearing)** | **1426** | **14.26%** |

`ceiling_over_pool = 0.1426` — the census names this ceiling itself, because no
amount of downstream verification can exceed it.

### Stage 2 — gate: do those candidates survive execution and verification?

A 200-record sample stratified across the whole pool (indices 1–9943), drawn only
from the 1426 candidate-bearing records. Each candidate is executed by the
production `SympyTool` in rank order and judged by the production `verify_answer`
against the hidden reference (`pool-gate.json`):

| | count | denominator |
|---|---|---|
| sample drawn from candidate-bearing pool | 200 | 1426 |
| accepted (tool ran **and** verifier agreed) | **32** | 200 (accept rate 0.16) |
| projected over the whole pool | **228.2** | `32/200 × 1426` |
| **preregistered bar** | **300** | — |

The projection is the *most favorable* reading available: it assumes the whole
10000-record pool is compiled and run, at the sampled accept rate.

### Where the execution attempts went (320 executions over 200 problems)

| verdict | count | share |
|---|---|---|
| `incorrect` (ran fine, wrong value) | 279 | 87.2% |
| `correct` | 32 | 10.0% |
| `invalid_prediction` | 8 | 2.5% |
| `internal_error` | 1 | 0.3% |

| outcome | count |
|---|---|
| `no_candidate_verified` (every candidate failed) | 162 / 200 |
| `tool_timeout` | 8 |
| `quarantine_unverifiable_reference` | 6 |
| `tool_execution_error` | 2 |

## 4. Where the yield is lost

The census failure reasons over all 10000 records (a record may carry more than
one reason):

| reason | count | what it means |
|---|---|---|
| `unconvertible_span` | 4706 | the problem's math span is outside the compiler's LaTeX→sympy grammar |
| `unbuildable_numeric` | 1975 | a span compiled, but it is not the quantity the problem asks for |
| `no_math_span` | 1431 | no math span identified in the problem text |
| `unbuildable_solve` | 426 | solve operation could not be built |
| `unbuildable_diff` / `integrate` / `simplify` / `factor` / `expand` | 21 / 7 / 4 / 3 / 1 | likewise |

**The binding constraint is the compiler, not the verifier.** The single largest
bucket is grammar coverage (`unconvertible_span`, 47% of the pool), and the
second is span selection — picking a piece of the problem text that compiles
cleanly but is not the answer (`unbuildable_numeric`, 20%). Raising yield means
widening the grammar and improving span selection; verification is downstream of
both and is not what is failing.

Cost: `elapsed_seconds = 2639.7` (44 minutes) of CPU, zero GPU. That is the
entire price of the veto.

## 5. What the 32 accepted records actually are

The gate counts them as passes because the tool really ran and the verifier
really agreed. That is the preregistered criterion and it is met for these 32.
But as *SFT demonstrations of tool use*, they are not uniform, and the
distinction matters for deciding whether to widen the grammar or abandon the
route. Operation mix: `numeric` 27, `solve` 3, `factor` 1, `simplify` 1. Rank at
which the accepted candidate was found: rank 0 → 22, rank 1 → 3, rank 2 → 5,
rank 3 → 2 (so the top-ranked candidate is right about 69% of the time).

The ones with real tool value look like this:

```
rank0 idx 4923   numeric   sqrt(4+sqrt(7))-sqrt(4-sqrt(7))                          ref=\sqrt{2}
rank0 idx 3986   factor    x^4+2021*x^2+2020*x+2021                                 ref=(x^{2}+x+1)(x^{2}-x+2021)
rank0 idx 8882   solve     (((2)/(x+8))+((5)/(x+9)))-(((3)/(x+15))+((4)/(x+6)))     ref=6,-\frac{33}{4}
rank0 idx 1756   numeric   (((7)/(3)))^999*sqrt(((3^1998+15^1998)/(7^1998+35^1998))) ref=1
```

(Verbatim from `pool-gate.accepted.jsonl`, so every line is greppable.)

The ones that are not: the full list is in
`pool-gate.accepted.jsonl`, and applying a mechanical test — *does the
expression contain any binary arithmetic operator at all?* — flags **13 of the
32** as bare values lifted verbatim out of the problem statement and handed to
the tool unchanged:

```
sqrt(2), 1/6, 3/2, 7, 64, 12/5, 9/4, 1/3, 62/3, sqrt(2)/3, 3/7, 3/20, 1/2
```

For these, the tool call carries no information the model did not already have:
input and output are the same number. Learning to emit them teaches "copy a
number out of the question into a tool call", not "use a tool to compute
something".

Two caveats on that test, stated so the count is not over-read. It is a
mechanical heuristic, not a judgement: `10^-3` is misclassified as non-trivial
by the operator rule, and `3*sqrt(2)` / `2+2*sqrt(6)` are counted as non-trivial
even though their source is plausibly an answer option copied verbatim. The
honest summary is therefore a range, not a point: **roughly 13–19 of the 32 are
substantive computations**, and the rest are transcriptions. Projected over the
pool at the same accept rate, that is on the order of **~136 substantive
demonstrations** — under half the preregistered bar, and fewer still once the
same proportion is removed from any widened pool.

## 6. What this establishes, and what it does not

Established:

- The `dp_v1` pool, as it stands, cannot supply 300 verifier-correct sympy
  demonstrations. The best available projection is 228.2, and the substantive
  subset of those is roughly 136.
- The failure is concentrated in compilation, not verification.
- Per the preregistration, the `sft_dp_v2` route stops here. **No GPU is opened
  for it**, and no SFT run is launched on a dataset that does not exist.

Not established — these are open, not settled:

- That the route is impossible. A wider grammar could raise the 14.26% ceiling,
  and better span selection could raise the 16% accept rate; the gate measured
  the pool as it is, with the compiler as it is.
- That the 32 accepted records are useless. They are protocol-valid, executable
  and verifier-correct; a smaller-scale demonstration set is not excluded by
  this gate, only the preregistered 300-record bar is missed.
- Anything about the three-arm evaluation itself. This gate ran in front of it
  and did not touch it.

One consequence carries forward and should be read together with the three-arm
results when they exist: the SFT arm has seen no sympy demonstrations, so a near
-zero `tool-call rate` and a flat mean-trajectory-steps number across all three
arms is the **expected** outcome, not evidence of a broken evaluation runner.

## 7. Reproducing

```
uv run python scripts/data/sympy_pool_census.py --limit 10000 --out <census-dir>
uv run python scripts/data/sympy_pool_gate.py --sample 200 --census <census-dir> --out <gate-dir>
```

The gate is resumable: accepted/rejected indices append to `pool-gate.done.jsonl`
and are skipped on re-run, because a full pass costs hours of CPU.

| artifact | sha256 |
|---|---|
| `artifacts/sympy_yield_gate/census/pool-census.json` | `35b9bfb0…f90d4a` |
| `artifacts/sympy_yield_gate/census/pool-census.rows.jsonl` | `d08d87fb…060b58` |
| `artifacts/sympy_yield_gate/gate200/pool-gate.json` | `a0de9500…f4d4ed` |
| `artifacts/sympy_yield_gate/gate200/pool-gate.accepted.jsonl` | `7b7c68e5…a40558` |
| `artifacts/sympy_yield_gate/gate200/pool-gate.done.jsonl` | `28ed861a…a9d858` |
| `artifacts/sympy_yield_gate/gate200/gate.log` | `a63f77e2…daf1b5` |

Pool pinning: `raw_records_sha256 = 969d778943acdc8b5845c02bb1b1f409af9ff22a18270cbffcf8e4cd92f46fa3`,
and the census confirms `matches_dp_v1_raw_sha256 = true` against the `dp_v1`
manifest (`dp_v1_canonicalized = 2944`).

Provenance note: the two scripts were executed while still outside the
repository and were committed afterwards as `8a36fe4`, so the scripts are not
themselves the preregistration — the criterion was written into the plan before
the run, but the code that applies it landed after. Their only change between
execution and commit is the removal of unused `# noqa: E402` directives and one
import-block reorder; the diff was reviewed line by line and no statement
changed.
