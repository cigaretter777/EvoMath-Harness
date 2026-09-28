# Six-arm metric table on the frozen Omni-MATH-200 (strict protocol)

**Date:** 2026-09-29
**Scope:** every arm that has already been run and stored, re-collated under one
set of definitions. No GPU, no regeneration, no new generations.
**Script:** `scripts/analysis/arm_metrics.py`
(tests: `tests/unit/analysis/test_arm_metrics.py`)
**Output:** `artifacts/results/mvp-20260929/arm-metrics.json`,
sha256 `f2290fae2c2eaff2a69bac7e8765fa639601c4da0a4ac6525d787c2ac091f871`
(re-running the script reproduces this file byte for byte; an earlier revision of
this doc quoted a hash from before the composition fix in §2)

```bash
python scripts/analysis/arm_metrics.py --out artifacts/results/mvp-20260929/arm-metrics.json
```

## 1. The table

| arm | mode | n | correct | accuracy | final action | verifier-valid | tool calls | calls/task | steps mean | steps p95 | invalid/task |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_direct` | direct | 200 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0.000 | 1.00 | 1 | n/a |
| `base_agent` | agent | 200 | 13 | 6.5% | 13.5% | 11.0% | 7 | 0.035 | 5.76 | 6 | 5.59 |
| `rule_strategy` | agent | 200 | 10 | 5.0% | 8.0% | 6.5% | 16 | 0.080 | 3.98 | 6 | 3.44 |
| `sft_direct` | direct | 200 | 27 | 13.5% | 66.5% | 66.5% | 0 | 0.000 | 1.00 | 1 | n/a |
| `r0_grpo_direct` | direct | 200 | 22 | 11.0% | 74.5% | 72.5% | 0 | 0.000 | 1.00 | 1 | n/a |
| `r2_grpo_direct` | direct | 200 | 25 | 12.5% | 66.0% | 65.5% | 0 | 0.000 | 1.00 | 1 | n/a |

Column definitions, because two of them are easy to conflate:

* **accuracy** -- `verifier_status == correct` / n.
* **final action** -- a legal final action came out at all. Agent arms: the loop
  closed with a final answer, which it only records if the strict parser
  accepted it. Direct arms: the extractor returned a value.
* **verifier-valid** -- the stored `valid_answer_rate`: the answer was judged
  correct or incorrect, so the verifier had something to compare. Rows that
  extracted cleanly and then hit a verifier-side status (reference invalid,
  timeout, internal error) count in *final action* and not here; the two columns
  differ by 4 rows on `r0` and 1 row on `r2`.
* **tool calls / calls per task** -- summed from the trajectories' recorded
  `usage.tool_calls`; the strict envelope's count, see §5.
* **steps mean / p95** -- from `usage.steps`; nearest-rank percentile, no
  interpolation.
* **invalid/task** -- `usage.invalid_actions` per task. `n/a` for direct arms
  because a single-pass generation has no action channel to violate: absent
  measurement, not a measured zero.

**Two modes are not directly comparable.** The three `direct` arms are
single-pass generations, so their one step and zero tool calls are true by
construction. Only accuracy and the final-action column are comparable across
modes.

**`rule_strategy` is composed.** The fixed rule routed 76 tasks to a direct
answer and reused those stored `base_direct` rows verbatim; the other 124 were
rolled out. Its step count is therefore a mixture (1.00 for the reused rows,
5.81 for the rolled-out ones) and its accuracy is 10/124 on the rolled-out part,
0/76 on the reused part.

## 2. How these numbers are held to the runs

`arm_metrics.py` fails rather than prints when a computed count disagrees with
the count the arm's own `summary.json` recorded. On this run all checks pass; the
counts checked per arm are tasks, correct, verifier-valid, rolled-out, reused,
tool calls, invalid actions, generated tokens, and mean output tokens. Sources:

| arm | authority |
|---|---|
| `base_direct`, `sft_direct` | `artifacts/eval/thesis_e0_base_direct_b1/summary.json` |
| `base_agent` | `artifacts/rollout_health/thesis_e0_base_tool/summary.json` |
| `rule_strategy` | `artifacts/rollout_health/thesis_e0_rule_strategy/summary.json` + `routing.jsonl` |
| `r0_grpo_direct` | `artifacts/eval/r0_omnimath_200/summary.json` |
| `r2_grpo_direct` | `artifacts/eval/r2_omnimath_200/summary.json` |

The composition check earned its place: the reused pool for `rule_strategy` holds
a `base_direct` row for all 200 tasks, but the arm only reused the 76 its routing
file marks `direct`. Counting the whole pool gives a 400-row arm with a 3.1%
accuracy -- a plausible-looking wrong table. The script now reads the routing
file and refuses to run if a routed task has no row in the pool.

## 3. Paired comparisons (stored, not recomputed here)

Deltas are task-paired against `base_direct` (0/200), from
`artifacts/results/campaign-20260926/summary.json` (`pairings`), written by
`scripts/campaign-20260926/analyze_e0_baselines.py`:

| comparison | delta | 95% CI | discordant (up / down) | McNemar p |
|---|---:|---|---:|---:|
| `base_agent` vs `base_direct` | +6.5 pt | 0.035 – 0.100 | 13 / 0 | 2.4e-4 |
| `rule_strategy` vs `base_direct` | +5.0 pt | 0.020 – 0.080 | 10 / 0 | 2.0e-3 |
| `rule_strategy` vs `base_agent` | −1.5 pt | −0.050 – 0.020 | 5 / 8 | 0.58 |
| `sft_direct` vs `base_direct` | +13.5 pt | 0.090 – 0.185 | 27 / 0 | 1.5e-8 |

What the pairs support: multi-turn agent loops beat the single-pass base on this
frozen set, and SFT beats it by twice as much. What they do **not** support:
attributing the agent arm's gain to *tool execution*. The strict envelope
executed 7 tool calls in that arm; the gain came from the multi-turn loop and its
format. That distinction is the point of §5 and of
`docs/results/protocol-gap-2026-09-29.md`.

## 4. Reproduction statement, corrected

The claim "reproduction over a two-week gap reached 200/200 byte-identical
outputs" is not what the artifacts show, and it is worth stating precisely
because it is easy to state wrongly:

* **Same generation configuration, two days apart:** 200/200 outputs
  byte-identical. The 09-26 campaign regenerated the SFT arm in slot 1 and
  compared it with the 09-24 E2 champion arm:
  `artifacts/results/campaign-20260926/summary.json` → `identity_check`
  `{n_shared: 200, byte_identical: 200, pass: true}`. This is the pipeline
  cross-check, and it passed.
* **Across generation configurations, it is not invariant.** The 09-24 four-way
  campaign compared batch-8 against batch-1 greedy decoding on the same arm:
  3/200 outputs byte-identical, verifier labels agreeing on 130/200, 15 tasks
  moving (`docs/results/campaign-2026-09-19/fourway/evaluation_report.md`,
  §"raw outputs byte-identical 3 (1.5%)").

So the defensible sentence is: *"re-running the same arm under the same
generation configuration reproduced 200/200 outputs byte-for-byte; across batch
configurations only 3/200 outputs and 130/200 labels agree, which is why every
number in this table comes from a single frozen configuration."*

## 5. Caveats that change how the table reads

1. **The tool-call column is a property of the protocol, not of the model.** The
   strict envelope executes a tool call only when the turn's text is exactly one
   action block with a valid payload. Re-parsing the stored turns shows 422 tool
   attempts on `base_agent`, 393 of them one mechanical repair from executing,
   and 193/200 tasks with at least one such attempt
   (`docs/results/protocol-gap-2026-09-29.md`). "Barely used tools" is the wrong
   reading of the 7; "the envelope discarded the reach" is the right one.
2. **The SFT arm has never seen a sympy demonstration.** The sympy-demo yield
   gate failed (32/200 usable, projected 228 < 300 required), so all three arms
   are being asked to use a tool they were not trained on:
   `docs/results/sympy-demo-yield-gate.md`. Low tool-call rates and flat
   tool-related differences are the expected consequence, and must be stated
   before the tool columns are compared.
3. **The `r2` reward is not the one the plan names.** The plan calls it
   cost-aware GRPO; `configs/reward/r2.yaml` is a tool-weighted shaping reward
   (tool 0.15, python 0.10, invalid 0.10) and the repo's own label is
   "tool-weighted". Both arms are single direct-evaluation passes of a
   short RL run (r0: 50 steps, r2: 12 steps), so neither is a converged method
   comparison -- they are a smoke signal, and the table should present them as
   such.
4. **Neither GRPO arm is significantly above SFT.** 11.0% and 12.5% against
   13.5%: the differences are a handful of tasks and no paired test on this
   table separates them. The honest headline is that RL did not yet beat SFT on
   this set, not that it "landed between".

## 6. Protocol sensitivity: all six arms under three extraction rulers

**Artifacts:** `artifacts/results/mvp-20260929/tolerance-rescore.json`
(sha256 `891ffb80befae03294380ce60b36779951c50e2baeb94af4799afdd82872b002`,
script `scripts/analysis/tolerance_rescore.py`, tests
`tests/unit/analysis/test_tolerance_rescore.py`) and
`artifacts/results/mvp-20260929/paired-protocols.json`
(sha256 `7f2e8ba2e5d51b99a03b340527777f6a4b5ed2c9b08c984efea39314a84a5481`,
script `scripts/analysis/paired_protocols.py`, tests
`tests/unit/analysis/test_paired_protocols.py`). CPU only: every number is read
off stored generations and trajectories; no model was called and nothing was
regenerated.

The question is how much of each arm's accuracy is a property of the model and
how much of the *ruler*. Three escalating readings of the same stored text,
applied identically to all six arms, each rung adding only where the one below
found nothing:

* **strict** -- the frozen protocol. Re-run here and required to reproduce every
  recorded status; a disagreement is fatal rather than reconciled, because then
  this harness is not the one that scored the file. For a trajectory the strict
  rung is the answer the loop closed with, and the runtime's own parser is
  applied to the last model turn and must agree with it -- it did, on all 324
  trajectories; had it not (a tolerant parser in the run, say), the script would
  have stopped instead of reporting a number measured against a different ruler.
* **coerce** -- mechanical: read the literal digits of a numeric `answer` inside
  a single well-formed `<final>` block. It adds no information and searches
  nothing, and it preserves the literal (`1.50` stays `1.50`, `1e3` stays
  `1e3`) instead of re-rendering a float.
* **lenient** -- the repo's own `extract_solution_answer` (last boxed group in
  the tail, prose anchor, whole-text math span). This one *interprets*, so it is
  an upper bound, not a score the arm would have earned.

### 6.1 The rulers

| arm | mode | n | strict | +coerce | +lenient | coerced answers | verifier-valid (recorded) |
|---|---|---:|---:|---:|---:|---:|---:|
| `base_direct` | direct | 200 | 0 (0.0%) | 0 (0.0%) | 15 (7.5%) | 0 | 0 |
| `sft_direct` | direct | 200 | 27 (13.5%) | 27 (13.5%) | 28 (14.0%) | 0 | 133 |
| `r0_grpo_direct` | direct | 200 | 22 (11.0%) | 22 (11.0%) | 22 (11.0%) | 0 | 145 |
| `r2_grpo_direct` | direct | 200 | 25 (12.5%) | 25 (12.5%) | 25 (12.5%) | 0 | 131 |
| `base_agent` | agent | 200 | 13 (6.5%) | **35 (17.5%)** | 40 (20.0%) | 22 | 22 |
| `rule_strategy` | agent | 124 | 10 (8.1%) | **24 (19.4%)** | 31 (25.0%) | 14 | 13 |

The "verifier-valid" column is the same notion as in §1: rows whose recorded
verdict was correct *or* incorrect. On `base_agent` it coincides with the
coerced count (22) -- both are 13 correct + 9 incorrect -- which is a coincidence
of this table, not a derivation.

### 6.2 The earlier hypothesis was wrong, and the data says why

§6 of the protocol-gap note predicted that the scalar-answer defect would also
be found in the direct arms. It is not there. It is an agent-mode artifact of
the prompt's JSON contract, and that note now carries the correction:

* **The direct arms are not parser-limited.** `base_direct` emits no `<final>`
  tag at all in 200 generations (15 contain a `<tool_call>` tag instead, 7
  contain a numeric answer inside some wrapper, none inside a well-formed final
  block). `sft_direct`, `r0` and `r2` emit `<final>` in 133/148/132 generations;
  the extractor reads a value from 133/149/132 of them, of which 131/148/131 come
  from a final tag and the rest from math delimiters. **No generation in any of
  the three arms writes a numeric `answer`.** Coercion therefore moves all four
  direct arms by exactly **0 rows**, and the strict column is for them a
  mathematics number: SFT satisfies the wire contract 133 times and converts
  that into 27 correct answers.
* **The agent arms are the opposite case.** 28 of `base_agent`'s 200 last turns
  end in a well-formed, fully closed `<final>{"answer": <number>}</final>`; the
  runtime parser refuses all 28 with `invalid_schema` (27 int, 1 float) because
  the action schema types `answer` as a string. **22 of those 28 verify correct**
  against the frozen references, which is where 13/200 becomes 35/200.
  `rule_strategy`: 19 such turns, 14 correct, 10/124 becomes 24/124.
* **The lenient gains are interpretive, and they favour the base model.** 14 of
  `base_direct`'s 15 gains contain no math delimiter at all -- the value was read
  by the prose anchor ("So the answer is 71") -- which is why the rung is an
  upper bound. The agent arms gain 5 more rows on top of coercion.

### 6.3 Paired comparisons under each ruler

From `paired-protocols.json`, the same paired test the campaigns report (exact
McNemar, paired bootstrap with the repo's seed), run once per ruler against
`base_direct`. `base_agent` and `sft_direct` cover all 200 tasks; `rule_strategy`
covered 124 and is paired on those (the 76 excluded tasks are reported in the
artifact: base scored 0 of them correct at strict).

| arm | ruler | delta | 95% CI | up/down | McNemar p |
|---|---|---:|---|---:|---:|
| `sft_direct` | strict | +13.5 pt | 0.090 – 0.185 | 27 / 0 | 1.5e-8 |
| `sft_direct` | lenient | +6.5 pt | 0.020 – 0.110 | 18 / 5 | 0.011 |
| `r0_grpo_direct` | strict | +11.0 pt | 0.070 – 0.155 | 22 / 0 | 4.8e-7 |
| `r0_grpo_direct` | lenient | +3.5 pt | −0.010 – 0.085 | 15 / 8 | **0.21** |
| `r2_grpo_direct` | strict | +12.5 pt | 0.080 – 0.175 | 25 / 0 | 6.0e-8 |
| `r2_grpo_direct` | lenient | +5.0 pt | 0.005 – 0.095 | 16 / 6 | 0.053 |
| `base_agent` | strict | +6.5 pt | 0.035 – 0.100 | 13 / 0 | 2.4e-4 |
| `base_agent` | coerce | **+17.5 pt** | 0.125 – 0.230 | 35 / 0 | **5.8e-11** |
| `base_agent` | lenient | +12.5 pt | 0.075 – 0.175 | 27 / 2 | 1.6e-6 |
| `rule_strategy` | strict | +8.1 pt | 0.040 – 0.129 | 10 / 0 | 2.0e-3 |
| `rule_strategy` | coerce | +19.4 pt | 0.129 – 0.266 | 24 / 0 | 1.2e-7 |
| `rule_strategy` | lenient | +17.7 pt | 0.113 – 0.250 | 22 / 0 | 4.8e-7 |

(The `coerce` rows omitted where they equal `strict`, which is every direct arm.)

What survives every ruler: **SFT over base, and the agent runtime over base.**
What does not: **`r0` over base**, which is significant under the frozen
protocol (p=4.8e-7) and indistinguishable from it under the interpretive upper
bound (p=0.21, and 8 tasks regress). `r2` sits on the boundary (p=0.053). That
is the same conclusion §5.4 draws from the strict numbers, now with a second,
independent ruler agreeing with it.

What the agent arms' two numbers mean together: 6.5% is the frozen protocol's
answer, and it is *also* mostly a schema-conformance number. 17.5% is what the
same trajectories support when a JSON number is accepted where the contract
demands a string -- a repair that cannot invent a value. Both are true; quoting
only the first understates the method and quoting only the second misstates the
protocol. The defensible sentence carries both.

### 6.4 Provenance disclosure

`tolerance_rescore.py` re-verifies every stored answer with the current
`verifier/service.py`. `r0_grpo_direct` (4 rows) and `r2_grpo_direct` (1 row)
have verdicts the current verifier does not reproduce -- all of them moves
between two *non-correct* statuses (`incorrect` where the record says
`invalid_prediction`), e.g. `omni_math:0986e00a217544428e57`, so no accuracy
moves. The cause is a verifier drift the arms' own manifests record: both were
scored at git `b4ca1604` with `verifier_source_sha256 b690cc7d…`, three commits
before the `verifier/service.py` at HEAD (`3cfc5e61…`); the extractor
(`029570bd…`) and prompts (`9db23ccf…`) are byte-identical everywhere. Zero rows
anywhere cross the correct boundary -- had any, the script would have refused to
score that arm rather than report it. This is disclosed here because the
"reproduce the recorded statuses" check is what makes the strict column
trustworthy, and it is weaker than it sounds on those two arms.
