# Pre-registration — job-seeking agentic-RL MVP campaign 2026-09-27

Committed **before** any GPU minute of this campaign. Decision rules live here;
if a result contradicts this document, the document wins. Every number already
in hand was produced on CPU by re-reading stored artifacts — §0 lists them with
their hashes so that nothing below rests on a re-run.

Scope: the three-arm agent-mode evaluation of the job-seeking MVP design
(`docs/superpowers/specs/2026-09-27-job-seeking-agentic-rl-mvp-design.md` §6.1).
RL training is **not** in scope; no arm here is trained by this campaign.

Host: AutoDL, single RTX 4090 24GB. Sandbox: SandboxFusion over the SSH forward
on `127.0.0.1:8080`, probed immediately before the first tool slot (§7).

## 0. What is already measured, and where

| what | artifact | sha256 |
|---|---|---|
| the frozen task set | `artifacts/eval/thesis_e0_base_direct_b1/task_ids.txt` | `task_ids_sha256 1fc257f2…` |
| six-arm four-metric table | `artifacts/results/mvp-20260929/arm-metrics.json` | `f2290fae…` |
| protocol funnel (attempts vs executions) | `artifacts/results/mvp-20260929/protocol-gap.json` | `90c5d2df…` |
| three-ruler re-scoring | `artifacts/results/mvp-20260929/tolerance-rescore.json` | `891ffb80…` |
| paired tests per ruler | `artifacts/results/mvp-20260929/paired-protocols.json` | `7f2e8ba2…` |

These are re-parses of generations and trajectories that already existed. The
one GPU-shaped hole in them is stated in §2 and is the entire reason this
campaign exists.

## 1. Pool

Frozen Omni-MATH 200: `artifacts/task_pools/rl_r0_200.jsonl` =
`data/processed/v1/frozen_eval.parquet` (sha256 `3b3255316c7e57480982e041d5e7172ca63b6052b15f7a5269c9ede574e05c43`)
restricted to the stored task list, `task_ids_sha256 = 1fc257f2…daee2`
(sha256 of the ids joined by `"\n"` with no trailing newline).

Every stored arm and every arm below draws from this list, in this order. Any
arm that cannot produce all 200 ids, once each, is void — not narrowed: the
re-scoring scripts already refuse a task set whose sha does not match, and
`paired_protocols.py` refuses to pair two arms whose task sets cross.

**The 2026-09-26 failure this rule exists for:** that campaign ran its agent
arms on the RL training pool while the direct arms evaluated the frozen set.
Zero overlap, so no pair was possible and 3.5 GPU hours bought nothing. The
same trap is one default away today: the pooled runner
`scripts/eval/run_rollout_health.py` defaults `--data` to
`data/processed/v1/rl_dev.parquet`. This campaign passes
`--pool artifacts/task_pools/rl_r0_200.jsonl` explicitly and the gate in §7
compares the recorded `pool`/`data`/`task_ids_file` fields against the stored
arm's before any GPU minute is spent.

## 2. Arms

The design's three arms are Base / SFT / SFT+GRPO. What is missing from the
six-arm table is those arms **in agent mode**: the protocol column for SFT and
for the GRPO arm has no tool-call or step cell, because neither has ever been
rolled out through the loop.

| id | arm | weights | channel | status |
|---|---|---|---|---|
| A | base + agent | Qwen3-1.7B `70d244cc` | agent loop, both tools | **stored**, reused (see §3) |
| B | SFT + agent | `artifacts/models/qwen3_1_7b_sft_dp_v1_merged` | agent loop, both tools | **new this run** |
| C | SFT+GRPO(r0) + agent | `artifacts/models/qwen3_1_7b_sft_dp_v1_merged` | agent loop, both tools | **new this run** |

Reference rows, already measured, not re-run: `base_direct` 0/200,
`sft_direct` 27/200, `r0_direct` 22/200, `r2_direct` 25/200, `rule_strategy`
10/200 (composed: 76 routed-direct rows reused verbatim + 124 rolled out).

Shared configuration for A, B, C — this is the identity being claimed, not a
description of intent:

| field | value |
|---|---|
| runtime / prompt | `runtime-v1` + `agent-v1` |
| agent config | `configs/agent/default.yaml` (max_steps 6, max_tool_calls 4, max_python_seconds 12.0, max_observation_chars 8000) |
| reward config | `configs/reward/r0.yaml` (correctness only — no tool-cost or invalid penalty; a baseline must not bake in an RL reward shape) |
| tools | sympy + python, unrestricted (`--mode all-tools`) |
| decoding | greedy, `temperature 0.0` → `do_sample False`, `max_new_tokens 1024` |
| base model revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (tokenizer same) |
| pool | `artifacts/task_pools/rl_r0_200.jsonl`, 200 tasks, once each |

**R0's name.** `configs/reward/r2.yaml` is tool-weighted shaping; r0 is
correctness-only. Neither is a cost-aware reward. Any claim written from this
campaign uses the repo's own labels.

## 3. Identity declaration, and the one reuse decision

**Decision (reversible in one line): arm A is reused, not re-run.** The stored
`base+tool` arm at `artifacts/rollout_health/thesis_e0_base_tool/` was recorded
at commit `fdf9276` on 2026-09-26. Re-running it would cost ~1 GPU hour and
produce a second row that is not a new fact. Reversal: add A to the run list.

The reuse is argued, not assumed, and the argument is committed in
`docs/results/campaign-2026-09-27/source-drift-justification.md`:

1. **Static audit.** Eight files on the evaluation path changed between
   `fdf9276` and HEAD (six under `agent/`, plus `tools/registry.py` and a new
   `configs/harness/champion_v1.yaml`). Every change is a new optional field, a
   new API, a type annotation, or a config nothing existing loads. No prompt, no
   verifier rule, no budget, and no decoding parameter was modified.
   `agent/loop.py`'s `runtime_version` falls back to the identical
   `f"{RUNTIME_VERSION}+{PROMPT_VERSION}"` string when no `HarnessSpec` is
   passed; `agent/replay.py` deletes the three new `None` fields before hashing
   precisely so legacy content hashes do not move.
2. **Contract check on recorded data.** The stored 200 trajectories load
   unchanged under the new `Trajectory` model and their canonical content hash
   is bit-identical (200/200), and `runtime_version` is `runtime-v1+agent-v1`
   for all of them.
3. **Replay check, this campaign.** All **324** stored agent trajectories
   (`base_tool` 200 + `rule_strategy` 124) replay clean under HEAD's
   `replay()` with its structural and accounting rules enforced: event sequence
   monotone, usage equal to the events, final answer equal to the final event.
   324/324, zero failures.

**What the reuse does not establish**, and what the write-up must therefore
say: the argument is static plus contract-level. It does not establish that a
*fresh* rollout under HEAD would visit the same states as one under `fdf9276`.
Arm A is reported as *recorded at `fdf9276` under an audited-equivalent tree*,
never as run under the identical tree.

**Known drift, disclosed in advance.** The r0/r2 direct rows were scored by the
verifier at `b4ca1604` (`verifier_source_sha256 b690cc7d…`); HEAD's verifier is
`3cfc5e61…`. Re-scoring under HEAD leaves r0 4 rows and r2 1 row
non-reproducible, all between two non-correct statuses
(`incorrect` ↔ `invalid_prediction`); **no row crosses the correct boundary**,
so no accuracy in §0 moves. The re-scoring script refuses to score an arm at all
if a correct-boundary row fails to reproduce. The extractor (`029570bd…`) and
prompt (`9db23ccf…`) hashes are identical across every stored arm.

**The two new arms' weights are fixed here.** Both B and C load the merged SFT
model `artifacts/models/qwen3_1_7b_sft_dp_v1_merged` — the same `base_model`
that `r0_direct` and `r2_direct` recorded, and the model r0's adapter was
trained from. Declared consequence: `sft_direct` was evaluated as
snapshot + adapter (`adapter_sha256 2868f83e…`), so the `sft_direct` ↔
`sft_agent` contrast carries a merge-rounding confound. It is bounded (the same
weights up to merge arithmetic) and it is disclosed rather than papered over;
the alternative — adding an `--adapter` flag to the agent runner to load the
adapter path exactly as the direct arm did — is a CPU-only ~10-line change and
is listed as prerequisite P1 so that the option stays open until the GPU is
booked.

## 4. The four metrics, with definitions and denominators

Computed by `scripts/analysis/arm_metrics.py` — the definition is code, and the
script cross-checks every value against the run's own summary, so a
disagreement is a hard failure rather than a footnote. Denominator is **200 for
all four**, in all three arms.

| metric | definition |
|---|---|
| **Answer accuracy** | tasks whose recorded verifier status is `correct`, / 200 |
| **Final action 合法率** | tasks on which a legal final action came out at all: agent arms → the trajectory's `final_answer` is non-empty; direct arms → the extractor returned a value (`extract_status == "ok"`), / 200 |
| **Tool-call rate** | `usage.tool_calls` summed over the arm, / 200 tasks (executed calls — the strict envelope's count, not attempts) |
| **平均轨迹步数** | mean `usage.steps` over 200 trajectories; a direct arm is single-turn and records 1.0 |

Declared diagnostics, reported beside but never substituted for a headline:
`verifier_addressable_rate`, `invalid_actions_per_task` (None for a direct arm —
absence of measurement, not a measured absence), `steps_p95`,
`termination_reasons`, `tool_calls_total`.

## 5. Parser protocol

**Strict is the primary protocol.** The strict ruler is the runtime's own
`parse_action` fullmatch — optional `<think>`, exactly one action block whose
payload satisfies its tag's schema — applied to the last model turn, with the
runtime parser's reading required to equal the recorded `final_answer` or the
scoring is refused. Every headline number in §6 is strict.

`coerce` (mechanical: a numeric `answer` literal inside one well-formed
`<final>`, read as its string form) and `lenient` (interpretive
`extract_solution_answer`, an upper bound) are **secondary sensitivity rows**,
already quantified in §0 and re-reportable for the new arms without new
generation. They may appear in a table and in prose about the protocol; they
may not appear as an arm's accuracy in a claim. If a tolerant rung changes a
conclusion, the conclusion is reported under that rung's name and the strict
number stays in the same sentence.

This is not a formality: at `coerce`, `base_agent` moves from 13/200 to 35/200
(paired +17.5 pt, p=5.8×10⁻¹¹) while all four direct arms move by exactly 0
rows. The gap between the two protocols is itself one of this campaign's
findings, and it is only legible while the two are kept apart.

## 6. Pre-committed tests and decision rules

Paired, exact McNemar, two-sided α = 0.05, over the same 200 tasks for every
comparison below; paired bootstrap CI from the repo's own
`_paired_statistics` (seed 20260913, 10 000 resamples). No comparison uses an
unpaired test, and no arm is compared on a subset without the excluded task
count and the excluded correct count being printed next to it.

* **H1 (the headline).** `SFT+agent` strict accuracy > `base+agent` 13/200.
  If it is not significant, the claim is "not significant", not "trend".
* **H2 (the protocol hypothesis — the reason this campaign is worth running).**
  `SFT+agent` closes part of the envelope gap: `final_action_rate` above
  `base_agent`'s 27/200, and executed tool calls above `base_agent`'s 7.
  The sharp form: `sft_direct` produces an extractable `<final>` in 133/200
  generations and never once types `answer` as a number, whereas 28 of
  `base_agent`'s last turns are closed, schema-valid `<final>` blocks refused
  for typing it as a number. Either SFT carries that format discipline into the
  loop — and the claim is "SFT learned the format and the mathematics" — or it
  does not, and the claim is "format and mathematics are separable bottlenecks".
  **Both outcomes are results and are reported with equal prominence.**
* **H3 (exploratory).** `R0+agent` vs `SFT+agent`. The direct rows give no
  reason to expect a difference (22 vs 27, never significant), so any p<0.05
  here is reported as exploratory and is not the campaign's headline.
* **H4 (exploratory, re-scoring only).** `SFT+agent` under `coerce` and
  `lenient`, to place it on the same three-ruler ladder as the stored arms.

**Stop rules.** (a) 200/200 per arm, one pass, no early stopping and no
peeking: partial results are never used to decide whether to continue, change
config, or add an arm. (b) No prompt, parser, budget, tool-set or decoding
tuning after any result from this campaign is seen. A tolerant rung is a
*re-scoring of stored trajectories* under the rungs already committed in §0,
never a new run with a looser parser. (c) A shard that dies is resumed at the
same shard id and the interruption is recorded; a shard that cannot be resumed
voids that arm. (d) If the recorded configuration deviates from §2 in any field
the gate compares, the arm is **void** — reported as void, not re-tuned and
re-run. (e) If the identity gate fails, there is no run at all: the fork
returns to the user rather than to a GPU.

## 7. Preconditions — blocking, CPU-only, before any GPU minute

| # | precondition | state |
|---|---|---|
| P0 | frozen task set, 200 ids, sha `1fc257f2…`; 324/324 stored trajectories replay clean | **done**, §0/§3 |
| P1 | agent runner (`scripts/campaign-20260926/rule_baseline.py`) can record the arm's own identity under `--dry-run` in the shape `verify_arm_identity.py` compares against the stored shard manifest; optionally the `--adapter` flag of §3 | **open** |
| P2 | `verify_arm_identity.py --baseline artifacts/rollout_health/thesis_e0_base_tool.shard0/manifest.json --candidate <identity.json> --justify docs/results/campaign-2026-09-27/source-drift-justification.md` exits **0** for each new arm | **open** |
| P3 | sandbox reachable at `127.0.0.1:8080` and answering, probed immediately before the run | **open** |
| P4 | the arm's `--adapter`/`--model` resolves to the weights fixed in §3, with the adapter sha recorded where the direct manifest records it | **open** |

P2 is the gate and it is not a formality. Note one honest limit found while
writing this: the stored agent manifest records `model` but **not**
`adapter_path`/`adapter_sha256`/`adapter_kind`, while the direct manifests
record all three. For the agent arms the gate therefore cannot by itself catch
a wrong adapter — it would see the candidate's extra keys as advisory, not
blocking. That is why §3 fixes the weights in prose **and** P4 requires the sha
to be written down: the gate covers the model, the pool, the configs, the
generation parameters, the sandbox and the source drift; the adapter identity
rests on the declaration plus the recorded sha.

## 8. What this campaign will NOT claim

* **That GRPO beats SFT.** r0 and r2 never significantly exceeded SFT in the
  direct rows, and r0 vs base under the lenient ruler is p=0.21 (15 up, 8 down).
  H3 is exploratory and cannot become the headline by being significant.
* **That the agent arm's accuracy comes from tool execution.** The strict
  envelope executed 7 of 422 tool-call attempts in `base_agent`; 609 of its
  1152 turns carry no action tag at all. The measured gap is about the
  envelope, and `protocol-gap-2026-09-29.md` §7 bounds what repairs can be
  claimed from re-read text (a repair produces an observation the model never
  saw; the trajectory diverges immediately after it).
* **That SFT learned to use sympy.** The demonstration-yield gate failed
  (32/200, projected 228.2 < 300; `docs/results/sympy-demo-yield-gate.md`), and
  all three arms use tools they were never trained on — so a low tool-call
  rate is an expected reading, not a finding about tool learning.
* **"200/200 byte-identical across weeks" in the unqualified form.** The
  same-configuration re-run reproduced 200/200 outputs byte for byte; across a
  batch-configuration change it is 3/200 byte-identical and 130/200
  verdict-identical. All numbers come from the single frozen configuration.
* **Any accuracy computed on a narrower task set than the frozen 200** without
  the excluded-task count printed beside it.
* **That a repaired trajectory's score is a score.** The `coerce`/`lenient`
  columns are ceilings over recorded text.
