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

*Amended — see §8b: B and C do not load a merged checkpoint. B loads the base
snapshot plus the SFT adapter, C loads merged SFT plus the r0 adapter, so that
each arm's weights are the ones its direct row was measured with.*

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
weights up to merge arithmetic) and it is disclosed rather than papered over.

*Amended — see §8b: this paragraph is superseded. Both arms load adapters
through a wrapper that drives the frozen runner, so the confound above is
removed rather than disclosed, and arm C becomes runnable at all. The paragraph
that follows about the runner's immutability stands as written; only its closing
conclusion was wrong.*

The alternative — an `--adapter` flag on the agent runner — is **not
available**, and the reason is worth stating because it constrains more than
this paragraph: `scripts/campaign-20260926/rule_baseline.py` writes
`git hash-object` of *its own committed file* into every manifest it produces
(`router_source_sha256`), a field the identity gate treats as blocking. The
stored arm pinned that blob (`e591c886…`), so the runner is **immutable by
contract**: adding a flag to it would change the hash every future agent arm
records and break the pair with the arm this campaign is built on. Loading the
adapter path would mean a second runner, which is a worse trade than a disclosed
merge-rounding confound. The merged model is therefore the only path, and the
frozen runner's blob is guarded by a test. See the amendment in §8.

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
| P1 | emit each new arm's identity on CPU in the shape the gate compares: `scripts/eval/emit_arm_identity.py`, which imports the runner's own loader and hashes rather than restating them | **done**, §8; for the arms of §8b it is called through `run_arm_with_adapter.py`, which writes the same bytes to `<output-dir>.identity.json` |
| P2 | `verify_arm_identity.py --baseline artifacts/rollout_health/thesis_e0_base_tool.shard0/manifest.json --candidate <identity.json> --justify docs/results/campaign-2026-09-27/source-drift-justification.md` exits **0** for each new arm | **done** for `all-tools` shard 0 on base and SFT weights, §8; **done** for both §8b arms (base+SFT adapter, merged SFT+r0 adapter); re-run after the campaign against each real manifest |
| P3 | sandbox reachable and *answering* (a ping is not evidence; the probe runs code), re-probed immediately before the run | **passed 2026-09-29**, both endpoint spellings, `pong+42`; expires — re-probe at launch |
| P4 | the weights actually loaded match those fixed in §3 — the gate reports the weights difference but cannot judge it | **split by §8b**: the *loading path* is closed by test (the adapter is asserted on the call the loader receives, and its sha256 is compared against the direct arm's), the *loaded weights* are checked after the run by the first-turn divergence rule in §8b |
| P5 | the runner is still the frozen blob (`git hash-object` equals the stored arm's `router_source_sha256`) | **guarded by test**, §8 |

P2 is the gate and it is not a formality. Two honest limits, both found while
writing this and both now written into the gate's behaviour (§8):

* The stored agent manifest records `model` but **not**
  `adapter_path`/`adapter_sha256`/`adapter_kind`, while the direct manifests
  record all three. For the agent arms the gate cannot by itself catch a wrong
  adapter.
* The weights are the one field the gate must not block on (§8 A2), so it
  reports them instead. A wrong checkpoint is visible in the output but is not
  fatal.

So the gate covers the pool, the configs, the generation parameters, the sandbox
and the source drift, and it *reports* the weights. The weights themselves rest
on the declaration in §3 plus the manifest the run writes; P4 is that check,
and it is a human one by construction.

*Amended — see §8b: the declaration in §3 is superseded by the adapter plan, and
P4 is no longer only human. The weights are pinned to the direct rows by hash on
the loading path, and the loaded weights get a deterministic post-run check.*

**P3's probe, and why it is not a curl.** The canonical probe already exists in
`scripts/cloud/preflight.py` (`_sandbox_evidence`), which ping plus a **real
execution**, and the launch wrapper says in as many words not to add a second
one because it could only drift from the first. Re-run it as:

```
$ .venv/bin/python -c "
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location('preflight', 'scripts/cloud/preflight.py')
m = importlib.util.module_from_spec(spec); sys.modules['preflight'] = m
spec.loader.exec_module(m)
os.environ['ADAPTIVE_MATH_SANDBOX_URL'] = 'http://localhost:8080'
print(m._sandbox_evidence(True))"
```

Result on 2026-09-29: `{'required': True, 'url': …, 'reachable': True, 'probe':
'pong+42'}` — the ping answered and an execution returned 42 — for both
`localhost:8080` and `127.0.0.1:8080`, so the two spellings the gate compares as
strings are in fact the same live sandbox. A liveness result is only evidence
about the moment it was taken, which is why the row above expires: a dead
sandbox degrades every Python tool call to `UNAVAILABLE` and would be recorded
in the run as a property of the model.

## 8. Amendment 1 (2026-09-29, before any GPU minute)

Recorded rather than silently edited, because two of the statements above were
wrong when they were committed and one of them changed a plan.

**A1. The agent runner cannot be edited.** §3 originally offered an `--adapter`
flag as "a CPU-only ~10-line change", and §7 P1 carried the same option. It is
not available: `rule_baseline.py` hashes its own blob into every manifest as
`router_source_sha256`, a blocking field, so any edit unpairs every future agent
arm from the stored one. The consequence is wider than the adapter question —
**the stored `base+tool` arm stays pairable only while that file is
byte-identical** to the blob it recorded (`e591c886…`, still true as of this
commit), and nothing in the repository would have noticed if it stopped being
true. `tests/unit/eval/test_emit_arm_identity.py::test_the_agent_runner_is_still_the_frozen_blob`
is now that alarm. The merged-model path of §3 stands as the only option.

**A2. The identity gate could not have passed this campaign's pairs.** `model`
was a blocking field, and the three arms of §2 differ in precisely that field,
so gating `SFT+agent` against the stored `base+agent` would have failed on the
one difference the pairing exists to measure. `model` and the adapter fields are
now `WEIGHTS_FIELDS`: reported in the gate's output, never blocking. Exempting
them from comparison does **not** exempt them from existing — a candidate that
omits a field the baseline recorded still blocks, weights included.

**A3. The pre-flight is implemented, and both arms pass it on CPU.** The
emitter is `scripts/eval/emit_arm_identity.py`. It refuses what the gate would
have refused later and less legibly: a task list that is not the frozen 200 once
each, a recorded path that is not absolute (the gate compares strings; the
stored arm recorded absolute paths), a sandbox URL spelled differently from the
stored one, and a `--shard-id` without `--shard-count`. It loads no weights and
opens no sandbox, and a test proves that by replacing both with raising stubs.

```
$ ADAPTIVE_MATH_SANDBOX_URL=http://localhost:8080 python scripts/eval/emit_arm_identity.py \
    --mode all-tools --shard-id 0 --shard-count 3 \
    --data <repo>/data/processed/v1/frozen_eval.parquet \
    --task-ids-file <repo>/artifacts/eval/thesis_e0_base_direct_b1/task_ids.txt \
    --model <base snapshot | merged SFT> \
    --agent-config <repo>/configs/agent/default.yaml \
    --reward-config <repo>/configs/reward/r0.yaml \
    --output-dir <the run's output dir> --out identity.json

$ python scripts/eval/verify_arm_identity.py \
    --baseline artifacts/rollout_health/thesis_e0_base_tool.shard0/manifest.json \
    --candidate identity.json \
    --justify docs/results/campaign-2026-09-27/source-drift-justification.md
```

Both invocations exit **0** with 9 identity fields aligned. The base-weights run
reproduces `router_source_sha256 e591c886…` and the routing distribution
`{direct 76, python 66, sympy 58}` exactly; the SFT-weights run adds one note —
`model: baseline '<base snapshot>' != candidate '<merged SFT>' (weights: expected
to differ between arms)` — which is the weights disclosure of §3 made visible in
the gate's own output instead of living only in this document.

**What the amendment does not change.** §4's metrics, §5's strict-first protocol,
§6's hypotheses and stop rules, and §9's list of things this campaign may not
claim are untouched. Nothing here was written after seeing a result: no arm of
§2 has been rolled out.

## 8b. Amendment 2 (2026-09-29, same day, still before any GPU minute)

**A4. The two new arms load adapters through a wrapper, not merged checkpoints.**
§2's table and §3's "the two new arms' weights are fixed here" are superseded by
this, and one sentence of A1 with them.

A1 was right that the runner cannot be **edited** and wrong to conclude that the
adapter path was therefore closed. The runner can be **driven**.
`scripts/campaign-20260927/run_arm_with_adapter.py` (commit `3010662`) parses the
run's argv with the emitter's parser — the runner's own parser plus
`--adapter`/`--adapter-kind` — patches exactly one call inside the runner
(`TransformersModelClient.from_pretrained`, to pass `adapter=`), and hands the
namespace to the runner's own `run()`. The manifest is still written by the
unmodified runner, `router_source_sha256` is still `e591c886…`, and A1's blob
guard still applies.

| id | arm | weights | channel | status |
|---|---|---|---|---|
| A | base + agent | Qwen3-1.7B `70d244cc` | agent loop, both tools | **stored**, reused (§3) |
| B | SFT + agent | base snapshot `70d244cc` + `artifacts/sft/qwen3_1_7b_sft_dp_v1/adapter` (`adapter_sha256 2868f83e…`, kind `sft`) | agent loop, both tools | **new this run** |
| C | SFT+GRPO(r0) + agent | `artifacts/models/qwen3_1_7b_sft_dp_v1_merged` + `artifacts/runs/grpo_qwen3_1_7b_r0/r0_adapter` (`adapter_sha256 21a3f4aa…`, kind `rl`) | agent loop, both tools | **new this run** |

Three consequences, each an improvement on what §3 declared:

1. **Arm B is `sft_direct`'s weights, loaded `sft_direct`'s way.** The
   merge-rounding confound is not bounded, it is absent — there is no merge in
   arm B's path. Arm B's `model` field is byte-identical to stored arm A's, so in
   the gate's own output the only difference between A and B is the adapter, which
   is precisely the difference H2 exists to measure.
2. **Arm C becomes runnable.** Under §3 it could not be: r0 exists only as an
   adapter, and the runner had no way to load one.
3. **The weights are pinned to the direct rows by hash.** The agent arm's adapter
   and the direct arm's adapter are the same bytes — `2868f83e…` (sft) and
   `21a3f4aa…` (rl) — asserted by test against the values `run_model_eval.py`
   recorded in `artifacts/eval/thesis_e0_base_direct_b1/eval_manifest.json` and
   `artifacts/eval/r0_omnimath_200/eval_manifest.json`. **P4 closes mechanically**,
   on the loading path, instead of resting on a declaration.

**Both arms pass the pre-flight on CPU**, against the same stored shard-0
manifest, gated exactly as §7 P2 prescribes:

```
$ ADAPTIVE_MATH_SANDBOX_URL=http://localhost:8080 \
  python scripts/campaign-20260927/run_arm_with_adapter.py \
    --mode all-tools --shard-id 0 --shard-count 3 \
    --data <repo>/data/processed/v1/frozen_eval.parquet \
    --task-ids-file <repo>/artifacts/eval/thesis_e0_base_direct_b1/task_ids.txt \
    --model <base snapshot> \
    --adapter <repo>/artifacts/sft/qwen3_1_7b_sft_dp_v1/adapter --adapter-kind sft \
    --agent-config <repo>/configs/agent/default.yaml \
    --reward-config <repo>/configs/reward/r0.yaml \
    --output-dir <repo>/artifacts/rollout_health/thesis_e0_sft_tool.shard0 --dry-run

$ python scripts/eval/verify_arm_identity.py \
    --baseline artifacts/rollout_health/thesis_e0_base_tool.shard0/manifest.json \
    --candidate artifacts/rollout_health/thesis_e0_sft_tool.shard0.identity.json \
    --justify docs/results/campaign-2026-09-27/source-drift-justification.md
```

Arm B: `GATE OK — 9 identity fields`, four notes, all of them weights
(`adapter`, `adapter_path`, `adapter_sha256`, `adapter_kind`) and no `model` note,
because its model is the baseline's. Arm C: the same four plus the `model` note,
because its base is the merged checkpoint. **Neither arm has a blocking
difference.** Both candidates reproduce `router_source_sha256 e591c886…` and the
routing distribution `{direct 76, python 66, sympy 58}` — from the unmodified
runner's own functions.

Both sidecars carry every field that has to survive to the launch — 200 frozen
tasks, `mode all-tools`, shard 0 of 3, the frozen `--data`/`--task-ids-file`
paths, `configs/agent/default.yaml`, `configs/reward/r0.yaml`, `temperature 0.0`
and `max_new_tokens 1024`, `router_source_sha256 e591c886…`, the routing
distribution above — and one pair of weights each:

| sidecar (read by the gate) | model | adapter sha256 (kind) |
|---|---|---|
| `artifacts/rollout_health/thesis_e0_sft_tool.shard0.identity.json` | base snapshot `70d244cc` | `2868f83e…` (sft) |
| `artifacts/rollout_health/thesis_e0_r0_tool.shard0.identity.json` | merged SFT | `21a3f4aa…` (rl) |

**The sidecar's own sha256 is deliberately not cited**, and this is worth one
sentence because citing it is the obvious thing to do and it is wrong: the
identity records the emitting commit as `git_sha`, so its bytes move whenever
HEAD moves and say nothing about the arm. Emitting this argv at `2f3e6d2` and
again at `f18ebd6` produced `f4dc7779…` and then `88a1011e…` from otherwise
identical content, while re-emitting at a fixed HEAD is byte-identical. `git_sha`
is bookkeeping in the gate (it is in `ALLOWED_TO_DIFFER`); the source drift it
would otherwise stand for is checked separately and more strictly, by diffing the
evaluation path. The sidecars are pre-flight evidence and they expire at the
launch — the authoritative identity is the manifest each run writes, and the gate
is re-run against it afterwards.

**Two interpreters, and it matters.** Every command above ran under
`.venv/bin/python`, which has neither torch nor peft — the identity has no use for
either, and a pre-flight that needs the GPU stack is not a pre-flight. The arm
does not have that freedom: the runtime is
`/root/autodl-tmp/conda-envs/adaptive-math/bin/python` (torch 2.8.0+cu128,
transformers 4.57.3, peft 0.20.0). Launching the arm with the CPU interpreter is
a live mistake, because the runner imports `PeftModel` *after* it has already read
the base checkpoint — 3.4 GB and then a traceback. The wrapper therefore checks
for `peft` before it calls the runner (on the run path only, so a dry run still
works anywhere), and the failure it produces in the CPU environment is:

```
RUNTIME FAIL: this interpreter cannot load a sft adapter (No module named 'peft');
sys.executable=/root/autodl-tmp/Adaptive-Solver-main-git/.venv/bin/python.
Run the arm with the runtime environment -- the one that has torch and peft.
```

The two interpreters agree on the identity: the same argv emitted under both
produced byte-identical sidecars, so the pre-flight is a function of the argv and
the files, not of the environment. The launch is that argv with the runtime
interpreter and without `--dry-run`, after which the wrapper copies the identity
to `<output-dir>/identity.json`, beside the manifest the frozen runner writes.

**The failure mode this opens, and the check that closes it.** An adapter arm that
silently runs without its adapter would produce a base arm wearing an SFT label,
and *no artifact of the run would show it*: the frozen manifest has no adapter
field, and the sidecar records what the wrapper intended, not what the loader
received. Two tests cover the wrapper on CPU — the adapter is asserted on the call
the loader receives (not on the flag the wrapper parsed), and the class is
restored by identity afterwards — but a test cannot see the run. So the run's
first-turn model outputs are compared against the arm they must differ from, in
the same channel, same prompts, greedy decoding:

| comparison | alarm | reading |
|---|---|---|
| arm B vs stored arm A, first `model_output` of each shared task | ≥ 50% byte-identical | the sft adapter did not load |
| arm C vs arm B, same | ≥ 50% byte-identical | the r0 adapter did not load |

The threshold is set against measured poles. On these 200 tasks under this
decoding, two genuinely different weight sets agree byte-for-byte on **0/200**
(base vs sft) and **12/200** (sft vs r0) of their direct generations
(`thesis_e0_base_direct_b1/`, `r0_omnimath_200/`), while identical weights and
configuration reproduce **200/200** (§0). 50% is more than eight times the worst
observed different-weights rate, and the check reads only artifacts already on
disk. It joins §6's stop rules: an arm that trips it is **void** and is
reported as void. Re-running a void arm is a harness fix and never a config
change — §6(b) still forbids the latter — and the void run is reported beside the
result.

**What the amendment does not change.** §4's metrics, §5's strict-first protocol,
§6's hypotheses and stop rules, and §9's list of things this campaign may not
claim. §7's P0, P3 and P5 are untouched; P1 and P2 are the same commands with two
more flags; P4 is closed on the loading path and remains open on the loaded
weights, which is what the first-turn check above is for. Nothing here was written
after seeing a result: no arm of §2 has been rolled out.

## 9. What this campaign will NOT claim

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
