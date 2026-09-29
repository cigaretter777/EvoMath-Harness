# MVP packaging — the four entry points (2026-09-29)

**Date:** 2026-09-29
**Scope:** the §7 code surface of the MVP design —
`scripts/mvp/{prepare_data,train_sft,train_grpo,evaluate}.py` and
`configs/mvp/{sft,grpo}.yaml`, with `tests/unit/mvp/`. Everything here is
**CPU-only**: no GPU was booked, no training run, no new generations. The runs
this round are dry-runs and re-reads of stored artifacts.
**Constraint that shaped it:** the frozen runner
(`scripts/campaign-20260926/rule_baseline.py`) may not be modified — its blob
hash is a blocking identity field (`e591c886…`) in every stored arm's manifest.
Nothing in this round touches it, the stored arms, or the released weights.

## 1. What was added

| file | what it is | wraps |
|---|---|---|
| `configs/mvp/sft.yaml` | field-for-field mirror of `configs/sft/qwen3_1_7b_dp_v1.yaml`, i.e. of the config that produced the released SFT adapter, plus a provenance header | — |
| `configs/mvp/grpo.yaml` | field-for-field mirror of `configs/grpo/qwen3_1_7b_r0.yaml` (the released R0 run) | — |
| `scripts/mvp/train_sft.py` | SFT entry: default config; `--merge-to DIR` merges the finished adapter into the base weights (the merged model GRPO trains on) | `scripts/train/run_sft.py` |
| `scripts/mvp/train_grpo.py` | GRPO entry on the only launch path verified on this host; `--export-adapter DIR` lifts the newest checkpoint's LoRA into an evaluable adapter | `scripts/train/run_grpo_direct.py` → `export_verl_lora.py` |
| `scripts/mvp/prepare_data.py` | CPU pre-flight for the data: regenerates the frozen 200-task list and verifies the parquet and the SFT split manifest against their tracked pins | `scripts/data/build_dataset.py` (`--build`) |
| `scripts/mvp/evaluate.py` | runs the three arms (Base / SFT / SFT+GRPO), one subprocess each, and collates the four metrics | `rule_baseline.py` (base) and `run_arm_with_adapter.py` (sft, grpo), then `arm_metrics.py` |
| `tests/unit/mvp/*.py` | 34 tests: the mirrors, the wrappers' forwarding, the data checks, the arm/weights resolution, the metrics path | — |

Two things are deliberately *not* new: the evaluation logic (the runner, the
identity emitter, the metrics collator are the ones the campaign used) and any
change to a script the campaigns ran. The wrappers add a default config and, for
the two training entries, the one post-step each product chain was missing (the
merge, and the LoRA export).

## 2. Evidence, all on this host and all on CPU

**The mirrors cannot drift.** Both configs are asserted equal to their sources
as parsed data, and both load through the trainer's own validator
(`test_configs_pinning.py`). The GRPO mirror's `actor_rollout_ref.model.path`
is asserted, against `evaluate.py`'s default, to be the merged SFT checkpoint —
the field that decides what the GRPO arm's base weights are.

**Both training entries run their config workflow with no GPU stack present.**
`uv run python scripts/mvp/train_sft.py --dry-run` prints the resolved mirror
config; `scripts/mvp/train_grpo.py --dry-run` prints
`"verl_agent_sha": "20bd331bdbc9026a5668e11362178e10ab7400c8"` (the pinned
backend). Neither imports torch in the dry-run path, which is what lets the CI
suite pin them (`test_entrypoints.py`).

**The data checks pass on the real artifacts, and the task list regenerates
byte-identically.**

```bash
uv run python scripts/mvp/prepare_data.py
# PREPARE OK: frozen_eval.parquet 3438 rows and task-ids hash match data/manifests/v1.json
#   task list: 200 ids, id-hash 1fc257f26fbf80ab..., .../task_ids.txt unchanged
#   sft data: train 2649 / validation 295 rows, manifest hash matches the pin in configs/mvp/sft.yaml
```

`frozen_eval.parquet` matches the tracked `data/manifests/v1.json` on both its
file hash (`3b325531…`) and its task-ids hash (`5b1612f7…`); the regenerated
200 ids hash to `1fc257f2…`, the constant every stored arm and analysis script
pins; `data/manifests/sft_dp_v1_split.json` (`f9cc4c4a…`) matches the
`data_manifest_sha256` in `configs/mvp/sft.yaml`, and both split parquets match
the hashes recorded inside it.

**The three arm identities gate against the stored arms — before any GPU
minute.** `scripts/mvp/evaluate.py --dry-run` emits one identity per arm and
prints the gate command. Run against the stored shard manifests:

| arm | candidate identity says | gate vs stored arm |
|---|---|---|
| `base` | base snapshot `70d244cc…`, no adapter | **GATE OK** (rc=0), with the campaign's `source-drift-justification.md` — the stored base arm predates the 2026-09-27 agent-loop changes |
| `sft` | base snapshot + adapter `2868f83e…` | **GATE OK** (rc=0), 9 identity fields aligned |
| `grpo` | merged SFT + adapter `21a3f4aa…` | **GATE OK** (rc=0), 9 identity fields aligned |

The two adapter hashes are the released adapters' hashes: the arms
`evaluate.py` would run are the arms the campaign measured.

**The metrics path reproduces the released rows on CPU.** `--metrics-only` over
the stored runs (symlinked under `artifacts/mvp/released-recheck/`) re-derives
the released table exactly, and writes
`artifacts/mvp/released-recheck/four-metrics.json`
(sha256 `ddc9f53290107c4bb1b8257ab25c06825a2f39d2400419ab417125654fb5a45e`):

| arm | n | correct | accuracy | final action | verifier-valid | tool calls | steps mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base` | 200 | 13 | 6.5% | 13.5% | 11.0% | 7 | 5.76 |
| `sft` | 200 | 28 | 14.0% | 79.5% | 78.0% | 0 | 2.17 |
| `grpo` | 200 | 27 | 13.5% | 94.0% | 93.0% | 0 | 1.51 |

Every count is checked against the run's own `summary.json` before the table is
printed, and the verifier-valid column — which `summary.json` does not record —
is counted independently and agrees with the pins registered in
`scripts/analysis/arm_metrics.py` (22 / 156 / 186).

**CI gates.** `ruff` and `mypy` are clean; the local suites are 596 unit tests
and 47 contract tests (4 skipped). The same four gates pass in a clean clone with
a fresh `uv sync --dev`: 576 unit tests passed / 20 skipped (the 20 extra skips
are the local-artifact tests, which is the point of rehearsing in a clone) and
47 contract tests passed / 4 skipped. That rehearsal is what caught — and the
fix landed for — one local-green/CI-red gap this round: a test comparing the
GRPO mirror's absolute `actor_rollout_ref.model.path` against a checkout-relative
default.

## 3. The experiment results (unchanged — quoted, not recomputed)

This round produced **no new experiment results**. The MVP's three arms are the
arms of campaign 2026-09-27; their numbers, from
[`campaign-2026-09-27/stage-summary.md`](campaign-2026-09-27/stage-summary.md)
and [`mvp-20260929-arm-table.md`](mvp-20260929-arm-table.md):

* **H1 — confirmed.** `sft_agent` 28/200 vs `base_agent` 13/200: +7.5 pt,
  95% CI [+3.0, +12.5], 20 up / 5 down, p = 0.0041 (exact McNemar). The
  headline holds.
* **H2 — half-confirmed.** The final-action half: 159/200 (79.5%) vs
  `base_agent`'s 27/200 (13.5%). The tool-call half does **not** hold: executed
  tool calls went 7 → 0, tool attempts 422 → 13. SFT carried its format
  discipline into the loop as *"close with a legal final answer"*, not as
  *"call the tools"*.
* **H3 — null, as pre-committed.** `r0_agent` vs `sft_agent`: −0.5 pt,
  95% CI [−4.0, +3.0], p = 1.0.
* **H4 — disclosed.** Under `lenient` re-scoring, `base_agent` (40/200) beats
  `sft_agent` (28/200), −6.0 pt, p = 0.0227. Strict remains the headline; the
  tolerant rulers are ceilings over recorded text, not scores the arms earned.

What this round adds is packaging, not evidence: the same three rows can now be
produced by four commands, two of which are new.

## 4. What this round does not claim

* **No GPU was booked, and `evaluate.py` has never taken its real path.** The
  three arms have not been rolled out through the MVP entry point; only the
  pre-flight, the identity gate and the metrics passthrough are verified. A real
  end-to-end run needs the GPU card and the sandbox, ~1.5–2 h at the released
  scale (B 51 m + C 41 m for the two adapter arms).
* **No README, no quick start, no trajectory spot-checks.** Out of scope by
  instruction this round.
* **No change to the frozen runner, the stored arms or the released artifacts.**
  The adapter hashes above are the released ones precisely because none of them
  moved.
* **Single-shard metrics.** `evaluate.py` computes metrics only for a
  single-shard run: the runner's `summary.json` always describes all 200 tasks,
  so a shard's counts could not be checked against it. Multi-shard runs are
  launched but must be merged first
  (`scripts/campaign-20260926/merge_rule_shards.py`), as the released arms were.

## 5. Reproduce

```bash
# 1. data pre-flight (CPU, seconds)
uv run python scripts/mvp/prepare_data.py

# 2. the two training entries, config workflow only
uv run python scripts/mvp/train_sft.py --dry-run
uv run python scripts/mvp/train_grpo.py --dry-run

# 3. the three arms' identities, then gate them (CPU, no weights)
#    (the dry-run prints each arm's exact gate command)
uv run python scripts/mvp/evaluate.py --dry-run
uv run python scripts/eval/verify_arm_identity.py \
    --baseline artifacts/rollout_health/thesis_e0_sft_tool.shard0/manifest.json \
    --candidate artifacts/mvp/eval/sft.identity.json
# base: the stored arm predates the 2026-09-27 agent-loop changes, so the gate
# needs the campaign's justification file (evaluate.py prints this line too)
uv run python scripts/eval/verify_arm_identity.py \
    --baseline artifacts/rollout_health/thesis_e0_base_tool.shard0/manifest.json \
    --candidate artifacts/mvp/eval/base.identity.json \
    --justify docs/results/campaign-2026-09-27/source-drift-justification.md

# 4. re-derive the released table from the stored runs (CPU)
#    (needs artifacts/mvp/released-recheck/{base,sft,grpo} -> the stored runs)
uv run python scripts/mvp/evaluate.py --metrics-only \
    --out-root artifacts/mvp/released-recheck

# 5. a real three-arm run (GPU + sandbox; ~1.5-2 h) -- not run this round
uv run python scripts/mvp/evaluate.py --out-root artifacts/mvp/eval
```
