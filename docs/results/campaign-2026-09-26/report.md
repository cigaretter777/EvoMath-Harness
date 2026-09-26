# Thesis baseline campaign 2026-09-26 — results (E0/E1)

Pre-registration: `preregistration.md` in this directory, including addendum A
(pool misalignment correction). Every number below comes from
`scripts/campaign-20260926/analyze_e0_baselines.py` reading artifacts on disk;
raw evidence in `artifacts/results/campaign-20260926/summary.json`.

Host: AutoDL, single RTX 4090 24GB. All four arms share the frozen Omni-MATH
200 (task_ids from the stored base-direct arm, verified by the alignment gate),
greedy, cap 1024, batch 1; agent arms run the production AgentLoop with the
live sandbox and reward config r0 (no tool-cost / invalid penalties).

## 1. Verdicts

| id | question | result |
|---|---|---|
| identity | is the regenerated SFT arm byte-identical to the E2 champion arm? | **200/200 byte-identical — pipeline cross-check PASS** |
| E0b vs E0a | does tool access move correctness for the base model? | +6.5% (13 vs 0), McNemar p=0.00024, CI [3.5%, 10.0%] — significant, but see §4 |
| E0c vs E0a | does rule-routed tool access move correctness? | +5.0% (10 vs 0), p=0.002 — significant, same caveat |
| E0c vs E0b | does restricting the tool set by rule help? | −1.5%, p=0.58 — not significant |
| E1 vs E0a | SFT's contribution | +13.5% (27 vs 0), p=1.5e-8 — the dominant effect |

## 2. Per-arm statistics

| arm | correct | incorrect | invalid | tool_calls (mean/median/p95) | invalid_actions (mean) | tokens (mean/p95) | rollout (mean/p95 s) |
|---|---|---|---|---|---|---|---|
| E0a base-direct | 0 | 0 | 200 | — | — | 966 / 1024 | 21.6 / 23.1 |
| E1 sft-direct | 27 | 106 | 67 | — | — | 680 / 1024 | 22.7 / 34.4 |
| E0b base+tool | 13 | 9 | 178 | 0.035 / 0 / 0 | 5.59 | 3842 / 6144 | 100 / 148 |
| E0c fixed-rule (124 rolled out) | 10 | 3 | 111 | 0.129 / 0 / 1 | 5.56 | 3925 / 5444 | 101 / 141 |
| E0c composite (200) | 10 | 3 | 187 | — | — | — | — |

The composite adds the 76 direct-routed tasks, which reuse the base-direct
arm (0 correct). The agent arms' 200 vs 124 n is by construction: the
fixed-rule strategy routes 76 tasks to the direct channel.

## 3. Tool use in the baselines

- Base+tool executed **7 tool calls across 200 trajectories** (197/200 made
  zero calls). Fixed-rule executed 16 across 124 (113/124 zero). The
  single-tool whitelist roughly quadruples the call rate but leaves it tiny.
- Both agent arms terminate overwhelmingly at `max_steps` with ~5.6 invalid
  actions per trajectory and ~3.9k generated tokens (vs 680 for SFT-direct):
  the base model does not know the protocol and loops.
- The direct arm's 200/200 invalid confirms the same at batch 1: the base
  model emits no parseable final envelope at all.

**Consequence for the E0b/E0c significance results:** the correctness gains
(+6.5%, +5.0%) come from the multi-turn agent format (extra turns to reach a
parseable final), not from tool execution — tools were almost never used. The
thesis claim "工具本身是否有效" cannot be answered by these arms beyond: the
base policy is not yet capable of using tools, so no-tool vs tool differs
mostly in loop format. This is the honest reading and the correct RL baseline.

## 4. What this means for the RL phase (E2-E4)

1. SFT (27/200) is the accuracy floor and the protocol floor; RL starts from
   the SFT adapter, not from base.
2. The base model's zero protocol compliance means "does the policy learn to
   call tools" is measurable from a near-zero floor in the base-pool
   diagnostics, but RL diagnostics should track tool-call rate per step as a
   leading indicator.
3. The archived openr1-pool runs (`*_openr1_pool`, base+tool: 21/200, 10 tool
   calls, 1065 invalid actions) are training-pool diagnostics, not part of the
   paired campaign.

## 5. Reproduction

```bash
# alignment gate (CPU, seconds)
bash scripts/campaign-20260926/verify_eval_alignment.sh

# analysis (CPU)
env -u OMP_NUM_THREADS /root/autodl-tmp/conda-envs/adaptive-math/bin/python \
  scripts/campaign-20260926/analyze_e0_baselines.py
```

Artifacts: `artifacts/eval/thesis_e0_base_direct_b1/` (E0a/E1),
`artifacts/rollout_health/thesis_e0_base_tool/` (E0b),
`artifacts/rollout_health/thesis_e0_rule_strategy/` (E0c, with per-shard
evidence), `artifacts/results/campaign-20260926/summary.json`,
`artifacts/audit/relabel-20260925/summary.json` (verifier-fix relabel audit).
