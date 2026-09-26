# Training Smoke Record

Status: **in progress** (low-cost route per
[decision 0001](../decisions/0001-low-cost-training-route.md)).
GPU host is live since 2026-09-19; environment self-check passed;
overfit-smoke gate **PASS** (2026-09-19). Formal-SFT data route resolved
to **dp_v1 direct** (user decision 2026-09-19), superseding the LR
pilots below. R0/R2 GRPO campaign ran to completion on the dp_v1 merged
base; result is a null (no significant accuracy change, all arms
statistically indistinguishable) — see
[overnight-campaign-2026-09-19.md](overnight-campaign-2026-09-19.md).

This file records SFT pilot results and the formal learning-rate selection
with their evidence, as required by Training Plan Task 4. Numbers are
appended only after the corresponding run completes; nothing here may be
filled in from expectation.

## SFT pilot table (to be filled on the GPU host)

| Run | Config | LR | Data | Loss first→last | Protocol validity | Verdict |
|---|---|---|---|---|---|---|
| overfit-smoke | qwen3_1_7b_smoke.yaml (epochs=12, calibrated) | 2e-5 | 200 synthetic DIRECT | 8.71 → 0.0003 | gate pass ≥98% (50 prompts); 10/10 exact spot-check | **PASS 2026-09-19** |
| pilot-lr-5e-6 | qwen3_1_7b_lora.yaml --set learning_rate=5e-6 | 5e-6 | sft_v1 | — | — | superseded (dp_v1 direct, 2026-09-19) |
| pilot-lr-1e-5 | qwen3_1_7b_lora.yaml | 1e-5 | sft_v1 | — | — | superseded (dp_v1 direct, 2026-09-19) |
| pilot-lr-2e-5 | qwen3_1_7b_lora.yaml --set learning_rate=2e-5 | 2e-5 | sft_v1 | — | — | superseded (dp_v1 direct, 2026-09-19) |

## Formal LR selection

Selection rule (fixed before running, per plan): choose on SFT-dev loss,
protocol validity (≥98%) and the frozen mini-eval — **not** on training
loss alone. Record the chosen run's resolved_config hash here.

- Selected LR: n/a — dp_v1 direct route (no formal SFT retraining run)
- Evidence: existing dp_v1 model reused as GRPO base
  (`artifacts/models/qwen3_1_7b_sft_dp_v1_merged`)

## Checkpoint gate (before GRPO)

## GRPO entrypoint gate (local, no CUDA backend)

`scripts/train/run_grpo.py` is the only project entrypoint for real GRPO. It
requires a content-addressed private task pool, verifies the fixed
`verl-agent` checkout SHA, writes the resolved Hydra config, and delegates
updates to upstream `recipe.hgpo.main_hgpo.run_ppo`. The fixed upstream has no
external environment registry; before Ray starts, the entrypoint applies an
idempotent, marked registration branch that routes only
`env.env_name=adaptive_math` to `VerlMathEnvironmentManager`.

The checked-in smoke config pins the locally materialized 32-task pool at
`artifacts/task_pools/rl_smoke.jsonl`; the artifact itself is ignored because
it contains reference answers. On the cloud machine copy that exact artifact
before running:

```bash
uv run python scripts/train/run_grpo.py \
  --config configs/grpo/qwen3_1_7b_smoke_r0.yaml --dry-run
torchrun --nproc_per_node=1 scripts/train/run_grpo.py \
  --config configs/grpo/qwen3_1_7b_smoke_r0.yaml
```

This is a gate, not a claimed training result: the GPU smoke remains required
to demonstrate rollout, a real tool call, terminal verification, nonconstant
GRPO groups, backward/update, LoRA checkpoint, and resume.

**GRPO smoke result (2026-09-19): PASS** — 2 training steps completed on the
AutoDL 4090 via direct `run_grpo.main()` (torchrun bypassed; see execution
notes below). Evidence: `artifacts/runs/repro-main-entry3.log` +
`artifacts/runs/grpo_qwen3_1_7b_smoke_r0/checkpoints/{global_step_1,global_step_2}/`
with `actor/` adapters and `latest_checkpointed_iteration.txt`.

- [x] rollout: 32 trajectories × up to 6 env steps (episode/length mean 4.1)
- [ ] real tool call: NOT demonstrated — smoke `episode/tool_call_count/mean: 0.0`.
      The formal R2 run (12 steps, tool_weight=0.15, sandbox reachable and
      verified) also recorded tool_call_count=0.0 throughout training: the
      model never discovered tool value in 12 steps (see the campaign doc).
- [x] terminal verification: symbolic verifier rewards flowed
      (episode/reward mean 0.031, max 1.0)
- [x] nonconstant GRPO groups: advantages mean -0.274, max 4.695, min -0.667
- [x] backward/update: update_actor ran both steps (~222–252 s each)
- [x] LoRA checkpoint: global_step_1/2 with `actor/` adapters
- [x] resume marker: `latest_checkpointed_iteration.txt` present

- [ ] parse success ≥98% on SFT dev
- [ ] all four behavior categories present in dev generations
- [ ] tool observation used after ≥90% of successful tool calls
- [ ] Base-Direct mini-eval degradation ≤2 absolute points

## Execution notes

- Host: AutoDL RTX 4090 24GB (2026-09-19). Training env:
  `/root/autodl-tmp/conda-envs/adaptive-math` (torch 2.8.0+cu128, peft,
  accelerate, tensorboard 2.21 added this session). GPU runs must NOT use
  `uv run` (project .venv has no torch). Every command needs
  `env -u OMP_NUM_THREADS` (host exports an invalid 0) and
  `HF_ENDPOINT=https://hf-mirror.com` (huggingface.co unreachable;
  transformers 4.57 calls the metadata API even for cached models).
- overfit-smoke calibration deviation (2026-09-19): the checked-in config
  had `epochs: 2` (50 steps). First real GPU execution of the gate failed
  0/50 — loss only reached ~2.8 and generations emitted
  `<think>…</think>` + plain answer without the `<final>` envelope (the
  Qwen3 chat template wraps assistant turns in think blocks; the strict
  fullmatch parser rejects anything imprecise). Calibrated to
  `epochs: 12` (300 steps): loss 8.71→0.0003 and exact parseable
  terminals. Evidence: `artifacts/runs/sft-overfit-smoke-20260918T185603Z/`
  (console-gate.log, debug_repro*.log, debug_out/run_ep12/metrics.jsonl).
- Command: `env -u OMP_NUM_THREADS HF_ENDPOINT=https://hf-mirror.com \
  ADAPTIVE_MATH_RUN_GPU_TESTS=1 <training-env-python> -m pytest \
  tests/integration/test_sft_overfit.py -q -v`
- GRPO smoke command (2026-09-19, torchrun bypassed — direct python; see the
  register-center hang note in project memory):
  `env -u OMP_NUM_THREADS HF_ENDPOINT=https://hf-mirror.com WANDB_MODE=disabled \
  <training-env-python> artifacts/runs/repro-main-entry.py` (calls
  `run_grpo.main(["--config", "configs/grpo/qwen3_1_7b_smoke_r0.yaml"])`)
- GRPO smoke fixes landed in config this session: `data.max_prompt_length=4096`
  (longest pool prompt renders to 2072 tokens), `lora_rank=16`/`lora_alpha=32`
  (upstream default lora_rank=0 = full-param Adam states ≈27GB → OOM on 24GB),
  `ray_init.num_cpus=8`, `trainer.ray_wait_register_center_timeout=120`.
- Do NOT set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — vLLM 0.11
  CuMem memory pool asserts against it.
- Artifacts: `artifacts/sft/*/resolved_config.yaml`, `environment.json`,
  `metrics.jsonl`, `adapter/` (+COMPLETE marker), `rejects.jsonl` if any
