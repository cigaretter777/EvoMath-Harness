# 50 → 100 步续训机制只读核对

- 核对日期：2026-10-23
- 状态：**只读核对完成**。未启动 A/B、未启动任何续训、未执行 R2 计时诊断、未修改奖励/数据/任务集/训练目标、未删除任何产物。
- 对象：`midterm_uniform_reward_v1` 的 R0/R1/R2 三臂 `global_step_50` 检查点，以及运行根目录当前源码。
- 结论先行：**机械上可以从 `global_step_50` 恢复并继续到 100 步；但这不是“纯步数延长”。** 第 1–50 步使用的是修复前验证器，第 51–100 步将使用修复后验证器；此外还有 vLLM 采样 RNG 重启、R2 工具计时抖动、审计 ID 重复、数据第二轮重复等语义差异。应称为“验证器变更后的续训”，不能当作同一协议的 100 步曲线直接解释。

---

## 1. 证据与边界

### 1.1 运行树与固定版本

| 项 | 值 |
|---|---|
| 运行根目录 | `/root/autodl-tmp/BudgetConstrainedReasoningMathAgent` |
| 训练入口 | `scripts/train/run_grpo.py` |
| pin 定的 verl-agent | `20bd331bdbc9026a5668e11362178e10ab7400c8` |
| R2 训练配置 | `configs/grpo/midterm_unified_r2.yaml` |
| R2 奖励配置 | `configs/reward/r2_unified.yaml` |
| 预算配置 | `configs/agent/unified_tool_budget_v1.yaml` |
| 冻结清单 | `artifacts/experiments/midterm_uniform_reward_v1/manifests/frozen_inputs.json` |
| R2 检查点 | `artifacts/experiments/midterm_uniform_reward_v1/training/r2/checkpoints/global_step_50` |

运行时使用的是运行根目录的 `src`（启动器把 `PYTHONPATH` 指向 `BudgetConstrainedReasoningMathAgent/src`）；`/root/autodl-tmp/Adaptive-Solver-main-git` 是另一个 git 工作树/模型镜像，不是本次续训的运行树。不要把两个目录的 `src` 混用。

### 1.2 源码树差异：只有 `verifier/symbolic.py`

对运行根目录的 `src/adaptive_math` 按冻结脚本的 canonical tree hash 规则重算：

| 树 | `src/adaptive_math` 文件数 | canonical SHA-256 |
|---|---:|---|
| 训练时冻结树（manifest） | 64 | `b53c0b334df52024d4f5b9e9cb2f9867b258d279e3f5d6d6dc2e9af4311430d5` |
| 当前运行树（已修复验证器） | 64 | `24cb484ea926b843be098e22c51ea464a483d705cc17d3e74aab4c7ef795488b` |
| 当前树 + 换回 pre-fix `symbolic.py` | 64 | `b53c0b334df52024d4f5b9e9cb2f9867b258d279e3f5d6d6dc2e9af4311430d5` |

也就是说：**在 `src/adaptive_math` 的 64 个 `.py` 中，当前运行树与训练冻结树的唯一差异是 `verifier/symbolic.py`。** 换回 pre-fix 文件后，canonical tree hash 与 manifest 精确一致。

同时核对：

| 源码子树 | 当前值 | 冻结值 | 是否一致 |
|---|---|---|---|
| `scripts/train` | `a3690ebc9041fbfd658432037afcbda490fbe164cf14dafde665125cbb3ef3d6` | 同左 | 是 |
| `scripts/data` | `c214ef0929bb9f9e0deb470e8a7506b684d5ea2d8d98bec9f6628f15000f0cc2` | 同左 | 是 |

因此，续训的源码语义差异可以归因到验证器模块，而不是训练脚本或数据处理脚本。

### 1.3 验证器模块哈希

| 版本 | SHA-256 |
|---|---|
| 训练时 pre-fix | `f4f21ac39e66ee21ae8fea4625b9a9522a7e0f92747b6ac3bc1fef393a6b2194` |
| 当前 fixed | `1de775613d9cb3fe8a9e0c91eacb59b239ac3609ee7c15265f2b579f83088fd7` |

修复说明与回归证据见：
`artifacts/experiments/midterm_uniform_reward_v1/audit/verifier_bug_20261023/README.md`。

---

## 2. 检查点清单

`global_step_50` 是完整 FSDP 检查点，不是只有 adapter：

| 文件 | 字节数 | 说明 |
|---|---:|---|
| `actor/model_world_size_1_rank_0.pt` | 8,197,332,351 | FSDP 模型分片（含 LoRA 权重） |
| `actor/optim_world_size_1_rank_0.pt` | 139,806,363 | AdamW 状态 |
| `actor/extra_state_world_size_1_rank_0.pt` | 15,077 | LR scheduler + RNG |
| `actor/lora_adapter/adapter_model.safetensors` | 69,782,352 | 便于单独取 LoRA 的副本 |
| `data.pt` | 1,947 | torchdata 状态 |
| 目录总计 | 7.9G | `du -sh` |

`adapter_config.json`：`r=16`，`lora_alpha=32`，target modules 为 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`。

`extra_state` 顶层键只有 `lr_scheduler` 与 `rng`：

```text
lr_scheduler:
  base_lrs: [1e-06]
  last_epoch: 50
  _step_count: 51
  _is_initial: false
  _last_lr: [1e-06]
  lr_lambdas: [null]
rng:
  cpu, numpy, random, cuda 四类状态
```

`data.pt` 的关键内容：

```json
{
  "_snapshot": {
    "_snapshot_step": 50,
    "_main_snapshot": {
      "_num_workers": 8,
      "_sampler_iter_state": {"samples_yielded": 400},
      "_index_sampler_state": null,
      "_sampler_iter_yielded": 50,
      "_base_seed": 2822172502284261475
    },
    "_worker_snapshots": {"worker_0": {"dataset_state": null, "fetcher_state": null}, "...": "..."}
  },
  "_steps_since_snapshot": 0,
  "_iterator_finished": false
}
```

`data.pt` 的 `samples_yielded=400` 是**样本数**；`_sampler_iter_yielded=50` 是**batch 数**。8 个 worker 的 dataset/fetcher 状态均为 `null`，说明这里没有需要恢复的 worker 内部游标。

磁盘：`/root/autodl-tmp` 当前 `160G total / 150G used / 11G avail`。单个完整检查点 7.9G，**只够再放一个**，余量约 3G，见 §8 的风险说明。

---

## 3. 学习率调度

### 3.1 配置与状态

Resolved 配置（`training/r2/resolved_config.yaml`）中 actor optimizer：

```text
lr: 1.0e-06
lr_warmup_steps: -1
lr_warmup_steps_ratio: 0.0
warmup_style: constant
```

`lr_warmup_steps=-1` 表示走比例；比例为 `0.0`，所以 `num_warmup_steps=0`。上游 `fsdp_workers.py` 因此构造：

```python
get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=0)
```

该 scheduler 的 lambda 在 `current_step >= 0` 时恒返回 `1.0`，所以 LR 始终是 `base_lr = 1e-6`，没有 warmup、没有 decay。

检查点中的 `last_epoch=50`、`_step_count=51`、`_last_lr=[1e-6]` 与此一致：

- 构造时 `_initial_step()` 把 `last_epoch` 从 `-1` 推到 `0`，`_step_count` 变为 1；
- 每个 global update 在 `update_actor` 末尾调用一次 `actor_lr_scheduler.step()`；
- 50 个 update 后 `last_epoch=50`，`_step_count=51`；
- 下一个 update 调用 `step()` 后 `last_epoch=51`，LR 仍然是 `1e-6`。

### 3.2 把 `total_training_steps` 从 50 改成 100 的影响

**对 LR 没有影响。** 这里 `lr_warmup_steps_ratio=0.0`，`total_training_steps` 只会在 warmup 比例为正值时影响 warmup 步数；本实验的 constant schedule 不依赖总步数。

但要注意：如果续训配置同时改了 `lr`、`lr_warmup_steps_ratio`、`warmup_style` 或 `base_lrs`，那就不是同一优化器轨迹，不能称为续训。续训配置必须保持这些字段与 50 步运行一致。

### 3.3 恢复行为

`FSDPCheckpointManager.load_checkpoint` 会：

1. 加载模型分片；
2. 加载 optimizer state；
3. 调用 `load_rng_state(extra_state["rng"])`；
4. 调用 `lr_scheduler.load_state_dict(extra_state["lr_scheduler"])`。

所以 scheduler 恢复到 `last_epoch=50`，不是从 0 重新开始。下一次 `step()` 从 51 开始，LR 仍为 `1e-6`。

---

## 4. 优化器恢复

### 4.1 状态内容

`optim_world_size_1_rank_0.pt`：

- `param_groups`：421 个 parameter id；`lr=1e-6`；`betas=(0.9, 0.999)`；`eps=1e-8`；`weight_decay=0.01`；`initial_lr=1e-6`；`decoupled_weight_decay=True`。
- `state`：392 个参数有 Adam 状态；每个状态含 `exp_avg`、`exp_avg_sq`，`step` 均为该臂的计数器值。
- 不是所有 421 个参数都有状态，这是正常的：没有梯度/从未更新的参数不会产生 `exp_avg`/`exp_avg_sq`。

三臂的 optimizer `step`：

| 臂 | active-turn 行数（审计重算） | `sum(ceil(active_rows / 8))` | 检查点 `optimizer step` |
|---|---:|---:|---:|
| R0 | 3487 | 456 | 456.0 |
| R1 | 3277 | 431 | 431.0 |
| R2 | 3245 | 422 | 422.0 |

这张表是精确对上的。解释：

- `actor_rollout_ref.actor.ppo_epochs=1`；
- `ppo_mini_batch_size=8`；
- `update_policy` 对进入 actor 的 active-turn 行按 8 行分块，每个 mini-batch 调用一次 `optimizer.step()`；
- 因此 optimizer step 是**按 mini-batch 计数的数据依赖计数器**，不是 global update 计数器；
- R2 的 422 = 28 轮 × 8 个 mini-batch + 22 轮 × 9 个 mini-batch（因为每轮 active 行数在 63–68 之间，分块边界导致 8 或 9 块）；
- R0/R1 的 456/431 同理，分别对应 8–11 块/轮和 8–10 块/轮。

所以不能把 422 误读为“之前跑过 422 个 global step”，也不能在恢复时把 optimizer step 重置为 0。恢复后第一次 `optimizer.step()` 会把它推到 423（R0 457、R1 432），动量继续累积。

### 4.2 恢复顺序与不可恢复项

已恢复：

| 状态 | 是否恢复 | 来源 |
|---|---|---|
| FSDP 模型/LoRA 权重 | 是 | `actor/model_*.pt` |
| AdamW `exp_avg`/`exp_avg_sq`/`step` | 是 | `actor/optim_*.pt` |
| LR scheduler | 是 | `extra_state["lr_scheduler"]` |
| torch/numpy/random/cuda RNG | 是 | `extra_state["rng"]` |
| dataloader 游标 | 是 | `data.pt` |

未恢复/不可恢复：

| 状态 | 说明 |
|---|---|
| vLLM 采样 RNG | 每次启动用 `seed=config.get("seed", 0)` 新建 engine；不在 `extra_state` 中。第 51 步的采样流不是第 50 步之前流的延续。 |
| SandboxFusion 服务状态/计时抖动 | 外部服务，不在检查点内；R2 的 `tool_seconds` 每次都重新测量。 |
| 审计 ID 的跨段唯一性 | `global_update/group_id/prompt_slot_id/trace_id` 来自 parquet 行，第二轮会重复。 |
| critic | 本实验 GRPO 不用 critic；无 critic 检查点。 |
| KL 参考模型 | `use_kl_loss=false`、`use_kl_in_reward=false`；无参考模型状态需要恢复。 |

### 4.3 实验条件与局限：三臂实际优化器更新次数

三臂名义上都是 50 个 global update，但 checkpoint 中的 optimizer `step` 是 **R0 456、R1 431、R2 422**。它按 PPO mini-batch 计数，精确等于 `sum(ceil(active_rows / 8))`，不是 global update 数，也不是“都跑了 50 步”。

这意味着：

- 三臂的优化器更新次数并未对齐；R2 比 R0 少 34 次（456 → 422）。
- 任何跨臂比较都不能只归因于 R0/R1/R2 的奖励差异；优化器更新次数是同一实验条件的一部分。
- 续训时 optimizer state 按原值恢复；R2 下一次更新变为 423，R0/R1 变为 457/432。不能把 422 重置为 0。
- 在论文/报告的实验条件与局限中必须写明 456/431/422，而不是只写“50 步”。

---

## 5. Dataloader 游标

### 5.1 数据结构

- `data.train_files` 指向 400 行的 `math_midterm_v1_rl_schedule_50x8.parquet`；
- `data.train_batch_size=8`；
- 400 / 8 = 50 个 batch = 1 个 epoch；
- `data.shuffle=false`，`create_rl_sampler` 选择 `SequentialSampler`；
- `StatefulDataLoader`（torchdata 0.11.0）用 8 个 worker，`drop_last=True`。

`data.pt` 保存的是第 50 个 batch 之后的状态：`samples_yielded=400`，即 epoch 0 的 400 个样本已经全部产出；`_sampler_iter_yielded=50`，即 50 个 batch。

### 5.2 恢复后的实际行为

`train_dataloader.load_state_dict(data.pt)` 把 `_main_snapshot` 设为下一次迭代的恢复点。第一次 `iter()` 会恢复 `_sampler_iter_yielded=50` 和 `samples_yielded=400` 的 batch sampler 状态；由于数据集正好 400 行，epoch 0 已经没有剩余样本。

`fit()` 的结构是：

```python
self.global_steps = 0
self._load_checkpoint()          # -> self.global_steps = 50
progress_bar = tqdm(total=..., initial=50)
self.global_steps += 1           # -> 51

for epoch in range(self.config.trainer.total_epochs):   # 默认 30
    for batch_dict in self.train_dataloader:
        ...
```

因此：

1. 恢复后的第一个外层 epoch 是**空迭代**（epoch 0 已耗尽），不产生 update；
2. 第二个外层 epoch 重新从 row 0 开始，第一个 batch 是 rows 0–7；
3. 第 51 个 global update 使用 rows 0–7，而不是 rows 400–407（后者不存在）；
4. 第 51–100 步是**第二轮完整的 400 题**，顺序与第 1–50 步完全相同。

`total_epochs` 默认为 30，足够覆盖“空的第一轮 + 第二轮 50 步”。不要把 `total_epochs` 覆盖成 1，否则恢复后可能一个 update 都不发生。

### 5.3 必须设置的 `total_training_steps`

`fit()` 里：

```python
is_last_step = self.global_steps >= self.total_training_steps
```

`total_training_steps` 默认由 `len(train_dataloader) * total_epochs` 推出，但本配置会显式覆盖它。若续训仍保留 `trainer.total_training_steps=50`：

- `_load_checkpoint` 把 `global_steps` 设为 50；
- 进入循环后先 `global_steps += 1` → 51；
- 第一个可用的 batch（第二轮 row 0–7）执行时 `is_last_step = 51 >= 50` 为真；
- 执行完这一个 update 后保存并 `return`。

所以**必须把 `trainer.total_training_steps` 改成 100**，否则只会多跑 1 个 update 就停。

### 5.4 恢复配置

`_load_checkpoint` 对 `resume_mode=resume_path` 的要求：

- `trainer.resume_mode=resume_path`；
- `trainer.resume_from_path` 必须是字符串，且路径中包含 `global_step_`；
- 传 `.../training/r2/checkpoints/global_step_50`；
- 它会加载该目录下的 `actor` 和 `data.pt`；
- 如果 `data.pt` 不存在，只打印 warning 并从 scratch 开始 dataloader——这会破坏题目顺序，必须确保文件存在。

---

## 6. 实际题目顺序

### 6.1 行号映射

parquet 的 `extra_info` 中：

- `global_update` 0..49；
- `slot_index` 0..7；
- `schedule_index` 0..399；
- row 0 是 `math:bd8f844e577125925d85`，`global_update=0`，`slot_index=0`；
- row 8 是下一轮的 slot 0；
- rows 392–399 是 `global_update=49` 的 8 个 slot。

trainer 的第 `s` 个 global update（1-based）对应：

```text
row_start = 8 * ((s - 1) mod 50)
row_end   = row_start + 7
global_update = (s - 1) mod 50
slot_index    = 0..7
```

| trainer step | epoch | parquet rows | parquet `global_update` |
|---:|---:|---:|---:|
| 1 | 1 | 0–7 | 0 |
| 2 | 1 | 8–15 | 1 |
| 49 | 1 | 384–391 | 48 |
| 50 | 1 | 392–399 | 49 |
| 51 | 2 | 0–7 | 0 |
| 52 | 2 | 8–15 | 1 |
| 100 | 2 | 392–399 | 49 |

### 6.2 已完成的 50 步核对

对 `rollout_audit.jsonl` 的 reset 事件做只读重放：

- 每个 global update 有 32 条 reset = 8 个 task × `env.rollout.n=4`；
- 按 reset 顺序去重 task 后，50 个 update 的 8 题序列与 parquet rows 0–399 **全部一致，0 个 mismatch**；
- 总计 1600 条 reset；
- `sample_index` 为 0..3；
- `group_id = g{global_update}:{task_id}#{task_slot}`，`task_slot=0..7`；
- `prompt_slot_id = g{global_update}:s{slot_index}`；
- `trace_id = rollout:{task_id}:{sample_index}`。

这说明第 1–50 步的实际题目顺序确实是 parquet 顺序，不是随机顺序，也不是 dataloader 重新洗牌。

### 6.3 第 51–100 步的 ID 重复

因为第 51–100 步重复 rows 0–399：

- `global_update` 会再次出现 0..49；
- `group_id`、`prompt_slot_id` 会与第 1–50 步完全相同；
- `trace_id` 只含 `task_id` 与 `sample_index`，也会重复；
- trainer 自己的 `global_step` 是 51..100，但**审计记录里没有这个 trainer global step 字段**，只有 parquet 的 `global_update`。

因此，如果要把 50 步和 51–100 步的审计合并分析：

- 必须按文件/目录分段（例如 `segment=1-50` 与 `segment=51-100`），或额外写入 trainer `global_step`；
- 不能只用 `global_update` 或 `group_id` 去重；
- 否则第 51 步会被误认为第 1 步的重复记录，或被错误合并。

---

## 7. 为什么这不是“纯步数延长”

### 7.1 验证器协议不同

| 阶段 | 验证器 | 影响 |
|---|---|---|
| 第 1–50 步 | pre-fix `f4f21ac...` | 训练时的真实判定 |
| 第 51–100 步（若续训） | fixed `1de775...` | 续训时的真实判定 |

已完成的 50 步中，训练审计记录到 9 个 `internal_error`：

| 臂 | `internal_error` 步数 | 修复后语义正确的数量 |
|---|---:|---:|
| R0 | 2 | 0 |
| R1 | 4 | 2 |
| R2 | 3 | 0 |

这 9 个里只有 2 个（都在 R1）在修复后是正确；其余 7 个仍是错误。训练没有回溯重跑，50 步结果保留为中期结果。

但这仍然意味着：

- 第 1–50 步的 reward 是在 pre-fix 判定下产生的；
- 第 51–100 步的 reward 是在 fixed 判定下产生的；
- 即使奖励公式、模型、数据完全不变，**验证器行为不同**；
- 未来步中可能遇到 50 步里没出现过的集合/元组/关系式边界，fixed 版会给出不同 verdict；
- 因此 100 步曲线是“两种验证器协议的拼接”，不是单一协议的 100 步延长。

### 7.2 vLLM 采样 RNG 重启

vLLM engine 在每次启动时用配置的 seed 新建；采样 RNG 不在 FSDP `extra_state` 中。恢复后第 51 步的采样流不是“如果一直跑到 51 步”的同一流。即使模型权重、优化器、scheduler、dataloader 都恢复了，**生成轨迹的随机序列仍然不连续**。

### 7.3 R2 的计时不可复现

R2 的 `time_cost` 来自 `tool_seconds`（当前数据中是 SandboxFusion 的 `execution_time`）。同一程序在不同时间执行会有抖动；这不是模型或 seed 能控制的。第 51–100 步的 R2 奖励因此不是权重和数据的确定性函数。R2 诊断方案见同目录的 `r2_cost_signal_diagnostic_plan.md`。

### 7.4 数据第二轮重复

第 51–100 步不是 400 个新题，而是同一 400 题的第二轮。它是“在相同任务集上多优化 50 步”，不是“更多唯一数据”。如果研究问题是“更多步数是否带来更好泛化”，这一点必须在解释时写明。

### 7.5 审计 ID 冲突

见 §6.3。`global_update`、`group_id`、`prompt_slot_id`、`trace_id` 都会重复；合并日志时必须加段标识。

### 7.6 结论

可以做的准确表述是：

> 从 R0/R1/R2 各自的 `global_step_50` 机械恢复，使用修复后验证器，在同一 400 题上再训练 50 个 global update。

不能做的表述是：

> 把 50 步和 100 步拼成同一条“纯步数延长”曲线，并据此声称“多训练 50 步的效果”。

如果论文/报告需要干净的步数效应，优先考虑：从同一 SFT、用统一修复版本、从头训练 100 步；把当前 50 步预实验保留为中期结果。续训可作为工程 smoke 或补充证据，但必须标注验证器变更和 RNG 重启。

---

## 8. 操作风险与续训配置草案（未执行）

### 8.1 启动器限制

- `scripts/train/run_grpo.py` **没有 `--resume` 参数**；恢复只能通过 YAML 的 `upstream_overrides` 设置 `trainer.resume_mode` / `trainer.resume_from_path`。
- `run_grpo.py` 会在 `config.output_dir` 写 `resolved_config.yaml`、`backend_adapter.json`、`task_runner_patches.json`。如果复用 `training/r2` 作为 `output_dir`，会覆盖 50 步运行的元数据。**续训必须用新的 `output_dir`**。
- `scripts/experiments/watchdog_midterm_training.py` 会拒绝非空输出目录，并按 R0→R1→R2 串行跑三臂；不适合单臂续训。
- `scripts/cloud/resume_latest.sh` 是通用 helper，要求 `COMPLETE` 标记和 `ADAPTIVE_MATH_RESUME_COMMAND`；`run_grpo.py` 不使用它，不能直接套用。
- `validate_inputs` 只校验 `task_pool_path`（jsonl）的 SHA-256；**不校验 parquet schedule、SFT、reward/budget 配置**。续训前要手动核对：
  - schedule parquet `3154e578308910fa837bb965955bc867f5923f3e52a838de6a4cc34f53b17ca0`
  - SFT `model.safetensors` `047ac5c7eac92bed957891162bc8a693524c6f42dfb97618093cdcc475432077`
  - R2 reward `5cdf9218d7ace71781aea5e263277edde846d453db53b98661c0c08ac4e99a87`
  - budget `9e0b8f6e9a420066500799f35e2724cbc79a991e2162505c5806e6cc41474817`
  - R2 config `087eec9b918ed991283c494c5ab73ab505f42f11f733aa5301a651fdabbcdd79`

### 8.2 检查点轮转与磁盘

- `trainer.max_actor_ckpt_to_keep=1`。
- `FSDPCheckpointManager.previous_saved_paths` 在**新进程**中是空列表；它不会把启动前已存在的 `global_step_50` 当作自己的上一份检查点。因此第一次保存时不会主动删除 `global_step_50`；之后轮转只删本进程保存的上一份。
- 但磁盘只有 11G 空闲，一个完整检查点 7.9G；`global_step_50` 已占 7.9G。写新检查点时只剩约 3G 余量，风险高。
- 建议续训配置用 `trainer.save_freq=50`，只在第 100 步保存一次，减少中间检查点写入；不要自行删除 `global_step_50` 或其他产物。若空间不足，应先停下由用户决定。
- 启动前记录 `global_step_50` 的文件大小/哈希；启动后确认它仍然存在且未被改写。

### 8.3 配置草案（仅说明，不执行）

复制 `configs/grpo/midterm_unified_r2.yaml` 为新的续训配置，至少改：

```yaml
output_dir: /root/autodl-tmp/BudgetConstrainedReasoningMathAgent/artifacts/experiments/midterm_uniform_reward_v1/training/r2_continue100_metadata

upstream_overrides:
  # ... 保留原有全部 overrides ...
  - trainer.total_training_steps=100
  - trainer.resume_mode=resume_path
  - trainer.resume_from_path=/root/autodl-tmp/BudgetConstrainedReasoningMathAgent/artifacts/experiments/midterm_uniform_reward_v1/training/r2/checkpoints/global_step_50
  - trainer.save_freq=50
  - trainer.default_local_dir=/root/autodl-tmp/BudgetConstrainedReasoningMathAgent/artifacts/experiments/midterm_uniform_reward_v1/training/r2/checkpoints
```

保持不变的字段：

- `actor_rollout_ref.model.path`（同一 SFT）；
- `data.train_files`（同一 parquet）；
- `data.train_batch_size=8`、`data.shuffle=false`、`data.seed=20261022`；
- `actor_rollout_ref.actor.ppo_mini_batch_size=8`、`ppo_epochs=1`；
- `env.rollout.n=4`、`env.max_steps=6`；
- `reward_path`、`budget_path`；
- `algorithm.adv_estimator=grpo`；
- `trainer.total_epochs`（不要改成 1）；
- `trainer.max_actor_ckpt_to_keep=1`。

不要做：

- 不要复用 `training/r2` 作为 `output_dir`（会覆盖 50 步元数据）；
- 不要把 `total_training_steps` 留在 50（只会多跑 1 步）；
- 不要改 reward/budget/task pool/schedule；
- 不要用 `watchdog_midterm_training.py`；
- 不要在无 GPU 的当前机器上直接启动；
- 不要在 SandboxFusion 隧道未验证健康时启动（训练会调用 python 工具）。

### 8.4 启动前检查清单

1. `global_step_50` 的 `actor/model_*.pt`、`optim_*.pt`、`extra_state_*.pt`、`data.pt` 均存在；
2. `data.pt` 的 `samples_yielded=400`、`_sampler_iter_yielded=50`；
3. `extra_state` 的 `last_epoch=50`、`base_lrs=[1e-6]`；
4. optimizer step = 456/431/422（按臂）；
5. parquet/SFT/config 哈希与 manifest 一致；
6. `ADAPTIVE_MATH_SANDBOX_URL` 可达，`/v1/ping` 成功；
7. GPU 可见，且 verl-agent SHA 为 `20bd331b...`；
8. 新 `output_dir` 为空；
9. 磁盘预留至少 8G，且明确 `global_step_50` 不会被删除；
10. 启动后先确认第一个 update 的 `global_step=51`、`global_update=0`、题目 rows 0–7。

---

## 9. 建议

1. **机械可行性：是。** 三个检查点都包含模型、优化器、scheduler、RNG、dataloader，按上述配置可以恢复。
2. **科学语义：不是纯步数延长。** 主要因为验证器从 pre-fix 变为 fixed；其次是 vLLM 采样 RNG 重启、R2 计时抖动、审计 ID 重复、第二轮数据重复。
3. **保留当前 50 步预实验为中期结果。** 不回溯重跑、不修改其检查点/日志/报告。
4. **R2 诊断优先。** 先按 `r2_cost_signal_diagnostic_plan.md` 判断时间项是稳定程序差异还是执行噪声；诊断结论出来前，不做“为追正结果”而改奖励系数或任务集的决定。
5. **如果目标是干净的 100 步比较：** 优先从同一 SFT、用统一修复版本从头训练 100 步；续训更适合工程验证或补充材料，并需明确标注“验证器变更后的续训”。
