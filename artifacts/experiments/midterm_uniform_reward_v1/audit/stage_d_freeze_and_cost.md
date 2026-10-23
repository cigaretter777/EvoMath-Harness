# 阶段 D：冻结清单、实测速率与三组正式训练成本确认

实验：`midterm_uniform_reward_v1`  
冻结时间（UTC）：2026-10-22T04:20:51Z  
本文件是启动三组 50 轮正式训练前的最后一份核对材料。**正式训练尚未启动**；需用户明确确认后，才按 §7 的顺序执行。

## 1. 冻结输入与哈希

完整机器可读清单：`manifests/frozen_inputs.json`。冻结脚本：`scripts/experiments/freeze_midterm_inputs.py`（本次重新执行，8 个配置、8 个数据/调度文件、3 棵源码树全部重新哈希）。

| 对象 | SHA-256（前 16 位） |
|---|---|
| `configs/grpo/midterm_unified_r0.yaml` | `f66605368d8eb1f2` |
| `configs/grpo/midterm_unified_r1.yaml` | `52bb3a4ed3647dd1` |
| `configs/grpo/midterm_unified_r2.yaml` | `087eec9b918ed991` |
| `configs/agent/unified_tool_budget_v1.yaml` | `9e0b8f6e9a420066` |
| `configs/reward/r0_unified.yaml` | `8e6b5c98f7c0f5dc` |
| `configs/reward/r1_unified.yaml` | `35a788d95ff3b666` |
| `configs/reward/r2_unified.yaml` | `5cdf9218d7ace717` |
| `configs/harness/unified_tool_reward_v1.yaml` | `1cf7b851dec161cf` |
| 训练池 `math_midterm_v1_rl_train.jsonl` | `635f9571b056f395` |
| 轮次表 `math_midterm_v1_rl_schedule_50x8.parquet` | `3154e578308910fa` |
| `data/processed/midterm_math_v1/train.parquet` | `574f0f259dd11931` |
| `data/processed/midterm_math_v1/dev.parquet` | `51cb0be79c6bb959` |
| `data/processed/midterm_math_v1/test.parquet` | `42aadb047de53704` |
| `src/adaptive_math`（64 个 `.py`） | `b53c0b334df52024` |
| `scripts/train`（4 个文件） | `a3690ebc9041fbfd` |
| `scripts/data`（14 个文件） | `c214ef0929bb9f9e` |
| SFT `model.safetensors`（3,441,185,608 B） | `047ac5c7eac92bed` |

SFT 检查点：`/root/autodl-tmp/Adaptive-Solver-main-git/artifacts/models/qwen3_1_7b_sft_tir_merged`，10 个文件全部记录字节数与 SHA-256（`config.json`、`tokenizer.json`、`chat_template.jinja`、`generation_config.json` 等）。三组均从该原始 SFT 启动，不读取任何冒烟检查点。

冻结副本已逐字节核对：`artifacts/experiments/midterm_uniform_reward_v1/configs/` 下 8 个文件与仓库源文件 `cmp` 全部一致（OK）。

三组正式配置均写入 `trainer.rollout_data_dir=.../training/<group>/rollout_dump`，保留每轮逐动作生成，用于阶段 F 的奖励复算；该键与冒烟使用同一上游字段，冒烟中已验证可写出 `0.jsonl`、`1.jsonl` 等逐轮文件。

## 2. 环境与测试

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D，**1 张**，24,564 MiB，驱动 595.71.05；实测训练峰值 22.92 GB |
| 主机内存 | 503 GB，训练进程峰值约 127 GB，可用约 402 GB |
| Python 环境 | `/root/autodl-tmp/conda-envs/adaptive-math`（Python 3.12） |
| 关键依赖 | torch 2.8.0、transformers 4.57.3、vllm 0.11.0、ray 2.50.0、peft 0.20.0、accelerate 1.15.0 |
| 上游后端 | verl-agent manifest SHA `20bd331bd…`，滚动补丁记录在每次运行的 `task_runner_patches.json` / `backend_adapter.json` |
| 数据盘 | `/dev/md0` 160 GB，已用 125 GB，可用 36 GB（78%） |

2026-10-22 12:38 启动前复查：`nvidia-smi -L` 只列出 **GPU 0**；显存占用 1 MiB / 24,564 MiB，利用率 0%，温度 35 °C，功率 45 W / 425 W，无计算进程。主机 16 CPU、503 GB 内存（可用 401 GB）。单卡条件不支持三组同时训练，故采用串行；串行只表示执行顺序，不表示权重传递（见 §7 第 1 条）。

测试（最终代码，2026-10-22）：

```bash
PYTHONPATH=src /root/autodl-tmp/conda-envs/adaptive-math/bin/python -m pytest tests -q \
  --deselect tests/unit/eval/test_verify_arm_identity.py::test_source_drift_is_empty_for_a_commit_against_itself
# 711 passed, 26 skipped, 1 deselected, 2 warnings in 52.80s
```

唯一被排除的用例要求仓库是含 `HEAD` 的 git 克隆；本目录不是 git 仓库（`git rev-parse HEAD` 不可用），属环境性跳过，与代码正确性无关。该用例在加 `-x` 的首轮运行中同样只报这一个错误。

## 3. 实测速率（不按显存猜速度）

三组正式运行与冒烟共用同一批次结构：每轮 8 个槽位 × `env.rollout.n=4` = 32 条轨迹，最多 6 个动作回合，生成上限 2048 token。以下均为真实 GPU 更新，非模拟：

| 运行 | 更新轮次 | `timing_s/gen` | `timing_s/update_actor` | `timing_s/step` | 响应长度均值/最大 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| `smoke/grpo_2step_prefixcache_off_v3` | 1 | 36.2 | 115.7 | **181.0** | 142 / 677 | 22.92 GB |
| 同上 | 2 | 29.2 | 116.0 | **191.1** | 100 / 387 | 22.92 GB |
| `smoke/grpo_resume_from_step2` | 3 | 245.6 | 97.8 | **380.5** | 201 / 2048 | 22.92 GB |

- 单轮实测范围 **181.0–380.5 s**，均值 **250.9 s**（4.18 min）。长尾来自单条轨迹触达 2048 token，生成阶段需等待该批最长序列。
- 检查点保存实测 13.2–15.2 s/次（7.9 GB），每 10 轮一次，共 5 次；适配器归档约 1 s/次，可忽略。
- 冷启动门（同一 SFT、同一沙箱、每轨迹最多 6 回合）实测 **80 条轨迹 / 609.9 s = 7.62 s/条**，作为 800 条贪心评测的实测速率基准。

## 4. 时间估计

训练（150 个全局更新）：

| 口径 | 每组 50 轮 | 三组合计 |
|---|---:|---:|
| 按实测最快 181.0 s/轮 | 2.51 h | 7.54 h |
| 按实测均值 250.9 s/轮 | 3.48 h | 10.45 h |
| 按实测最慢 380.5 s/轮 | 5.28 h | 15.85 h |

加上每组 5 次检查点保存（约 75 s）与启动/加载开销（约 2 min/组），训练合计取 **7.7–16.0 h**。

评测（4 组 × 200 题 = 800 条贪心轨迹）：按冷启动实测 **1.69 h**；考虑 HF 贪心逐题请求、4 次模型/适配器加载与长尾，给 2× 余量，取 **1.7–3.4 h**。

总计（含约 10% 排队/日志/分析余量）：

| 口径 | GPU 小时 |
|---|---:|
| 乐观（全按最快 + 评测 1×） | 约 9.5 h |
| 基准（均值 + 评测 1×） | 约 12.5 h |
| 保守（全按最慢 + 评测 2×） | 约 19.5 h |

**建议授权窗口：20 GPU 小时**（单卡 4090D）。计费单价以实例实际账单为准；本报告只给 GPU 小时，不猜单价。达到 20 h、或 §7 的保存前空间检查不通过时暂停并报告，不静默续跑。

## 5. 存储估计与保留方案（GiB 口径，已核实后端轮换顺序）

单位统一为 **GiB**（1024³ 字节）；`df -h`/`du -h` 的 `G` 也是 GiB。实测基准：

| 对象 | 字节 | GiB |
|---|---:|---:|
| 完整 FSDP 检查点（含模型/优化器/extra） | 8,422,827,514 | 7.844 |
| 单个 LoRA 适配器 | 69,782,352 | 0.065 |
| 每 50 轮的 dump+audit+train.log | 约 14,000,000 | 0.013 |
| 当前数据盘可用 | 38,548,885,504 | 35.90 |

保留方案为每组 5 个适配器（第 10/20/30/40/50 轮）加第 50 轮完整检查点。终态细目：

| 项 | 计算 | GiB |
|---|---|---:|
| 3 份第 50 轮完整检查点（新增） | 3 × 7.844 | 23.53 |
| 15 份适配器归档（新增，第 10/20/30/40/50 轮） | 15 × 0.065 | 0.97 |
| 3 组的日志、dump、审计（新增） | 3 × 0.013 | 0.04 |
| **训练新增小计** | | **24.54** |
| 已有实验审计/清单/冒烟产物（现已占用） | 282,370,506 B | 0.26 |
| 全部实验产物合计 | | 24.81 |
| 当前可用 | | 35.90 |
| 训练写完后可用 | 35.90 − 24.54 | **11.35** |
| 安全余量要求 | | 3.00 |
| 余量之上的空间 | | 8.35 |

日志/dump 的 0.013 GiB/组不是拍脑袋的“小于 1 GB”：它来自 2 步真实冒烟实测（审计 JSONL 139,376 B、rollout dump 373,252 B、`train.log` 45,687 B）按 50 步线性外推；即使实际放大 10 倍，三组也只有约 0.4 GiB，仍被 3 GiB 安全余量覆盖。检查点保存为直接写入目标目录，没有额外的整份暂存副本；3 GiB 余量同时覆盖 `torch.save` 的临时文件与评测轨迹写入。

**峰值的关键是后端轮换时机。** 已核对 pinned verl-agent 的 `verl/utils/checkpoint/fsdp_checkpoint_manager.py::save_checkpoint`：它在 `torch.save` **之前**调用 `remove_previous_save_local_path`。`max_actor_ckpt_to_keep=1` 因此不是“保存后删除”，而是**先删除上一份，再写入新一份**，同一组的两个完整检查点不会共存。按该真实顺序，r2 第 50 轮保存后的峰值就是上表的全部实验产物 24.81 GiB，训练写完后仍余 11.35 GiB。

如果后端是“先保存、后删除”（或把 `max_actor_ckpt_to_keep` 设为 2），峰值会多出一份 7.844 GiB，达到 **32.64 GiB**，仅剩 3.26 GiB，扣掉 3 GiB 安全余量只剩 0.26 GiB——这正是评审中“峰值接近 33.5 GB”的口径。本实验**不依赖**这种共存场景，而是依赖上面已核实的“先删后写”顺序；因此不把 `max_actor_ckpt_to_keep` 改为 2，也不启用系统盘暂存。

### 5.1 若将来启用系统盘暂存（当前未启用）

按评审意见，正确顺序必须是：

1. 第 50 轮保存完整结束后，先核验检查点完整性（文件清单、字节数、哈希）；
2. 复制到数据盘**临时目录**，此时不删除任何来源文件；
3. 对副本核对文件清单、字节数、SHA-256，并做一次实际加载验证；
4. 验证通过后，在数据盘内原子改名为正式目录；
5. 最后才删除系统盘上的临时检查点。

注意两个交互点：第一，后端在第 50 轮写入前已经删除了第 40 轮，若要在第 50 轮验证完成前保留第 40 轮，必须在保存前把第 40 轮复制到别处，或把 `max_actor_ckpt_to_keep` 设为 2；第二，必须另行确认 AutoDL 在关机、重启、释放实例时是否保留系统盘。当前方案不依赖系统盘，故不涉及这两个问题。

### 5.2 已完成的清理（不涉及任何正式结果）

1. 可再生缓存：`hf-cache`（6.3 GB）与 `pip-cache`（1.5 GB）；本实验全部从本地合并权重加载，不使用 HF 缓存。
2. 旧项目的**冒烟中间检查点** `Adaptive-Solver-main-git/artifacts/runs/grpo_qwen3_1_7b_smoke_r0/checkpoints/global_step_1`（7.9 GB），经用户按确切路径授权删除；保留该冒烟的最终轮 `global_step_2`、`resolved_config.yaml`、`backend_adapter.json`，`latest_checkpointed_iteration.txt` 仍为 `2`，状态一致。旧项目的 R0/R2/R4 正式检查点全部未动。
3. 更早删除的本实验三份冒烟完整检查点见 `audit/implementation_corrections.md` §6.1。

存储不再是启动正式训练的限制条件。仍需按 §7 的公式在每次保存前检查，而不是只看“磁盘低于 3 GB”。

## 6. 阶段 F 评测器的现状（不影响训练启动）

现有 `scripts/eval/run_model_eval.py` 与 `src/adaptive_math/evaluation/model_eval.py` 绑定的是上一个项目的 Omni-MATH `frozen_eval`、无工具直答和 SFT 适配器清单，不能直接用于本实验（MATH 200 题、最多 6 回合工具交互、四组检查点）。阶段 F 需要新增评测运行器，复用本实验的 `AgentLoop`、`ProductMathEnv`/`OfflineMathEnv`、沙箱工具与 PEFT 适配器加载，并在评测前把运行器源码哈希写入评测清单。

该缺口不改变训练输入、不影响阶段 E 启动；本次不在训练前临时改动训练代码。训练启动后、进入阶段 F 前完成评测器及其测试。

## 7. 执行顺序与中断策略

1. **r0 → r1 → r2 仅表示执行顺序，不表示权重传递。** 三组分别从**同一个原始 SFT 检查点**初始化（`resume_mode=disable`），各自使用独立的优化器、训练状态与 `training/<group>` 输出目录；不跨组续训、不共享任何冒烟权重。当前只有 1 张 RTX 4090 D（24,564 MiB，冒烟峰值约 22.9 GB），不足以同时运行三组，故串行；若将来有三张资源条件一致的 GPU，可以并行，但必须保持工具执行资源与计时条件可比。
2. 每组 50 个全局更新，`save_freq=10`，`max_actor_ckpt_to_keep=1`，适配器归档到 `training/<group>/adapters/step_NN`；启动命令设置 `ADAPTIVE_MATH_AUDIT_LOG=training/<group>/rollout_audit.jsonl`，保留逐组展开审计（只追加日志，不改变训练行为）。
3. 中断与风险窗口：pinned 后端在**每次**保存时先删除上一份完整检查点，再写入新一份。因此每次保存（第 10/20/30/40/50 轮，三组共 15 次）都存在一个短暂的“该组无完整检查点”窗口，而不只是第 40–50 轮。窗口内若进程或机器中断，该组只能从原始 SFT 重跑；适配器和日志不含优化器/调度器状态，不能替代完整检查点。恢复时优先用 `global_step_N` 完整状态；若缺失则从 SFT 重跑该组，旧残缺目录保留但不并入主结果。当前方案不使用系统盘暂存，因此不依赖 AutoDL 在关机/重启/释放实例时是否保留系统盘；若将来启用暂存，风险窗口从该组首次保存一直延伸到副本验证完成，且必须先确认平台持久化行为。
4. 每次保存前检查（不能只看剩余是否低于 3 GB）：`free_now + rotated >= expected_new + reserve`。`rotated` 为本组将被后端删除的上一份完整检查点（首次保存为 0），`expected_new` 取 7.9 GiB，`reserve` 取 3 GiB。按当前实测，r2 第 50 轮保存前约 `11.35 + 7.844 = 19.19 GiB`，要求为 `7.844 + 3 = 10.84 GiB`，余量 8.35 GiB。任一次不满足即在保存前暂停并报告，不静默续跑。
5. 暂停条件：NaN/Inf、组展开异常、沙箱大面积失联、第 4 条保存前检查不通过、累计超过 20 GPU 小时、峰值显存 OOM 连续两次。
6. 主检查点固定为第 50 轮；评测阶段不按结果挑选轮次。

## 8. 待确认

- [ ] 用户确认按本文件的顺序与 20 GPU 小时上限启动三组正式训练。
- [x] 存储方案：已清理可再生缓存并经用户授权删除旧冒烟中间检查点，三组检查点直接写数据盘（§5）；`frozen_inputs.json` 不受影响。
- [x] §5 已按评审四点修正：GiB 统一口径并逐项列出检查点/适配器/日志/已有产物；如实记录 pinned 后端“先删后写”的轮换顺序与对应的 24.81 GiB 峰值；补全系统盘暂存的正确复制顺序（当前未启用）；把风险窗口扩展到每次保存；加入每次保存前的空间不等式检查。训练写完后预计剩余 11.35 GiB。
- [ ] 启动前再次核对 `frozen_inputs.json` 未被改动，并记录启动时刻与 GPU 状态。
