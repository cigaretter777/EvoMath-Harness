# R2 续训 50→100 步：性能探索报告

| 项目 | 内容 |
|---|---|
| 实验标识 | `midterm_uniform_reward_v1` / `r2_continue100` |
| 报告日期 | 2026-10-24 |
| 训练窗口 | 2026-10-23 21:51:57 – 2026-10-24 01:22:41（CST，约 3 h 31 min） |
| 评测窗口 | 2026-10-24 01:25 – 03:14（CST，约 1 h 49 min） |
| 续训起点 | `training/r2/checkpoints/global_step_50`（原检查点未改动） |
| 训练终点 | `global_step_100` |
| 训练输出 | `artifacts/experiments/midterm_uniform_reward_v1/training/r2_continue100/` |
| 评测输出 | `artifacts/experiments/midterm_uniform_reward_v1/evaluation/r2_continuation_20261023/` |
| 报告状态 | 训练、四组评测、配对分析均已完成；本报告不启动任何新训练或评测 |

> **结论先行。** 在本次条件下，R2 从 50 步续训到 75/100 步**没有提供可靠证据**表明正确率提升或工具成本下降。主目标（200 题冻结测试集、贪心评测）正确率为 SFT 55.0%、R2-50 54.0%、R2-75 56.0%、R2-100 54.0%；三个预设比较（R2-75−R2-50、R2-100−R2-50、R2-100−R2-75）经 Holm 校正后均不显著（校正后 p=0.867、1.000、0.867）。工具调用次数在所有组都饱和于 0/1 次/题；工具时间中位数四组几乎相同（0.2659–0.2673 s/题），均值差异由个别 ~29–30 s 慢调用造成，剔除 ≥5 s 调用后四组均为约 0.315–0.322 s/题。R2-100 的非法动作率升至 6.15%（R2-50 1.98%、R2-75 0.75%），属于协议行为漂移，未转化为正确率收益。
>
> **本次不是单纯的“增加步数”消融。** 第 1–50 步使用修复前验证器训练，第 51–100 步使用修复后验证器训练；评测四组统一使用修复后验证器。第 51–100 步还复用了同一 400 题调度表的第二个 epoch，并非新数据。因此本报告**不能**用来证明“单纯增加步数有效”，也不能把任何差异单独归因于步数。

---

## 1. 本次做了什么

### 1.1 续训设置

- 仅续训 R2，不训练 R0/R1。
- 从原 R2 的 `global_step_50` 恢复：`trainer.resume_mode=resume_path`，`trainer.resume_from_path=.../training/r2/checkpoints/global_step_50`，`trainer.total_training_steps=100`。
- 新配置：`configs/grpo/midterm_unified_r2_continue100.yaml`。与冻结的 `configs/grpo/midterm_unified_r2.yaml` 相比，仅以下字段不同：
  - `output_dir`、`trainer.experiment_name`、`trainer.default_local_dir`、`trainer.rollout_data_dir`；
  - `trainer.total_training_steps`（50→100）、`trainer.save_freq`（10→25）；
  - `trainer.resume_mode` / `trainer.resume_from_path`。
  - 奖励配置、训练数据、prompt、解析规则、工具预算与 GRPO 超参数逐项相同（已在启动前 diff 核对）。
- 续训日志确认：加载 `global_step_50` 的 model/optimizer/extra_state； scheduler `last_epoch=50`；数据游标 `samples_yielded=400`；第 51 步从调度表第二遍第 0 行开始（`training/epoch=1.000`）。

### 1.2 验证器变化（必须披露）

| 阶段 | 验证器 | SHA-256 |
|---|---|---|
| 原 R2 第 1–50 步训练 | 修复前 | `f4f21ac...`（原树） |
| 第 51–100 步续训 | 修复后 | `1de775613d9cb3fe8a9e0c91eacb59b239ac3609ee7c15265f2b579f83088fd7` |
| 本次四组评测 | 修复后（同一份） | 同上 |

修复内容是集合/区间/元组答案的成员比较崩溃（`Symbol - Interval` 等）。因此本次续训同时改变了“训练步数”和“训练时验证器”，两者无法分离；本报告不把结果解释为验证器修复的效应，也不解释为步数的纯效应。

### 1.3 数据与随机性

- 训练调度仍是 `math_midterm_v1_rl_schedule_50x8.parquet` 的同一 400 行；第 51–100 步是第二个 epoch，不是新数据。
- 恢复时 vLLM 采样 RNG、优化器/调度器与 dataloader 状态从 step 50 重新接续；工具时间存在抖动。
- 单种子（`data.seed=20261022`），未换种子、未调奖励、未延长到 100 步以上。

### 1.4 检查点与产物保留

| 对象 | 状态 |
|---|---|
| 原 R2-50 完整检查点 | **未改动**；`latest_checkpointed_iteration=50`，adapter SHA `12744a0aede8feeb102913dea143a6d760ce4adbfd8de744a7eadb482a7e01e3` |
| R2-75 LoRA adapter | 保留于 `training/r2_continue100/adapters/step_75/`；SHA `c68b05940044f4060f6643fc9865962a9a96b629305a9fd5aa070c053c122ac5` |
| R2-75 完整 FSDP 检查点 | 训练器在写 step 100 前按 `max_actor_ckpt_to_keep=1` 自动轮转；原目录只剩 `data.pt`。轮转前已做逐字节备份：`/root/r2_continue100_step75_full/global_step_75`（model SHA `f6a6c4fe87d731ffaf7589b57fc8a8c23c1983d4bbdc5fdc306624ec14cd5b36`，全部 15 个文件校验一致） |
| R2-100 完整检查点 | `training/r2_continue100/checkpoints/global_step_100/`；model SHA `e1b2412a71a07dd950746344e0ada9250db362d376a740b84de4ef758c923b3e`，optim SHA `e8fc18cb4eaeca677e9fe999fb7e5da2b276e96ce8e43a69aac50177e491f54b` |
| R2-100 LoRA adapter | `training/r2_continue100/adapters/step_100/`；SHA `c502c4ffbe3571375b6340158dfd0eeaf750a6eaf8c9dec2c64cff30b50e9ebe` |

> 备份位于系统盘（overlay），不在 `autodl-tmp` 数据盘；数据盘当时只剩约 2.5 GB，无法同时容纳两个 7.9 GB 完整检查点。未删除任何实验产物；备份删除需明确授权。

### 1.5 训练过程诊断（仅作背景，不作为目标）

- 50 个更新全部完成，无 Traceback / OOM / 磁盘错误；平均 251.9 s/step。
- 训练 rollout 的 `critic/score/mean` 从第 51–60 步均值 0.226 升至第 91–100 步均值 0.308；`episode/success_rate` 从 26.3% 升至 38.8%。趋势并非单调：第 81–90 步 score 均值回落到 0.218、成功率 26.3%。
- 这是训练分布（同一 400 题第二遍）上的指标，不能替代冻结测试集的正确率；下文评测显示后者没有相应提升。

---

## 2. 评测协议

### 2.1 冻结条件

- 测试集：`data/processed/midterm_math_v1/test.parquet`，200 题，SHA `42aadb047de5370406a6d3fe4d8358cfd7d119b8613fda2573bc07731ed5c769`。
- 说明：需求中的“相同开发集”按与 v2 正式评测相同的 200 题冻结 test split 执行；评测器只接受 `frozen_inputs.json` 注册的该集合（`EXPECTED_TASKS=200`），未使用 50 题 dev split。
- 基础模型：`/root/autodl-tmp/Adaptive-Solver-main-git/artifacts/models/qwen3_1_7b_sft_tir_merged`。
- 工具预算：6 步、最多 4 次调用、共享工具时间 60 s、单次 30 s、观察上限 8,000 字符（`configs/agent/unified_tool_budget_v1.yaml`，SHA 未变）。
- 生成：贪心，`do_sample=False`，bf16，`max_new_tokens=2048`，`max_prompt_tokens=4096`。
- Prompt 版本 `agent-v2`；parser tolerance `unclosed_think`、`bare_final_scalar`；sandbox `http://127.0.0.1:8080`。
- 四组均在同一会话/同一 GPU（RTX 4080 SUPER，32 GB）上重跑：SFT、R2-50、R2-75、R2-100。组顺序为 `sft → r2 → r2_100 → r2_75`（`order_seed=20261022`）。

### 2.2 协议一致性

- 与 v2 正式评测的 manifest 逐项比较，任务数/任务 ID 哈希、test parquet 与哈希、模型文件哈希、预算及哈希、奖励配置、初始 prompt token、生成参数、prompt 版本、parser tolerance、sandbox URL、frozen inputs 哈希**全部一致**。
- 唯一差异是 evaluator 脚本 SHA：v2 `8bcc6378...`，本次 `359e0d02...`；差异仅来自新增 `r2_75`/`r2_100` 组选择，不改变生成、解析、预算或验证器行为。分析脚本的 protocol check 结果为 **PASS**（`mismatches=[]`）。

### 2.3 评测可靠性

- 评测器按设计在遇到 `sandbox_unavailable` 的非法轨迹时暂停，保留该无效行，恢复后重试同一任务。
- 本次共 9 次暂停、11 条无效尝试（SFT 3、R2-50 1、R2-75 5、R2-100 2），全部为 `infrastructure_error`；均未计入目标指标。重试后每组恰好 200 条有效轨迹（`latest_valid_by_task`）。
- 暂停/恢复记录：`audit/continuation_prep_20261023/eval_resume_history.log`、`eval_pause_events.jsonl`；自动恢复驱动仅调用冻结评测器的 `--resume`，未修改评测器或任何 journal 行。
- v2 正评中也出现过 5 条同类 `sandbox_unavailable` 无效行（2 个 PAUSED 文件）；本次隧道空闲连接瞬断更频繁，因此暂停次数较多。

### 2.4 统计口径

- 按任务配对（200 对）；10,000 次配对 bootstrap，seed `20261022`，四组与所有比较共用同一组重采样索引。
- 正确率差异使用双侧精确 McNemar；主比较族（R2-75−R2-50、R2-100−R2-50、R2-100−R2-75）内做 Holm 校正；与 SFT 的次要比较另做 Holm 校正。
- 报告 95% CI；不对 CI 再做多重性校正。
- 目标指标：正确率、工具调用次数/题、工具时间/题。平均奖励、非法动作率、截断率、工具失败率只作为诊断，不替代目标。

---

## 3. 主结果

### 3.1 主表

| 组别 | 正确 | 调用/题 | 工具时间/题（均值） | 工具时间/题（中位数） | 工具失败率 | 非法动作率 | 截断率 | 0/1/2+ 次调用 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 110/200 (55.00%) | 0.9900 | 0.4701 s | 0.2665 s | 19.70% | 1.00% | 1.00% | 2/198/0 |
| R2-50 | 108/200 (54.00%) | 0.9850 | 0.6070 s | 0.2659 s | 20.30% | 1.98% | 1.00% | 3/197/0 |
| R2-75 | 112/200 (56.00%) | 0.9850 | 0.4545 s | 0.2673 s | 17.26% | 0.75% | 1.00% | 3/197/0 |
| R2-100 | 108/200 (54.00%) | 0.9950 | 0.4551 s | 0.2666 s | 21.61% | 6.15% | 0.00% | 1/199/0 |

- 工具调用在四组都饱和于 0 或 1 次/题，没有任何题达到 2 次以上；调用次数的组间差异 ≤0.01 次/题。
- 工具失败主要是模型代码的 `execution_error`（34–43 次/组），另有 SFT/R2-50 各 1 次 `timeout`；不是本轮新增的沙箱基础设施故障（后者只出现在被排除的无效尝试中）。
- 非法动作主要是 `action_parse_error`（“Use exactly one <tool_call> or <final> action”）：SFT 4 次、R2-50 8 次、R2-75 3 次、R2-100 25 次（另有 1 次 `tool_validation_error`）。R2-100 的 26 次非法动作分布在 18 个任务上。

### 3.2 工具时间的异常值敏感性

均值被个别慢调用主导：

| 组别 | ≥5 s 调用数 | 均值（全部） | 均值（剔除含 ≥5 s 调用的任务） | 中位数 |
|---|---:|---:|---:|---:|
| SFT | 1（30.012 s） | 0.4701 s | 0.3217 s | 0.2665 s |
| R2-50 | 2（30.003 s、28.729 s） | 0.6070 s | 0.3165 s | 0.2659 s |
| R2-75 | 1（28.328 s） | 0.4545 s | 0.3145 s | 0.2673 s |
| R2-100 | 1（28.334 s） | 0.4551 s | 0.3150 s | 0.2666 s |

R2-50 的均值比 R2-75/R2-100 高约 0.15 s/题，恰好接近多出的一次 ~28.7 s 调用分摊到 200 题的量级；中位数与剔除慢调用任务后的均值都显示四组没有实质差异。若只把 ≥5 s 的调用按 0 计（而不是剔除整个任务），四组均值为 SFT 0.3201、R2-50 0.3133、R2-75 0.3129、R2-100 0.3135 s/题，结论相同。因此**不能**把“R2-75/R2-100 比 R2-50 工具成本更低”作为结论。

### 3.3 配对比较

主比较族（treated − baseline；Δ正确率为百分点）：

| 比较 | Δ正确率 (95% CI) | McNemar p（Holm） | 改善/退步题数 | Δ调用/题 (95% CI) | Δ工具时间/题 (95% CI) |
|---|---:|---:|---:|---:|---:|
| R2-75 − R2-50 | +2.0 pp [−0.5, +5.0] | 0.289（0.867） | 6 / 2 | 0.000 [−0.020, +0.020] | −0.152 s [−0.456, +0.002] |
| R2-100 − R2-50 | 0.0 pp [−3.0, +3.0] | 1.000（1.000） | 5 / 5 | +0.010 [0.000, +0.025] | −0.152 s [−0.455, +0.003] |
| R2-100 − R2-75 | −2.0 pp [−5.0, +1.0] | 0.344（0.867） | 3 / 7 | +0.010 [0.000, +0.025] | +0.001 s [−0.006, +0.007] |

与 SFT 的次要比较（Holm 校正后均 p=1）：

| 比较 | Δ正确率 (95% CI) | 改善/退步题数 | Δ调用/题 (95% CI) | Δ工具时间/题 (95% CI) |
|---|---:|---:|---:|---:|
| R2-50 − SFT | −1.0 pp [−4.0, +2.0] | 4 / 6 | −0.005 [−0.025, +0.010] | +0.137 s [−0.024, +0.443] |
| R2-75 − SFT | +1.0 pp [−2.0, +4.0] | 6 / 4 | −0.005 [−0.025, +0.010] | −0.016 s [−0.036, −0.002] |
| R2-100 − SFT | −1.0 pp [−4.5, +2.5] | 5 / 7 | +0.005 [0.000, +0.015] | −0.015 s [−0.035, −0.002] |

> R2-75/R2-100 相对 SFT 的工具时间 CI 略低于 0，但绝对差约 15–16 ms/题，且中位数完全相同；结合 §3.2，这同样由 SFT 的 30 s 慢调用造成，不应解读为实质成本下降。

---

## 4. 如何解读

### 4.1 正确率

- R2-75 相对 R2-50 的点估计为 +2.0 pp（6 题改善、2 题退步），但 95% CI 跨 0（[−0.5, +5.0]），McNemar p=0.289，Holm 校正后 0.867。**不能认为 R2-75 更好。**
- R2-100 相对 R2-50 为 0.0 pp（5/5），相对 R2-75 为 −2.0 pp（3 改善/7 退步），均不显著。**没有证据表明续训到 100 步提升正确率。**
- 与 SFT 的三个次要比较也都接近 0，方向不一致且不显著。**本次没有复现出 R2 相对 SFT 的可靠优势或劣势。**
- 训练 rollout 的成功率从 26.3% 升到 38.8%，但冻结测试集正确率保持在 54–56%。这一分离提示：训练奖励/成功率的上升主要发生在同一 400 题的第二遍数据上，不能外推为泛化提升。

### 4.2 工具成本

- 调用次数已经饱和于 0/1 次/题；R2 的成本项在本数据上没有“减少多次调用”的空间。R2-100 相对 R2-50/R2-75 的 +0.01 次/题只是零调用题从 3 题降到 1 题，实际影响可忽略。
- 工具时间的中位数四组几乎相同；均值差异由 ~29–30 s 的个别慢调用造成。剔除含 ≥5 s 调用的任务后，四组均在 0.315–0.322 s/题（若只把慢调用按 0 计则为 0.313–0.320 s/题）。**没有可靠的证据表明 R2-75/R2-100 降低了工具成本。**
- R2-100 的工具失败率（21.61%）和非法动作率（6.15%）都高于 R2-50/R2-75；这属于协议/行为漂移，不是成本改善，也没有带来正确率收益。

### 4.3 与 v2 四组评测的对照

| 组别 | v2 正确率（2026-10-22） | 本次正确率（2026-10-24） | 说明 |
|---|---:|---:|---|
| SFT | 56.00% (112/200) | 55.00% (110/200) | 差 2 题；跨会话/工具时间抖动 |
| R2-50 | 54.00% (108/200) | 54.00% (108/200) | 完全一致 |
| R2-75 | — | 56.00% (112/200) | 新增 |
| R2-100 | — | 54.00% (108/200) | 新增 |

跨会话比较（SFT 55% vs v2 56%）有 1–2 题的差异，属于评测非确定性/工具时间边界效应；本次所有组间比较均使用同一会话内的重跑结果，未复用 v2 的 SFT/R2-50 行。v2 使用 RTX 4090 D，本次使用 RTX 4080 SUPER，硬件不同；四组内部共享同一硬件。

---

## 5. 不能支持的结论

以下结论**不能**由本次实验支持：

1. “继续训练到 75/100 步提升了正确率。”——三个主比较均不显著，CI 覆盖 0 或负值。
2. “R2-75 或 R2-100 优于 SFT / R2-50。”——点估计在 ±2 pp 内，校正后均不显著。
3. “续训降低了工具成本。”——调用次数饱和、工具时间中位数不变，均值差异是慢调用伪影。
4. “验证器修复带来了提升。”——第 1–50 步与 51–100 步同时改变了步数和验证器，没有分离设计。
5. “单纯增加步数有效/无效。”——本次不是纯步数消融：验证器变化、第二 epoch 数据、RNG 重启、单种子都与步数混杂。
6. 对多随机种子、更长训练、其他数据分布或更大模型的泛化结论。

可以支持的结论仅限于：**在当前冻结测试集与评测协议下，R2-75/R2-100 相对 R2-50 没有显示出可靠的正确率提升或工具成本下降；R2-100 的非法动作率明显升高；训练集上的奖励/成功率上升没有转化为测试集正确率。**

---

## 6. 产物与复现

### 6.1 路径

| 内容 | 路径 |
|---|---|
| 续训配置 | `configs/grpo/midterm_unified_r2_continue100.yaml` |
| 续训运行 | `artifacts/experiments/midterm_uniform_reward_v1/training/r2_continue100/` |
| 训练日志 | `training/r2_continue100/train.log`、`rollout_audit.jsonl`、`rollout_dump/` |
| R2-75 adapter | `training/r2_continue100/adapters/step_75/` |
| R2-100 adapter / 完整检查点 | `training/r2_continue100/adapters/step_100/`、`checkpoints/global_step_100/` |
| R2-75 完整检查点备份 | `/root/r2_continue100_step75_full/global_step_75/`（系统盘，逐字节校验） |
| 四组评测 | `evaluation/r2_continuation_20261023/`（`run_manifest.json`、`evaluation_summary.json`、`group_*.jsonl`、`progress.json`） |
| 配对分析 | `evaluation/r2_continuation_20261023/analysis/`（`main_table.csv`、`paired_comparisons.csv`、`main_table.md`、`analysis.json`） |
| 预检/验证证据 | `audit/continuation_prep_20261023/`（`step75_verification.json`、`step100_verification.json`、`eval_preflight.json`、`eval_resume_history.log`、`eval_pause_events.jsonl`） |
| 自动恢复驱动 | `audit/continuation_prep_20261023/run_eval_with_resume.sh` |
| 本次报告 | `report/r2_continuation_20261023_report.md` |
| 校验和 | `report/SHA256SUMS.r2_continuation_20261023` |

### 6.2 复现命令（分析）

```bash
cd /root/autodl-tmp/BudgetConstrainedReasoningMathAgent
PYTHONPATH=src /root/autodl-tmp/conda-envs/adaptive-math/bin/python \
  scripts/eval/analyze_r2_continuation.py \
  --eval-dir artifacts/experiments/midterm_uniform_reward_v1/evaluation/r2_continuation_20261023
```

### 6.3 成本与磁盘

- GPU 时间：训练约 3 h 31 min，评测约 1 h 49 min，合计约 **5 h 20 min**。按 ¥2/h 估算约 **¥10.7**（以控制台实际计费为准）。
- 数据盘：`/root/autodl-tmp` 160 GB，当前约 158 GB 已用、2.5 GB 可用（99%）；step-100 完整检查点约 7.9 GB。
- 系统盘：`/` 30 GB，当前约 14 GB 已用、17 GB 可用；其中包含 R2-75 完整检查点备份约 7.9 GB。
- 未删除任何实验产物；原 R2-50 未改动。step-75 完整检查点在训练器轮转时自动移除，轮转前已完成并校验备份。

---

## 7. 披露清单

1. **验证器变化**：第 1–50 步训练使用修复前验证器，第 51–100 步使用修复后验证器 `1de775...`；评测统一使用修复后验证器。本次不是纯步数消融。
2. **第二 epoch**：第 51–100 步复用同一 400 行调度表，不是新数据。
3. **随机性重启**：vLLM 采样 RNG、优化器/调度器、dataloader 从 step 50 接续；工具时间抖动。
4. **评测暂停/重试**：11 条 `sandbox_unavailable` 无效尝试被保留并排除，任务经 `--resume` 重试；每组最终 200 条有效轨迹。
5. **硬件变化**：本次评测为 RTX 4080 SUPER；v2 为 RTX 4090 D。四组内部同硬件。
6. **检查点轮转**：step-75 完整检查点被训练器自动轮转；逐字节备份在系统盘，adapter 仍在实验树。
7. **跨会话差异**：SFT 本次 55.0% vs v2 56.0%（差 2 题）；R2-50 本次与 v2 完全一致（108/200）。组间比较只使用本次同会话结果。
8. **不作因果声明**：本报告不把任何差异归因于“单纯增加步数”或“验证器修复”，也不外推到其他种子、数据或模型。
