# 续训准备材料（2026-10-23）

状态：**准备材料完成；续训、四组评测与配对分析已于 2026-10-24 完成。** 50 步预实验保留为中期结果。最终报告见 `../../report/r2_continuation_20261023_report.md`。

| 文件 | 内容 | 状态 |
|---|---|---|
| `50_to_100_resume_mechanics.md` | 从 `global_step_50` 恢复到 100 步的 LR、优化器、dataloader 游标、实际题目顺序，以及为什么这不是纯步数延长 | 已完成 |
| `r2_cost_signal_diagnostic_plan.md` | R2 时间字段来源、固定工具程序电池、重复计时设计、方差分解与预注册判读规则 | 已完成；按冻结规则执行 |
| `../r2_cost_signal_20261023/diagnostic_report.md` | 实际计时与判读报告 | 已完成，判定 **C** |

关键结论：

1. 机械上可从三个 `global_step_50` 检查点恢复；LR 恒为 `1e-6`，优化器状态按 checkpoint 恢复，dataloader 第二轮从 row 0 重新开始，题目顺序重复。
2. 第 1–50 步为修复前验证器，第 51–100 步为修复后验证器；叠加 vLLM 采样 RNG 重启、R2 计时抖动、审计 ID 重复，不能称为“纯步数延长”。
3. R2 诊断：3,125/3,125 次调用有效，无缺失/超时；33/33 组可评估。原组内 `D_obs=0.514s`，重放保留程序均值的 `D_sim=0.007s`（比值 0.014），`S_spread=0.000`、`S_order=0.520`、`S_adv=0.169`，`p_D=0.0001`；完整 27 组同向。按冻结规则判 **C**：不是 A（幅度/排序/优势不可重现），也不是 B（`p_D` 显著，差异并非纯程序内噪声）。
4. 该诊断没有修改奖励、数据、任务集或训练目标，也没有启动 A/B 或续训；结论仅针对当前重放条件和所选 all-correct 组。

## 续训执行记录（2026-10-23 → 2026-10-24）

| 文件 | 内容 |
|---|---|
| `step75_verification.json` | step-75 adapter/完整检查点哈希；轮转前逐字节备份 |
| `step100_verification.json` | step-100 哈希、训练退出状态、评测预检（协议 mismatches=[]、sandbox pong+42） |
| `eval_preflight.json` | 四组 adapter 与评测协议核对 |
| `run_eval_with_resume.sh` | 冻结评测器的自动 `--resume` 驱动（不修改评测器/journal） |
| `eval_resume_history.log`、`eval_pause_events.jsonl`、`eval_resume_*.log` | 9 次暂停、11 条无效尝试的完整记录 |

执行结果摘要：训练 50 个更新全部完成（3 h 31 min）；评测四组各 200 条有效轨迹（1 h 49 min）；主目标与配对检验见最终报告。未删除任何产物，原 R2-50 未改动。
