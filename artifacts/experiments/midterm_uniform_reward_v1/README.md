# `midterm_uniform_reward_v1`

**Status: complete.** 三组独立 GRPO 训练、四组统一评测、配对分析与完整报告均已完成；本目录不再启动新的训练或评测。

## 最终报告

- [`report/midterm_uniform_reward_v1_report.md`](report/midterm_uniform_reward_v1_report.md)
- 报告校验和：[`report/SHA256SUMS`](report/SHA256SUMS)

## 结果速览

| 组别 | 正确率 | 非法动作率 | 工具调用/题 | 工具时间/题（秒） | 截断率 |
|---|---:|---:|---:|---:|---:|
| SFT | 56.00% (112/200) | 1.00% | 0.990 | 0.444 | 1.00% |
| GRPO+R0 | 53.00% (106/200) | 12.42% | 0.990 | 0.459 | 5.50% |
| GRPO+R1 | 51.50% (103/200) | 2.22% | 0.990 | 0.918* | 1.00% |
| GRPO+R2 | 54.00% (108/200) | 0.25% | 0.995 | 0.457 | 0.00% |

\* R1 的 0.918 s/题包含三次连续 30 秒沙箱超时（合计 93.069 s）；剔除后为 0.452 s/题，与 R2 的 0.457 s/题无实质差异。

三个预设正确率比较（R0−SFT、R1−R0、R2−R1）经 Holm 校正后均不显著（p=0.328、0.549、0.453）。不支持“正确率提升”“R2 降低工具成本”“任一 GRPO 组非劣于 SFT”等结论。详见报告 §6、§9。

## 关键产物

| 内容 | 路径 |
|---|---|
| 冻结输入与哈希 | [`manifests/frozen_inputs.json`](manifests/frozen_inputs.json) |
| 训练时间线、中止与重启 | [`audit/formal_run_launch.md`](audit/formal_run_launch.md) |
| 实现修正与冒烟证据 | [`audit/implementation_corrections.md`](audit/implementation_corrections.md) |
| 训练指标 / rollout 诊断 | [`audit/training_metrics_summary_20261023.json`](audit/training_metrics_summary_20261023.json)、[`audit/training_rollout_diagnostics_20261023.json`](audit/training_rollout_diagnostics_20261023.json) |
| 验证器修复证据 | [`audit/verifier_bug_20261023/`](audit/verifier_bug_20261023/) |
| 正式训练检查点/适配器 | `training/{r0,r1,r2}/checkpoints/global_step_50`、`training/{r0,r1,r2}/adapters/step_50` |
| 统一评测 v2 | [`evaluation/formal_unified_tool_eval_20261023_v2/`](evaluation/formal_unified_tool_eval_20261023_v2/) |
| 配对分析 | [`evaluation/formal_unified_tool_eval_20261023_v2/analysis/`](evaluation/formal_unified_tool_eval_20261023_v2/analysis/) |
| 作废的 v1 评测证据 | `evaluation/formal_unified_tool_eval_20261023/`（不参与结果） |

## 重要披露

- 训练完成后修复了验证器集合比较崩溃（`Symbol - Interval`）；v1 评测作废，v2 四组全部用修复后的同一验证器重跑。`frozen_inputs.json` 保留原始源码树哈希 `d910...`，未重写。
- v2 期间发生 5 次沙箱瞬断（SFT 第 5、184 题，R1 第 38、122、140 题）；无效行保留，同一后端探针恢复后 `--resume`，未选择性删除结果。
- R1 三次 30 秒超时为基础设施停滞，主表按协议保留，报告中给出剔除敏感性分析。
- 训练期 9 条验证器 `internal_error` 事件、8 条轨迹，其中 2 条修复后应判正确（均在 R1）；训练未回溯重跑。
- 单训练种子、50 个全局更新、200 道测试题；没有正式非劣效设计，不能声称“非劣效/等价”。

## 复现与环境

```bash
cd /root/autodl-tmp/BudgetConstrainedReasoningMathAgent
export PYTHONPATH=src
export ADAPTIVE_MATH_SANDBOX_URL=http://127.0.0.1:8080
PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
```

完整命令、哈希清单与偏差记录见报告 §10 与附录 C。正式训练/评测均已结束，不要在本目录启动新的运行。
