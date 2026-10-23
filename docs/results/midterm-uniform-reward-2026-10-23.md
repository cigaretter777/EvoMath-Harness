# 统一工具奖惩三臂预实验(midterm_uniform_reward_v1)

2026-10-22 训练、2026-10-23 评测与收尾。**状态 COMPLETE**(三组训练各 50 updates、四组各 200 题评测),研究判读:未检出正确率收益。

- 实验目录:`artifacts/experiments/midterm_uniform_reward_v1/`(本轮随提交强制加入 git,`.gitignore` 中 `artifacts/` 默认忽略)。
- 完整报告:[`report/midterm_uniform_reward_v1_report.md`](../../artifacts/experiments/midterm_uniform_reward_v1/report/midterm_uniform_reward_v1_report.md)(1,153 行,sha256 `f909e4ec…c0c34`)。
- 离线审计:[`audit/posthoc_20261023/audit_report.md`](../../artifacts/experiments/midterm_uniform_reward_v1/audit/posthoc_20261023/audit_report.md);逐条修正清单:[`report_corrections.md`](../../artifacts/experiments/midterm_uniform_reward_v1/audit/posthoc_20261023/report_corrections.md)。
- 主报告正文未按审计清单改写;引用其中训练期统计与 R2 机制表述时,以审计报告和修正清单为准。

## 一句话结论

R1 相对 R0 的**协议行为改善**是本轮最有依据的结果(评测非法动作率 −10.19 pp,95% CI [−14.82, −5.65];截断率 −4.5 pp [−8.0, −1.0]);未获得正确率提升或工具成本下降的充分证据;R2 无收益不能用“成本项没有学习信号”解释。

## 主表(200 题/组,统一工具环境,评测 v2)

| 组别 | 正确率 | 非法动作率 | 工具调用/题 | 工具时间/题(s) | 截断率 |
|---|---:|---:|---:|---:|---:|
| SFT | 56.00% (112/200) | 1.00% | 0.990 | 0.444 | 1.00% |
| GRPO+R0(仅正确性) | 53.00% (106/200) | 12.42% | 0.990 | 0.459 | 5.50% |
| GRPO+R1(正确性−非法动作惩罚) | 51.50% (103/200) | 2.22% | 0.990 | 0.918* | 1.00% |
| GRPO+R2(正确性×成本−非法动作惩罚) | 54.00% (108/200) | 0.25% | 0.995 | 0.457 | 0.00% |

\* R1 的 0.918 s/题含三次连续 30 秒沙箱超时(合计 93.069 s),是 `client_round_trip` 回退计费值而非实测执行时间;四组同剔三道题后,R2−R1 = **+0.0005 s/题 [−0.0065, +0.0082]**,无成本下降。

## 正确率配对比较(精确 McNemar,Holm 校正)

| 比较 | 改善/退步题数 | 正确率差 | 原始 p | Holm p |
|---|---|---:|---:|---:|
| R0 − SFT | 2 / 8 | −3.0 pp | 0.109 | 0.328 |
| R1 − R0 | 4 / 7 | −1.5 pp | 0.549 | 0.549 |
| R2 − R1 | 8 / 3 | +2.5 pp | 0.227 | 0.453 |

三项均不显著。n=200、π_d≈0.05–0.055 时,80% 功效的精确 McNemar 最小可检出效应约 **4.5–5 pp**;报告中的 6–7 pp 是 CI 宽度,不是 MDE。

## 离线审计的关键修正(2026-10-23)

- **训练期非法动作率按 active rows 重算**为 R0 **9.03%**、R1 **3.48%**、R2 **2.74%**(原报告 5.01% / 2.36% / 1.97% 把终局后的重复行计入了分母)。首 10 轮→末 10 轮趋势:R0 2.01%→16.05%,R1 1.40%→4.65%,R2 1.86%→2.46%;R1 改善方向不变。
- **R0 奖励均值 0.2826 vs 0.2981 是口径差异**:`critic/rewards/mean` 按 active turn 行加权,`episode/reward/mean` 按轨迹等权(0.29814);逐轨迹复算 0.298125,按行复现 0.2826391。不是奖励实现错误。
- **R2 存在 O(1) 归一化优势**:33 个全对采样组中 33 个有时间成本变异、1 个有调用次数变异;组内最大 |A| 中位数 1.344,132 条样本中 36 条 |A|>1.0。信号主要来自时间分量;其来源(真实效率差异还是计时噪声被归一化放大)现有单次 rollout 无法识别。
- **训练器输入核验通过**:4,800/4,800 条轨迹的奖励公式、状态与 correct 分量一致;150/150 个 update 的 score 序列与“终局奖励×active turns”在 1e-6 内逐行一致;无缺失 `done`。这不等于重新裁决数学正确性。
- **两条训练假阴性局部影响不可忽略**:gu3 的优势从全零变为 `[−0.5, −0.5, 1.5, −0.5]`;gu39 其余样本变化 −0.934 / +0.874 / −0.934。不能按 2/4,800 的占比认定影响很小。
- **优势标准差口径为 G−1**:150/150 个 update 在 3 位小数下与 G−1 精确匹配,G 口径 0/150。开题/论文中需写明。

## 重要披露

- 训练完成后修复了验证器集合比较崩溃(`Symbol - Interval`);v1 评测作废,v2 四组用修复后的同一验证器重跑。`frozen_inputs.json` 保留原始源码树哈希 `d910…`,未重写。
- v2 期间发生 5 次沙箱瞬断;无效行保留,同一后端探针恢复后 `--resume`,未选择性删除结果。
- 训练期 9 条验证器 `internal_error` 事件、8 条轨迹,其中 2 条修复后应判正确(均在 R1);训练未回溯重跑。
- 单训练种子、50 个全局更新、200 道测试题;没有正式非劣效设计,不能声称“非劣效/等价”。

## 复现入口

```bash
cd /root/autodl-tmp/BudgetConstrainedReasoningMathAgent
export PYTHONPATH=src
export ADAPTIVE_MATH_SANDBOX_URL=http://127.0.0.1:8080
PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
$PY scripts/audit/posthoc_midterm_audit.py
$PY scripts/audit/check_advantage_std.py
```

完整训练/评测命令、哈希清单与偏差记录见报告 §10 与附录 C。正式训练/评测均已结束,不要在该目录启动新的运行。
