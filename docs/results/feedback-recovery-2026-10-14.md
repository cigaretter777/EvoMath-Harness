# 工具反馈与错误恢复探索性验证(fr-20261014T1520Z)

2026-10-14 运行、2026-10-15 收尾。**状态 COMPLETE(324/324 无缺失),研究判读 INCONCLUSIVE。**

- 协议(执行设计,文档在仓库外):`/root/autodl-tmp/BudgetConstrainedReasoningMathAgent/2026-10-14-feedback-recovery-validation.md`;副本随本轮提交加入 `docs/experiments/2026-10-14-feedback-recovery-validation.md`,sha256 与 `protocol.lock.json` 所钉一致。
- 运行产物:`artifacts/diagnostics/feedback_recovery/fr-20261014T1520Z/`(本轮随提交强制加入 git,`.gitignore` 中 `artifacts/` 默认忽略);结论与六问逐条回答见该目录 `REPORT.md`,偏离见 `DEVIATIONS.md`(D1–D13)。
- 协议指纹:`spec_hash=4ca6c6c51c29`;模型 `artifacts/models/qwen3_1_7b_sft_tir_merged`;分析脚本 sha256 `3ca9b54d5dd4…`(`analysis/summary.json.analysis_script_sha256`)。

## 一句话结论

在 TIR 模型自然产生的 27 个独立"工具执行错误"状态上,F(完整反馈)与 M(纯失败标记)的最终正确率差 **0.000**,CI95 [−0.037, +0.037];P(反馈+固定修复提示) 相对 F 为 **−0.009**,CI95 [−0.065, +0.046]。三臂正确率都在 8% 附近,属协议 §9"所有臂几乎失败"行,因此判 **INCONCLUSIVE**(不能宣称反馈无用,更不能宣称 RL 无效)。可复用的机制证据:锚点后真正执行成功的修复全实验仅 1/324 条(且答案仍错);90%+ 分支直接作答;25 次 `sympy` 工具调用全部被 `invalid_arguments` 挡回。

## 关键数字

| 项 | 值 |
|---|---|
| B 自然轨迹 | 40 题 × 4 = 160;正确率 0.100;含 E_EXEC 轨迹 54(33.8%),E_EXEC 事件 56 |
| 错误状态池 | 54 候选 → 27 锚点(每题 1,稳定哈希);`pool_verdict=C`(≥20) |
| C 续跑 | 27 × 3 臂 × 4 = 324/324,0 缺失、0 重试、0 协议违规 |
| 臂正确率 | F 8.33%(9/108)、P 7.41%(8/108)、M 8.33%(9/108) |
| 主/次对比 | F−M = 0.000 [−0.037, +0.037];P−F = −0.009 [−0.065, +0.046](锚点级配对 bootstrap 10000 次) |
| 恢复行为 | 锚点后再发 python 调用:F 6/108、P 8/108、M 1/108 条分支;执行成功 1 条(全实验) |
| 墙钟 | A 208 s + B 1797 s + C 6543 s ≈ 2 h 37 min(RTX 4090D;沙箱经 SSH 隧道在远端) |
| token | A 7,178 / B 58,165 / C 后缀 265,465(F 79,297、P 83,865、M 102,303) |

## 结论与下一步(详见 REPORT.md Q6)

不进入方法研究也不是否定——本轮判读是"当前池/power 下不可判读"。若继续,先把诊断做成可判读:提高基线恢复成功率(更强 TIR 模型或收窄到"模型有能力修"的错误类型)、把锚点数提到能分辨 10pp 以上的量级、修 `sympy` 工具入参协议(当前 25/25 全废)。

## 复现入口

```bash
CONDA=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
RUN=artifacts/diagnostics/feedback_recovery/fr-20261014T1520Z
$CONDA scripts/analysis/analyze_feedback_recovery.py --run-dir $RUN   # 会重写 audit/labels.jsonl,见 D12
$CONDA scripts/analysis/write_run_manifest.py --run-dir $RUN
```

精确启动命令(A/B/C)见 `$RUN/run_plan.md` §6。
