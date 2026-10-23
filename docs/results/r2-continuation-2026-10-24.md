# R2 续训 50→100 步(r2_continue100)

2026-10-23 训练、2026-10-24 评测与配对分析。**状态 COMPLETE**(R2 从 `global_step_50` 续训 50 个更新;SFT / R2-50 / R2-75 / R2-100 四组各 200 题)。研究判读:**未检出正确率提升或工具成本下降**。

- 实验目录:`artifacts/experiments/midterm_uniform_reward_v1/`(本轮新增文件随提交强制加入 git,`.gitignore` 中 `artifacts/` 默认忽略)。
- 完整报告:[`report/r2_continuation_20261023_report.md`](../../artifacts/experiments/midterm_uniform_reward_v1/report/r2_continuation_20261023_report.md)(sha256 `f62f997d…eb8b73`)。
- 校验和:[`report/SHA256SUMS.r2_continuation_20261023`](../../artifacts/experiments/midterm_uniform_reward_v1/report/SHA256SUMS.r2_continuation_20261023)(100 个文件)。
- 评测与配对分析:[`evaluation/r2_continuation_20261023/`](../../artifacts/experiments/midterm_uniform_reward_v1/evaluation/r2_continuation_20261023/)。
- 本分支基于 `feature/midterm-uniform-reward-v1`(commit `467df29`),延续同一实验族。

## 一句话结论

在 200 题冻结测试集上,R2-75 相对 R2-50 的点估计为 +2.0 pp(6 题改善 / 2 题退步),但 95% CI [−0.5, +5.0]、Holm 校正后 p=0.867;R2-100 相对 R2-50 为 0.0 pp,R2-100 相对 R2-75 为 −2.0 pp,均不显著。工具调用饱和于 0/1 次/题,工具时间中位数四组几乎相同(0.2659–0.2673 s/题),均值差异由个别 ~29–30 s 慢调用造成。**本次不能用来证明“单纯增加步数有效”。**

## 主表(200 题/组,贪心,同会话)

| 组别 | 正确率 | 工具调用/题 | 工具时间中位数(s) | 非法动作率 |
|---|---:|---:|---:|---:|
| SFT | 110/200 (55.00%) | 0.990 | 0.2665 | 1.00% |
| R2-50 | 108/200 (54.00%) | 0.985 | 0.2659 | 1.98% |
| R2-75 | 112/200 (56.00%) | 0.985 | 0.2673 | 0.75% |
| R2-100 | 108/200 (54.00%) | 0.995 | 0.2666 | 6.15% |

## 主比较(精确 McNemar,Holm)

| 比较 | 改善/退步题数 | 正确率差 (95% CI) | 原始 p | Holm p |
|---|---|---:|---:|---:|
| R2-75 − R2-50 | 6 / 2 | +2.0 pp [−0.5, +5.0] | 0.289 | 0.867 |
| R2-100 − R2-50 | 5 / 5 | 0.0 pp [−3.0, +3.0] | 1.000 | 1.000 |
| R2-100 − R2-75 | 3 / 7 | −2.0 pp [−5.0, +1.0] | 0.344 | 0.867 |

与 SFT 的三个次要比较也均不显著(Holm p=1):R2-50 −1.0 pp、R2-75 +1.0 pp、R2-100 −1.0 pp。

## 关键披露

- **验证器变化**:第 1–50 步用修复前验证器训练,第 51–100 步用修复后验证器 `1de775613d9c…`;四组评测统一用修复后验证器。本次不是纯步数消融。
- **第二 epoch**:第 51–100 步复用同一 400 行调度表,不是新数据。
- **随机性与硬件**:vLLM 采样 RNG、优化器/调度器、dataloader 从 step 50 接续;单种子;评测 GPU 为 RTX 4080 SUPER(v2 为 RTX 4090 D),四组内部同硬件。
- **评测暂停/重试**:9 次暂停、11 条 `sandbox_unavailable` 无效尝试全部保留并排除,经 `--resume` 重试后每组 200 条有效轨迹;评测器未修改。
- **跨会话差异**:SFT 本次 55.0% vs v2 56.0%(差 2 题);R2-50 本次与 v2 完全一致(108/200)。组间比较只使用本次同会话结果。
- **检查点**:原 R2-50 未改动;R2-75 完整 FSDP 检查点被训练器在写 step 100 前自动轮转,轮转前的逐字节备份在 `/root/r2_continue100_step75_full/global_step_75`(7.9 GB,实验树外,未入库);R2-100 完整检查点与 adapter 在实验树内。checkpoints/adapters 均未入库,哈希钉在 `audit/continuation_prep_20261023/step75_verification.json` 与 `step100_verification.json`。
- **R2-100 协议漂移**:非法动作 26 次/18 题(25 次 `action_parse_error` + 1 次 `tool_validation_error`),未转化为正确率收益。

## 复现入口

```bash
cd /root/autodl-tmp/BudgetConstrainedReasoningMathAgent
PYTHONPATH=src /root/autodl-tmp/conda-envs/adaptive-math/bin/python \
  scripts/eval/analyze_r2_continuation.py \
  --eval-dir artifacts/experiments/midterm_uniform_reward_v1/evaluation/r2_continuation_20261023
```

评测器 `scripts/eval/run_unified_tool_eval.py` SHA `359e0d02…`、分析脚本 `scripts/eval/analyze_r2_continuation.py` SHA `bdd09895…` 为本地工作区版本,未随本分支入库;已入库的 `evaluation/.../analysis/` 为实际输出。
