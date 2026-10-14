# Round-1 预算诊断 — Formal(100 题)发现

2026-10-14。冻结策略 `qwen3_1_7b_sft_dp_v1_merged`,cap {0,1,2,4} 四臂 × 100 题(官方四层 ×25)
→ 4 采样/题/臂 = 1600 条。temp 0.7,`--parser-tolerance unclosed_think,bare_final_scalar`(与训练环境一致)。
预注册:`docs/experiments/thesis-round1-prereg.md`。数据:`analysis_formal/`(table_a/b/c.csv、tables.md、summary.json)。
运行 20:15Z→23:13Z(2h58m),四臂各 ~2.7-3h;C0-C3 门禁在启动日志中,沙箱 pong+42。

## 门禁/完整性

四臂 400/400,任务池配对精确,post-check 0 flags;final 96.0%/94.8%/95.0%/94.0%,
max_steps 4.0%/5.3%/5.0%/6.0%;0 条 `tool_result` 事件;全部 793 个 invalid 步均为 `action_parse_error`(下详)。

## Table A(按题聚合)

| cap | 正确率(题级) | Wilson 95% CI | 平均调用 | P(calls=0) | 平均 token | invalid/条 |
|---|---|---|---|---|---|---|
| b0 | 11.75% (47/400) | [9.0%, 15.3%] | 0.000 | 1.00 | 716 | 0.44 |
| b1 | 14.00% (56/400) | [10.9%, 17.7%] | 0.000 | 1.00 | 760 | 0.53 |
| b2 | 13.00% (52/400) | [10.1%, 16.7%] | 0.000 | 1.00 | 781 | 0.46 |
| b4 | 13.75% (55/400) | [10.7%, 17.5%] | 0.000 | 1.00 | 812 | 0.56 |

四臂 CI 全重叠、无单调趋势;**调用数(执行态)四臂全部 = 0**,`share_calls_0` = 1.00。

## 机制普查(1600 条全量)

invalid 步:b0 174 / b1 213 / b2 184 / b4 222,100% `action_parse_error`;0 个 `tool_call_budget_exhausted`
(即从未出现解析合法的单动作工具调用)。分类(multi-action / no-action / unclosed-think):
16/97/143、30/103/171、16/89/145、14/97/173。1024 截断步占全部生成步的 19%/21%/22%/25%。
固障路径同 pilot:截断在 think 中(无动作)、LaTeX `<` 挡 `unclosed_think` 修复、少量多调用重复环。

## 组内奖励可见性

| arm | all-equal 组 | 有正确率混合的题 | correct-but-diff-cost | invalid 后恢复的题 |
|---|---|---|---|---|
| b0 | 80/100 | 20/100 | 0 | 1 |
| b1 | 86/100 | 14/100 | 0 | 1 |
| b2 | 78/100 | 22/100 | 0 | 0 |
| b4 | 81/100 | 19/100 | 0 | 1 |

**78-86% 的题组内奖励零方差**(r0 成本权重全 0,奖励=0/1 二值);成本维度恒定(0 调用、0 s),
correct-but-diff-cost 四臂全 0 —— 决策树 item 2("不同成本同正确")的触发条件**不成立**。
全轮 100 题中"四采样里至少对过一次"的题只有 22-26 条。

## 难度分层(官方 strata,25 题/层;acc = 题级 macro)

| arm | S0[1,4] | S1[4,5] | S2[5,5.25] | S3[5.25,9.5] |
|---|---|---|---|---|
| b0 | 0.320 | 0.020 | 0.110 | 0.020 |
| b1 | 0.420 | 0.040 | 0.100 | 0.000 |
| b2 | 0.340 | 0.040 | 0.110 | 0.030 |
| b4 | 0.390 | 0.040 | 0.100 | 0.020 |

层内同样平坦(每层四臂差 ≤ 5pp);能力集中在易层 S0,难度 ≥4 后接近地板。这是"曲线平坦"在
组成受控条件下的加强版。

## pilot ↔ formal 一致性(同题复采样)

formal 在 pilot-20 子集上的题级 macro:b0 0.212 / b1 0.263 / b2 0.225 / b4 0.275,
对照 pilot 轮 0.250 / 0.263 / 0.213 / 0.275 —— 逐臂吻合(独立抽样重测,噪声范围内);
其余 80 题只有 9.4-10.9%(构成更难),解释了 formal 总体(~12-14%)低于 pilot(~21-28%)。

## 对照预注册的判定

命中第三条主判读原文:"curve flat and call counts pinned at one value across caps → the budget
is structurally non-binding for this policy; the multi-step-budget premise is NOT supported at this
stage (a valid negative result)"。决策树:item 1(调用钉死 → 查冷启动/探索空间)已完成机制普查;
item 4(大量全同奖励组 → 成功稀疏性/奖励分辨率)材料齐全;item 2 **未触发**。

## 口径备注

- 生成种子未接线(`seed_enforced_by_transformers_client: false`);每臂 4 采样为随机抽取,非 4 个独立题项;
  统计全部按题聚合。Wilson CI 为描述性补充(非预注册项)。
- 方差/优势类数值为重算诊断值(ddof=0,对齐 runner `group_advantages`),非训练器实际优势。
