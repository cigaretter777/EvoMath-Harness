# Round-1 预算诊断 — Pilot(20 题)发现

2026-10-14。冻结策略 `qwen3_1_7b_sft_dp_v1_merged`,cap {0,1,2,4} 四臂 × 20 题(strata 5×4)→ 4 采样/题/臂,
temp 0.7,`--parser-tolerance unclosed_think,bare_final_scalar`(与训练环境一致)。预注册:
`docs/experiments/thesis-round1-prereg.md`。数据:`analysis_pilot/`(table_a/b/c.csv、tables.md、summary.json)。

## 门禁结果(预注册四条全过)

1. 四臂各 80/80 条,任务池配对精确一致(四臂 pool == 预计算选择)。
2. 同上(配对即门禁 2)。
3. 终止原因:final 96.3%/95.0%/91.3%/95.0%,max_steps 3.8%/5.0%/8.8%/5.0% —— **步数上限未主导任何臂**;
   python 时间上限未被触碰(0.0 s)。步级截断:21%/29%/31%/22% 的生成在 1024 token 上限处截断(停在 think 中)。
   先验证据(09-24 E2:cap 2048 → 200/200 结果不变、纯改标签)+ 本次机制普查 → **配置冻结进正式轮**,不改步数/token 上限。
4. 工具错误排除:全程 0 条 `tool_result` 事件(0 次工具调用),沙箱独立探活 pong+42 —— 观察到的行为是策略侧
   (envelope 解析失败),不是死沙箱。

## Table A(按题聚合,cap 列)

| cap | 正确率(题级 macro) | 平均调用 | P(calls=0) | final 占比 | max_steps 占比 | 平均 token | invalid/条 |
|---|---|---|---|---|---|---|---|
| b0 | 25.0% (20/80) | 0.000 | 1.00 | 96.3% | 3.8% | 721 | 0.43 |
| b1 | 26.3% (21/80) | 0.000 | 1.00 | 95.0% | 5.0% | 893 | 0.66 |
| b2 | 21.3% (17/80) | 0.000 | 1.00 | 91.3% | 8.8% | 1011 | 0.81 |
| b4 | 27.5% (22/80) | 0.000 | 1.00 | 95.0% | 5.0% | 815 | 0.54 |

正确率曲线平坦、无单调趋势(n=20 题,臂间差异远小于二项噪声);**调用数四臂全部钉死在 0**。

## 机制普查(为什么调用从不落地)

共 195 个 invalid 步,**100% 为 `action_parse_error`**;0 个 `tool_call_budget_exhausted`
(即从未产生过一次解析合法的单动作工具调用,b0 若有会被预算拒绝计数——没有)。逐条分类(前一个
model_output 的原始文本):

| 臂 | multi-action(≥2 动作标签) | no-action(0 标签) | unclosed think | 1024 截断步占比 |
|---|---|---|---|---|
| b0 | 6 | 22 | 27 | 21% |
| b1 | 8 | 37 | 44 | 29% |
| b2 | 8 | 29 | 51 | 31% |
| b4 | 11 | 21 | 28 | 22% |

三条固障路径:①数学 think 长文撞 1024 上限被截断在 think 中 → 整个回合无动作;②LaTeX 全文含 `<`
(`i < j` 等)→ 保守版 `unclosed_think` 修复不生效(其触发条件要求动作标签前无 `<`);③少量多调用
重复环(格式正确但违反"恰一个动作")。与 09-24 P1 裁决(严格 parser 全丢 = 协议伪影)在容错开启后
**仍然成立**:容错只修"单侧 think 且体无 `<`"这一种形状,本例的失败形状不在修复域内。

## 组内奖励可见性(Table C 聚合)

- all-equal-reward 组:b0 14/20、b1 13/20、b2 16/20、b4 14/20(**65-80% 的题组内奖励零方差**);
  题内有正确率混合(有梯度信号)的题只有 6/7/4/6 条。
- correct-but-different-cost 组:**四臂全 0** —— 成本维度实际恒定(0 调用、0 s python),r0 成本权重全 0,
  奖励 = 正确性 0/1 二值。在当前温度/组规模下,GRPO 式优势仅对约 1/4-1/3 的题存在。
- 失败后恢复(two-column):tool-error 恢复 0/臂;invalid 后仍正确的题仅 b0 有 1 条。恢复机制基本不存在。

## 对照预注册的三条主判读

命中第三条原文:**"curve flat and call counts pinned at one value across caps → the budget is
structurally non-binding for this policy; the multi-step-budget premise is NOT supported at this
stage (a valid negative result)"**。先验 0/400 执行 ≥2 次调用,本次 pilot 在 cap 4 下依旧 0 次执行。
预注册决策树 item 1(调用钉死 → 先查冷启动与探索空间)与 item 4(大量全同奖励组 → 成功稀疏性/奖励
分辨率)对应的材料均已在上面给出。

## 时机与口径备注

- 生成本轮为探索性抽样(生成种子未接线,`seed_enforced_by_transformers_client: false`);4 条采样是
  随机抽取,不是 4 个独立题项;n=20 题仅 pilot 定位,**正式 100 题已启动**为证据轮。
- 本文件所有"优势/方差"类数值均为**重算诊断值**(ddof=0,对齐 runner 的 group_advantages),非训练器
  实际优势。
