# 晨报 — Round-1 预算诊断(pilot + formal 全部完成)

2026-10-14 07:20 本地。你睡前启动的两轮都跑完了,零事故。

## TL;DR

1. **正式轮(100 题 × 4 cap × 4 采样 = 1600 条)已完成**,四臂 400/400、配对一致、0 flags(2h58m,20:15Z→23:13Z)。
2. **结论 = 预注册的有效阴性**:正确率对工具预算平坦(11.75/14.00/13.00/13.75%,CI 全重叠、无趋势),
   **工具调用执行数四臂全部为 0** —— "预算"对当前 SFT 策略结构性非绑定。b0(预算 0)与 b4(预算 4)等价。
3. **原因已定位**(1600 条全量机制普查):策略从未产出过一次解析合法的单动作调用;全部 793 个 invalid 步
   都是 `action_parse_error` —— ①数学 think 长文撞 1024 截断(19-25% 的生成步)整回合无动作;
   ②LaTeX 里的 `<` 使 `unclosed_think` 容错修复不生效;③少量多调用重复环。沙箱独立探活 pong+42,
   排除死沙箱。
4. **奖励侧**:78-86% 的题组内奖励零方差(R0 成本权重全 0、0/1 二值);correct-but-different-cost 全 0。
   预注册决策树 item 2(不同成本同正确)未触发;item 1/4 的材料已齐。
5. 层级能力:易层 S0[1,4] 32-42%,难度 ≥4 后 0-4%(与"预算无关"完全一致)。
6. pilot↔formal 同题复采样一致(formal 复跑 pilot-20 子集:0.21/0.26/0.23/0.28 vs pilot 0.25/0.26/0.21/0.28)。

## 产物索引(全部未提交,按你的指示)

- `artifacts/diagnostics/r1/analysis_pilot/` — pilot 三表 + `FINDINGS.md`
- `artifacts/diagnostics/r1/analysis_formal/` — formal 三表 + `FINDINGS.md`(含分层表、机制普查、CI)
- `artifacts/diagnostics/r1/sft_b{0,1,2,4}_{pilot,100}/` — 原始 1600+320 条轨迹(含事件流、reward 分解、manifest)
- `artifacts/runs/diagnostics_r1_{pilot,formal}/` — 启动日志、门禁记录、状态
- 代码:`scripts/analysis/analyze_budget_diagnostics.py`、`scripts/diagnostics/run_round1_{pilot,formal}.sh`
- 预注册:`docs/experiments/thesis-round1-prereg.md`

## 对下一步的含义(按预注册决策树,等你拍板,不擅动)

- **预算感知 RL(BC-GRPO/双优势等)的前提("策略能用工具预算")在 SFT 冷启动点不成立**;
  直接上 RL 对比前,先决定是否处理上游三选一(或组合):
  ①动作格式/冷启动数据(策略会尝试调用但 envelope 恒不合法;`sft_tir_v1` 数据是纯工具轨迹,可评估换臂);
  ②容错域扩展(现容错只修"单侧 think 且体内无 `<`";本例失败形状不在域内);
  ③探索空间/奖励分辨率(全同奖励组 78-86%,同温度下 GRPO 梯度覆盖很窄)。
- 门禁 3 相关冻结决定:pilot 后按先验证据(E2:改 cap 不变结果)+机制普查,**未改任何预算参数**,pilot/formal 同配方可比。
- 未触碰:BC-GRPO 实现、200 步训练、生成种子接线 —— 均待你指示。
