# AdaptiveMath-Evo v2.0 对齐设计：PDF 技术设计 → 本仓库落地方案

版本：1.0
日期：2026-09-18
状态：设计草案（待评审）；不含实现
上游文档：[AdaptiveMath-Evo Technical Design v2.0（PDF）](../../../AdaptiveMath_Evo_Technical_Design.pdf)、[Harness Evolution 设计报告 v1.0](./2026-09-18-harness-evolution-design.md)、[Phase H1 实施计划](../plans/2026-09-18-harness-evolution-h1-agent-loop.md)

## 0. 执行摘要

外部 PDF《AdaptiveMath-Evo · Technical Design v2.0》描述了比本仓库 v1.0 设计更完整的愿景：在 Tool-Using Math Agent + Agentic RL 基线上，增加 **受约束、可验证、可回滚的 Harness Evolution 外环**，核心闭环为 Rollout → Failure Mining → **Attribution** → Patch Proposal → **Quality Gate** → Regression Evaluation → Promote/Rollback。

对照结论：

- **PDF 的 P0（Harness 参数化）与版本治理纪律，本仓库已完成**（Phase H0/H1：`HarnessSpec` 内容寻址、四条件统计门禁、append-only 账本、轨迹溯源），且在统计严格性上**高于** PDF 的示例 promotion rule（PDF 用固定阈值，仓库用配对 bootstrap CI + 精确 McNemar）。
- **PDF 超出仓库现状的四块能力**：① 最小单元 `HarnessPatch`（schema 约束的字段级补丁）；② Failure Attribution（规则分诊 + counterfactual replay + CAPABILITY_FAILURE 隔离）；③ Quality Gate（泄漏/bloat/安全不变量/复杂度预算的静态检查）；④ 数据切分协议（evolution dev / regression / final held-out 物理隔离）。
- **两处设计冲突需要显式决策**（§4）：LLM proposer（PDF 允许 vs 仓库 v1.0 明确非目标）；reward 可进化性（PDF V1 禁止 vs 仓库允许 R0–R3 预设内 patch）。

本文档给出：概念映射（§2）、差距分析（§3）、冲突决策建议（§4）、目标模块结构（§5）、数据契约增补（§6）、分阶段路线图（§7，PDF P0–P5 → 仓库 H 阶段映射）、测试与验收（§8）。

**采纳原则：PDF 是愿景上限，仓库 v1.0 是工程纪律下限。凡 PDF 更严的（attribution、数据隔离），吸收；凡仓库更严的（统计门禁、人工 proposer 边界），保留并作为 PDF 对应机制的实现方式。**

## 1. PDF 核心主张速览

PDF 的关键工程原则（v2.0 §0）：

1. 模型策略 θ（SFT/Agentic RL）与 Harness H（trajectory-driven search + regression gate）**两条优化链严格解耦**；
2. Evolution 的最小单元不是代码 diff，而是 **schema 约束的 `HarnessPatch`**；V1 禁止 LLM 任意修改源码；
3. **Attribution 是核心而非 patch 生成**：错误归因会把模型能力缺陷编码成 prompt/tool hack，导致 shortcut learning 与 benchmark overfitting；`CAPABILITY_FAILURE` 不进入 evolution queue，只进 RL hard-example pool；
4. Proposal 与 evaluator 解耦；evolution set 与 held-out regression set 物理隔离；所有 Harness 版本 immutable、可复现、可回滚；
5. θ 与 H 不同时在线更新，采用 **alternating optimization**（Stage A cold-start → B Agentic RL → C Harness Evolution → D 短程 re-adaptation）。

## 2. 概念映射：PDF ↔ 本仓库

| PDF v2.0 概念 | 本仓库对应物 | 状态 |
|---|---|---|
| Agent A = (πθ, H, E) | `AgentLoop` + `HarnessSpec` + `Environment` | ✅ 已对齐（H1 完成接入） |
| Harness 参数化 / typed config（P0） | `harness/spec.py`：`HarnessSpec` 不可变、全字段显式、`spec_hash()` 内容寻址 | ✅ 已实现，且更严格（`extra="forbid"`、一致性校验、yaml↔preset 双向锁定） |
| Harness Registry / immutable version / alias（§12） | `harness/registry.py`：`RegistryLedger` 单 JSON 原子写、append-only events、champion 单点、rollback 审计 | ✅ 已实现（alias 语义 = champion 指针 + promoted/rejected 状态） |
| Performance Gate / promotion rule（§8.4） | `harness/gate.py`：`evaluate_gate` 四条件（点估计 floor + CI 下界 + McNemar + metric guards） | ✅ 已实现，统计上严于 PDF 示例规则；PDF 指标向量 M(H) 由 `MetricGuard` + runner 双臂 summary 承载 |
| Trajectory 绑定 harness_version + model_version（§6.1） | `Trajectory.harness_spec_hash` + 三段式 `runtime_tag()` | ⚠️ 部分：harness 绑定已完成；**model/checkpoint 绑定目前只在评测 manifest，不在轨迹** |
| Trace Schema：per-step `error_code`（§6.1） | `Trajectory.events`（`TraceEvent`）+ `usage` + `termination_reason` | ⚠️ 部分：缺**机器可读的 per-step error_code 字段**（当前错误信息在 observation 文本里，无法直接聚类） |
| Failure Taxonomy（§6.2，8 类） | 评测计划 Task 4 的 12 类机器标签（`failure_taxonomy.py`） | ❌ 未实现；类别体系需做映射（§3.2） |
| Failure Attribution（§7）：rule triage + counterfactual replay + reference-assisted | 无 | ❌ 未实现；`agent/replay.py` 的确定性 replay 是 counterfactual 复用的现成底座 |
| `HarnessPatch` 最小单元（§8.2） | v1.0 设计：patch = 整 spec bump | ❌ 未实现；需增加 patch 描述层（§6.2） |
| Quality Gate（§8.3）：泄漏/bloat/安全/复杂度 | 无（现有 gate 只做性能回归） | ❌ 未实现 |
| Weakness Mining：minimum_support 聚簇（§8.1） | v1.0 `FailureCluster` 契约（样本量 ≥ 10 阈值） | ⚠️ 契约已定义，实现待 H3 |
| 数据切分（§9）：RL train / Evolution Dev / Regression Set / Final Held-out | train / rl_dev / mini-eval / frozen_v1 | ⚠️ 可映射，但"Evolution Dev 可迭代使用、Regression Set 仅 evaluator"的访问隔离未成文 |
| Alternating optimization Stage A–D（§10） | 主链路 SFT → GRPO 已有；Stage C/D（H 更新后 re-adaptation）无 | ❌ Stage D 未实现 |
| LLM Patch Proposer（§8.2 允许，输出过 schema 校验） | v1.0 明确非目标（人工 proposer） | ⚡ 冲突，决策见 §4.1 |
| Reward function V1 不可进化（§5 表格） | v1.0 允许 R0–R3 预设内 reward patch | ⚡ 冲突，决策见 §4.2 |

## 3. 差距分析

### 3.1 Trace Schema 增补（PDF §6.1 → P1 前置）

PDF 要求每个 step 携带 `error_code`，且轨迹绑定 `model_version`。当前差距与最小改动：

- `TraceEvent` 增加可选 `error_code: str | None`（受控词表，见 §3.2），缺省 `None` 且在 canonical hash 中剔除 None 键——沿用 H1 的 hash 兼容手法，保证 golden trace `content_hash`（851f9da5…）逐字节不变；
- `Trajectory` 增加可选 `model_version: str | None`（checkpoint hash 或训练 run id），同样 None 时剔键；评测/rollout 入口负责写入；
- `verifier_result` 已由私有验证边界的逐题结果承载，不进轨迹（维持 hidden verifier 边界，见 `tests/contract/test_hidden_verifier_boundary.py`）。

### 3.2 Failure Taxonomy 映射（PDF 8 类 ↔ 仓库计划 12 类）

| PDF 类别 | 仓库 12 类对应 | 归因方向（PDF） |
|---|---|---|
| TOOL_SELECTION_ERROR | WRONG_TOOL_CHOICE | Routing / Tool Policy |
| TOOL_ARGUMENT_ERROR | FORMAT_INVALID（工具参数面） | Tool Schema / Prompt |
| TOOL_EXECUTION_ERROR | TOOL_CODE_ERROR / TOOL_TIMEOUT | Tool Runtime / Retry |
| SYMPY_PARSE_ERROR | TOOL_CODE_ERROR（SymPy 子类） | Preprocessor / Tool Contract |
| PREMATURE_TERMINATION | PREMATURE_STOP / NO_FINAL | Termination / Verification |
| LOOP_OR_REDUNDANCY | （新增，当前 12 类无对应） | Loop / Context / Stop Rule |
| VERIFICATION_FAILURE | TOOL_RESULT_IGNORED / VERIFIER_UNSUPPORTED | Verification Policy |
| CAPABILITY_FAILURE | WRONG_REASONING | **Model θ，不触发 H patch** |

决策：以仓库 12 类为实现基底，增补 `LOOP_OR_REDUNDANCY`；每类标注 `attribution_direction ∈ {harness, model, ambiguous}`，`WRONG_REASONING` 默认归因 model（= PDF 的 CAPABILITY_FAILURE），**不进入 evolution queue，只进 RL hard-example pool**。

### 3.3 Failure Attribution 模块（PDF §7，全新）

三层结构，全部确定性优先：

1. **Rule-based triage**：由 `error_code` + 工具异常 + `termination_reason` 做第一层组件映射（纯函数，golden traces 可复现）；
2. **Counterfactual replay**：固定 θ，仅替换候选 harness component 重放失败样本。复用 `agent/replay.py` 的确定性 replay 与 `ScriptedModel` 模式；产出 `counterfactual_delta`（重放前后逐题正确性差）；
3. **Reference-assisted analysis**（V1 可选）：对有 ground-truth 的题构造 reference tool path，比较结构差异。

产出契约（PDF §7.1，直接采纳）：

```python
class AttributionResult(BaseModel):  # frozen, extra="forbid"
    failure_type: str
    target_component: str            # HarnessSpec 字段路径，如 "budget.max_tool_calls"
    confidence: float                # [0, 1]
    evidence_run_ids: tuple[str, ...]
    counterfactual_delta: float | None
    eligible_for_harness_patch: bool # CAPABILITY_FAILURE → False
```

### 3.4 HarnessPatch 最小单元（PDF §8.2）

PDF 的 patch 是字段级操作（`path` / `operation` / `old_value` / `new_value`）；仓库 v1.0 的 patch 是"整 spec bump"。两者不矛盾，分层调和：

- **`HarnessSpec` 仍是唯一可执行表示**（内容寻址、可注册、可过门禁）——不变；
- **新增 `HarnessPatch` 作为描述/审计层**：`apply_patch(spec, patch) -> HarnessSpec` 是纯函数，deterministic，且能生成 inverse patch（PDF §15.1 单元测试要求）；`harness_version` bump 由 apply 自动完成；
- 保留 v1.0 硬性规则"**一次 patch 只改一类字段**"：patch 的 `path` 必须落在单一字段族（prompt / budget / tools / generation / reward），registry 注册时必须引用 `cluster_id`。

```python
class HarnessPatch(BaseModel):  # frozen, extra="forbid"
    patch_id: str
    base_spec_hash: str              # 内容寻址，而非 PDF 的人类可读 base_version
    target_component: str
    operation: Literal["update", "add", "enable", "disable"]
    path: str                        # 如 "budget.max_tool_calls"
    old_value: JSONValue
    new_value: JSONValue
    evidence_cluster_ids: tuple[str, ...]
    expected_effect: str
    risk_level: Literal["low", "medium", "high"]
```

### 3.5 Quality Gate（PDF §8.3，全新）

静态检查，先于性能门禁运行；任一不过即 reject，不消耗成对重跑预算：

| 检查 | 规则 | 仓库实现依托 |
|---|---|---|
| Schema validity | patch path 存在、值域合法、allowlist（processor / 工具名） | `HarnessSpec` 校验器复用 |
| Data leakage | patch 的 prompt/字段中不得出现冻结集题目文本或答案 | 复用 `scripts/data/audit_dataset.py` 泄漏检查能力 |
| Prompt bloat | `prompt_version` 对应模板的 token 增长 ≤ 显式预算（如 +20%） | `agent/prompts.py` 模板计量 |
| Safety invariant | 不得关闭 verifier、不得放宽 sandbox/工具权限边界、`tools` 不得引入未知工具 | spec 一致性校验扩展 |
| Complexity budget | 每轮 evolution 的 patch 数与受影响 component 数上限 | evolution runner 配置 |

### 3.6 数据切分协议（PDF §9 → 仓库映射）

| PDF split | 仓库 split | 访问规则（成文化） |
|---|---|---|
| RL Train / Rollout | train | evolver 可读 trace；不用于 harness 结论 |
| Evolution Dev | rl_dev | failure mining + candidate 迭代使用 |
| Regression Set | mini-eval | **仅 evaluator/runner**；proposer（人或 LLM）不得接触其题目内容与逐题结果 |
| Final Held-out | frozen_v1 | 仅最终报告；禁止 harness 选择（既有纪律，不变） |

新增纪律：每次门禁运行的 manifest 记录所用 split 的 `task_ids_hash`；同一 patch 假设不得对 Regression Set 重复"试错式"重跑（次数计入账本事件，可审计）。

## 4. 冲突决策点

### 4.1 LLM Proposer：PDF 允许 vs 仓库 v1.0 禁止

- PDF：proposer 可由 LLM 实现，但输出必须过 JSON Schema/Pydantic 校验，且禁止直接写 production config。
- 仓库 v1.0（§4.2 非目标）："机器负责定位，人负责提出 patch"，不做自动 patch 搜索。

**建议（维持 v1.0，PDF 机制作为受控扩展）**：V1 保持人工 proposer——这不是能力缺口而是有意边界，防止 prompt bloat 与 benchmark overfitting（PDF §7 自己也以此为由强调 attribution 优先）。LLM proposer 若要引入，必须同时满足：① 输出即 `HarnessPatch`（pydantic 校验）；② 过 Quality Gate 全部静态检查；③ 过双 Gate（Quality + Performance）；④ 每个机器提议的 patch 在账本中标注 `proposer_version`。在此约束下 LLM proposer 与 PDF 设计完全兼容，可作为 H5+ 阶段实验，不改变门禁语义。

### 4.2 Reward 可进化性：PDF V1 禁止 vs 仓库允许预设内 patch

- PDF：Reward function 不可进化，防止 evolver 直接改评分规则。
- 仓库 v1.0：允许 reward patch，但仅限 R0–R3 预设权重组合、不改公式、且只在下一轮 GRPO 生效。

**建议（保留仓库决定，向 PDF 收窄一档）**：reward patch 继续允许但施加 PDF 精神的两条额外约束：① reward patch 必须走 `significant_improvement` 模式（声称有效必须举证），不接受 `non_regression` 放行；② reward patch 的晋升永不单独生效，必须绑定下一次 GRPO run 的 `resolved_config.yaml`（既有 spec hash 咬合机制）。改 reward 公式本身仍属主报告层面的新实验，非 harness patch——两边一致。

## 5. 目标模块结构

在既有 `harness/` 旁新增 `evolution/` 包（PDF §13 目录建议的仓库化适配）：

```text
src/adaptive_math/
├── harness/                     # 已实现（H0/H1），不变
│   ├── spec.py  presets.py  gate.py  registry.py
├── evolution/                   # 新增（taxonomy.py 已于 H2.5 落地，其余待 H3+）
│   ├── taxonomy.py              # ✅ 13 类机器标签 + attribution_direction（§3.2）
│   ├── miner.py                 # 聚簇 → FailureCluster（minimum_support，§3.4 v1.0 契约）
│   ├── attribution.py           # rule triage + counterfactual replay（§3.3）
│   ├── patch.py                 # HarnessPatch + apply_patch / inverse_patch（§3.4）
│   ├── quality_gate.py          # 5 项静态检查（§3.5）
│   ├── runner.py                # 成对回归 runner（= v1.0 的 Phase H2）
│   └── promoter.py              # 晋升/回滚 CLI 逻辑（= v1.0 的 Phase H4）
└── agent/
    ├── trace.py                 # ✅ +model_version（可选，None 剔键）
    └── state.py                 # ✅ TraceEvent +error_code（可选，None 剔键）
```

依赖方向：`evolution/` → `harness/` → `agent/`；训练侧只经公开接口读 harness，不依赖 evolution 内部（PDF §15.2 集成测试要求）。

## 6. 数据契约增补汇总

新增：`FailureCluster`（v1.0 §5.4 已定义）、`AttributionResult`（§3.3）、`HarnessPatch`（§3.4）、`QualityGateReport`（§3.5，字段：checks、passed、reasons、patch_id）。
修改：`TraceEvent.error_code`、`Trajectory.model_version`（均可选、None 剔键、hash 兼容）。
不变：`HarnessSpec` / `GatePolicy` / `GateDecision` / `RegistryLedger` / `TraceEnvelope` hash 语义。

## 7. 分阶段路线图（PDF P0–P5 → 仓库映射）

| PDF 阶段 | 仓库阶段 | 状态 / 门禁 |
|---|---|---|
| P0 Harness 参数化 | H0 + H1 | ✅ 完成（395 tests 全绿，golden hash 锚定） |
| P1 Trace/Failure | **H2.5：trace 增补 + taxonomy** | ✅ 完成（2026-09-18，439 tests 全绿；`model_version`/`error_code` 入轨迹且旧 hash 不变，13 类标签 + 归因方向 + `label_trajectory` 落地，golden 正确轨迹打标为空契约锚定；实施记录见 [H2.5 计划](../plans/2026-09-18-harness-evolution-h2-5-trace-taxonomy.md)） |
| P2 Constrained Evolver | **H3：miner + patch + quality_gate** | 门禁：能从失败簇生成合法 patch；非法 patch（泄漏/bloat/越权）全部被判reject |
| P3 Regression Gate | **H2（runner）+ 已有 gate/registry** | 门禁：两次相同运行产出逐字节一致 `GateDecision`；candidate 崩溃自动 reject 不伤 champion |
| P4 Attribution Enhancement | **H4：attribution（counterfactual replay）** | 门禁：错误 patch rate 下降可量化；CAPABILITY_FAILURE 零进入 evolution queue |
| P5 RL Re-adaptation | **H5：Stage D 短程 RL + 端到端演练** | 门禁：固定 θ 的 H0/H1 ablation 报告；产物互相引用闭合 |

（H2/H3/H4 编号与 v1.0 设计 §10 保持兼容：runner 仍是 H2，taxonomy 提前为 H2.5 的一部分，promoter CLI 并入 H4。）

## 8. 测试与验收

沿用 PDF §15 并落到仓库既有测试分层：

- **单元**：patch apply deterministic + inverse 存在；taxonomy 对 golden fixtures 分类稳定；quality gate 拦截答案泄漏、超限 prompt、非法 processor、verifier disable（PDF §15.1 逐条对应）；
- **契约**：trace 增补字段后旧轨迹永续可回放（沿用 `test_harness_trace_compat.py` 模式）；registry 账本 schema 演进兼容；
- **集成**：固定 θ + H0/H1 对同一 regression set 并行运行，逐样本 diff；candidate 崩溃/timeout 自动 reject；promotion 后一键 rollback 到 parent；训练进程只经公开接口读 harness（PDF §15.2 逐条对应）；
- **治理**：报告中每个 harness 数字可追溯到 `GateDecision` hash（verify_release 类检查）。

## 9. 与 PDF 表述边界的一致声明

遵循 PDF §17：当前完成度为 P0 + P1 的 trace/taxonomy 内核（+ P3 的门禁/账本内核），对外表述仅限"实现 trajectory-driven constrained Harness Evolution 的版本化、回归门禁与失败标签基础设施"；在 H2–H5 完成前不使用"自进化 Agent""Co-Evolution"等表述。实验数字只出自 §16 式实验表的可复现记录，不使用估计值。
