# Harness Evolution 设计报告：失败挖掘 → Harness Patch → 回归评测 → 晋升/回滚

版本：1.0
日期：2026-09-18
状态：设计基线；核心数据契约与门禁/账本模块已初版实现（见 §11），runner 与 CLI 未实施
上游文档：[AdaptiveMath-RL 产品与技术设计报告](./2026-09-09-adaptive-math-rl-product-design.md)、[Master Roadmap](../plans/2026-09-10-adaptive-math-rl-master.md)

## 0. 执行摘要

AdaptiveMath-RL 的主链路（数据 → SFT → Agent → Rollout → Reward → GRPO）解决的是"改模型权重"。本设计解决它的对偶问题：**不改权重时，如何系统地改进模型外面的那圈外壳（harness），并且每一次改动都可度量、可比较、可回滚**。

Harness Evolution 是一个闭环：

~~~text
GRPO / 评测产出轨迹与逐题结果
        ↓
Failure Mining      失败机器标签 + 聚簇，定位系统性失败类别
        ↓
Harness Patch       针对失败类别构造新 HarnessSpec（prompt/预算/工具/reward/解码）
        ↓
Regression Evaluation  同一 checkpoint、同一冻结任务集、同一 decode seed 上
                       champion vs candidate 成对重跑
        ↓
Promotion / Rollback   统计门禁通过才晋升；线上劣化则有审计地回滚
        ↓
（新 champion 进入下一轮 rollout / 训练 / 评测）
```

三条核心原则，全部继承自仓库既有纪律：

1. **Harness 是一等公民**：像 `RewardConfig` 一样全字段显式、版本化、内容寻址（hash），不存在"隐性默认配置"。
2. **比较必须成对且冻结**：像 `model_eval` 一样，同一任务集、同一顺序、同一 decode 预算，配对 bootstrap CI + 精确 McNemar，不接受单次最好点。
3. **晋升必须有证据，回滚必须有审计**：像 `EvaluationJournal` 一样 append-only，任何状态转移都有不可篡改的事件记录。

本设计不引入新的训练算法、不改动 verifier 语义、不扩大产品范围；它是一套**工程治理闭环**，让"改 prompt / 调预算 / 调 reward 系数"这类操作从拍脑袋变成可复现的实验。

## 1. 为什么需要 Harness Evolution

### 1.1 问题

主链路的实验实践中，以下操作会频繁发生：

- GRPO 后在冻结 dev 集上看到某类失败聚簇（如 `BUDGET_EXHAUSTED` 集中在高难题、`TOOL_RESULT_IGNORED` 集中在 SymPy 输出较长时）；
- 直觉修复方式是改 harness：把 `max_tool_calls` 从 4 调到 6、在 system prompt 中强调"必须读取工具输出"、把 reward 的 `python_weight` 从 0.10 调到 0.05；
- 但如果没有治理，这些改动会变成：改了一个 yaml → 手动跑了几道题觉得变好了 → 直接用到下一轮训练 → 无法回答"这次改动到底带来了什么"。

这正是 reward 版本化（R0–R3）出现之前 reward 调参的状态。Harness Evolution 把同一套纪律推广到整个外壳。

### 1.2 与训练的关系

Harness Evolution **不是** GRPO 的替代品，而是它的互补层：

| | 改什么 | 迭代成本 | 证据形式 |
|---|---|---|---|
| GRPO | 模型权重 | 小时—天（GPU） | 冻结集 accuracy、Pareto |
| Harness Evolution | 权重之外的外壳 | 分钟—小时（仅推理） | 成对回归门禁 |

两类改动共享同一份评测纪律，且互相产生输入：GRPO 的轨迹喂给失败挖掘；harness patch 后的新外壳直接进入下一轮 GRPO 的 rollout 环境（此时 reward 配置的一致性由 spec hash 保证，见 §8.2）。

### 1.3 研究/工程命题

> 在冻结任务集与固定 checkpoint 下，harness 层面的补丁（prompt、预算、工具策略、reward 成本系数）能否以统计显著的方式消除失败挖掘定位出的系统性失败类别，且不引入成本或格式上的隐性回退？

该命题与主报告的三个研究问题正交，可以被实验支持或否定；即使结论为负，闭环基础设施（spec、门禁、账本）仍是可信资产。

## 2. 定义：什么是 Harness

**Harness = 在同一 checkpoint 下会改变 Agent 行为的一切运行时因素。**

属于 harness（纳入 `HarnessSpec`）：

- **Prompt**：system/user 模板（以 `prompt_version` 标识，对齐 `agent.prompts.PROMPT_VERSION`）；
- **动作协议与运行时**：解析规则、终止条件（以 `runtime_version` 标识，对齐 `agent.loop.RUNTIME_VERSION`）；
- **工具集合**：启用的工具名列表（`python` / `sympy` / 未来的新工具）；
- **预算**：`Budget`（max_steps / max_tool_calls / max_python_seconds / max_observation_chars）；
- **解码参数**：`GenerationConfig`（max_new_tokens / temperature / seed）；
- **Reward 配置**：`RewardConfig`（R0–R3 权重；仅在训练/离线评测链路生效，但对 rollout 行为有因果影响，必须纳入 spec）。

不属于 harness（不在 spec 内，但会被记录为运行上下文）：

- 模型权重 / checkpoint（由 `checkpoint_hash` 标识，成对比较时**必须相同**）；
- 任务数据本身（由冻结评测集的 `task_ids_hash` 标识，成对比较时**必须相同**）；
- SandboxFusion 等基础设施版本（记录在环境元数据，不进 spec hash，但门禁运行记录必须附带）。

边界判定规则：**换一个因素，如果需要在 HarnessSpec 里改字段，它就是 harness 的一部分；如果它要求重新训练或重新构建数据集，它就不是。**

## 3. 设计依据与既有资产

### 3.1 直接复用的仓库机制

| 既有机制 | 位置 | 在 Harness Evolution 中的角色 |
|---|---|---|
| `RewardConfig` 全字段必填 + `config_hash()` | `src/adaptive_math/reward/types.py` | `HarnessSpec` 的设计模板与内嵌组件 |
| reward yaml ↔ 代码 preset 双向锁定测试 | `tests/unit/reward/test_functions.py` | `configs/harness/*.yaml` ↔ `presets.py` 的同款锁定 |
| `TraceEnvelope.content_hash` + 确定性 replay | `src/adaptive_math/agent/replay.py` | 轨迹溯源；门禁运行的轨迹必须 hash 可校验 |
| `Trajectory.runtime_version`（`runtime+prompt` 两段式） | `src/adaptive_math/agent/loop.py` | 扩展为含 harness 版本与 spec hash 的溯源串 |
| 成对统计（paired bootstrap CI + 精确 McNemar，种子记录） | `src/adaptive_math/evaluation/model_eval.py` | 回归门禁的统计内核 |
| Append-only journal + 原子写 | `model_eval.EvaluationJournal._atomic_json` | registry 账本与 runner 断点续跑的持久化模式 |
| 数据治理（冻结 split、task_ids_hash、泄漏拒绝） | `src/adaptive_math/data/`、评测计划 Task 1 | 门禁只接受冻结集上的成对结果 |
| 失败机器标签（12 类 taxonomy，计划中） | 评测计划 Task 4 `failure_taxonomy.py` | Failure Mining 的标签内核 |

### 3.2 项目自主设计（无外部固定实现可抄，需实验验证）

- harness 的内容寻址表示与版本规则（§5）；
- "点估计 floor + CI 下界 + McNemar 显著性 + 成本护栏"四条件门禁（§7.3）；
- champion/candidate 账本与晋升/回滚规则（§7.4）；
- 失败类别 → patch 类型的映射目录（§6.2）。

### 3.3 不可声称的内容

- 不声称 harness 调优可替代训练；任何提升声明必须注明"固定 checkpoint 下的 harness 效应"；
- 不声称门禁通过等于线上提升——门禁只在冻结集上成立，泛化由后续独立评测确认；
- 不在成对重跑完成前声称任何 patch 有效；
- 不把"harness 版本很多"当作工程成果——每次注册都必须对应一次失败挖掘证据。

## 4. 范围与非目标

### 4.1 范围

- 单 checkpoint 下的 harness 版本化、成对回归评测、晋升/回滚治理；
- 失败挖掘的机器标签与聚簇（复用计划中的 failure_taxonomy）；
- 与 GRPO rollout 环境的 spec 一致性（reward 系数进 spec hash）；
- 本地可复现的门禁运行与审计产物。

### 4.2 非目标

- 不做自动 patch 搜索（AutoML 式的 prompt/超参搜索）；patch 由失败挖掘证据驱动、人工提出，门禁负责证伪；
- 不做在线 A/B 或多 champion 并行服务；任一时刻只有一个 champion；
- 不修改 verifier 语义与 reward 公式本身（改公式 = 主报告层面的新实验，不是 harness patch）；
- 不在测试集上做 harness 选择（沿用全局约束：选择只发生在 train/rl_dev/mini-eval）；
- 不做跨 checkpoint 的 harness 结论迁移声明（换 checkpoint 必须重新过门禁）。

## 5. 核心数据契约

### 5.1 HarnessSpec（已实现 `src/adaptive_math/harness/spec.py`）

```python
class HarnessSpec(BaseModel):  # extra="forbid", frozen=True
    schema_version: str          # "harness-v1"
    harness_version: str         # 人类可读版本，如 "h-v1"；任何字段改动必须 bump
    prompt_version: str          # 对齐 PROMPT_VERSION
    runtime_version: str         # 对齐 RUNTIME_VERSION
    tools: tuple[str, ...]       # 启用工具，排序去重，拒绝未知工具名
    budget: Budget               # 复用 core.types.Budget
    generation: GenerationConfig # 复用 agent.model_client.GenerationConfig
    reward: RewardConfig         # 复用 reward.types.RewardConfig（含 version）
```

关键性质：

- `spec_hash()`：canonical JSON（orjson + 排序键）的 SHA-256，即 harness 的内容地址。**Harness Patch 的表示 = 一个 `harness_version` bump 后的新 spec**；
- 一致性校验：`tools=()` 禁止 `max_tool_calls > 0`；未启用 `python` 禁止 `max_python_seconds > 0`；
- `runtime_tag()`：`runtime+prompt+harness` 三段式版本串，写入 `Trajectory.runtime_version`；轨迹同时携带 `harness_spec_hash`（Phase H1 已接入，旧两段式轨迹因字段缺省 None 而保持 hash 不变）；
- 版本名唯一性：registry 拒绝同一 `harness_version` 注册到不同 hash，防止"同名不同物"的隐性 patch；
- `configs/harness/*.yaml` 与 `presets.py` 代码 preset 双向锁定（同 reward 纪律）。

### 5.2 成对结果与门禁决策（已实现 `src/adaptive_math/harness/gate.py`）

```python
class PairedOutcome(BaseModel):
    task_id: str
    champion_correct: bool
    candidate_correct: bool

class GatePolicy(BaseModel):
    policy_version: str
    mode: Literal["non_regression", "significant_improvement"]
    max_regression: float      # 点估计 delta >= -max_regression
    ci_lower_bound: float      # bootstrap CI 下界 >= 此值
    mcnemar_alpha: float       # 显著性水平
    metric_guards: tuple[MetricGuard, ...]  # 成本/格式护栏

class GateDecision(BaseModel):
    policy_version: str
    passed: bool
    reasons: tuple[str, ...]   # 失败原因（通过时为空）
    task_count: int
    task_ids_sha256: str       # 证明双臂跑的是同一冻结集
    accuracy_delta: float
    paired_bootstrap_ci95: tuple[float, float]
    bootstrap_seed: int        # 统计可复现
    bootstrap_resamples: int
    mcnemar_pvalue: float
    metric_deltas: dict[str, float]
```

### 5.3 晋升/回滚账本（已实现 `src/adaptive_math/harness/registry.py`）

```python
class HarnessRecord(BaseModel):
    spec: HarnessSpec
    state: Literal["candidate", "promoted", "rejected"]
    registered_at: str
    gate: GateDecision | None

class RegistryLedger(BaseModel):   # 单 JSON 文件，原子写
    champion: str | None           # 当前 champion 的 spec hash
    records: dict[str, HarnessRecord]
    events: tuple[RegistryEvent, ...]  # register/gate/promote/reject/rollback
```

### 5.4 失败挖掘报告（待实现，契约先行）

```python
class FailureCluster(BaseModel):   # failure_taxonomy 的产出，门禁的输入
    cluster_id: str                # 内容寻址：labels + 切片条件的 hash
    labels: tuple[str, ...]        # 机器标签，如 ("BUDGET_EXHAUSTED",)
    slice: dict[str, JSONValue]    # 聚簇条件：难度/题型/工具等
    task_ids: tuple[str, ...]      # 命中的冻结集任务
    count: int
    example_trace_hashes: tuple[str, ...]  # 可回放证据
    proposed_patch_hypothesis: str # 人工填写的 patch 假设（非机器生成）
```

## 6. 阶段设计

### 6.1 Failure Mining（失败挖掘）

输入：GRPO rollout 记录、冻结评测的逐题 prediction record、轨迹库。
处理：

1. 用 `failure_taxonomy` 的 12 类机器标签（FORMAT_INVALID、NO_FINAL、WRONG_REASONING、WRONG_TOOL_CHOICE、TOOL_CODE_ERROR、TOOL_TIMEOUT、TOOL_RESULT_IGNORED、PREMATURE_STOP、BUDGET_EXHAUSTED、VERIFIER_UNSUPPORTED、VERIFIER_TIMEOUT、INFRASTRUCTURE_FAILURE）给每条失败轨迹打标签——规则必须确定、可从 golden traces 复现；
2. 按（标签 × 难度 × 题型 × 工具）聚簇，产出 `FailureCluster`；只保留样本量达到阈值（默认 ≥ 10）的聚簇，避免对噪声打补丁；
3. 人工审阅每个聚簇的可回放轨迹，写下 patch 假设。**机器负责定位，人负责提出 patch**——这是本设计的有意边界（见 §4.2 非目标）。

输出：按影响面排序的 `FailureCluster` 列表，作为 patch 的唯一合法来源。

### 6.2 Harness Patch（补丁构造）

每个 patch 必须：针对至少一个 `FailureCluster`、只改 spec 中可枚举的字段、bump `harness_version`。patch 类型目录：

| Patch 类型 | 改动字段 | 典型适应症 |
|---|---|---|
| Prompt patch | `prompt_version` | FORMAT_INVALID、TOOL_RESULT_IGNORED、PREMATURE_STOP |
| Budget patch | `budget.*` | BUDGET_EXHAUSTED、PREMATURE_STOP |
| Tool patch | `tools` | WRONG_TOOL_CHOICE（如下线某工具做消融） |
| Reward patch | `reward.*`（仅限预设权重组合） | 工具成本失衡（下一轮 GRPO 生效） |
| Decoding patch | `generation.*` | 截断导致的 NO_FINAL、采样噪声 |

硬性规则：

- **一次 patch 只改一类字段**，否则门禁通过也无法归因；
- reward patch 不改变公式与 `variant` 语义，只在 R0–R3 预设空间内移动；改公式属于主报告层面的新实验；
- 每个 patch 在 registry 注册时必须引用其针对的 `cluster_id`。

### 6.3 Regression Evaluation（回归评测）

成对重跑协议（runner 待实现，协议如下）：

1. 固定：同一 checkpoint、同一冻结评测集（rl_dev / mini-eval，非测试集）、同一 decode seed 集合、同一 SandboxFusion 版本；
2. champion 与 candidate 两个 spec 分别驱动 `AgentLoop` 跑完全部任务，轨迹落盘（hash 可校验、可 replay）；
3. 由私有验证边界计算逐题正确性，产出 `list[PairedOutcome]` 与双臂指标（invalid rate、工具调用、Python 秒、token、延迟）；
4. 断点续跑复用 `EvaluationJournal` 模式：append-only 逐题记录 + 不可变 manifest（含两个 spec hash + checkpoint hash + task_ids_hash），manifest 不匹配拒绝续跑。

### 6.4 Promotion / Rollback（晋升与回滚）

门禁四条件（`non_regression` 模式，全部满足才 passed）：

1. 点估计：`accuracy_delta >= -max_regression`；
2. 区间：bootstrap CI 下界 `>= ci_lower_bound`；
3. 显著性：delta < 0 时 McNemar p 必须 > alpha（不允许显著回退）；
4. 护栏：所有 `MetricGuard` 通过（如 `invalid_rate`、`mean_tool_calls`、`mean_python_seconds` 不得涨过阈值）。

若 patch 的假设是"修复了某失败类别"，改用 `significant_improvement` 模式：必须统计显著地变好才放行——**声称有效就必须举证**。

晋升/回滚规则（registry 强制）：

- 晋升必须有记录在案的 passed 门禁；无 gate 或 gate 失败不可 promote；
- 任一时刻只有一个 champion；晋升时旧 champion 自动降为 promoted（历史）状态；
- 回滚只能指向历史 promoted hash，且必须填写 reason；回滚是新的审计事件，不是删除历史；
- 账本 append-only：条目永不删除，事件永不修改，可重建全部状态转移。

## 7. 与主链路的集成

### 7.1 轨迹溯源

`AgentLoop` 当前写 `runtime_version=f"{RUNTIME_VERSION}+{PROMPT_VERSION}"`。接入后改为从 `HarnessSpec.runtime_tag()` 取值，并在 `Trajectory` 增加 `harness_spec_hash` 字段。效果：任何一条轨迹（训练 rollout、评测、产品 demo）都能回答"它是在哪个 harness 下产生的"。

### 7.2 与 GRPO 的一致性

- GRPO 的 reward 桥已按 `RewardConfig` 版本化；harness spec 内嵌同一份 `RewardConfig`，训练 run 的 `resolved_config.yaml` 记录 `harness_spec_hash`，保证 rollout 环境、reward 计算、训练配置三者一致；
- harness patch 晋升后，下一轮 GRPO 的 rollout 环境从新 champion spec 构造——harness 迭代与训练迭代通过 spec hash 咬合，而不是通过口头约定。

### 7.3 数据治理约束

- 门禁只接受 rl_dev / mini-eval 上的成对结果；测试集（frozen_v1）只用于最终报告，禁止用于 harness 选择；
- runner 的 manifest 必须含 `task_ids_hash`，与 SFT/RL manifest 做泄漏检查（复用 `audit_dataset.py` 的既有能力）；
- 门禁产物（predictions、决策、账本）纳入版本化 artifact 存储，报告中的数字必须能追溯到 `GateDecision`。

## 8. 指标与可观测性

- 每次门禁运行产出：成对预测 JSONL、双臂 summary、`GateDecision`、运行 manifest（两个 spec hash + checkpoint hash + 环境元数据）；
- 账本面板（后续）：champion 历史时间线、每次晋升的 delta 与 CI、回滚原因列表；
- 失败挖掘面板：聚簇大小随 harness 版本的变化趋势——**某个 patch 晋升后，其目标聚簇应在下一次挖掘中缩小**，这是闭环有效性的元指标。

## 9. 测试策略

- 单元（已实现）：spec 校验器/hash 稳定性/yaml 锁定；门禁的合成配对数据（已知 delta 的手算用例、两种模式、护栏、确定性）；registry 全生命周期与全部非法操作拒绝；
- 契约（待实现）：registry 账本 schema 演进兼容；`Trajectory` 增加 harness 字段后旧轨迹仍可 replay；
- 集成（待实现）：runner 用 scripted/fake model 跑通"两个 spec → 成对结果 → 门禁 → 登记"全链路；
- 治理（待实现）：`verify_release` 类检查——报告中每个 harness 相关数字都能追溯到门禁产物 hash。

## 10. 实施阶段与门禁

- **Phase H0（已完成）**：`HarnessSpec` / `GatePolicy` / `evaluate_gate` / `HarnessRegistry` + yaml preset + 29 个单元测试；ruff、mypy strict、全仓 322 测试通过。
- **Phase H1（已完成）**：`AgentLoop` 接入 HarnessSpec（runtime_tag + `harness_spec_hash` 入轨迹）；契约测试证明旧轨迹可 replay。
  门禁已通过：golden trace 的 `content_hash` 逐字节不变；同一剧本下 legacy 与 harness 驱动的 loop 逐事件一致。
- **Phase H2**：成对 rollout runner（fake model + 小规模真实模型 smoke）+ `scripts/eval/run_harness_gate.py`。
  门禁：两次相同运行产出逐字节一致的 `GateDecision`。
- **Phase H3**：`failure_taxonomy.py`（评测计划 Task 4）+ `FailureCluster` 产出，接入 metric guards。
  门禁：golden traces 上的机器标签与人工标注一致率达标（阈值随 Task 4 定义）。
- **Phase H4**：晋升/回滚 CLI + 账本报告；完成一次真实的"挖掘 → patch → 门禁 → 晋升（或拒绝）"端到端演练并写入 runbook。
  门禁：演练产物（聚簇、spec、决策、账本事件）齐全且互相引用闭合。

## 11. 当前实施状态

已实现（Phase H0 + H1，2026-09-18）：

~~~text
src/adaptive_math/harness/
├── spec.py            # HarnessSpec：不可变、内容寻址（§5.1）
├── presets.py         # CURRENT_PRODUCTION preset（= agent/default + r2 + agent-v1）
├── gate.py            # evaluate_gate：四条件回归门禁（§5.2、§6.4）
└── registry.py        # HarnessRegistry：append-only 晋升/回滚账本（§5.3）
configs/harness/champion_v1.yaml
tests/unit/harness/    # test_spec / test_gate / test_registry，29 个测试

# Phase H1（§7.1 接入）
src/adaptive_math/agent/trace.py       # Trajectory.harness_spec_hash（可选，缺省 None）
src/adaptive_math/agent/replay.py      # trajectory_hash：None 时剔除该键，旧 hash 不变
src/adaptive_math/agent/loop.py        # AgentLoop(harness=…)：runtime_tag + spec hash 入轨迹，
                                       #  generation/budget/tools 与 spec 不一致即 ValueError
src/adaptive_math/tools/registry.py    # ToolRegistry.names（环境工具集合的显式读口）
tests/unit/agent/test_trace.py         # 字段校验（+3）
tests/unit/agent/test_replay_hash.py   # hash 兼容锚点（3）
tests/unit/tools/test_registry.py      # names（+1）
tests/integration/test_harness_loop.py # 接入与 parity（6）
tests/contract/test_harness_trace_compat.py  # 旧轨迹永续可回放（3）
~~~

验证（2026-09-18）：`pytest` 全仓 395 passed / 2 skipped 无回归；
golden trace `content_hash`（851f9da5…）逐字节不变；
`ruff check src tests` 与 `mypy`（strict，61 源文件）通过。

未实现：runner（§6.3）、failure_taxonomy（§6.1）、CLI 与面板（Phase H2–H4）。

## 12. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 对噪声打补丁（小样本聚簇误导） | 聚簇样本量阈值；`significant_improvement` 模式强制显著性 |
| 门禁过松导致隐性回退累积 | 四条件门禁 + metric guards；账本支持事后审计与回滚 |
| 门禁过严导致无法迭代 | `max_regression` / `ci_lower_bound` 是显式策略参数，按 patch 类型可调，但调整本身记入 `policy_version` |
| 成对比较不公平（seed/预算漂移） | runner manifest 锁定 decode seed 与预算；manifest 不匹配拒绝续跑 |
| harness 结论被错误迁移到新 checkpoint | spec 与 checkpoint hash 在门禁运行中绑定；换 checkpoint 必须重新过门禁 |
| 版本名混淆（同名不同物） | registry 拒绝同 `harness_version` 不同 hash |

## 13. 完成定义

Harness Evolution 闭环"建成"的标准：

1. 任一轨迹可通过 `harness_spec_hash` 回溯到完整 HarnessSpec；
2. 任一 champion 变更都对应一条 passed `GateDecision` 与一条账本事件；
3. 至少完成一次端到端演练：从真实 GRPO 轨迹挖掘出失败聚簇 → 构造 patch → 成对回归 → 门禁判定 → 晋升或拒绝，全部产物可审计；
4. 即使演练结论是"patch 无效、已拒绝"，闭环本身仍作为可信工程资产成立。
