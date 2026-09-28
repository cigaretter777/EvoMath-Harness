# 求职版 Agentic RL 最小 MVP 设计

> 日期：2026-09-27  
> 状态：待评审  
> 目标：从现有 EvoMath Harness 中提炼一条范围小、可复现、适合面试讲解的 Agentic RL 主线。

## 1. 项目定位

求职版项目不再试图同时证明数据平台、安全沙箱、复杂数学验证、云端调度、BC-GRPO 和无人值守 Harness Evolution。核心定位收缩为：

> A minimal reproducible Agentic RL system for tool-using mathematical reasoning.

项目只回答一个问题：

> 在统一的多步 Agent 环境中，经过 SFT 建立动作协议后，标准 GRPO 能否改善数学任务的正确率、协议合法率或工具使用行为？

成功不要求 GRPO 必须显著提高准确率。只要训练闭环真实运行、评测口径一致，并能通过轨迹解释结果，就构成完整 MVP。

## 2. 最小叙事闭环

```text
数学题
  ↓
SFT：建立 <tool_call> / <final> 协议
  ↓
Agent：直接回答或调用 SymPy
  ↓
Trajectory：记录动作、观察、用量和终止原因
  ↓
Verifier：判断最终答案是否正确
  ↓
Standard GRPO：同题多轨迹、组内相对优势、策略更新
  ↓
Frozen Evaluation：比较 Base / SFT / SFT+GRPO
```

主叙事到冻结评测结束。失败分类与 Harness Candidate 评测只能作为可选扩展，不得阻塞 Agentic RL 主闭环交付。

## 3. STAR 叙事

### Situation

基础数学模型能够生成解题文本，但不能稳定遵循工具调用协议，也不能自主决定何时使用符号工具。固定工作流可以编排模型，却无法让训练、评测和实际 Agent 运行共享同一行为接口。

### Task

构建一个最小的工具增强数学 Agent 环境，使模型能够在直接回答和调用 SymPy 之间决策；使用 SFT 建立动作协议，再通过标准 GRPO 优化策略，并在固定评测集上验证训练前后的行为变化。

### Action

1. 定义 `<tool_call>` 与 `<final>` 两种动作。
2. 实现预算受限的多步 AgentLoop。
3. 只接入 SymPy 一个工具。
4. 使用隐藏答案 Verifier 生成终局奖励。
5. 使用 SFT 教会模型输出合法动作。
6. 对每道题采样一组轨迹，通过标准 GRPO 计算组内相对优势并更新策略。
7. 在相同任务、Prompt、Verifier、batch size 和生成配置下比较 Base、SFT 与 SFT+GRPO。
8. 使用保存的轨迹解释准确率、格式和工具行为的变化。

### Result

结果分两类，均视为闭环完成：

- 正向结果：GRPO 在相同评测条件下提高准确率、协议合法率，或减少无效工具调用。
- 负向结果：GRPO 没有显著提升，但训练、Checkpoint、恢复和评测全部跑通，并通过轨迹将原因定位到奖励稀疏、组内方差不足、协议退化或工具探索不足。

未经实验验证，不得声称 GRPO 或 BC-GRPO 提升了模型能力。

## 4. MVP 功能范围

### 4.1 必须保留

- 一个小型指令模型，默认使用现有 Qwen3-1.7B 路线。
- 一个经过固定切分的数学任务集。
- `<tool_call>` 和 `<final>` 动作协议。
- 一个 SymPy 工具。
- 一个多步 AgentLoop。
- 步数和工具调用次数预算。
- 一个满足当前任务答案类型的最小 Verifier。
- 可序列化的 Trajectory。
- SFT 数据与训练入口。
- 标准 GRPO 训练入口。
- Base、SFT、SFT+GRPO 三臂评测。
- 一条从准备数据到生成报告的复现路径。
- 核心单元测试与 CI。

### 4.2 可选扩展：最小 Harness Evolution

只有 Agentic RL 主闭环稳定后，才增加以下薄层：

```text
失败轨迹
  ↓
提出一个 Prompt 或 Budget Candidate
  ↓
Champion / Candidate 冻结集配对评测
  ↓
简单 Gate 决定 Promote 或 Reject
```

Candidate 每次只能改变一个字段，例如：

- Prompt 增加“符号问题优先考虑 SymPy”；或
- `max_steps` 从 4 调整为 6；或
- `max_tool_calls` 从 1 调整为 2。

最小 Gate：

```text
candidate_accuracy >= champion_accuracy
且
candidate_mean_steps < champion_mean_steps
```

该扩展不包含自动 Patch 生成、复杂注册表或无人值守演化。

## 5. 明确不做

以下能力不属于求职 MVP 主线：

- V1 LangGraph Router、Planner、Critic 和 Executor。
- 通用 Python 与 SandboxFusion。
- 多工具自动路由。
- 大规模多源数据治理与近似去重。
- R0–R3 多套奖励产品化。
- BC-GRPO、dual-λ 和多预算约束。
- 多类 Harness Failure Taxonomy。
- 自动 Harness Patch 生成。
- Harness Registry、版本晋升和多级回滚。
- 无人值守 Harness Evolution。
- 多云训练、复杂自愈调度和产品服务部署。

现有仓库可以继续保留这些历史能力，但 README、演示和面试叙事不以它们为主线。

## 6. 最小实验设计

### 6.1 实验臂

| 实验臂 | 目的 |
|---|---|
| Base | 测量原始模型能力与协议遵从率 |
| SFT | 测量监督学习对动作协议和任务表现的贡献 |
| SFT + GRPO | 测量标准 Agentic RL 的增量贡献 |

三臂必须使用相同：

- 冻结任务 ID；
- Prompt 版本；
- Verifier 版本；
- batch size；
- `max_new_tokens`；
- 解码策略；
- 工具和预算配置。

### 6.2 核心指标

只保留四个主指标：

1. Answer accuracy；
2. Final action 合法率；
3. Tool-call rate；
4. 平均轨迹步数。

辅助诊断可以记录 invalid action 和 termination reason，但不扩展成大型指标体系。

### 6.3 最小完成标准

MVP 完成必须满足：

- Base、SFT、SFT+GRPO 三臂都能在同一冻结集运行；
- 至少一次真实 GRPO 更新完成；
- 产生可加载的 LoRA Checkpoint；
- Checkpoint 能恢复或用于评测；
- 评测报告包含四个核心指标；
- 至少抽查成功、错误、非法动作和工具调用四类轨迹；
- README 提供不超过三条主命令的复现入口；
- 不把无显著结果包装成能力提升。

## 7. 最小代码表面

目标不是立即删除现有模块，而是为求职主线提供清晰入口：

```text
src/adaptive_math/
├── agent/          # actions, parser, environment, loop, trajectory
├── tools/          # MVP 只暴露 sympy
├── verifier/       # 最小终局验证
├── training/       # SFT 与标准 GRPO 桥接
└── evaluation/     # 三臂统一评测

scripts/mvp/
├── prepare_data.py
├── train_sft.py
├── train_grpo.py
└── evaluate.py

configs/mvp/
├── sft.yaml
└── grpo.yaml
```

允许内部复用现有实现，但对外只暴露这条路径。Campaign 脚本、云端运维脚本和 V3 原型不出现在 MVP 快速开始中。

## 8. 面试交付物

最终求职材料只需要：

1. 一张 Agentic RL 数据流图；
2. 一个三分钟运行演示或短视频；
3. Base/SFT/GRPO 对比表；
4. 一张典型成功轨迹；
5. 一张典型失败轨迹及原因分析；
6. 一页 README，说明问题、方法、结果和局限；
7. 可复现配置和测试结果。

简历主句：

> 构建工具增强数学 Agent 的统一运行与训练环境，打通 SFT、标准 GRPO、隐藏答案验证和冻结集评测链路，并通过可回放轨迹分析模型的协议遵从与工具使用行为。

如果 GRPO 没有显著提升，结果表述为：

> 跑通端到端 Agentic RL 训练闭环，并定位小规模 GRPO 的主要限制来自奖励稀疏、组内有效优势不足或工具探索不足。

## 9. 后续扩展顺序

严格按照以下顺序推进：

1. 整理并验证 Agentic RL 最小路径；
2. 统一 Base/SFT/GRPO 评测身份；
3. 生成求职版结果和演示；
4. 可选增加一个 Harness Candidate 对照；
5. 只有求职 MVP 已完成且仍有明确研究需求时，才考虑 BC-GRPO 或完整 Harness Evolution。

任何新增功能如果不能直接提高主叙事的可运行性、可比较性或可解释性，都不进入 MVP。
