# AdaptiveMath-Evo：14 天面试速成知识手册

> 目标读者：具备不扎实的 Python、PyTorch、SFT、LoRA 和强化学习基础，希望在 14 天内形成能够应对面试追问的最小知识闭环。
>
> 建议投入：每天 4～6 小时，总计约 70 小时。
>
> 完成标准：面对项目中的任何模块或数字，都能说明它的定义、来源、控制变量、实现位置、实验局限和下一步验证方法。

---

## 0. 使用方式

这份手册不是完整的机器学习课程。学习顺序围绕一条可执行链路展开：

```text
公开数学数据
  → 清洗、去重、答案自校验
  → Direct / Tool-Integrated SFT 轨迹
  → LoRA 冷启动
  → 同题多路 Agent Rollout
  → Python / SymPy 沙箱执行
  → Deterministic Verifier
  → R0～R3 Reward
  → Group-relative Advantage
  → GRPO Policy Update
  → 冻结集配对评测
  → Failure Mining
  → Harness Patch
  → Champion / Candidate 门禁与回滚
```

每天都要完成四件事：

1. 读懂当天的核心概念；
2. 在项目中找到对应实现；
3. 手算或运行一个最小例子；
4. 不看笔记回答当天的面试问题。

### 真实结果与理想结果必须分开

当前远端 `campaign-2026-09-19` 战役的真实结果是：

| Arm | 正确题数 | Verifier Accuracy | Invalid Prediction | Valid-answer Rate |
|---|---:|---:|---:|---:|
| SFT | 27/200 | 13.5% | 71 | 64.5% |
| R0，50-step | 22/200 | 11.0% | 55 | 72.5% |
| R2，12-step | 25/200 | 12.5% | 69 | 65.5% |

配对结果：

- SFT→R0：准确率 `−2.5pp`，McNemar `p=0.302`，95% CI `[-6.0pp,+1.0pp]`；
- SFT→R2：准确率 `−1.0pp`，McNemar `p=0.625`，95% CI `[-3.0pp,+1.0pp]`；
- 三组准确率差异均不显著；
- R0 的可见收益是 Invalid Prediction 从 71 降到 55，Valid-answer Rate 提升 8 个百分点。

本手册中用于练习表达的高指标，只能作为下一轮实验目标。没有对应 Manifest、逐题结果和统计报告时，不能作为真实项目成果。

---

## 第 1 天：建立项目全景

### 学习目标

- 说清项目要解决的问题；
- 区分模型、Agent、Environment、Verifier、Reward 和 Harness；
- 能在 3 分钟内讲完完整链路。

### 1.1 六个核心对象

#### Policy Model

根据上下文生成下一段推理、工具动作或最终答案。模型本身只看到消息和工具反馈，不应看到私有参考答案。

#### Agent Runtime

负责重复执行以下循环：

```text
构造上下文 → 调用模型 → 解析动作 → 执行动作 → 更新状态 → 判断终止
```

模型负责决策，Runtime 负责约束、执行和记录。

#### Tool Environment

提供 Python、SymPy 等外部能力，并负责：

- Schema 校验；
- 超时和资源限制；
- 返回结构化结果或错误；
- 不向模型泄露参考答案。

#### Verifier

在轨迹终止后，比较预测答案与私有参考答案，产生 `correct`、`incorrect`、`invalid_prediction`、`timeout` 等确定性结果。

#### Reward

将 Verifier 结果和轨迹成本转换为训练信号。Reward 不等于 Verifier：Verifier 判断事实，Reward 决定如何优化。

#### Harness

同一模型权重下，所有会改变 Agent 行为的运行时因素：

- Prompt；
- 动作协议；
- 工具集合；
- Step、Token、工具及执行时间预算；
- 解码参数；
- 训练时采用的 Reward 配置。

### 1.2 一句话项目定义

> AdaptiveMath-Evo 是一个工具增强数学 Agent：内环使用 SFT 和 GRPO 优化模型策略，外环使用失败归因、配对回归与版本门禁优化模型外部 Harness。

### 1.3 代码入口

- Agent 循环：`src/adaptive_math/agent/loop.py`
- 环境：`src/adaptive_math/agent/environment.py`
- 动作解析：`src/adaptive_math/agent/parser.py`
- 工具：`src/adaptive_math/tools/`
- Verifier：`src/adaptive_math/verifier/`
- Reward：`src/adaptive_math/reward/`
- GRPO 桥接：`src/adaptive_math/training/reward_bridge.py`
- verl 环境：`src/adaptive_math/training/verl_environment.py`
- Harness：`src/adaptive_math/harness/`
- Failure Taxonomy：`src/adaptive_math/evolution/taxonomy.py`

### 1.4 面试追问

**问：这和普通调用大模型 API 有什么区别？**

答：普通调用通常是输入到输出的单轮映射；本项目把模型放入有状态环境，模型可多轮选择工具，环境返回真实执行反馈，最终通过私有 Verifier 计算奖励并用于训练。

**问：为什么既要 GRPO，又要 Harness？**

答：GRPO改变模型权重，成本高；Harness优化模型外部配置，固定Checkpoint即可验证，成本低。两者解决不同问题，并使用相同的冻结集和统计纪律评估。

### 当日验收

- [ ] 画出完整架构图；
- [ ] 3分钟讲完项目；
- [ ] 能举例说明“属于Harness”和“不属于Harness”的因素。

---

## 第 2 天：PyTorch 最小训练闭环

### 学习目标

- 理解 Tensor、Parameter、Gradient 和 Optimizer；
- 理解语言模型的 Causal LM Loss；
- 能解释 Micro Batch 和 Gradient Accumulation。

### 2.1 最小训练过程

```python
optimizer.zero_grad()
logits = model(input_ids)
loss = cross_entropy(logits, labels)
loss.backward()
optimizer.step()
```

每一步的含义：

1. `forward` 根据参数计算预测；
2. Loss 衡量预测 Token 与目标 Token 的差异；
3. `backward` 根据链式法则计算梯度；
4. Optimizer 使用梯度更新可训练参数。

### 2.2 Causal LM 的 Token Shift

输入：

```text
[BOS, 我, 会, 调用, Python]
```

训练目标是：

```text
[我, 会, 调用, Python, EOS]
```

位置 `t` 的 Logits 用来预测位置 `t+1` 的 Token。

### 2.3 Attention Mask 与 Loss Mask

两者解决不同问题：

- Attention Mask：决定某个 Token 能看到哪些上下文；
- Loss Mask：决定某个 Token 是否产生梯度。

Agent 轨迹中通常应当：

| Token 来源 | 能进入上下文 | 计算 Loss |
|---|---:|---:|
| System/User Prompt | 是 | 否 |
| 模型推理 Token | 是 | 是 |
| 模型生成的 ToolAction | 是 | 是 |
| 工具返回 Observation | 是 | 否 |
| 模型生成的 FinalAction | 是 | 是 |
| Padding | 否 | 否 |

工具返回不是模型生成的。如果给它计算 Loss，相当于要求模型学习“预测环境输出”，会污染策略梯度。

### 2.4 Batch 概念

- Micro Batch：一次放入GPU并完成前后向的样本数；
- Gradient Accumulation：累计多次Micro Batch梯度后再更新；
- Effective Batch：`micro_batch × accumulation × GPU数`；
- GRPO Group Size：同一道题采样的轨迹数，不等于训练Micro Batch。

### 面试追问

**问：Group Size=4是否意味着4条长轨迹同时反向传播？**

答：不一定。Group Size描述奖励归一化的统计分组；实际更新可以用Micro Batch=1分批处理，通过梯度累积完成一个更新。

### 当日验收

- [ ] 手写一个三层训练循环；
- [ ] 解释 Attention Mask 和 Loss Mask；
- [ ] 解释 Batch、Micro Batch、Group Size 的区别。

---

## 第 3 天：SFT 与 LoRA

### 学习目标

- 理解 SFT 的作用和局限；
- 理解 LoRA 为什么节省显存；
- 能拆分 Base→SFT 和 SFT→GRPO 的收益。

### 3.1 SFT 学到了什么

SFT使用教师轨迹进行最大似然训练，主要提供：

- 输出格式；
- 基础推理模式；
- 何时以及如何调用工具的先验；
- 如何读取工具反馈并继续推理。

SFT擅长模仿数据中的行为，但无法仅凭模仿自动发现超出训练分布的最优工具策略。

### 3.2 LoRA 原理

对原权重矩阵 `W`，不直接更新 `W`，而是学习低秩增量：

\[
W' = W + \frac{\alpha}{r}BA
\]

其中：

- `r` 是 Rank；
- `A` 和 `B` 是可训练的小矩阵；
- 原始 `W` 冻结。

当线性层是 `d_in × d_out` 时：

- 全量训练参数：`d_in × d_out`；
- LoRA参数：`r × (d_in + d_out)`。

Rank远小于隐藏维度时，可训练参数大幅减少。

### 3.3 显存为什么下降

训练显存主要包含：

- 模型权重；
- 梯度；
- Optimizer State；
- 激活；
- KV Cache和临时Buffer。

全量Adam对每个参数通常需要权重、梯度、FP32主副本和两份动量状态。1.7B参数仅这些状态就可能超过24GB。LoRA只为少量参数保留梯度和Optimizer State。

### 3.4 数据来源的正确表达

不能说“Omni-MATH训练集”，因为Omni-MATH是评测Benchmark。可以说：

> 从固定Revision的OpenR1-Math-220k等带训练划分的公开数据中构建冷启动轨迹，经答案自校验、去重和泄漏审计后用于SFT；Omni-MATH或MATH-500只用于冻结评测。

### 3.5 归因实验

至少报告三行：

| Arm | 作用 |
|---|---|
| Base | 原始模型能力 |
| SFT | 冷启动数据与格式对齐收益 |
| GRPO | 在线探索与Outcome Reward增量 |

没有SFT中间结果，就无法判断最终提升来自教师数据还是强化学习。

### 面试追问

**问：为什么不直接GRPO？**

答：小模型若没有动作协议和工具使用先验，早期Rollout容易全部格式错误或全部答错，组内Reward无方差，GRPO得不到有效学习信号。SFT用于把策略带到可探索区域。

### 当日验收

- [ ] 手算一个线性层的全参和LoRA参数量；
- [ ] 能解释Base、SFT、GRPO三段归因；
- [ ] 能正确描述训练数据与评测数据来源。

---

## 第 4 天：数据治理与轨迹构造

### 学习目标

- 理解数据清洗、去重、Split和Manifest；
- 理解Direct、Python-TIR和教师Rollout；
- 能回答数据泄漏问题。

### 4.1 数据管线

```text
Source Registry
→ 固定Revision和License
→ Canonicalization
→ Reference Self-check
→ Quarantine
→ Deduplication
→ Train / RL-dev / Frozen-eval Split
→ Manifest与Hash
```

### 4.2 为什么参考答案要自校验

如果参考答案本身不可解析或错误：

- SFT会学习错误答案；
- RL会奖励错误轨迹；
- 评测会产生虚假结论。

因此每条保留数据都应满足：参考答案经过同一Verifier与自身比较时为正确。

### 4.3 Direct与Python-TIR

#### Direct

模型直接进行推理并输出最终答案，适合不需要外部计算的任务。

#### Python-TIR

Tool-Integrated Reasoning轨迹包含：

```text
模型推理
→ ToolAction
→ 真实Python执行结果
→ 模型继续推理
→ FinalAction
```

关键点是工具结果必须来自真实执行，而不是教师模型伪造。

### 4.4 去重与泄漏

只比较完全相同的字符串不够。数学题可能通过改空格、变量名或题面模板形成近重复，因此可使用MinHash/LSH检测近重复。

必须验证：

- 相同 `task_id` 不跨Split；
- 近重复题不跨训练与冻结评测；
- 评测集不参与SFT、RL、Reward调参和Checkpoint选择；
- Manifest固定源版本、记录数与Hash。

### 面试追问

**问：5,000条轨迹为什么不是简单随机抽样？**

答：需要按难度、答案类型和Direct/Tool行为做均衡采样，否则数据会被简单算术或Direct轨迹主导，模型学不到工具策略。

### 当日验收

- [ ] 能解释数据Manifest的作用；
- [ ] 能列举三种数据泄漏；
- [ ] 能画出Python-TIR轨迹。

---

## 第 5 天：Agent Runtime 与动作协议

### 学习目标

- 理解Agent Loop；
- 理解动作Schema和错误恢复；
- 精确定义“动作协议解析成功率”。

### 5.1 动作类型

模型每轮只能输出一种受支持动作：

- `ToolAction`：调用某个工具并传入结构化参数；
- `FinalAction`：提交最终答案并终止轨迹。

解析器应拒绝：

- 同时出现多个动作；
- JSON不合法；
- 未知工具名；
- 缺少必填字段；
- 标签未闭合；
- Final后继续输出工具调用。

### 5.2 Agent Loop

伪代码：

```python
while not terminated:
    raw = model.generate(context)
    parsed = parse_action(raw)
    if parsed.invalid:
        observation = structured_error(parsed.error)
    elif parsed.is_tool:
        observation = await environment.execute(parsed.action)
    else:
        final_answer = parsed.answer
        terminated = True
    trace.append(raw, parsed, observation)
```

### 5.3 动作协议解析成功率

推荐定义：

> 在全部模型决策轮次中，模型输出可被唯一解析为Schema合法的 `ToolAction` 或 `FinalAction` 的轮次占比。

公式：

\[
\text{Parse Success Rate}
=\frac{\text{合法动作轮次}}{\text{全部模型决策轮次}}
\]

它不代表答案正确率。一个格式完全合法但答案错误的FinalAction，协议解析成功但Verifier判错。

### 5.4 轨迹级与动作级指标

- 动作级解析成功率：分母是全部模型决策轮次；
- 轨迹级协议有效率：分母是全部轨迹，一条轨迹中任意动作非法即可视为失败；
- Valid-answer Rate：最终答案能被Verifier解析的轨迹占比；
- Accuracy：最终答案正确的轨迹占比。

面试中必须说明分母。

### 面试追问

**问：模型输出非法动作后为什么不立即终止？**

答：将结构化错误作为Observation回填，可以让模型学习自修正；但重试次数受预算限制，避免无限循环。

### 当日验收

- [ ] 跟踪一条Direct轨迹；
- [ ] 跟踪一条Tool轨迹；
- [ ] 跟踪一条非法动作恢复轨迹；
- [ ] 能区分四种成功率指标。

---

## 第 6 天：工具、安全沙箱与预算

### 学习目标

- 理解Python和SymPy工具边界；
- 理解SandboxFusion的作用；
- 能解释预算与终止条件。

### 6.1 为什么不能在宿主机直接执行模型代码

模型生成的代码属于不可信输入，可能包含：

- 文件系统访问；
- 网络访问；
- 无限循环；
- 大量内存分配；
- 进程创建；
- 恶意读取环境变量。

生产路径应将Python发送到Linux沙箱，并设置CPU、内存、Wall-time和输出长度限制。

### 6.2 Python与SymPy的职责

#### Python Tool

适合：枚举、数值计算、组合搜索和程序化验证。

#### SymPy Tool

适合：方程求解、因式分解、化简、微积分和符号等价。

SymPy输入同样需要限制，复杂表达式可能造成指数级计算或超时。

### 6.3 Budget

典型预算：

- `max_steps`：最多决策轮数；
- `max_tool_calls`：最多工具调用数；
- `max_python_seconds`：累计Python执行时间；
- `max_observation_chars`：工具输出回填长度；
- `max_new_tokens`：模型生成Token上限。

预算同时服务三个目标：

1. 防止无限循环；
2. 控制推理成本；
3. 形成可比较的实验条件。

### 面试追问

**问：截断工具输出会不会损害正确率？**

答：会，因此Observation预算属于Harness，需要在固定Checkpoint下做配对消融；可以优先保留结构化结果、尾部错误和关键数值，而不是简单截断全部内容。

### 当日验收

- [ ] 列出五类沙箱风险；
- [ ] 解释每种预算的作用；
- [ ] 能说明工具错误如何进入下一轮模型上下文。

---

## 第 7 天：Deterministic Verifier

### 学习目标

- 理解答案抽取、标准化和类型化验证；
- 区分无效答案与错误答案；
- 理解Hidden Verifier边界。

### 7.1 三层结构

```text
Final文本
→ Extractor：提取候选答案
→ Normalizer：清理表面格式
→ Typed Verifier：按答案类型判断等价
```

### 7.2 类型验证

#### 整数和有理数

将 `0.5`、`1/2`、`\frac{1}{2}` 转成统一精确表示后比较，避免浮点误差。

#### 实数

在明确容差规则下比较，容差必须预注册，不能按题目临时修改。

#### 集合

元素顺序无关，但元素重复和类型转换需要明确规则。

#### 元组

顺序相关，`(1,2)` 与 `(2,1)` 不等价。

#### 区间

比较左右端点及开闭性，如 `[0,1)` 与 `(0,1]` 不等价。

#### 符号表达式

使用受限SymPy过程进行化简或等价判断，并设置独立进程和超时。

### 7.3 状态区别

- `correct`：成功解析且等价；
- `incorrect`：成功解析但不等价；
- `invalid_prediction`：预测格式无法形成合法答案；
- `invalid_reference`：参考答案本身不可验证；
- `timeout`：符号验证超时；
- `infrastructure_failure`：进程或服务异常。

### 7.4 Hidden Verifier

产品Agent只接收题目和公开环境，不能获得参考答案或Verifier句柄。参考答案只在轨迹终止后进入私有边界。

否则模型可能：

- 直接读取标签；
- 根据Verifier反馈反复猜答案；
- 学会利用验证器漏洞，而不是解决问题。

### 面试追问

**问：为什么不用另一个LLM当Judge？**

答：数学任务具有可验证终值，规则Verifier成本更低、方差更小且可复现；LLM Judge可用于分析推理质量，但不应替代确定性终值奖励。

### 当日验收

- [ ] 为六类答案各写一个正例和反例；
- [ ] 解释Invalid Prediction与Incorrect；
- [ ] 解释为什么Verifier必须是私有边界。

---

## 第 8 天：GRPO 最小理论闭环

### 学习目标

- 理解为什么同题采样多条轨迹；
- 会手算Group Advantage；
- 理解GRPO与PPO的差别。

### 8.1 为什么需要同题多路Rollout

同一道题采样 `G` 条轨迹，使模型在相同任务难度下比较不同策略。这样可减少不同题目难度造成的Reward尺度偏差。

### 8.2 Group Advantage

对同题Reward：

\[
R=[1.0,0.8,0.0,0.0]
\]

均值：

\[
\mu=0.45
\]

标准差：

\[
\sigma=\sqrt{\frac{\sum_i(R_i-\mu)^2}{4}}
\]

每条轨迹的相对优势：

\[
A_i=\frac{R_i-\mu}{\sigma+\epsilon}
\]

高于组内平均的轨迹得到正Advantage，低于平均的轨迹得到负Advantage。

### 8.3 为什么全零/全一Group无效

若：

```text
R = [0,0,0,0]
```

或：

```text
R = [1,1,1,1]
```

则标准差为0，所有Advantage应设为0。模型无法判断哪条轨迹更好。

因此训练前要监控：

- All-zero Group Rate；
- All-one Group Rate；
- Mixed Group Rate；
- Effective Group Rate。

### 8.4 GRPO与PPO

共同点：都属于On-policy方法，使用旧策略采样，并通过概率比率和Clip限制更新幅度。

主要区别：

- PPO通常训练Value Model估计优势；
- GRPO使用同题组内Reward归一化作为相对优势，可以省去独立Critic。

### 8.5 Policy Version一致性

同一Group内的轨迹必须由同一Policy Version采样。混用新旧策略会破坏组内比较和概率比率的含义。

### 面试追问

**问：GRPO一定比PPO省显存吗？**

答：通常省去Value Model及其Optimizer State，但多路长序列Rollout仍会消耗大量KV Cache和生成时间，瓶颈可能从训练显存转移到Rollout吞吐。

### 当日验收

- [ ] 手算两组Reward的Advantage；
- [ ] 解释全零Group为什么无梯度；
- [ ] 用一分钟比较GRPO和PPO。

---

## 第 9 天：R0～R3 Reward 与 Reward Hacking

### 学习目标

- 30秒讲清R0～R3；
- 理解成本为何只作用于正确轨迹；
- 能列举Reward Hacking及防护。

### 9.1 Reward定义

#### R0：Outcome-only

\[
R_0=\mathbb{1}[\text{correct}]
\]

只验证GRPO是否能提高正确率。

#### R1：增加非法动作惩罚

\[
R_1=R_0-w_i\min(N_{invalid},C)
\]

用于改善动作协议合规性。

#### R2：增加工具成本

\[
R_2=\mathbb{1}[correct]\left(1-w_t\frac{N_{tool}}{B_{tool}}-w_p\frac{T_{python}}{B_{python}}\right)-P_{invalid}
\]

成本只作用于正确轨迹。

#### R3：增加Token成本

\[
R_3=R_2-\mathbb{1}[correct]\cdot w_{tok}\frac{N_{token}}{B_{token}}
\]

用于优化正确率—成本Pareto前沿。

### 9.2 为什么成本只作用于正确轨迹

如果错误轨迹也获得“少调用、短输出”的成本奖励，模型可能学会：

- 不调用必要工具；
- 立即猜答案；
- 提前输出Final；
- 用最短格式稳定答错。

正确性必须是主导目标，成本只在正确解之间排序。

### 9.3 Reward Hacking清单

| 风险 | 现象 | 防护 |
|---|---|---|
| 格式投机 | 格式正确但答案错误 | Accuracy主奖励 |
| 工具回避 | 少调用但猜错 | 成本仅作用于正确轨迹 |
| 工具滥用 | 重复调用工具 | 工具成本、循环检测 |
| Verifier利用 | 构造解析漏洞 | 对抗样例、进程隔离 |
| Token投机 | 极短错误答案 | 正确性优先 |
| Reward饱和 | 全零或全一Group | 难度采样、健康检查 |

### 面试追问

**问：为什么R2可能降低准确率？**

答：如果工具成本权重过大，策略会回避必要工具；即使成本只作用于正确轨迹，也可能在有限探索中改变正确轨迹的相对排序，因此必须做R0/R2消融和Accuracy非劣门禁。

### 当日验收

- [ ] 30秒讲清R0～R3；
- [ ] 手算两条轨迹的R2；
- [ ] 列举五种Reward Hacking。

---

## 第 10 天：verl-agent 接入与多轮训练

### 学习目标

- 能说明使用框架之外自己实现了什么；
- 理解Rollout、环境与训练器的数据接口；
- 能解释Token Mask和Policy Version。

### 10.1 不能只说“接入verl-agent”

框架提供分布式训练和GRPO基础设施，项目实现的关键适配包括：

- 数学任务池加载；
- 同题Group与环境实例映射；
- 多轮模型生成暂停和恢复；
- Python/SymPy真实执行；
- Tool Observation回填；
- 终止轨迹的Verifier调用；
- Reward Bridge；
- 生成Token ID与Action Mask保存；
- Group内Policy Version一致性检查；
- Adapter导出、配置固化和断点恢复。

### 10.2 多轮Rollout数据流

```text
Trainer采样Prompt
→ Policy生成ToolAction
→ Environment暂停生成
→ 执行工具并返回Observation
→ Observation追加上下文
→ Policy继续生成
→ FinalAction
→ Hidden Verifier
→ Reward Breakdown
→ Group Advantage
→ Policy Update
```

### 10.3 Loss Mask

模型真正控制的是它生成的Token，因此：

```text
模型生成：[1,1,1,1]
工具反馈：[0,0,0,0]
Prompt：  [0,0,0,0]
Padding： [0,0,0,0]
```

必须保存生成时的Token ID。事后重新分词可能因Chat Template、空格或特殊Token处理不同而改变边界。

### 10.4 KL配置

KL用于限制新策略偏离参考策略。关闭KL的收益：

- 少加载一份Reference Model；
- 降低显存和计算成本。

风险：

- 策略更新过大；
- 格式或通用能力漂移；
- 过拟合Reward。

关闭KL时应依靠较小学习率、Clip、LoRA、短训练和冻结集门禁控制风险。

### 面试追问

**问：你对verl的贡献是不是只写了一个配置文件？**

答：不是。主要工作是将多轮工具环境接入固定宽度Rollout：维护每条环境状态、处理已终止轨迹的No-op、回填工具Observation、保存Token Mask、计算终止Verifier Reward，并确保同组轨迹来自同一Policy Version。

### 当日验收

- [ ] 画出verl与环境交互时序；
- [ ] 能说明至少六个项目自定义点；
- [ ] 能解释关闭KL的收益和风险。

---

## 第 11 天：24GB 显存与单卡工程取舍

### 学习目标

- 能解释为什么1.7B Agentic GRPO能在4090运行；
- 能完成显存预算；
- 能对齐Step、Batch与轨迹数量。

### 11.1 显存预算

以Qwen3-1.7B、BF16、LoRA Rank=16、Group Size=4为例：

| 项目 | 估算显存 |
|---|---:|
| Actor BF16权重 | 约3.2 GiB |
| Rollout推理权重 | 约3.2 GiB |
| LoRA参数、梯度、Adam状态 | 约0.1 GiB |
| 反向激活 | 约4～6 GiB |
| Paged KV Cache | 约2～3 GiB |
| Logits与临时Buffer | 约2～3 GiB |
| CUDA、Ray与碎片 | 约2～3 GiB |
| 总峰值 | 约17～21 GiB |

这是设计预算，不是实测值。正式简历中的Peak Memory应来自日志或 `nvidia-smi` 采样。

### 11.2 单卡策略

- BF16基础权重；
- LoRA Rank=16；
- Micro Batch=1；
- LogProb Micro Batch=1；
- Gradient Checkpointing；
- Group Size=4；
- Paged KV Cache；
- 工具运行在CPU沙箱；
- Prompt和Observation限长；
- 必要时关闭Reference KL。

### 11.3 轨迹数量对账

若：

- `train_batch_size=8`；
- `group_size=4`；
- `total_steps=50`；

则名义Rollout数量：

\[
50\times8\times4=1600
\]

若训练200 Step，则为：

\[
200\times8\times4=6400
\]

需要说明重试、截断、基础设施失败和恢复是否会改变最终有效轨迹数。

### 11.4 Full Parameter为何不可行

仅FP32 Master Weight和Adam一、二阶状态的粗略占用：

\[
1.7B\times(4+4+4+4)\text{ Bytes}\approx27.2GB
\]

还未包括模型BF16权重、激活、KV Cache和运行时Buffer，必然超过24GB。

### 面试追问

**问：Group Size=4，Batch=8，是否需要同时保存32条完整KV Cache？**

答：逻辑上每Step有32条Rollout，但推理引擎可使用Paged KV和调度分批处理；显存决定并发度，吞吐会受影响。Group统计不要求32条序列同时常驻GPU。

### 当日验收

- [ ] 不看表格讲出显存组成；
- [ ] 手算50和200 Step的轨迹数；
- [ ] 解释LoRA、Checkpointing和Micro Batch分别节省什么。

---

## 第 12 天：评测、Bootstrap 与 McNemar

### 学习目标

- 区分绝对提升、相对提升和百分点；
- 理解配对Bootstrap和McNemar；
- 能识别小样本指标陷阱。

### 12.1 三种变化口径

从13.5%到22.0%：

- 绝对变化：`+8.5个百分点`；
- 相对变化：`8.5/13.5≈63.0%`；
- 200题中：正确题数从27到44，多17题。

简历优先报告“题数＋百分点”，相对提升放在括号中。

### 12.2 为什么使用配对比较

两个模型评测同一批题，结果不是独立样本。每题可以形成：

| Champion | Candidate | 类型 |
|---|---|---|
| 对 | 对 | Unchanged |
| 错 | 错 | Unchanged |
| 错 | 对 | Improved |
| 对 | 错 | Regressed |

真正决定差异的是Improved与Regressed的数量，而非两个总准确率独立比较。

### 12.3 Paired Bootstrap

对任务ID进行有放回采样，每次重新计算准确率差值，重复10,000次，得到经验95% CI。

- CI完全大于0：支持显著提升；
- CI跨0：不能排除无提升或轻微回退；
- 非劣门禁：CI下界高于预注册Margin，例如 `−1pp`。

### 12.4 McNemar Exact Test

只使用Improved和Regressed数量，检验二者是否显著不对称。适合同一批题上的二分类正确/错误结果。

### 12.5 AIME陷阱

AIME每年只有30题。单次Pass@1多对4题就可能变化13.3个百分点，方差很大。

如果使用AIME，应明确：

- 每题采样次数；
- Temperature与Top-p；
- 报告Avg@K还是Pass@1；
- Bootstrap应以题目为Cluster，不能把同题32次采样当作960道独立题。

主指标更适合使用MATH-500或更大的冻结集，AIME作为高难补充。

### 12.6 当前战役结论

当前SFT、R0、R2三组差异均不显著。因此正确说法是：

> 短程GRPO没有产生可验证的准确率增益；R0在协议合规性上出现信号，但需要更大任务池、更多训练Step和独立复现实验。

### 面试追问

**问：p值不显著是否说明GRPO无效？**

答：只能说明当前样本量、训练预算和配置下，没有足够证据证明差异。需要结合效应大小、置信区间、训练健康指标和统计功效决定下一步。

### 当日验收

- [ ] 能计算百分点和相对提升；
- [ ] 能解释Bootstrap与McNemar分别回答什么；
- [ ] 能解释AIME为何不能按960个独立样本统计。

---

## 第 13 天：Harness 与失败驱动迭代

### 学习目标

- 理解HarnessSpec；
- 理解失败分类、Patch和门禁；
- 能回答“这是不是包装后的Prompt Engineering”。

### 13.1 HarnessSpec

HarnessSpec将以下因素统一为不可变配置：

- `prompt_version`；
- `runtime_version`；
- `tools`；
- `budget`；
- `generation`；
- `reward`。

Canonical JSON经SHA-256生成 `spec_hash`。轨迹记录该Hash，确保任何行为结果都能追溯到唯一Harness配置。

### 13.2 Failure Taxonomy

典型标签：

- `FORMAT_INVALID`；
- `NO_FINAL`；
- `WRONG_REASONING`；
- `WRONG_TOOL_CHOICE`；
- `TOOL_CODE_ERROR`；
- `TOOL_TIMEOUT`；
- `TOOL_RESULT_IGNORED`；
- `PREMATURE_STOP`；
- `BUDGET_EXHAUSTED`；
- `VERIFIER_UNSUPPORTED`；
- `INFRASTRUCTURE_FAILURE`。

失败分类的目的不是生成好看的标签，而是决定问题归属于：

- 模型能力；
- Harness配置；
- Verifier；
- 基础设施。

模型能力失败不应通过Prompt Hack掩盖，而应进入RL Hard-example Pool。

### 13.3 Patch主体

当前合理定位是：

> 机器执行确定性失败分类和统计门禁，人工根据失败聚簇提出受Schema约束的单变量Patch。

这不是自动搜索，也不应声称LLM自主改进系统。

### 13.4 为什么一次只改一类字段

若同时修改Prompt、工具预算和Temperature，即使指标提升，也无法判断哪个改动有效。单变量Patch用于保持因果归因。

### 13.5 数据隔离

推荐四层：

| 数据集 | 用途 |
|---|---|
| Train | SFT和RL训练 |
| Evolution Dev | 发现失败和提出Patch |
| Regression | Harness晋升门禁 |
| Final Held-out | 最终一次性报告 |

不能在Regression或Final Held-out上反复修改Harness。

### 13.6 Gate

候选Harness应同时满足：

1. Accuracy点估计不低于允许Floor；
2. Bootstrap CI下界高于非劣Margin；
3. Significant Improvement模式下McNemar达标；
4. Invalid Rate、Token、工具调用和延迟不违反Guard。

### 13.7 Registry与回滚

Registry采用Append-only事件：

```text
register → gate → promote / reject → rollback
```

任何状态变化记录Spec Hash、时间、统计证据和理由。Champion指针只能指向通过门禁的版本，回滚目标必须是历史Promoted版本。

### 面试追问

**问：Harness Evolution不就是Prompt Engineering吗？**

答：Prompt只是Harness的一部分。项目将Prompt、预算、工具、解码和Reward统一版本化，每个Patch由失败证据驱动，在固定Checkpoint和隔离数据集上成对评测，并经过统计与成本门禁后才能晋升，同时保留审计和回滚记录。

### 当日验收

- [ ] 解释HarnessSpec和Spec Hash；
- [ ] 列出四类失败归因；
- [ ] 解释Patch主体和防过拟合策略；
- [ ] 解释Champion/Candidate生命周期。

---

## 第 14 天：项目陈述与压力面试

### 学习目标

- 形成3分钟项目陈述；
- 准备30秒、1分钟和3分钟三种答案；
- 能诚实解释负结果和下一步实验。

### 14.1 三分钟项目陈述模板

#### 30秒：问题与结果

> 我构建了一个工具增强数学Agent，打通数据治理、LoRA SFT、多轮Python/SymPy工具交互、确定性Verifier和Agentic GRPO。模型通过结构化动作协议与沙箱交互，训练只更新模型生成Token。项目还把Prompt、工具、预算和解码参数封装为可版本化Harness，通过失败分类、配对回归和晋升/回滚治理运行时变化。

#### 1分钟：个人核心工作

> 我的核心工作有三部分。第一是Agent Runtime和工具环境，包括动作解析、沙箱执行、反馈回填、预算终止和可回放轨迹。第二是训练链路，将终止Verifier结果转换为R0到R3奖励，对同题轨迹计算Group Advantage，并完成verl-agent的多轮环境适配和Token Loss Mask。第三是实验治理，固定数据、Prompt、Verifier和Checkpoint Hash做配对Bootstrap与McNemar检验，并为Harness建立内容寻址配置和版本门禁。

#### 3分钟：实验与反思

> 当前真实战役在OmniMath冻结200题上比较SFT、R0和R2。SFT、R0、R2分别答对27、22、25题，配对差异均不显著。R0把无效预测从71降到55，有效答案率提升8个百分点，说明短程GRPO首先改善了协议合规性，但没有形成准确率增益。这个负结果帮助我定位下一轮重点：扩大RL任务池和训练Step、提升Mixed-reward Group比例、验证工具反馈是否真正被利用，并在任何能力提升声明前扩大冻结评测集。

### 14.2 高频压力问题

#### 数据

1. 数据从哪里来，Revision和License是什么？
2. 参考答案如何自校验？
3. 如何避免OpenR1与评测集近重复？
4. SFT数据中Direct和Tool轨迹比例是多少？

#### Agent

5. 动作协议解析成功率的分母是什么？
6. 工具执行失败后如何恢复？
7. 为什么Python不能在宿主机运行？
8. Agent如何避免无限工具循环？

#### Training

9. SFT和GRPO分别贡献多少？
10. 工具Observation为什么Mask？
11. GRPO Advantage如何计算？
12. 全零Reward Group如何处理？
13. 为什么关闭KL，风险是什么？
14. verl-agent具体修改了什么？

#### Reward

15. R0～R3分别解决什么问题？
16. 为什么错误轨迹不奖励成本节省？
17. 如何检测Reward Hacking？
18. R2为什么可能比R0差？

#### Evaluation

19. 为什么使用配对Bootstrap？
20. McNemar检验输入是什么？
21. AIME 30题如何报告可信结果？
22. `p>0.05`应该如何解释？

#### Harness

23. Harness与Prompt Engineering的区别？
24. Patch由谁提出？
25. 如何避免在Evolution Dev上过拟合？
26. 非劣Margin为什么是1pp？
27. Registry为什么必须Append-only？

#### Resource

28. 24GB显存如何分配？
29. 50 Step、Batch=8、Group=4有多少Rollout？
30. Rollout吞吐和训练吞吐哪个是瓶颈？

### 14.3 不知道答案时的正确应对

不要编造。使用以下结构：

> 这个值当前没有独立测量，我能确认的是……；如果验证，我会固定……，记录……，再通过……判断。

例如：

> 当前没有独立记录KV Cache峰值，我只能给出基于模型结构和并发度的预算。正式验证会采集 `nvidia-smi` 时间序列，并把峰值、并发序列数和实际上下文长度写入Run Manifest。

### 当日验收

- [ ] 完成两轮45分钟模拟面试；
- [ ] 30个问题全部能在1分钟内回答；
- [ ] 对每个简历数字说明定义、分母和证据文件；
- [ ] 能主动讲出项目负结果和下一轮实验。

---

## 附录 A：面试最小公式表

### LoRA

\[
W'=W+\frac{\alpha}{r}BA
\]

### Group Advantage

\[
A_i=\frac{R_i-\mu_R}{\sigma_R+\epsilon}
\]

### R0

\[
R_0=\mathbb{1}[correct]
\]

### R2

\[
R_2=\mathbb{1}[correct](1-C_{tool}-C_{python})-P_{invalid}
\]

### R3

\[
R_3=R_2-\mathbb{1}[correct]\cdot C_{token}
\]

### 相对提升

\[
\text{Relative Improvement}=\frac{new-old}{old}
\]

### 名义Rollout数

\[
N_{rollout}=N_{step}\times B_{prompt}\times G
\]

---

## 附录 B：简历数字证据清单

每个数字至少应关联以下证据：

| 数字 | 必要证据 |
|---|---|
| 数据条数 | Dataset Manifest、Hash、过滤报告 |
| SFT准确率 | Checkpoint、评测Manifest、逐题Prediction |
| GRPO提升 | 同题配对结果、Bootstrap CI、McNemar p值 |
| Invalid Rate | 状态定义、分母、逐题Verifier结果 |
| 工具调用下降 | Trace聚合脚本、相同任务和预算 |
| Token下降 | Tokenizer Revision、生成Token统计 |
| 延迟下降 | 硬件、并发、Warm-up和计时边界 |
| 峰值显存 | `nvidia-smi`或PyTorch Memory Snapshot |
| 训练时长 | Start/End时间、故障恢复记录 |

---

## 附录 C：最终自测评分

每项0～2分：

- 0：不会；
- 1：能背定义，不能结合代码；
- 2：能结合代码、实验和局限解释。

| 模块 | 分数 |
|---|---:|
| PyTorch训练闭环 | /2 |
| SFT与Loss Mask | /2 |
| LoRA与显存 | /2 |
| 数据治理 | /2 |
| Agent Runtime | /2 |
| Tool Sandbox | /2 |
| Verifier | /2 |
| GRPO | /2 |
| R0～R3 Reward | /2 |
| verl环境适配 | /2 |
| 统计检验 | /2 |
| HarnessSpec | /2 |
| Failure Taxonomy | /2 |
| Gate与Registry | /2 |
| 实验复盘 | /2 |

总分：

- 24～30：可以参加项目深挖面试；
- 18～23：可以讲项目，但需要继续补代码和实验；
- 12～17：容易在第二层追问暴露；
- 0～11：应先完成最小代码跟踪和Smoke实验。

---

## 最终原则

面试官真正判断的不是你是否记住GRPO公式，而是你是否理解每个工程选择的因果关系：

- 为什么需要这个模块；
- 它接受什么输入、产生什么输出；
- 它可能在哪些条件下失败；
- 你如何证明改动有效；
- 你如何避免把别人的开源结果或理想目标写成自己的实测结果。

当你能对项目中的每个数字回答“定义、来源、分母、控制变量、证据和局限”时，这个项目就从简历包装变成了可以防守的工程经历。
