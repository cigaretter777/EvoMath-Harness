# R2 工具成本信号诊断方案

- 日期：2026-10-23
- 状态：**已按 2026-10-23 用户确认修订并冻结；2026-10-23 已执行，判定 C，结果见 `../r2_cost_signal_20261023/diagnostic_report.md`。** 本轮只执行计时诊断，不启动续训，不修改奖励/数据/任务集/训练目标，不删除任何产物。
- 目标：回答一个有限的问题——**R2 的 `time_cost` 在 all-correct 组里产生的差异，是程序级稳定差异（信号），还是同一程序重复执行的计时波动（噪声）？**
- 不回答的问题：时间成本是否提高准确率、是否值得保留、权重是否应调整。这些问题不在本诊断的决策范围内；无论结果如何，本轮都不改 `time_weight`、`max_tool_seconds`、奖励公式、任务池或 schedule。

---

## 1. 现有证据

### 1.1 R2 的计分公式

`configs/reward/r2_unified.yaml`：

```yaml
version: r2-v2-unified-tool-time
tool_weight: 0.15
python_weight: 0.0
time_weight: 0.10
invalid_weight: 0.10
clip_min: -1.0
clip_max: 1.0
```

`configs/agent/unified_tool_budget_v1.yaml`：

```yaml
max_steps: 6
max_tool_calls: 4
max_python_seconds: 0.0
max_observation_chars: 8000
max_tool_seconds: 60.0
max_tool_call_seconds: 30.0
```

`src/adaptive_math/reward/functions.py` 中，正确轨迹的 time term 是：

```text
time_cost = time_weight * tool_seconds / max_tool_seconds
          = 0.10 * tool_seconds / 60
          = tool_seconds / 600
```

只有 `correct=1` 时才计费；总奖励再裁剪到 `[-1, 1]`。因此 1 秒的差异只值 `1/600 ≈ 0.001667` 的原始奖励；0.584 秒的中位 spread 只值约 `0.000973`。

### 1.2 50 步 R2 的 all-correct 结构

`artifacts/experiments/midterm_uniform_reward_v1/audit/posthoc_20261023/r2_cost_advantage.json`：

| 指标 | 值 |
|---|---:|
| 组总数 | 400 |
| mixed | 187 |
| all_correct | 33 |
| all_wrong | 180 |
| all-correct 轨迹数 | 132（33 × 4） |
| all-correct 组中 tool_cost 有变化的组 | 1 |
| all-correct 组中 time_cost 有变化的组 | 33 |

时间 spread（秒）：

| 统计量 | 值 |
|---|---:|
| min | 0.124824 |
| median | 0.583980 |
| p75 | 0.753614 |
| max | 1.213758 |
| >0.5s 的组数 | 22 / 33 |

组内 std（原始奖励）：

| 统计量 | 值 |
|---|---:|
| min | 8.526e-05 |
| median | 0.0004163 |
| max | 0.019235 |

GRPO 组内标准化后的最大绝对 advantage：

| 统计量 | 值 |
|---|---:|
| per-group min | 0.968127 |
| per-group median | 1.344169 |
| per-group max | 1.499787 |
| 132 个样本中 abs > 0.5 | 85 |
| 132 个样本中 abs > 1.0 | 36 |

反事实：

| 反事实 | median max abs advantage |
|---|---:|
| 只用 tool_count 变化 | 0.0 |
| 只用 time_cost 变化 | 1.334225 |
| 实际 | 1.344169 |

### 1.3 关键解释

原始奖励的组内差异只有约 `1e-4` 到 `1e-3`；GRPO 用同一量级的组内 std 去做标准化，所以 advantage 变成 O(1)。**这本身不是错误**：GRPO 的设计就是按组内标准差归一化。但它意味着：如果 `tool_seconds` 的组内差异主要是计时噪声，GRPO 会把噪声放大成 O(1) 的 advantage。

因此，关键不是“time_cost 的绝对值小”，而是：

> 同一段代码重复执行时，`tool_seconds` 的程序内波动有多大？程序间的稳定差异有多大？观测到的组内 spread 是否超出“只有程序内噪声”的零假设？

### 1.4 程序计数：只读复核结果

从 `training/r2/rollout_audit.jsonl` 的 `action_text` 中解析 `<tool_call>`：

| 项 | 数量 |
|---|---:|
| all-correct 轨迹 | 132 |
| 可解析的 Python `code` 出现次数 | 126 |
| 唯一 Python 程序数 | 125 |
| 含 `import sympy` 的出现次数 | 103 |
| 含 `import sympy` 的唯一程序数 | 102 |
| `action_text` 被审计截断到 800 字符、无法恢复 code 的轨迹 | 6 |
| 唯一程序代码长度 min / median / max | 74 / 227 / 669 字符 |

补充：

- 132 条轨迹各有 1 条 active `tool_calling=1` 行；其中 **126 条**可解析出 Python `code`，**6 条**因 `rollout_audit.jsonl` 的 `action_text` 被截断到 800 字符而无法恢复代码。截断不等于模型调用非法，但在本诊断中只能按缺失处理。
- 只有 **1 个** all-correct 组出现 `tool_cost` 变化：该组 1 条轨迹 `tool_cost=0`，另外 3 条 `tool_cost=0.0375`。这不是“模型选择了 0 次工具调用”：该轨迹有 `tool_calling=1` 行，但工具未执行（基础设施失败），所以环境没有计 `tool_calls`/`tool_seconds`。该轨迹的代码仍可解析，但**没有正的时间观测**，不进入时间稳定性统计。
- 其余 32 个组四条轨迹的 `tool_cost` 都是 `0.0375`（各 1 次成功调用），只有 `time_cost` 变化。
- 按“有可解析程序 **且** 有正的 `sandbox_execution_time`”计算可用轨迹，33 组分成：**27 组 n=4、5 组 n=3、1 组 n=2**；共 125 条可用轨迹。缺失的 7 条 = 6 条截断 + 1 条基础设施失败。这个分布是主分析的可用案例集；缺失规则见 §3.3。
- 这里的“含 `import sympy`”是 **Python 程序内部 import sympy**，不是调用 `sympy` 工具；它们仍然通过 SandboxFusion 的 Python 执行路径计时。

### 1.5 实验条件与局限：三臂实际优化器更新次数

从三个 `global_step_50` 的 `actor/optim_world_size_1_rank_0.pt` 只读核对：

| 臂 | active-turn 行数 | `sum(ceil(active_rows / 8))` | checkpoint optimizer `step` |
|---|---:|---:|---:|
| R0 | 3487 | 456 | 456.0 |
| R1 | 3277 | 431 | 431.0 |
| R2 | 3245 | 422 | 422.0 |

这说明：

- 三个臂名义上都是 50 个 global update，但实际 optimizer update 次数是 **R0 456、R1 431、R2 422**；它们不是同一个数。
- 差异来自 `ppo_mini_batch_size=8` 对每轮 active-turn 行的分块：R2 为 28 轮 8 块 + 22 轮 9 块，R0/R1 同理落在 8–11 / 8–10 块。
- 这是实验条件的一部分，也是解释局限：**不能把 R0/R1/R2 的差异只归因于奖励变体**，因为三个臂的优化器更新次数并未对齐。本诊断只分析 R2，不做三臂比较；后续报告必须把 456/431/422 写入条件表和局限，而不是把它们误写成“都是 50 步”。

---

## 2. 时间字段来源核查

### 2.1 字段链路

| 字段/来源 | 产生位置 | 语义 | 是否进入 R2 `time_cost` |
|---|---|---|---|
| SandboxFusion `execution_time` | `SandboxFusionClient.run_code` 解析响应 `run_result.execution_time` | 沙箱内命令的执行时长；服务端字段 | 是，优先 |
| `PythonTool` 的 `tool_seconds` | `tools/python_tool.py::_annotate_timing` | 若 `execution_time` 存在且非负：`tool_seconds=execution_time`，`timing_source="sandbox_execution_time"` | 是 |
| `latency_ms` | `SandboxFusionClient.run_code` 用 `time.perf_counter()` 包住 HTTP 请求 | 客户端观测的往返时间，含网络/排队/服务时间 | 仅作为 fallback |
| `client_round_trip` fallback | `_annotate_timing` 在 `execution_time` 缺失时 | `tool_seconds=max(latency_ms/1000, requested_timeout or 0)`，`timing_source="client_round_trip"`，`timing_missing=true` | 是，但标记缺失 |
| `UNAVAILABLE` | `run_code` 的传输/5xx 错误 | `executed=false`，`timing_missing=true`，不计工具时间 | 否（环境按基础设施错误处理） |
| SymPy `worker_process` | `tools/sympy_tool.py::_run_isolated` 用 `time.perf_counter()` 包住 `spawn + join` | 子进程 wall time，含 spawn、SymPy import、运算 | **当前 R2 数据中未实际发生**（见 §2.3） |
| `allowance_seconds` / `remaining_tool_seconds` | 环境预算 | 剩余可用时间，不是测量值；用于派生 per-call timeout | 否，仅作为 timeout 上限 |
| `max_tool_call_seconds=30` | 预算配置 | 单次调用上限 | 否，作为 `requested_timeout` |
| `monotonic_ms` | agent loop | 回合/动作的单调时钟时间戳 | 否 |
| `python_seconds` | 环境 usage | legacy Python-only 计量；本实验 `max_python_seconds=0`、`python_weight=0` | 否（`python_cost=0`） |
| trainer `timing_raw`（gen/step/save 等） | verl trainer | 训练阶段耗时 | 否，不是奖励输入 |

环境侧取值：

```python
# src/adaptive_math/agent/environment.py
def _tool_seconds(result):
    value = result.metadata.get("tool_seconds")
    if isinstance(value, int | float) and value >= 0:
        return float(value)
    return max(0.0, result.latency_ms / 1000.0)
```

即：**R2 的 `tool_seconds` 优先读 `metadata["tool_seconds"]`；只有缺失时才退回客户端 `latency_ms`。** 训练审计的 step 记录没有逐调用的 `timing_source`/`timing_missing`，但有 `reward_components.time_cost`，可反推 `tool_seconds = time_cost * 600`（仅对 correct 样本）。逐调用的来源信息在评测 v2 的 JSONL 中有；训练侧需要 replay 才能直接记录。

### 2.2 评测 v2 的来源事实

`timing_measurement_audit.json` 的评测部分：

- SFT / R0 / R2：所有 Python 调用都是 `sandbox_execution_time`，`timing_missing=false`；没有 timeout。
- R1：153 次正常 `sandbox_execution_time`，42 次 `execution_error` 但仍有 `sandbox_execution_time`，另有 **3 次** `client_round_trip` + `timing_missing=true` + `infrastructure_error=true`，`tool_seconds≈31.03s`（请求 timeout 30s + 往返）。
- 这 3 次不是“真实执行了 31 秒”，而是 fallback 计费；分析时必须单独标记。

### 2.3 必须更正的一条审计注记

`timing_measurement_audit.json` 的 `notes` 写：

> “Training is predominantly python; a small number of sympy calls use worker_process wall time. R2's time term therefore mixes sandbox execution time and worker-process wall time.”

只读复核后，这条推断**不成立**：

- 训练期 active tool 名称统计：R0 `python=1588, sympy=10, unparsed=10`；R1 `python=1583, unparsed=1`；R2 `python=1567, sympy=1`。
- 审计计数的 R0 10 次 / R2 1 次 SymPy 调用中，所有可解析的顶层参数都不满足 `operation` + `expression`：多数只带 `code`，另有带 `arguments_schema` 或 JSON 截断的行；`SympyArguments` 的 `extra="forbid"` 使 `code`/`arguments_schema` 直接失败。即使某行带了 `operation`，也因缺 `expression` 而失败。`ToolRegistry` 在 `model_validate` 阶段返回 `INVALID_ARGUMENTS`，`executed=false`；没有一次能构造出合法的 `SympyArguments` 并进入 worker。
- 因此 `SympyTool._run_isolated` 从未被调用，`worker_process` 计时从未产生；环境也没有计 `tool_calls`/`tool_seconds`。
- 评测 v2 全部是 Python；没有 SymPy 调用。

**结论：当前 50 步 R2 的 `time_cost` 是纯 Python SandboxFusion `execution_time`，不是混合边界。** 这是本次诊断的主前提。

但这只是“当前数据未触发”，不是“设计上不会触发”：

- 如果第 51–100 步或未来训练出现合法的 SymPy 调用，`SympyTool` 会返回 `worker_process` wall time（含 spawn + import），环境会把它计入同一个 `tool_seconds`；
- 那时 time term 会混合“沙箱内 Python 执行时间”和“子进程 spawn/import/运算时间”，两者边界不同；
- 本方案把它列为**潜在风险**，并用一个可选的 SymPy 小电池量化其量级（见 §3.4），但不把它混入 primary 结论。

### 2.4 不能混淆的字段

- `execution_time` 是服务端执行时长；`latency_ms` 是客户端往返；两者不是同一量。R2 优先用前者，所以网络抖动通常不会直接进入奖励；但 timeout/缺字段时会退回后者。
- `allowance_seconds` 是“还剩多少预算”，不是“花了多少时间”；不能拿它当测量值。
- `monotonic_ms` 是 agent 回合时间戳，不是工具执行时间。
- `python_seconds` 在本实验为 0（`max_python_seconds=0`，且 `python_weight=0`）；不要把 `python_seconds` 与 `tool_seconds` 混用。
- trainer 的 gen/step 时间不是奖励输入；R2 只看 `tool_seconds`。

---

## 3. 固定程序电池

### 3.1 来源与选择

- 来源：`artifacts/experiments/midterm_uniform_reward_v1/training/r2/rollout_audit.jsonl` 中 33 个 all-correct 组的 132 条轨迹。
- 只读解析每条轨迹的 `action_text` 中的 `<tool_call>`，取 `name=="python"` 且 `arguments.code` 为字符串的调用。
- 去重：按**完全相同的代码字符串**去重；同一程序在不同轨迹/不同组出现时只测一次，但保留全部 occurrence 映射。
- 结果：126 次出现，125 个唯一程序。**全部 125 个都进入电池**，不按耗时、代码长度、是否 import sympy 做筛选，避免选择偏差。
- 唯一例外：6 条轨迹的 `action_text` 被审计截断到 800 字符，无法恢复 code；它们只作为缺失计数报告，不进入电池，也不参与稳定性统计。
- 另 1 条轨迹的 code 可解析，但工具未执行（基础设施失败），没有正的时间观测；该程序进入电池，但该轨迹不进入时间稳定性统计。

### 3.2 冻结与标识

为每个唯一程序生成：

```text
program_id = sha256(exact_code_utf8)[:16]
```

`program_battery.json` 的记录至少包含：

```json
{
  "program_id": "…",
  "code": "…",
  "code_sha256": "…",
  "code_length": 227,
  "imports_sympy": true,
  "occurrences": [
    {
      "global_update": 7,
      "group_id": "g7:…",
      "prompt_slot_id": "g7:s0",
      "trace_id": "rollout:…:0",
      "sample_index": 0,
      "step_index": 1,
      "observed_time_seconds": 0.5275,
      "observed_reward": 0.9603,
      "observed_time_cost": 0.000879,
      "observed_tool_cost": 0.0375,
      "case_status": "complete"
    }
  ]
}
```

`group_trajectory_program_map.json` 保留全部 33 组 × 4 条轨迹的映射，包括：

- `global_update`, `group_id`, `prompt_slot_id`, `trace_id`, `task_id`, `sample_index`;
- `program_id`（可解析时）或 `unparsed_reason="action_text_truncated_800"`;
- 训练观测的 `reward`, `reward_components`, `time_seconds=time_cost*600`;
- `available_case`（是否进入主分析）、`complete_case`（是否属于 27 个完整 4 条组）、`notes`（missing/zero_time）。

`round_plan.json` 在计时前冻结：25 轮，每轮为 125 个 `program_id` 的固定种子随机排列；`round_index=1..5` 为 warm-up，`6..25` 为 measure；每个元素记录 `round_index`, `phase`, `position_in_round`, `program_id`。

冻结规则：

1. 在计时前生成 `program_battery.json`、`group_trajectory_program_map.json`、`round_plan.json`；
2. 对 canonical JSON（`sort_keys=True`、固定 separators、UTF-8）分别计算 SHA-256，写入 `program_battery.sha256`、`group_trajectory_program_map.sha256`、`round_plan.sha256`；
3. 计时脚本启动时重新计算并比对；不一致则停止，不开始计时；
4. 电池和轮次计划一旦冻结，不因看到计时结果而增删程序或重排轮次。

### 3.3 组映射、可用案例与缺失规则

**全部 33 个 all-correct 组都保留在映射和报告中，不静默删除。** 主分析使用“有可解析程序且有正 `sandbox_execution_time`”的可用轨迹，按原组计算：

| 组类型 | 组数 | 每组可用轨迹 | 说明 |
|---|---:|---:|---|
| 完整可用 | 27 | 4 | 4 条轨迹均有 code 和正时间 |
| 部分可用 | 5 | 3 | 4 条中有 1 条 `action_text` 截断或工具未执行 |
| 最少可用 | 1 | 2 | 4 条中有 2 条 `action_text` 截断 |
| 合计 | 33 | 125 条可用轨迹 | 7 条缺失 = 6 条截断 + 1 条基础设施失败 |

处理规则：

- 每个组的观测优势**在可用案例集内重算**，使用与训练相同的组内均值、`stdev(ddof=1)` 和 `+1e-6`；27 个完整组的可用案例优势与训练原优势在浮点误差内一致，另外 6 组在报告中标注“可用案例重算”，并同时列出原始 4 条优势。
- 主判据聚合全部 33 个组（每组等权），`S_order`/`S_adv`/`S_spread` 按该组可用 `n_g` 计算；`n_g=2` 的组只有一个 pair，结果单独标注为低信息量。
- **完整案例敏感性分析**：只用 27 个 `n_g=4` 组重算同一组指标。若主分析（33 组）与完整案例（27 组）在 A/B 上冲突，判定归为 **C**，不强行选边。
- 6 条截断轨迹的 `group_id`、`sample_index`、`trace_id` 和缺失原因写入映射；1 条零时间轨迹记录 `zero_time_infrastructure_failure`。报告缺失数量、组分布和对聚合权重的影响。
- 不使用插补、不按耗时删程序、不把截断轨迹当作“模型未调用工具”。

### 3.4 扩展与口径核对（不进入 primary 结论）

| 扩展 | 目的 | 规模 | 用途 |
|---|---|---|---|
| 评测 v2 外部样本 | 检查训练 all-correct 子集是否特殊 | 20 个评测程序 × 10 reps | 可选；本轮不执行 |
| SymPy 口径核对 | 核对“请求的 sympy 调用”与“实际 worker 执行”的边界差 | 5 个固定表达式 × 5 reps × 2 路径 = 50 calls | 本轮执行，非 primary；见 §4.6 |
| 并发条件 | 模拟训练时多 env 同时调用 | 30 个程序 × 10 reps，8 并发 | 可选；本轮不执行，primary 只看顺序交错 |

这些扩展必须在报告里明确标注“非 primary”，不能用来替代主电池的结论。

---

## 4. 重复计时设计

### 4.1 主条件 C1：随机交错轮次、固定电池、顺序执行

- 对 125 个唯一程序，执行 **25 个完整轮次**；每轮包含全部 125 个程序各一次。
- 每轮顺序：用固定 seed `20261023` 生成 125 个 `program_id` 的随机排列；每轮独立重排。**正式计时采用随机交错轮次**，不是“每个程序连续跑 25 次”。
- 前 **5 轮**为 warm-up（625 次调用，不计入统计）；第 **6–25 轮**为 measure（20 轮 × 125 = **2,500 次**正式调用）。总调用数 **3,125**。
- Warm-up 的含义（预注册，不得事后改变解释）：
  - 让 SandboxFusion 进程、Python import/编译缓存和客户端连接经历 5 个完整轮次；
  - 保证每个程序至少执行 5 次，以暴露明显的冷启动/超时问题；
  - **不是**冷启动测量，也**不是**“每程序独立预热后进入稳态”的声明；
  - warm-up 数据一律不进入均值、方差、排序或优势稳定性统计；
  - 由于轮次交错，程序之间的缓存/负载影响不能完全隔离，这是本设计的已知局限。
- 执行方式：轮内严格顺序、一次一个 `ToolRegistry.execute("python", ...)`；primary 不使用并发。
- 执行路径：**必须走训练同款 `PythonTool` + `SandboxFusionClient`**，不要直接 curl `/run_code`，否则会绕过 `_annotate_timing`，测到的不是 R2 实际使用的 `tool_seconds`。
- 上下文：

```python
ToolContext(
    trace_id=f"diag:{program_id}:r{round_index}:p{position_in_round}",
    task_id=program_task_id,
    remaining_observation_chars=8000,
    remaining_python_seconds=None,
    remaining_tool_seconds=60.0,
    max_tool_call_seconds=30.0,
)
```

这与训练第一次工具调用的上下文一致：`_remote_timeout` 返回 `min(60, 30)=30` 秒；`_annotate_timing` 走 unified 路径。

- 记录字段（每条调用一行 JSONL）：
  - `timestamp_utc`, `condition="C1_interleaved"`, `program_id`, `round_index`, `phase`（warmup/measure）, `position_in_round`
  - `call_index`（全序列 1..3125）, `rep_index`（该程序第几次，1..25）
  - `ok`, `error_code`, `executed`
  - `tool_seconds`（`metadata["tool_seconds"]`，R2 实际计费值）
  - `latency_ms`
  - `timing_source`, `timing_missing`, `timeout_limit`
  - `output_sha256`, `output_length`, `truncated`
  - `requested_timeout`（应为 30.0）
  - `sandbox_url`, `sandbox_sha`（固定 `469de0845cb7d665bc68044d541aa3aa28c5b30a`）
  - `client_pid`, `host_loadavg`, `cpu_count`
- `round_plan.json` 预先记录每轮排列；`timing_repeats.jsonl` 逐行 flush，保留完整轮次记录。不保存完整输出，只保存 hash 与长度，控制日志体积。

### 4.2 预检（在正式 3,125 次之前）

1. `GET /v1/ping`（`SandboxFusionClient.health()`）必须成功；
2. 重新计算并比对 `program_battery.sha256`、`group_trajectory_program_map.sha256`、`round_plan.sha256`；不一致立即停止；
3. 用电池中代码长度最小的程序跑 1 次，确认 `ok=true`、`timing_source="sandbox_execution_time"`、`tool_seconds>0`；
4. 用代码长度最大的程序跑 1 次，确认不超 30s；
5. 记录 server URL、client 版本、Python 版本、CPU/内存、当前 loadavg、时间戳；
6. 确认输出目录为空；若预检失败，停止；不开始正式计时。

### 4.3 可选 C2：并发（本轮不执行）

- 30 个程序（按 program_id 排序后等距抽取，固定 seed）× 10 reps；
- 8 个并发 client（对应训练时多 env 同时调用）；
- 记录同样的字段，额外记录 `concurrent_group`、`start_offset_ms`；
- 这是 secondary；primary 结论只看 C1。本轮因单 CPU、当前 loadavg 高且用户要求聚焦，**不执行 C2**；若后续执行，C1/C2 冲突归为 C。

### 4.4 可选 C3：冷启动（本轮不执行）

- 5 个程序 × 5 次，在服务空闲 60s 后第一次调用；
- 只用于解释 warm-up 之外的冷启动分布；不进入 primary。本轮不执行。

### 4.5 失败与缺失处理（预注册）

**测量有效性定义**：

- `valid`：`timing_source == "sandbox_execution_time"`、`executed == true`、`tool_seconds` 为有限非负数；
- `missing`：`timing_missing == true`、`timing_source == "client_round_trip"`、`UNAVAILABLE`、无执行、非有限值或 timeout；记录但不进入 primary；
- `execution_error_with_time`：`executed == true` 且有 `sandbox_execution_time`；时间进入 primary，但单独报告，`ok=true` 子集作为 sensitivity。

**处理规则**：

- primary 中每次调度只尝试一次，**不自动重试**；失败不补跑，避免按结果选择性采样。
- warm-up 的 5 次中，若某程序出现 ≥2 次 missing/timeout，标记 `unstable`，跳过其所有 measure 调用；记录跳过原因。
- measure 中若某程序有效重复 <10 次，标记 `insufficient_reps`，不进入程序均值/排序；其所属组若可用轨迹 <2，整组标记 `not_evaluable`，不参与聚合。
- 每个程序若 `missing` 超过 2/25，报告并做 sensitivity；不按耗时删除程序。
- 全局保护：每轮结束后检查累计 missing 率；若总调用 ≥125 且 missing 率 >5%，停止并写 `status=paused_missing_rate`，不擅自提高 timeout、不改代码、不改设计。
- 若服务不可达或 `health()` 失败，立即停止并保留部分 JSONL；不把部分数据伪装成完整诊断。
- 所有缺失、跳过、失败和零时间都在 `diagnostic_report.md` 中列明；不静默丢弃任何组。

### 4.6 SymPy 请求与实际执行的口径核对（非 primary）

目的：回答“原训练里的 sympy 请求是否真的执行过”和“如果执行，`worker_process` 与 Python `execution_time` 的边界差多大”。

**静态核对**：

- 遍历 R0/R1/R2 的 `rollout_audit.jsonl`，统计 active `tool_calling` 行中 `name=="sympy"` 的请求；
- 用 `SympyArguments` 的同一套规则（`extra="forbid"`，需要 `operation` + `expression`）判定是否能通过 `ToolRegistry.model_validate`；
- 区分 `requested`、`schema_valid`、`executed`、`timing_source=worker_process`；当前 R2 预期为 requested 1、schema_valid 0、executed 0。
- 这一静态核对不修改任何轨迹，只读。

**动态口径小电池**：

- 5 个固定表达式，覆盖 `simplify`, `factor`, `expand`, `diff`, `solve`；
- 每个表达式走两条路径各 5 次：
  1. `SympyTool` 的 `worker_process` 路径（本地 spawn + import + 运算）；
  2. 等价的 Python 程序走 `PythonTool`/SandboxFusion 的 `sandbox_execution_time` 路径；
- 记录 `tool_seconds`, `timing_source`, `latency_ms`, `ok`, `output_sha256`；
- 目的仅是量化边界差（spawn/import 是否显著），**不进入 primary，不改变 R2 当前数据结论**；
- 如果 SymPy 路径因本机资源失败，记录失败，不把它解释为 R2 有 SymPy 计费。

---

## 5. 统计与判读

### 5.1 程序内统计

对每个程序 `p`，使用第 6–25 轮中所有 `valid` 的正式重复（计划 20 次；失败/缺失可能使 `n_p<20`）：

```text
n_p, mean_p, sd_p, median_p, MAD_p, CV_p = sd_p / mean_p
min_p, max_p, p05_p, p95_p
```

同时报告：

- 用 `tool_seconds` 的结果（primary）；
- 用 `latency_ms` 的结果（secondary，用于判断客户端/排队噪声）；
- 用 `median`/`MAD` 的稳健版本（sensitivity）；
- 每个程序的 `round_index` 序列，用于检查交错轮次中的漂移。

### 5.2 方差分解与 ICC

对程序 `p` 的第 `r` 次重复：

```text
t_{p,r} = μ + b_p + ε_{p,r}
b_p ~ N(0, σ_b²)      # 程序间稳定差异
ε_{p,r} ~ N(0, σ_w²)  # 程序内重复噪声
```

估计：

- `σ_w²`：各程序内方差的 pooled estimate（或 REML）；
- `σ_b² = max(0, Var(mean_p) - σ_w² / n_harmonic)`；
- `ICC = σ_b² / (σ_b² + σ_w²)`。

报告：

- `σ_b`, `σ_w`（秒；同时换算成 reward 单位：除以 600）；
- `ICC` 及按程序 bootstrap 的 95% CI（重采样程序，不重采样重复；B=2,000）；
- 若 `σ_b²` 被截断为 0，明确报告“程序间方差不可估计”，不要把它当作精确 0。

**注意：`σ_b`、`σ_w`、`ICC`、`r_full` 都是辅助量，不作为 A/B/C 的门槛。** 主判据是 §5.4 的原组内时间差异、排序和优势稳定性；全局 ICC 只用于解释和交叉检查。

### 5.3 分半可靠性

- 每个程序的 20 次重复随机分成两半（各 10 次，固定 seed）；
- 计算两半均值，跨 125 个程序做 Spearman 相关；
- 用 Spearman-Brown 校正：`r_full = 2r_half / (1 + r_half)`；
- 按程序 bootstrap 95% CI；
- 解释：`r_full` 高说明程序排序稳定；低说明单次/少数重复的排序不可靠。该指标是辅助量，不单独决定 A/B/C。

### 5.4 主判据：原 33 组内的时间差异、排序与优势稳定性

主分析单位是**原始 33 个 all-correct 组**，不是全局 125 个程序的相关性。每组使用 §3.3 的可用案例集（27 组 n=4，5 组 n=3，1 组 n=2），并在同一可用集内重算观测优势：

```text
time_cost_i = observed_time_seconds_i / 600
reward_i     = base_i - time_cost_i
base_i       = observed_reward_i + observed_time_cost_i
mean         = sum(reward_i) / n_g
std          = stdev(reward_i)        # ddof=1
adv_i        = (reward_i - mean) / (std + 1e-6)
```

对每个组 `g`，定义三个组内统计量：

1. **时间差异稳定性 `S_spread_g`**
   - `obs_spread = max(observed_time) - min(observed_time)`；
   - 从每个程序的有效测量值中有放回抽取（保留程序均值），得到 `sim_spread`；
   - `S_spread_g = P(0.5 * obs_spread <= sim_spread <= 1.5 * obs_spread)`。

2. **排序稳定性 `S_order_g`**
   - 对组内所有轨迹对 `(i,j)`，计算 `obs_sign = sign(t_i - t_j)`；
   - 从各程序测量分布中抽取 `(t_i, t_j)`，统计同号概率；若观测为平局，该 pair 记 0.5；
   - `S_order_g = mean over pairs P_sim(sign(t_i - t_j) == obs_sign)`；
   - 取值范围 0.5（随机）到 1.0（完全稳定）。

3. **优势稳定性 `S_adv_g`**
   - 对每条可用轨迹，使用重抽时间重算 GRPO 优势 `adv_sim_i`；
   - 若 `|adv_obs_i| < 0.10`，该样本记为中性，贡献 0.5；
   - 否则要求 `sign(adv_sim_i) == sign(adv_obs_i)` 且
     `|adv_sim_i - adv_obs_i| <= max(0.10, 0.25 * |adv_obs_i|)`；
   - `S_adv_g = mean over available samples P_sim(sample is stable)`。

重抽次数 `B=10,000`，固定 seed `20261023`。对每组计算 `S_spread_g`, `S_order_g`, `S_adv_g`，再取跨组中位数：

```text
S_spread = median_g S_spread_g
S_order  = median_g S_order_g
S_adv    = median_g S_adv_g
D_obs    = median_g obs_spread
```

**零假设（只有程序内噪声）**：对每个程序，把其测量值中心化为 `residual = t - mean_p`，再整体平移到 grand mean `μ_0`；按原组映射抽取 `t_null = μ_0 + residual`，重算奖励、优势、`S_spread_g`, `S_order_g`, `S_adv_g`。重复 `B=10,000` 次，得到各统计量的零分布。单侧 p 值：

```text
p_X = (1 + #{X_null >= X_obs}) / (B + 1),  X ∈ {spread, order, adv, D}
```

主判据同时看：

- 时间差异是否超出“只有程序内噪声”的零分布（`p_D`, `p_spread`）；
- 原组内的程序排序是否可重复（`S_order`）；
- 原组内的 GRPO 优势符号与量级是否可重复（`S_adv`）；
- 观测的时间 spread 至少达到 `0.10s`（实用下限；低于此即使显著也归 C）。

**完整案例敏感性**：只用 27 个 `n_g=4` 组重复上述计算，得到 `S_order_complete`, `S_adv_complete`, `S_spread_complete`, `p_D_complete`。主分析（33 组）与完整案例（27 组）若在 A/B 上冲突，归为 **C**。

### 5.5 程序均值保留的重模拟与原始优势对照

- `S_spread_g`, `S_order_g`, `S_adv_g` 本身已是“保留每个程序自己的测量分布”的重模拟；它们回答“这些程序在重复执行下是否还能重现原组内差异/排序/优势”。
- 另报告 `P_sim(obs_spread)`：观测 spread 在保留程序均值的模拟分布中的分位；若接近 0 或 1，说明原观测偏大/偏小，需在报告中解释，但不单独改变判定。
- 对 6 个非完整组，列出原始 4 条轨迹优势与可用案例重算优势的差异；不把重算优势冒充原始训练优势。
- 全局 ICC、`r_full`、`σ_b` 单列在辅助表中；它们与主判据冲突时只标注，不覆盖主判据。

### 5.6 代码特征与漂移

- 对 `mean_p` 与以下特征做 Spearman 相关：
  - 代码长度；
  - 是否 `import sympy`；
  - 是否包含循环/递归/大计算；
  - 是否产生大量输出。
- 目的：判断 time signal 是否只是 import/代码长度开销；这不改变“时间成本是否可测”的结论，但影响解释。
- 漂移：由于正式计时是交错轮次，对每个程序用 `tool_seconds ~ round_index` 回归；报告斜率、95% CI、首尾 5 轮差异。若漂移与 `sd_p` 同量级，使用轮次固定效应/去趋势作为 sensitivity，并在报告中说明。
- 若 `import sympy` 组与其余组的 `mean_p` 差异很大，报告分层结果；不把 import 开销误写成“推理效率”。

### 5.7 最小可检测差异（MDE）

在每组每条轨迹 n=20 次重复、双侧 α=0.05、80% power 下，两个程序均值之差的最小可检测量约为：

```text
MDD ≈ (z_{0.975} + z_{0.80}) * σ_w * sqrt(2/n)
    ≈ 2.802 * σ_w * sqrt(2/20)
    ≈ 0.886 * σ_w
```

单程序均值 95% CI 半宽约 `1.96 * σ_w / sqrt(20) ≈ 0.438 * σ_w`。

报告实测 `σ_w` 后的 MDE；例如：

- `σ_w=0.2s` → MDD≈0.18s；
- `σ_w=0.5s` → MDD≈0.44s；
- 若实测 `σ_w` 接近 0.5s，本设计只能勉强区分约 0.58s 的观测中位 spread，结论应偏保守。

若某程序有效重复 <10，或全局 MDE 大于 `D_obs`，判定不能为 A；归为 C（不可分辨），并在报告中说明需要更多重复。预注册后不因看到结果而改 n 或阈值。

### 5.8 预注册判读规则（冻结）

在计时前把以下规则写入 `preregistration.md`，计时后不改。主判据基于 §5.4 的 33 组可用案例；完整案例 27 组作为冲突检查。**全局 ICC/r_full 不参与门槛。**

| 判定 | 条件 | 解释 |
|---|---|---|
| **A. 信号** | `p_D < 0.05` 且 `S_order ≥ 0.75` 且 `S_adv ≥ 0.65` 且 `S_spread ≥ 0.60` 且 `D_obs ≥ 0.10s` 且 `MDE ≤ D_obs` 且完整案例敏感性不判 B | 原组内的时间差异、排序和优势在重复执行下可重现；R2 的 time term 至少有可测的组内程序级信号。仍需报告特征相关，区分“真实效率”与“import/长度开销”。 |
| **B. 噪声主导** | `p_D ≥ 0.05` 且下列至少两项成立：`S_order < 0.60`、`S_adv < 0.55`、`S_spread < 0.50`；或完整案例与主分析均满足 B | 原组内 time advantage 与同一程序重复执行的噪声一致；不能把 R2 的 time advantage 当作可靠效率信号。 |
| **C. 混合/不确定** | 不满足 A 也不满足 B；包括 `p_D < 0.05` 但排序/优势不稳定、`p_D ≥ 0.05` 但排序/优势很高、33 组与 27 组冲突、MDE > `D_obs`、可评估组 <90%、关键程序 `insufficient_reps` | 需要更多重复或更受控负载；不能支持“有信号”或“纯噪声”的强结论。 |

补充（全部预注册）：

- `S_order` 是 33 组中位数的 pairwise 排序一致概率；`S_adv` 是组内优势符号+量级稳定的平均概率；`S_spread` 是组内 range 落在观测值 0.5–1.5 倍内的概率。
- `p_D` 是主要零假设检验；`p_spread`, `p_order`, `p_adv` 只作支持性报告。若 `p_D` 与三个稳定性指标冲突，归 C。
- 若可评估组少于 30/33（<90%），或任一关键组只有 <2 条可用轨迹，归 C，不强行给 A/B。
- `σ_b`、`ICC`、`r_full` 只报告；例如 ICC 很高但排序稳定性低，仍按主判据判 C/B，并在报告中标注冲突。
- **任何判定都不导致修改奖励系数、任务集、schedule 或训练目标。** 判定只用于决定下一步是否/如何重训，由用户确认。

---

## 6. 资源与执行前置

### 6.1 资源估算

| 项 | 估算 |
|---|---:|
| 主电池调用数 | 125 × (5+20) = 3,125 |
| 现有 all-correct 观测均值 | `tool_seconds` 约 0.802s/次（131 个正时间样本；其中 125 条可映射到程序） |
| 主电池纯执行时间 | 3,125 × 0.802s ≈ 42 分钟 |
| 加客户端/调度开销（1.3–1.8×） | 约 55–75 分钟 |
| SymPy 口径小电池 | 50 calls ≈ 5–10 分钟 |
| 保守上限（含慢程序/失败记录） | 2.5 小时 |
| 本轮不执行 C2/C3/外部评测 | 0 |
| 总 wall time | 约 1–2 小时；保守 2.5 小时 |
| GPU | **0 GPU-hour**（无模型推理） |
| 日志/输出 | <50 MB |
| 依赖 | 现有 `adaptive-math` 环境、`httpx`、`numpy`；不新增依赖 |

### 6.2 执行前置

1. **用户已确认本方案（2026-10-23）**；
2. SandboxFusion 可达：`ADAPTIVE_MATH_SANDBOX_URL=http://localhost:8080`；执行前 `curl /v1/ping` 已返回 `200` + `"pong"`；正式预检仍用 `SandboxFusionClient.health()` 复核；
3. 如果隧道断开，需要用户在对端重连；不要在服务不可用时开始计时；
4. 记录 SandboxFusion SHA `469de0845cb7d665bc68044d541aa3aa28c5b30a` 和客户端代码哈希；
5. 磁盘足够（诊断输出 <50MB；不写检查点）；不需要 GPU；
6. 不启动训练、不加载模型、不调用 vLLM；
7. 只在新的诊断目录写文件；不改现有 `training/`、`evaluation/`、`audit/posthoc_20261023/` 内容。

### 6.3 输出目录与文件

建议新目录：

```text
artifacts/experiments/midterm_uniform_reward_v1/audit/r2_cost_signal_20261023/
```

| 文件 | 内容 |
|---|---|
| `preregistration.md` | 本方案的判定规则、电池/映射/轮次哈希、seed、时间戳、范围声明 |
| `program_battery.json` | 125 个程序及 occurrence 映射 |
| `program_battery.sha256` | 电池 canonical hash |
| `group_trajectory_program_map.json` | 33 组 × 4 条轨迹的组—轨迹—程序映射，含 7 条缺失原因 |
| `group_trajectory_program_map.sha256` | 映射 canonical hash |
| `round_plan.json` | 25 轮 × 125 程序的随机交错顺序 |
| `round_plan.sha256` | 轮次计划 canonical hash |
| `timing_repeats.jsonl` | 每条调用的原始记录（round/phase/position） |
| `sympy_caliber_check.jsonl` | SymPy/Python 两条路径的 50 次口径核对 |
| `sympy_caliber_summary.json` | SymPy 请求/schema valid/executed/口径差汇总 |
| `within_program_stats.json` | 每程序 mean/sd/median/MAD/CV 与有效 n |
| `group_stability.json` | 33 组的 S_spread/S_order/S_adv、观测时间与优势 |
| `complete_case_sensitivity.json` | 27 个完整 4 条组的敏感性结果 |
| `variance_decomposition.json` | σ_b/σ_w/ICC + bootstrap CI（辅助） |
| `noise_null.json` | 33 组零分布、p 值、观测值 |
| `advantage_resimulation.json` | 保留程序均值的重模拟与原始优势对照 |
| `feature_correlations.json` | 代码特征相关、轮次漂移、分层 |
| `run_manifest.json` | 版本、哈希、环境、耗时、调用数、缺失统计 |
| `diagnostic_report.md` | 中文结论、限制、判定、范围声明 |

`build_battery.py`、`run_timing.py`、`analysis.py` 作为新脚本放在诊断目录；只读现有 audit，不修改既有文件。

---

## 7. 预期产出与后续决策

诊断结束后只做三件事：

1. 给出判定 **A / B / C**，并报告 33 组可用案例的 `S_spread`、`S_order`、`S_adv`、`D_obs`、`p_D`、`p_spread`、`p_order`、`p_adv`，以及 27 组完整案例的同组指标；`σ_b`、`σ_w`、`ICC`、`r_full`、MDE 只作辅助报告；
2. 明确当前 50 步 R2 的 time term 在**当前重放条件和所选全对组**下是否可作为“组内程序级效率信号”；
3. 给出“是否/如何重训”的建议，但**不执行**重训，等待用户确认。

三种结果的解释：

- **A（信号）**：R2 的 time term 至少捕捉到稳定的程序间耗时差异；但这不等于“模型学会了有效推理”。还要看特征相关：如果差异主要由 `import sympy`/代码长度解释，应把它描述为“成本/实现开销信号”，不是“推理效率”。不因此改权重。
- **B（噪声主导）**：50 步 R2 的 all-correct 组内 time advantage 与同一程序重复执行的噪声一致；不能把 R2 与 R1 的差异归因于时间成本塑形。当前预实验保留为中期结果；不通过调大 `time_weight` 来“救”信号，因为那会放大噪声。
- **C（混合）**：需要更多重复或更受控负载；报告已观测的效应量与 MDE，不强行下结论。是否重训由用户决定。

### 7.1 与 50→100 续训的关系

- 本诊断不需要 GPU，也不依赖续训；可以先做。
- 如果诊断是 B（噪声主导），则“从同一 SFT 用统一修复版本重训 100 步”应优先考虑**重新设计/重新验证成本信号**，而不是直接续训 R2；但本轮不改奖励。
- 如果诊断是 A（信号），续训仍需标注验证器/RNG/审计 ID/同一 400 题重复等差异；不能把续训当成纯步数延长。
- 如果诊断是 C，先补重复/控制负载，再决定是否重训。

### 7.2 实验条件与局限

- **优化器更新次数不齐**：R0 456、R1 431、R2 422（checkpoint `optimizer step`，不是 global update 数）。三个臂名义上都是 50 个 global update，但优化器实际更新次数不同；任何三臂差异都不能只归因于奖励变体，本诊断不做三臂比较。
- **缺失**：6 条轨迹因 `action_text` 截断到 800 字符无法映射；1 条轨迹工具未执行（零时间）。33 组中只有 27 组是完整 4 条可用；主分析使用 33 组可用案例，完整案例 27 组作冲突检查。
- **重放条件**：所有时间来自 2026-10-23 当前 SandboxFusion 服务、当前客户端、单 CPU 高 load 环境、顺序交错轮次；不是原训练时的并发/负载条件。
- **测量路径**：primary 是 Python SandboxFusion `execution_time`；SymPy `worker_process` 只在口径小电池中检查，当前 R2 原数据没有合法 SymPy 执行。
- **统计限制**：每程序 20 次正式重复；若 `σ_w` 大，MDE 可能接近或超过观测 spread，此时只能判 C，不能强行区分。
- **不做因果推断**：时间成本是否导致准确率变化、是否值得保留、权重是否应调整，均不在本诊断范围。

### 7.3 结论范围

本诊断的结论**仅针对当前重放条件和所选 33 个 all-correct 组/125 个程序**：

- 不外推为“原训练全部信号”的来源；
- 不外推到 mixed/all-wrong 组、其他 global update、其他检查点、其他模型、其他任务集；
- 不外推到并发训练、不同 SandboxFusion 负载或其他时间字段实现；
- 不证明时间项与准确率之间的因果关系。

### 7.4 明确不做的事

- 不改 `time_weight`、`max_tool_seconds`、`max_tool_call_seconds`、`tool_weight`、`invalid_weight`、clip；
- 不改任务池、schedule、SFT、训练目标、GRPO 配置；
- 不删除 50 步检查点、日志、报告或审计；
- 不用“挑几个跑得快的程序”或“删掉慢程序”来制造正结果；
- 不把诊断结果当作调参依据直接开跑；重训决定必须由用户确认。
