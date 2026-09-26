# Phase H2.5 实施计划：Trace 增补 + Failure Taxonomy

日期：2026-09-18
状态：完成（2026-09-18，Task 1–6 全部执行；git 不可用，commit 步骤按计划约定跳过）
上游设计：[AdaptiveMath-Evo v2.0 对齐设计](../specs/2026-09-18-adaptivemath-evo-v2-alignment-design.md) §3.1、§3.2、§7（H2.5 行）、§8；[Harness Evolution 设计报告 v1.0](../specs/2026-09-18-harness-evolution-design.md) §5.4、§6.1

## 执行记录

- Task 1–5 均按 红 → 绿 执行：每个任务先运行失败测试（行为级失败确认后再实现），
  再实现至转绿。Task 5 中一处失败为测试自身的夹具不一致（FINAL 事件答案与
  `Trajectory.final_answer` 不符），修测试而非修实现。
- 全仓门禁（2026-09-18 收尾运行）：`pytest` **439 passed, 2 skipped**
  （基线 395 + 新增 44）；`ruff check src tests` **All checks passed**；
  `mypy`（strict，不带路径参数）**Success: no issues found in 63 source files**。
- 门禁锚点：golden trace `content_hash`（851f9da5…c3f95cedb）逐字节不变
  （`test_replay_hash.py` 与 `test_harness_trace_compat.py` 双锚点持续通过）；
  taxonomy 对 golden 正确轨迹打标为空（新增契约测试）。

## 目标

让失败挖掘有机器可读的输入：① 轨迹绑定 `model_version`（PDF §6.1 的 model/harness 双绑定补齐）；② `TraceEvent` 携带受控词表 `error_code`（当前 INVALID_ACTION 只有人类可读 message，无法聚类）；③ 新增 `evolution/taxonomy.py`——13 类非互斥机器标签 + 归因方向映射 + 确定性打标函数，作为后续 miner / attribution 的标签内核。

## 架构

trace 增补沿用 H1 的 hash 兼容手法：新字段可选、缺省 `None`、canonical hash 时剔键——golden trace `content_hash`（851f9da5…c3f95cedb）逐字节不变。`error_code` 是 event 级字段，剔键需逐 event 处理。环境在三条 INVALID_ACTION 路径上发射机器码；工具失败的机器码已存在于 `ToolResult.error_code`（TOOL_RESULT payload 内），taxonomy 直接读取，不重复发射。taxonomy 是纯函数模块：输入 `Trajectory` + 私有验证边界的正确性信号，输出确定性标签集合；需要人工判断的标签（如语义层面的 WRONG_TOOL_CHOICE）只做确定性代理，不做伪智能推断。

## 技术栈

Python 3.12、pydantic v2（frozen + extra="forbid"）、orjson canonical hash、pytest。

## 全局约束

- [ ] golden trace `content_hash = 851f9da5…c3f95cedb` 逐字节不变；
      `tests/integration/test_trace_replay.py::test_golden_trace_fixture_replays` 持续通过。
- [ ] legacy `AgentLoop()` 轨迹：`model_version is None`、事件无 `error_code`，行为逐字节不变。
- [ ] None 字段剔键：trajectory 级（`harness_spec_hash` 已有、`model_version` 新增）与
      event 级（`error_code`）；不得用 `exclude_none=True`（会误伤 `final_answer=None`）。
- [ ] taxonomy 不打标正确轨迹为失败；`correct=None` 时跳过所有依赖正确性的规则。
- [ ] 不修改 verifier / reward / 工具语义；不在 taxonomy 中访问隐藏标签——
      正确性以参数传入（私有验证边界产出），维持 `test_hidden_verifier_boundary` 契约。
- [ ] 仓库未初始化 git：commit 步骤跳过，以门禁命令输出为证据。
- [ ] mypy strict、ruff、全仓 pytest 收尾全绿。

## 文件结构

```text
src/adaptive_math/agent/state.py          # 修改：StepErrorCode 枚举 + TraceEvent.error_code
src/adaptive_math/agent/trace.py          # 修改：Trajectory.model_version
src/adaptive_math/agent/replay.py         # 修改：trajectory_hash 剔键（model_version / 逐 event error_code）
src/adaptive_math/agent/environment.py    # 修改：三条 INVALID_ACTION 路径发射 error_code
src/adaptive_math/agent/loop.py           # 修改：AgentLoop(model_version=...) 写入轨迹
src/adaptive_math/evolution/__init__.py   # 新建（空 docstring 包）
src/adaptive_math/evolution/taxonomy.py   # 新建：FailureLabel / AttributionDirection / label_trajectory
tests/unit/agent/test_trace.py            # 修改：model_version 字段校验
tests/unit/agent/test_replay_hash.py      # 修改：hash 兼容锚点扩展
tests/unit/agent/test_environment.py      # 修改：error_code 发射
tests/unit/evolution/__init__.py          # 新建
tests/unit/evolution/test_taxonomy.py     # 新建：逐规则合成轨迹测试
tests/integration/test_harness_loop.py    # 修改：model_version 接入用例
tests/contract/test_harness_trace_compat.py  # 修改：taxonomy 对 golden trace 打标为空
```

## Task 1 — `Trajectory.model_version` 字段 + hash 兼容

Files: `src/adaptive_math/agent/trace.py`, `src/adaptive_math/agent/replay.py`, `tests/unit/agent/test_trace.py`, `tests/unit/agent/test_replay_hash.py`

- [ ] 先在 `test_trace.py` 追加失败测试：
  - 缺省为 `None`；设置后 JSON round-trip 保持；空字符串被 pydantic 拒绝。
  - 失败命令：`.venv/bin/python -m pytest tests/unit/agent/test_trace.py -q`
- [ ] 在 `test_replay_hash.py` 追加失败测试：
  - golden 轨迹（无 `model_version`）hash 仍等于 `GOLDEN_CONTENT_HASH`；
  - 仅 `model_version` 不同 → hash 不同。
  - 失败命令：`.venv/bin/python -m pytest tests/unit/agent/test_replay_hash.py -q`
- [ ] 实现：`trace.py` 增加
  `model_version: str | None = Field(default=None, min_length=1, max_length=128)`
  （注释：checkpoint hash 或训练 run id；None = legacy）。`replay.py` 的
  `trajectory_hash` 在 `payload["model_version"] is None` 时 `del payload["model_version"]`，
  与 `harness_spec_hash` 同手法。
- [ ] 验证：上述两个测试文件通过。

## Task 2 — `TraceEvent.error_code` 字段 + hash 兼容

Files: `src/adaptive_math/agent/state.py`, `src/adaptive_math/agent/replay.py`, `tests/unit/agent/test_trace.py`, `tests/unit/agent/test_replay_hash.py`

- [ ] 先在 `test_trace.py` 追加失败测试：
  - `TraceEvent` 缺省 `error_code is None`；接受 `StepErrorCode` 成员并 round-trip；
    非法字符串被 pydantic 拒绝。
- [ ] 在 `test_replay_hash.py` 追加失败测试：
  - golden 轨迹（事件无 `error_code`）hash 不变；
  - 同一轨迹某事件加 `error_code` → hash 不同。
- [ ] 实现：`state.py` 增加

  ```python
  class StepErrorCode(StrEnum):
      ACTION_PARSE_ERROR = "action_parse_error"
      TOOL_CALL_BUDGET_EXHAUSTED = "tool_call_budget_exhausted"
      PYTHON_TIME_BUDGET_EXHAUSTED = "python_time_budget_exhausted"
  ```

  `TraceEvent` 增加 `error_code: StepErrorCode | None = None`。
  `replay.py` 的 `trajectory_hash`：dump 后遍历 `payload["events"]`，
  对 `event["error_code"] is None` 的逐项 `del event["error_code"]`。
- [ ] 验证：两个测试文件通过，且 golden fixture 回放测试仍通过。

## Task 3 — 环境发射 error_code

Files: `src/adaptive_math/agent/environment.py`, `tests/unit/agent/test_environment.py`

- [ ] 先写失败测试（复用现有 EchoTool 模式）：
  - 解析失败路径（`step(None, ...)`）→ INVALID_ACTION 事件
    `error_code == StepErrorCode.ACTION_PARSE_ERROR`；
  - tool-call 预算耗尽路径 → `TOOL_CALL_BUDGET_EXHAUSTED`；
  - python 时间预算耗尽路径 → `PYTHON_TIME_BUDGET_EXHAUSTED`。
  - 失败命令：`.venv/bin/python -m pytest tests/unit/agent/test_environment.py -q`
- [ ] 实现：`_invalid(self, content, monotonic_ms, *, error_code: StepErrorCode)`，
  `append_event(..., error_code=error_code)`；`AgentState.append_event` 增加同名可选参数
  （缺省 None，既有调用方不变）。三个调用点分别传入对应码。
- [ ] 验证：新测试 + 既有 environment 测试全过。

## Task 4 — `AgentLoop(model_version=...)` 接入

Files: `src/adaptive_math/agent/loop.py`, `tests/integration/test_harness_loop.py`

- [ ] 先写失败测试（复用 ScriptedModel 模式）：
  - `AgentLoop(model_version="ckpt-test")` 轨迹 `model_version == "ckpt-test"`；
  - legacy `AgentLoop()` 轨迹 `model_version is None`。
  - 失败命令：`.venv/bin/python -m pytest tests/integration/test_harness_loop.py -q`
- [ ] 实现：`AgentLoop.__init__(self, harness=None, *, model_version: str | None = None)`；
  构造 Trajectory 时写入 `model_version=self._model_version`。
- [ ] 验证：新测试 + `tests/integration/test_agent_loop.py` 全过。

## Task 5 — `evolution/taxonomy.py`

Files: `src/adaptive_math/evolution/__init__.py`, `src/adaptive_math/evolution/taxonomy.py`, `tests/unit/evolution/__init__.py`, `tests/unit/evolution/test_taxonomy.py`

- [ ] 先写失败测试：每条规则一个合成轨迹用例（ unhappy path 优先）：
  - `TOOL_RESULT` payload `result.error_code` 映射：TIMEOUT→TOOL_TIMEOUT；
    EXECUTION_ERROR/OUTPUT_LIMIT→TOOL_CODE_ERROR；INVALID_ARGUMENTS→FORMAT_INVALID；
    UNKNOWN_TOOL→WRONG_TOOL_CHOICE；UNAVAILABLE→INFRASTRUCTURE_FAILURE；
  - INVALID_ACTION `error_code`：ACTION_PARSE_ERROR→FORMAT_INVALID；
    两个预算码→BUDGET_EXHAUSTED；
  - termination：INFRASTRUCTURE_ERROR→INFRASTRUCTURE_FAILURE；
    MAX_STEPS/MAX_TOOL_CALLS/PYTHON_TIME_BUDGET→BUDGET_EXHAUSTED；
  - `final_answer is None`→NO_FINAL；
  - 连续两个相同 (name, arguments) 的 TOOL_CALL→LOOP_OR_REDUNDANCY；
  - 最后一个 TOOL_RESULT `ok=False` 后立即 FINAL 且 `correct=False`→PREMATURE_STOP；
  - `correct=False` 且某 ok=True 工具输出包含 `reference_answer` 而 final 不含→TOOL_RESULT_IGNORED；
  - `verifier_status`：TIMEOUT→VERIFIER_TIMEOUT；INVALID_REFERENCE→VERIFIER_UNSUPPORTED；
    INTERNAL_ERROR→INFRASTRUCTURE_FAILURE；INVALID_PREDICTION→FORMAT_INVALID；
  - `correct=False` 且无任何其他标签→WRONG_REASONING（唯一 MODEL 归因兜底）；
  - `correct=True`→空标签；`correct=None`→跳过正确性依赖规则；
  - 输出按枚举定义序排列、确定性。
  - 失败命令：`.venv/bin/python -m pytest tests/unit/evolution/test_taxonomy.py -q`
- [ ] 实现：

  ```python
  class FailureLabel(StrEnum):
      FORMAT_INVALID = "format_invalid"
      NO_FINAL = "no_final"
      WRONG_REASONING = "wrong_reasoning"
      WRONG_TOOL_CHOICE = "wrong_tool_choice"
      TOOL_CODE_ERROR = "tool_code_error"
      TOOL_TIMEOUT = "tool_timeout"
      TOOL_RESULT_IGNORED = "tool_result_ignored"
      PREMATURE_STOP = "premature_stop"
      BUDGET_EXHAUSTED = "budget_exhausted"
      VERIFIER_UNSUPPORTED = "verifier_unsupported"
      VERIFIER_TIMEOUT = "verifier_timeout"
      INFRASTRUCTURE_FAILURE = "infrastructure_failure"
      LOOP_OR_REDUNDANCY = "loop_or_redundancy"

  class AttributionDirection(StrEnum):
      HARNESS = "harness"
      MODEL = "model"
      AMBIGUOUS = "ambiguous"

  ATTRIBUTION_DIRECTION: dict[FailureLabel, AttributionDirection] = {
      FailureLabel.FORMAT_INVALID: AttributionDirection.HARNESS,
      FailureLabel.NO_FINAL: AttributionDirection.AMBIGUOUS,
      FailureLabel.WRONG_REASONING: AttributionDirection.MODEL,
      FailureLabel.WRONG_TOOL_CHOICE: AttributionDirection.HARNESS,
      FailureLabel.TOOL_CODE_ERROR: AttributionDirection.AMBIGUOUS,
      FailureLabel.TOOL_TIMEOUT: AttributionDirection.HARNESS,
      FailureLabel.TOOL_RESULT_IGNORED: AttributionDirection.HARNESS,
      FailureLabel.PREMATURE_STOP: AttributionDirection.HARNESS,
      FailureLabel.BUDGET_EXHAUSTED: AttributionDirection.HARNESS,
      FailureLabel.VERIFIER_UNSUPPORTED: AttributionDirection.AMBIGUOUS,
      FailureLabel.VERIFIER_TIMEOUT: AttributionDirection.HARNESS,
      FailureLabel.INFRASTRUCTURE_FAILURE: AttributionDirection.AMBIGUOUS,
      FailureLabel.LOOP_OR_REDUNDANCY: AttributionDirection.HARNESS,
  }

  def label_trajectory(
      trajectory: Trajectory,
      *,
      correct: bool | None,
      verifier_status: VerifierStatus | None = None,
      reference_answer: str | None = None,
  ) -> tuple[FailureLabel, ...]: ...
  ```

  规则即上述测试列表；`ATTRIBUTION_DIRECTION` 必须覆盖全部枚举成员（测试断言）。
- [ ] 验证：新测试通过。

## Task 6 — 契约锚点 + 全仓门禁 + 文档收尾

Files: `tests/contract/test_harness_trace_compat.py`, `docs/superpowers/specs/2026-09-18-adaptivemath-evo-v2-alignment-design.md`, 本计划

- [ ] 契约测试追加：golden fixture 轨迹经 `label_trajectory(correct=True)` 打标为空元组；
  golden `content_hash` 常数不变（已有锚点持续通过）。
- [ ] `.venv/bin/python -m pytest -q` 全绿（基线 395 passed / 2 skipped + 新增）。
- [ ] `.venv/bin/python -m ruff check src tests` 无错。
- [ ] `.venv/bin/python -m mypy`（strict，不带路径参数）无错。
- [ ] 对齐设计文档 §7 路线图 H2.5 标记完成；本计划状态改为"完成"，记录门禁输出摘要。

## 退出标准

1. 任一新轨迹可携带 `model_version` 与 event 级 `error_code`，旧轨迹 hash 逐字节不变；
2. `label_trajectory` 对 13 类标签的全部确定性规则有合成轨迹测试锚定，且对 golden 正确轨迹打标为空；
3. `ATTRIBUTION_DIRECTION` 覆盖全部标签，`WRONG_REASONING` 为唯一 MODEL 归因（= PDF 的 CAPABILITY_FAILURE 等价物）；
4. 全仓 pytest / ruff / mypy strict 全绿。
