# Phase H1 实施计划：AgentLoop 接入 HarnessSpec

日期：2026-09-18
状态：完成（2026-09-18，Task 1–5 全部执行；git 不可用，commit 步骤按计划约定跳过）
上游设计：[Harness Evolution 设计报告](../specs/2026-09-18-harness-evolution-design.md) §5.1、§7.1、§9、§10（Phase H1）

## 执行记录

- Task 1–4 均按 红 → 绿 执行：每个任务先运行失败测试（收集数逐任务递增），
  再实现至转绿；parity 与契约测试未要求任何生产代码返工。
- 计划外小幅增补：`ToolRegistry.names` 属性（环境工具集合的显式读口，
  供 loop 一致性校验与后续 H2 runner 复用），按同款 TDD 流程落地。
- 全仓门禁（2026-09-18 收尾运行）：`pytest` **395 passed, 2 skipped**；
  `ruff check src tests` **All checks passed**；`mypy`（strict，不带路径参数）
  **Success: no issues found in 61 source files**。
- 门禁锚点：golden trace `content_hash`（851f9da5…c3f95cedb）逐字节不变；
  legacy 与 harness 驱动 loop 在同一剧本下逐事件一致，差异仅限
  `runtime_version` 与 `harness_spec_hash`。

## 目标

让每一条 `Trajectory` 都能回答"它是在哪个 harness 下产生的"：`AgentLoop` 可选地
接受一个 `HarnessSpec`，轨迹写入三段式 `runtime_tag()` 与 `harness_spec_hash`；
同时保证旧格式轨迹（两段式 `runtime_version`、无 `harness_spec_hash` 字段）的
`content_hash` 逐字节不变、仍可 replay（Phase H1 门禁）。

## 架构

不改权重、不改行为语义：harness 只影响轨迹的溯源字段。`AgentLoop` 在 `run()`
入口校验运行实测参数（generation / budget / tools）与 spec 一致——不一致即
`ValueError`，杜绝"spec 说是 A、实际跑的是 B"的隐性漂移。`trajectory_hash`
在 `harness_spec_hash is None` 时从 canonical JSON 中剔除该键，旧 hash 不变。

## 技术栈

Python 3.12、pydantic v2（frozen + extra="forbid"）、orjson canonical hash、pytest。

## 全局约束

- [ ] 旧轨迹 hash 不变：`tests/fixtures/golden_traces/direct_correct.json` 的
  `content_hash = 851f9da5…c3f95cedb` 在改动后仍通过 `verify_hash`。
- [ ] `AgentLoop()` 无参构造（legacy 路径）行为逐字节不变：`runtime_version`
  仍为 `f"{RUNTIME_VERSION}+{PROMPT_VERSION}"`，不写 `harness_spec_hash`。
- [ ] 一次 patch 一类字段原则不适用本计划（本计划是接入，不是 patch）。
- [ ] 不修改 verifier / reward / 环境语义；不动 `teacher_rollout.py` 等既有调用方
  （它们走 legacy 路径，H2 runner 才切换）。
- [ ] 仓库未初始化 git：各任务的 commit 步骤跳过，以门禁命令输出为证据。
- [ ] mypy strict、ruff、全仓 pytest 在收尾时必须全绿。

## 文件结构

```text
src/adaptive_math/agent/trace.py          # 修改：Trajectory 增加 harness_spec_hash
src/adaptive_math/agent/replay.py         # 修改：trajectory_hash 向后兼容
src/adaptive_math/agent/loop.py           # 修改：AgentLoop(harness=...)
tests/unit/agent/test_trace.py            # 修改：字段校验测试
tests/unit/agent/test_replay_hash.py      # 新建：hash 兼容性单元测试
tests/integration/test_harness_loop.py    # 新建：harness 驱动 loop 的集成/parity 测试
tests/contract/test_harness_trace_compat.py  # 新建：旧轨迹可回放契约测试
```

## Task 1 — `Trajectory.harness_spec_hash` 字段

Files: `src/adaptive_math/agent/trace.py`, `tests/unit/agent/test_trace.py`

- [ ] 先在 `tests/unit/agent/test_trace.py` 追加失败测试：
  - 带合法 `harness_spec_hash`（64 位小写 hex）的 Trajectory 可 JSON round-trip；
  - 缺省为 `None`；非法值（长度不足 / 大写 / 非 hex）被 pydantic 拒绝。
  - 失败命令：`.venv/bin/python -m pytest tests/unit/agent/test_trace.py -q`
- [ ] 实现：`trace.py` 增加
  `harness_spec_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")`
  （导入 `pydantic.Field`）。
- [ ] 验证：上述 pytest 通过。

## Task 2 — `trajectory_hash` 向后兼容

Files: `src/adaptive_math/agent/replay.py`, `tests/unit/agent/test_replay_hash.py`（新建）

- [ ] 先写失败测试：
  - 固定轨迹（无 harness 字段）的 hash 等于 golden fixture 的
    `content_hash` 常数（逐字节回归锚点）；
  - 同一轨迹仅 `harness_spec_hash` 不同 → hash 不同；
  - 带字段的轨迹经 `TraceEnvelope.create` 后 `verify_hash` 通过、可 `replay`。
  - 失败命令：`.venv/bin/python -m pytest tests/unit/agent/test_replay_hash.py -q`
- [ ] 实现：`trajectory_hash` 中
  `payload = trajectory.model_dump(mode="json")`；若
  `payload["harness_spec_hash"] is None` 则 `del payload["harness_spec_hash"]`，
  再 orjson 排序键 dumps。注意不能用 `exclude_none=True`（会误伤
  `final_answer=None` 的旧轨迹 hash）。
- [ ] 验证：新测试通过，且
  `tests/integration/test_trace_replay.py::test_golden_trace_fixture_replays` 仍通过。

## Task 3 — `AgentLoop(harness=HarnessSpec)` 接入

Files: `src/adaptive_math/agent/loop.py`, `tests/integration/test_harness_loop.py`（新建）

- [ ] 先写失败测试（用 `ScriptedModel`，复用 `tests/integration/test_agent_loop.py` 的模式）：
  - `AgentLoop(harness=spec)` 产出的轨迹 `runtime_version == spec.runtime_tag()`、
    `harness_spec_hash == spec.spec_hash()`；
  - `generation != spec.generation` → `ValueError`；
  - 环境 budget != spec.budget → `ValueError`；
  - 环境 registry 工具名集合 != spec.tools → `ValueError`；
  - legacy `AgentLoop()` 轨迹 `harness_spec_hash is None` 且 `runtime_version`
    为两段式。
  - 失败命令：`.venv/bin/python -m pytest tests/integration/test_harness_loop.py -q`
  - 测试 spec 用 `tools=()` + 零工具预算构造（避免拉起 SandboxFusion），
    例如 `HarnessSpec(harness_version="h-test", …, tools=(), budget=Budget(max_steps=3, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100), generation=GenerationConfig(), reward=R2_DEFAULT)`。
- [ ] 实现：`AgentLoop.__init__(self, harness: HarnessSpec | None = None)`；
  `run()` 开头校验三项一致性（registry 工具名经
  `sorted(d["name"] for d in environment.registry.descriptions())` 取得）；
  构造 Trajectory 时按 harness 有无选择 `runtime_version` 与 `harness_spec_hash`。
  导入安全：`harness/__init__.py` 不依赖 `agent.loop`（仅 `presets.py` 依赖，
  不在 `__init__` 导出链上），`from adaptive_math.harness.spec import HarnessSpec`
  无循环导入。
- [ ] 验证：新测试 + `tests/integration/test_agent_loop.py` 全过。

## Task 4 — 契约测试：旧轨迹可回放 + 新旧 loop parity

Files: `tests/contract/test_harness_trace_compat.py`（新建）、
`tests/integration/test_harness_loop.py`（追加 parity 用例）

- [ ] 先写失败测试：
  - 契约：golden fixture JSON 解析后 `trajectory.harness_spec_hash is None`，
    `verify_hash` 与 `replay` 均通过（旧格式永续可回放）；
  - parity：同一 `ScriptedModel` 剧本分别喂给 legacy loop 与
    `AgentLoop(harness=spec)`，两者 `events`、`usage`、`final_answer`、
    `termination_reason` 完全相等；差异仅允许出现在 `runtime_version`（三段式
    vs 两段式）与 `harness_spec_hash`（有 vs None）。
  - 失败命令：`.venv/bin/python -m pytest tests/contract/test_harness_trace_compat.py -q`
- [ ] 实现：应无需改生产代码；若 parity 失败则修 loop 直至通过。
- [ ] 验证：contract + integration 全过。

## Task 5 — 全仓门禁与文档收尾

Files: `docs/superpowers/specs/2026-09-18-harness-evolution-design.md`（§10/§11 状态更新）

- [ ] `.venv/bin/python -m pytest -q` 全绿（基线 322 单测 + 全部集成/契约）。
- [ ] `.venv/bin/python -m ruff check src tests` 无错。
- [ ] `.venv/bin/python -m mypy src`（strict）无错。
- [ ] 设计文档 §10 Phase H1 标记完成、§11 更新实施状态与测试计数。
- [ ] 本计划文件状态改为"完成"，记录门禁输出摘要。

## 退出标准

1. 任一 harness 驱动的轨迹带 `harness_spec_hash` 且可经 `spec_hash()` 回溯到完整 spec；
2. golden trace 的 `content_hash` 常数逐字节不变（契约测试锚定）；
3. legacy 与 harness 驱动 loop 在同一剧本下逐事件一致；
4. 全仓 pytest / ruff / mypy strict 全绿。
