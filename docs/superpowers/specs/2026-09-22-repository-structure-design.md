# EvoMath Harness 仓库结构规范化设计

## 1. 目标

将仓库整理为边界明确、可独立运行且可持续演进的三代结构：

- V1：独立的 **LangGraph Workflow Agent**，作为可运行、可测试的历史基线；
- V2：基于 `adaptive_math` 的 Agentic RL Runtime；
- V3：继续作为 `adaptive_math` 内的 Harness Evolution 扩展能力。

结构调整只改变代码组织和工程入口，不在迁移中重写算法或改变运行行为。

## 2. 已确认的约束

1. V1 必须与 V2/V3 解耦。
2. V1 迁移后仍能独立安装、运行示例和执行测试。
3. V2/V3 保持现有 `adaptive_math.*` 导入路径。
4. V3 的 `harness` 与 `evolution` 继续作为 `adaptive_math` 的扩展模块，不拆成独立包。
5. 当前工作区中尚未提交的 Harness/Evolution 代码必须完整保留。
6. 迁移完成后，根项目不再通过 Ruff 排除项绕开 V1。

## 3. 目标目录

```text
EvoMath-Harness/
├── src/adaptive_math/              # V2 Agentic RL + V3 Harness Evolution
│   ├── core/
│   ├── verifier/
│   ├── reward/
│   ├── tools/
│   ├── agent/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── harness/                    # V3 扩展
│   └── evolution/                  # V3 扩展
├── legacy/langgraph-agent/         # V1 独立应用
│   ├── pyproject.toml
│   ├── src/langgraph_agent/
│   │   ├── config/
│   │   ├── router/
│   │   ├── workflow/
│   │   ├── llm/
│   │   └── rl/
│   ├── configs/
│   ├── scripts/
│   ├── examples/
│   ├── tests/
│   └── README.md
├── configs/                        # V2/V3 配置
├── scripts/                        # V2/V3 操作入口
├── tests/                          # V2/V3 测试
├── docs/
├── docker/
├── data/
├── reports/
└── pyproject.toml                  # adaptive_math 主项目
```

## 4. 文件迁移映射

| 当前路径 | 目标路径 |
|---|---|
| `src/main.py` | `legacy/langgraph-agent/src/langgraph_agent/cli.py` |
| `src/config/` | `legacy/langgraph-agent/src/langgraph_agent/config/` |
| `src/router/` | `legacy/langgraph-agent/src/langgraph_agent/router/` |
| `src/graph/` | `legacy/langgraph-agent/src/langgraph_agent/workflow/` |
| `src/llm/` | `legacy/langgraph-agent/src/langgraph_agent/llm/` |
| `src/rl/` | `legacy/langgraph-agent/src/langgraph_agent/rl/` |
| `examples/` | `legacy/langgraph-agent/examples/` |
| `tests/test_router.py` | `legacy/langgraph-agent/tests/test_router.py` |
| `tests/test_graph.py` | `legacy/langgraph-agent/tests/test_workflow.py` |
| `tests/test_rl.py` | `legacy/langgraph-agent/tests/test_rl.py` |
| `autodl_train.py` | `legacy/langgraph-agent/scripts/train.py` |
| `training_config.json` | `legacy/langgraph-agent/configs/training.json` |
| `demo_trajectories.json` | `legacy/langgraph-agent/examples/data/demo_trajectories.json` |

迁移时使用文件移动保留 Git 历史，并将所有 `src.*` 导入改为 `langgraph_agent.*`。

## 5. 运行与依赖边界

### 5.1 V1 LangGraph Workflow Agent

V1 拥有独立 `pyproject.toml`、依赖集合、CLI 和测试配置：

```bash
cd legacy/langgraph-agent
uv sync --dev
uv run langgraph-agent --help
uv run pytest
```

V1 不导入 `adaptive_math`，也不复用主项目内部模块。它只作为独立历史应用存在。

### 5.2 V2/V3 主项目

主项目保留现有包名和开发命令：

```bash
uv sync --dev
uv run pytest
uv run ruff check src tests scripts
uv run mypy
```

`adaptive_math` 不导入 `langgraph_agent`。主项目的构建目标仍只有 `src/adaptive_math`。

## 6. 架构边界

### V1

LangGraph Workflow Agent 包含动态路由、固定工作流、模型提供商封装和早期 GRPO 原型。其定位是展示项目从规则化编排出发的历史基线。

### V2

V2 是自研的、基于 Pydantic 契约的 Agent Runtime，包含动作协议、环境、工具、Verifier、Reward、Trace、SFT 和 GRPO 适配层。它不依赖 LangGraph。

### V3

V3 在 V2 契约上扩展 HarnessSpec、失败分类、质量门禁、注册表以及后续的失败挖掘和 Harness Patch 闭环。V3 与 V2 共享 `adaptive_math` 包，不复制运行时实现。

## 7. 迁移阶段

### 阶段一：V1 独立化

1. 创建 `legacy/langgraph-agent` 包结构。
2. 移动 V1 源码、测试、示例、配置和训练入口。
3. 修正包导入和文件路径。
4. 建立独立 CLI、依赖声明和 README。
5. 验证 V1 安装、CLI smoke 和测试。

### 阶段二：主项目清理

1. 删除根项目中已迁移的 V1 排除规则。
2. 确认 `src/` 只包含 `adaptive_math` 主包。
3. 清理根目录旧入口和旧数据文件。
4. 保持 V2/V3 导入路径、脚本和配置不变。

### 阶段三：文档与工程门禁

1. 更新根 README 的目录树、V1 运行方式和版本说明。
2. 更新架构文档及所有旧路径引用。
3. 增加跨包依赖扫描，阻止 V1 与 V2/V3 相互导入。
4. 分别运行主项目和 V1 的测试、Lint 与类型检查。

## 8. 测试与验收

### V1 门禁

- 独立 `uv sync --dev` 成功；
- `uv run langgraph-agent --help` 成功；
- V1 Router、Workflow 和 RL 测试通过；
- V1 示例不需要安装主项目即可运行；
- V1 源码中不存在 `adaptive_math` 导入。

### V2/V3 门禁

- 主项目完整 pytest 套件通过；
- Ruff 和 mypy 通过；
- `adaptive_math` 中不存在 `langgraph_agent` 导入；
- Agent trace、隐藏答案边界和 Harness 兼容契约保持通过；
- 主包 wheel 只包含 `adaptive_math`。

### 文档门禁

- README 与文档中的仓库内相对链接有效；
- 不存在指向已迁移旧路径的命令；
- V1、V2、V3 的状态描述与实际代码一致。

## 9. 风险与缓解

| 风险 | 缓解措施 |
|---|---|
| V1 导入路径变化导致测试或示例失效 | 先建立新包与测试，再移除旧位置 |
| V1 运行时依赖未完整声明 | 从现有 imports 和历史配置生成独立依赖集合，并执行干净安装验证 |
| 根目录路径被训练脚本硬编码 | 全仓搜索旧路径并增加路径门禁 |
| 当前未提交 V3 工作被覆盖 | 不移动或重写 `src/adaptive_math/{harness,evolution}` 及相关测试；提交范围逐阶段核对 |
| 大规模移动使审查困难 | 文件移动与行为修改分开提交，避免无关格式化 |
| 两套 uv 环境与锁文件混淆 | 根项目和 V1 各自维护 `pyproject.toml` 与 `uv.lock` |

## 10. 完成定义

仓库满足以下条件时，结构规范化完成：

1. V1 可从 `legacy/langgraph-agent` 独立安装、运行和测试；
2. 根项目源码只包含 `adaptive_math` 主包；
3. V2/V3 的公开导入和运行行为保持不变；
4. V1 与 V2/V3 不存在双向代码依赖；
5. 两套测试与静态检查全部通过；
6. README、架构文档和脚本路径与新结构一致；
7. 当前 Harness Evolution 工作完整保留并可继续开发。
