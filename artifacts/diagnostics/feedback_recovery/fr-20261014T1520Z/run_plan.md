# 运行计划：工具反馈与错误恢复探索性验证

- run_id：`fr-20261014T1520Z`
- 依据文档：`2026-10-14-feedback-recovery-validation.md`（执行设计）
- 仓库：`/root/autodl-tmp/Adaptive-Solver-main-git`，分支 `codex/feedback-recovery`，基线提交 `206750e`（本轮改动全部在工作树，未提交、未并入主干）
- 执行者：执行 agent（本人），非独立复核者；人工审计的复核独立性限制见 §8.3 与 DEVIATIONS.md
- 输出根：`artifacts/diagnostics/feedback_recovery/fr-20261014T1520Z/`

## 1. 资源与资产（执行前核实结果）

| 项目 | 实测值 |
|---|---|
| GPU | RTX 4090D 24 GB，运行前空闲（1 MiB / 0 %） |
| 解释器 | `/root/autodl-tmp/conda-envs/adaptive-math/bin/python`（torch 2.8.0+cu128，transformers 4.57.3）；`.venv` 无 torch，仅用于 CPU 测试 |
| 模型 | `artifacts/models/qwen3_1_7b_sft_tir_merged`（TIR/`behavior:"python"` 训练，7083 行） |
| 模型指纹 | 全目录逐文件 sha256，见 `manifest.{a,b,c}.json` 的 `model_fingerprint`；缓存于 `.model_fingerprint_cache.json` |
| chat template | `chat_template.jinja` 存在并生效，哈希记入 manifest；`enable_thinking` 从未显式传入（模板默认值生效，作为未解不确定性记录） |
| 沙箱 | SandboxFusion，`http://127.0.0.1:8080`（SSH 远端转发到 Mac 侧服务） |
| 沙箱探活 | `GET /v1/ping` → `pong`；`POST /run_code` `print(6*7)` → `42`（canonical pong+42 通过） |
| 沙箱状态语义 | 实测**跨调用无状态**：调用 1 设 `probe_marker`，调用 2 `print("probe_marker" in dir())` → `False`。故分支续跑无需沙箱快照，也不会跨分支污染 |
| AGENTS.md | 仓库内（含 3 层深度）与文件系统 4 层内均未找到；规则来源为 `README.md`、`docs/`、CI 四关 |

选型证据（`stage_a` 之前完成，非本轮题池结果）：同一 2×2 预检下，`sft_tir_merged` 4/4 条轨迹产生解析合法的工具调用、0 解析错误、均 230 token；`sft_dp_v1_merged` 0/4 调用、均 1288 token。据此选择 TIR 模型。预检占用 2 道题（`omni_math:8cbe0c48c29daa190277`、`omni_math:cfd1da8604a325e749e2`），已从 A/B 池排除。

## 2. 数据与题池（§4.1）

- 池：`data/processed/v1/frozen_eval.parquet`（3438 行，`answer_type` 全为 `expression`）
- 排除：E0/E1+MVP 已用 200 题、第一轮诊断 100 题、本实验预检 2 题，合计 300 题（有跨源重复计数说明，见选择 manifest）
- 排除后 3138 行；难度窗口取自数据集原有标签，两个相邻层：
  - L1 = {3.5, 3.75, 4.0}，可用 379 题
  - L2 = {4.25, 4.375, 4.5, 4.75}，可用 192 题
- 窗口依据（对本题池结果盲选）：TIR 模型在已花费的 200 题上，难度 3.5–4.5 段最终正确率 12.1%、32% 的题出现执行错误；1–3 段 45.2%（过易，恢复效应无空间）；5–7 段 9–12%（同错误率但更可能触底）
- 数值答案规则：池内 `answer_type` 全为 `expression`，故按 `reference_value` 的**值形式**做代理过滤（规则正则见 `diagnostic_feedback_recovery.json` 的 `numeric_filter`），排除分数/根式/关系式等未校验形式
- 抽样：层内按 `sha256(main_seed:layer:task_id)` 升序；A 取前 5（每层），B 取随后 20（每层）；对结果盲选、无重抽
- 结果：A = 10 题（难度 4.0×5 + 4.5×5），B = 40 题（两层各 20）
- 污染核查：与 `train.parquet`（129,828 条）task_id 重合 0、source_hash 重合 0；与已用 300 题重合 0
- A 与 B 互斥；A 不进入 B/C 统计；本轮用过的题不再作为未见测试题

## 3. 冻结配置（§4.2）

- 预算：`configs/agent/feedback_recovery.yaml`，steps=8、tool_calls=4、python_seconds=12.0、observation=8000 字符（未沿用 `default.yaml` 的 6 步，避免与旧运行混淆）
- 生成：temperature=0.7，max_new_tokens=2048/轮；无额外 token 总上限，无历史截断
- 解析器容错：`unclosed_think,bare_final_scalar`（与该模型训练协议一致，A 前指定，B/C 不更换）
- 主种子：20261014；派生方式 `sha256("seed-v1:20261014" : purpose : parts)`；生成 seed 键含阶段、题 ID、样本 ID、实验臂、生成轮次；用 `hashlib` 而非 `hash()`（避免 PYTHONHASHSEED 影响）
- 判定器：`verify_answer`（`pipeline_version=canonicalize-v1`），表达式题走独立 worker 进程；A 阶段冒烟测试结果记入 `summary.json.verifier_smoke`

## 4. 执行前必查项（§3）与基础条件修复

1. 模型/tokenizer/chat template：已核实并记录哈希（见 §1）。
2. 生成 seed 真实生效：原 `TransformersModelClient` 未把 seed 接入采样、`finish_reason` 恒为 `stop`。第一次实现误用 `model.generate(generator=...)`，transformers 4.57 直接抛 `model_kwargs are not used by the model: ['generator']`，A 首次运行 20 条全部 0 轮 `model_error`（该次结果原样保留于 `stage_a_attempt0_model_error/`，未覆盖）。最终实现为 `torch.random.fork_rng` + `torch.manual_seed`（用完恢复，不污染后续请求），`finish_reason` 按真实 EOS 判定；测试见 `tests/unit/agent/test_transformers_client.py`。A 通过后另做端到端复核（`stage_a/seed_check.json`）：同 seed 重放两次与记录文本逐字节相同（607 字符），换 seed 文本即变（655 字符），且**整条 episode 逐事件重放一致**（仅剥掉 `monotonic_ms`/`latency_ms` 等墙钟字段），即"输入与日志一致"是实证而非目测。
3. **错误反馈传递（基础条件修复）**：提交 `0d576aa`（2026-09-29）起，`ProductMathEnv.step` 在工具失败时用通用失败文案替换了真实输出，模型看不到 traceback；TIR 评测（`a4c79bb`）在其之后运行，因而 56 次执行错误对模型不可见。本轮已修复为 `Tool call failed (<code>).` + 真实 stdout/stderr，并对 A/B/C 统一生效（§3.3 要求声明为"基础条件修复"）。
4. 判定器版本与已知误判：表达式题由 symbolic worker 判定，数值代理已排除未校验题型；冒烟测试在 A 中执行并记录。已知误判（来自第一轮 FINDINGS：`free symbol mismatch` 等）在本轮 C 分析中按人工审计抽查。
5. 沙箱持久化：实测无状态（见 §1）。
6. 资产可用性：模型、GPU、沙箱均可用，无需 dry-run 交付。

## 5. 阶段与预估

| 阶段 | 规模 | 说明 |
|---|---|---|
| A | 10 题 × 2 = 20 轨迹 | 交互有效性门槛；通过后冻结 `protocol.lock.json` |
| B | 40 题 × 4 = 160 轨迹 | 自然错误与锚点池（每题至多 1 个锚点） |
| C | ≤40 锚点 × 3 臂 × 4 = ≤480 续跑 | F/P/M 同状态分支；基础设施故障按原 seed 重试 1 次 |

实测吞吐（A 运行后回填）：见本节末尾"回填"块。

预估（按预检 4 条轨迹 / 37 s、含模型加载的粗估，A 完成后按实测修正）：
- A ≈ 3–6 分钟
- B ≈ 40–80 分钟（更长轨迹，每题 4 采样）
- C ≈ 1.5–3 小时（每锚点 12 条续跑，后缀较短）
- 合计 ≈ 2.5–4.5 小时 GPU 时间；token 量以 B/C 实测为准记入各阶段 summary

**回填（A 完成后）**：

A 实测（`stage_a/summary.json`，重跑后那一次）：

| 项目 | 实测 |
|---|---|
| 轨迹数 / 墙钟 | 20 条 / 208.2 s（含模型加载，≈10.4 s 每条） |
| 生成 token | 7,178（≈359 每条、≈175 每生成步） |
| 工具调用 | 21（20 条轨迹中 20 条执行过合法调用） |
| 终止原因 | 全部 `final`；基础设施故障 0 |
| 错误计数 | E_EXEC 6、`call_form_errors` 1、`length_truncated_turns` 1 |
| 正确率 | 3/20（参考值，仅描述，不作结论） |

门槛判定（`stage_a/summary.json.gate`）：5 项自动检查全过——执行过合法调用的轨迹 20/20（阈值 ≥5）、基础设施故障 0、解析失败步率 0.000（≤0.10）、未成动作的长度触顶率 0.024（≤0.10）、隐藏标签泄漏 0；`reparse_agrees_with_environment=false`（环境计到 1 个 `call_form` 错误步、重解析器为 0 个解析错误步；两项各自都在阈值内，见 DEVIATIONS D13）；判定器冒烟通过（参考值自判 correct/1.0，错答 incorrect/0.0）。人工检查 3 条（`selected` 见 `protocol.lock.json.stage_a.manual_read`）：均含完整"调用—错误反馈—后续生成"，且错误反馈确实带 stderr 回溯到达模型。seed 生效与"输入与日志一致"见 §4.2 第 2 条。**A 通过**，`protocol.lock.json` 已生成，`spec_hash=4ca6c6c51c29`（短 12 位 `4ca6c6c51c29`）。

按 A 实测速率修正预估：
- B：160 条 × ≈10.4 s ≈ 28 分钟（轨迹更长则 25–45 分钟）
- C：每个锚点 12 条续跑；前缀更长（≈1.5–3k token 预填充），按 ≈8–12 s 每条估：20 锚点 ≈ 40 分钟，40 锚点 ≈ 80 分钟
- 合计剩余 GPU 时间 ≈ 1–2 小时；若 B 锚点不足 8 题则按 §7 停止规则终止 C

**回填（B/C 完成后，2026-10-15 收尾）**：

| 阶段 | 启动（UTC） | 墙钟 | 生成 token | 工具调用 | python 秒 | 轨迹/续跑 |
|---|---|---|---|---|---|---|
| A | 15:26:01Z | 208.2 s | 7,178 | 21 | 5.0 | 20 |
| B | 15:34:51Z | 1,796.7 s（29.9 min） | 58,165 | 172 | 219.6 | 160 |
| C | 16:14:16Z | 6,543.2 s（1 h 49 min；`executed_this_invocation=312`，另 12 条为单锚点冒烟） | 后缀 265,465（F 79,297 / P 83,865 / M 102,303） | 44（python 19、sympy 25） | — | 324 |

- 全轮墙钟：A 启动 15:26:01Z → C 结束 ≈ 18:03Z，**≈2 h 37 min**（不含预检与分析）；无缺失、无重试、无协议违规，`manifest.c.json.summary.integrity.problems = []`。
- 实际工程量小于 §5 预估上限（27 锚点而非 40，324 条而非 ≤480），因为 B 的 40 题中只有 27 题产出合格锚点（54 个候选，每题取 1）。
- 沙箱仍在另一台机器（Mac，SSH 隧道），C 的墙钟含沙箱往返；`python_seconds` 只计沙箱自报执行时间。
- 分析用同一解释器、纯 CPU：`scripts/analysis/analyze_feedback_recovery.py`。**注意**：该脚本以 `"w"` 打开 `audit/labels.jsonl`，重跑会覆盖其后追加的人工精读块（DEVIATIONS D12），需重追加。

## 6. 精确启动命令

```bash
CONDA=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
RUN=artifacts/diagnostics/feedback_recovery/fr-20261014T1520Z
CFG=configs/agent/feedback_recovery.yaml
MODEL=artifacts/models/qwen3_1_7b_sft_tir_merged
POOL=data/processed/diagnostics/feedback_recovery

# A（门槛；先于冻结，故 protocol-hash 记 pre-freeze）
$CONDA scripts/eval/run_feedback_recovery.py --stage a --run-id fr-20261014T1520Z \
  --run-dir $RUN --data $POOL/fr_a10.parquet --agent-config $CFG --model $MODEL \
  --max-new-tokens 2048 --temperature 0.7 \
  --parser-tolerance unclosed_think,bare_final_scalar --protocol-hash pre-freeze

# seed 生效与整集可复现（A 通过后的门槛条款证据，写 stage_a/seed_check.json）
$CONDA scripts/eval/check_seed_reproducibility.py --stage a --run-id fr-20261014T1520Z \
  --run-dir $RUN --data $POOL/fr_a10.parquet --agent-config $CFG --model $MODEL \
  --max-new-tokens 2048 --temperature 0.7 \
  --parser-tolerance unclosed_think,bare_final_scalar --protocol-hash pre-freeze

# 冻结
$CONDA scripts/eval/write_protocol_lock.py --run-dir $RUN

# B（A 通过、协议冻结后）
$CONDA scripts/eval/run_feedback_recovery.py --stage b --run-id fr-20261014T1520Z \
  --run-dir $RUN --data $POOL/fr_b40.parquet --agent-config $CFG --model $MODEL \
  --max-new-tokens 2048 --temperature 0.7 \
  --parser-tolerance unclosed_think,bare_final_scalar --protocol-hash 4ca6c6c51c29

# C（锚点池达到门槛后）
$CONDA scripts/eval/run_feedback_recovery.py --stage c --run-id fr-20261014T1520Z \
  --run-dir $RUN --data $POOL/fr_b40.parquet --anchors $RUN/anchors/snapshots.jsonl \
  --agent-config $CFG --model $MODEL --max-new-tokens 2048 --temperature 0.7 \
  --parser-tolerance unclosed_think,bare_final_scalar --protocol-hash 4ca6c6c51c29

# 分析
$CONDA scripts/analysis/analyze_feedback_recovery.py --run-dir $RUN
```

注意：`write_protocol_lock.py` 写出的 `spec_hash` 覆盖协议文档、配置、数据池、代码文件与 A 的门槛证据；若其中任何一项在 B/C 期间变化，B/C 结果作废并需新版本。A 自身用 `pre-freeze` 记录，属于设计内（协议要求 A 通过后才冻结）。

执行顺序约束（§7.1）：C 内每个锚点的三臂顺序由 `run_order` 稳定哈希决定（不是固定 F→P→M），避免臂与时段混杂；该顺序逐条记入 `stage_c/continuations.jsonl` 的 `execution_order`。

## 7. 停止规则

- A 不通过：只允许一次针对已定位基础问题的修正后重跑；仍不通过 → `BLOCKED_VALIDITY`，不进入 B/C。
- B 锚点 <8 题 → 停止 C，输出 `INSUFFICIENT_ERROR_STATES`；8–19 → 小样本探索（不做明确支持/否定判定）。
- 基础设施故障率 > 2% → 暂停排查，不继续批量采样。
- 不因结果不理想扩样、换模型、调难度或事后挑提示模板。
