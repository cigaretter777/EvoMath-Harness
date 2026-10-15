# 协议偏离记录（DEVIATIONS）

- run_id：`fr-20261014T1520Z`
- 协议：`/root/autodl-tmp/BudgetConstrainedReasoningMathAgent/2026-10-14-feedback-recovery-validation.md`
- 记录规则：每条给出「偏离什么 / 何时发生 / 为什么 / 影响范围 / 证据路径」。无偏离的项不写；本文件本身即为"是否偏离"的声明。

## D1 阶段 A 首次运行失败并保留（协议内允许的一次修正后重跑）

- **偏离**：A 不是一次通过，而是"失败 → 定位基础问题 → 修一次 → 重跑"。
- **何时**：2026-10-14 23:03（首次启动）—23:29（重跑完成），本地时间。
- **原因**：我为满足"生成 seed 真实生效"新增的 seed 通路误用 `model.generate(generator=...)`；transformers 4.57 不接收该参数，直接抛 `model_kwargs are not used by the model: ['generator']`。AgentLoop 当时把异常吞成 `model_error`，故 20 条轨迹全部 0 轮终止、0 工具调用、0 token。这是**我方实现缺陷**，不是模型或沙箱问题。
- **影响**：首次运行的 20 条轨迹全部无效，不进入任何统计。
- **证据**：`stage_a_attempt0_model_error/`（run.log、summary、task_pool、trajectories、manifest.a.json 原样保留，未覆盖）；修复后同目录 `stage_a/`。
- **协议依据**：§5「允许一次针对已定位基础问题的修正后重跑A，保存前后配置与结果」。

## D2 附带修复：模型调用异常原因入库

- **偏离**：在修 seed 的同时扩大了改动范围——`AgentLoop` 现在把异常类型与消息写进轨迹事件（`record_model_error`），此前只写 `model_error`。
- **原因**：没有这条记录，D1 那样的全 0 调用失败与"模型自己不动手"在轨迹上不可区分，而协议恰恰要求把基础设施故障与模型行为分开（§5.3、§7.2）。
- **影响**：仅新增事件字段，使失败可审计；不改变成功路径的任何行为。
- **证据**：`src/adaptive_math/agent/loop.py`、`src/adaptive_math/agent/environment.py`；哈希入 `protocol.lock.json`。

## D3 基础条件修复：错误反馈此前对模型不可见

- **偏离**：本轮修复了"工具失败时用通用失败文案替换真实输出"的既有行为，属于协议 §3.3 明确要求声明的"基础条件修复"。
- **何时**：修复自提交 `0d576aa`（2026-09-29）起存在；本轮 2026-10-14 修好。
- **原因**：TIR 评测（`a4c79bb`）在 `0d576aa` 之后运行，因此此前 56 次执行错误对模型不可见；若不修，本诊断的 F 臂根本没有"反馈"可谈。
- **影响**：**本轮的 F/M 臂观测与 MVP 轮次及第一轮零调用诊断不可比**，故不与它们合并汇总（协议本就禁止）。
- **证据**：`src/adaptive_math/agent/environment.py` 的 `Tool call failed (<code>).` + 真实 stdout/stderr；A 阶段 3 条人工读的轨迹可见 traceback 已到达模型（`protocol.lock.json.stage_a.manual_read`）。

## D4 数值答案规则改用代理过滤

- **偏离**：协议 §4.1 倾向数值答案，但本题池 `answer_type` 100% 为 `expression`，无法按类型筛。
- **做法**：改用 `reference_value` 的**值形式**正则做代理，并把规则原文写进 manifest；排除分数/根式/关系式等未校验形式。
- **影响**：样本不是纯数值答案；判定仍走既有 `verify_answer`（表达式由 symbolic worker 判定），未新造判定器。
- **证据**：`data/manifests/diagnostic_feedback_recovery.json` 的 `numeric_filter` 字段。

## D5 难度窗口由先验证据选定（协议未指定）

- **说明**：协议未规定难度；我按**已花费的 200 题**（E0/E1）的先验统计选了 3.5–4.75 两个相邻层，理由是 3.5–4.5 段正解率 12.1%、执行错误率 32%，既有错误又有恢复空间；1–3 段 45.2% 过饱和。
- **对结果的约束**：这是**基于先验（非本轮题池结果）**的选择，但确属对难度的事前挑选，故记录；难易结论只对本池成立。

## D6 人工审计无独立复核者

- **偏离**：§8.3 优先由不了解实验臂的复核者检查；本轮由**执行 agent 本人**自审。
- **影响**：审计结论有自我确认风险；`audit/labels.jsonl` 中 `reviewer` 字段逐条标注为 `executing agent (not independent)`，报告中不得把自动标注当作人工审计。
- **证据**：`audit/labels.jsonl` 首行即选择规则与复核者声明。

## D7 C 阶段单锚点冒烟使用正式锚点

- **说明**：§10 要求 CPU/mock 测试通过后先跑单锚点分支冒烟。冒烟使用**正式锚点 #1** 的 12 条分支（非临时锚点）。
- **处理**：冒烟记录纳入 `stage_c/continuations.jsonl`；批量运行时按 `logical_sample_id` 的 done-set 跳过已跑样本，**同一分支不会重复计数**，也不按结果替换。
- **证据**：`stage_c/summary.json.executed_this_invocation` 与 `stage_c/continuations.jsonl` 的 `logical_sample_id`/`attempt_id`。

## D8 分析层的重试去重

- **说明**：基础设施故障重试会在日志里为同一 `logical_sample_id` 留下两条记录（attempt 0 失败、attempt 1 成功）。协议 §10.6 要求"故障重试去重"。
- **处理**：分析脚本按 `logical_sample_id` 去重——优先取可用尝试；若全部尝试均为基础设施故障，则该分支记为**缺失**（不是答错）。重试次数与"重试后仍缺失"数量写入 `analysis/summary.json.attempt_dedupe`；若出现"非基础设施故障后重试"的协议违例，单独列出而不是悄悄合并。

## D10 冻结后对两个**非数据生成**脚本的修改（已逐文件核验）

- **事实**：`protocol.lock.json` 记录 37 个代码文件的 sha256。冻结后逐文件重算，**有 2 个文件哈希变了**：
  - `scripts/analysis/analyze_feedback_recovery.py`：新增按 `logical_sample_id` 的重试去重（协议 §10.6 要求）、按 §8.3 把审计抽样改为带 seed 的随机抽样 + 按臂轮转（此前是"文件前 N 条"）。
  - `scripts/eval/check_seed_reproducibility.py`：仅类型注解与 `# type: ignore` 清理（为过 mypy），运行行为未变；其产出的 `stage_a/seed_check.json` 仍是修改前那次运行的记录。
- **数据生成路径零改动（核验方法）**：对锁中全部 37 个文件重算 sha256 并逐一比对，`src/adaptive_math/**`、`scripts/eval/run_feedback_recovery.py`、`configs/agent/feedback_recovery.yaml`、数据池与 manifest **全部与冻结时一致**；变化的只有上面两个事后用途脚本。
- **时间与影响**：两处修改都发生在 B 仍在运行时，此时**C 数据尚不存在、任何分析产物尚未生成**，因此不存在"按结果调分析"的可能；分析脚本不参与 B/C 的数据生成，也不改变任何轨迹。
- **为什么不在冻结后更新锁**：更新 `spec_hash` 会使 B 的 manifest 记录的哈希失去意义，并等价于"B 跑到一半换协议"，正是协议 §5 禁止的。故保留原锁，把差异记在这里；报告中的分析结果按其脚本的**当前**哈希（`analysis/summary.json.analysis_script_sha256`）可复核。

## D11 已知的派生字段缺陷：`tasks_with_any_exec_error`

- **事实**：`stage_b/summary.json` 的 `tasks_with_any_exec_error` 字段实现有误——它统计的是"有任意轨迹的题数"（恒等于 40），不是"出现过 E_EXEC 的题数"。属我方实现缺陷。
- **为什么不改**：该字段在冻结代码内，修改即触发 D10 所述的同一问题；而它只是**描述性派生字段**，不影响任何轨迹、锚点选择或统计。
- **处置**：正确值由分析脚本从 `stage_b/trajectories.jsonl` 原始轨迹重算，写入 `analysis/summary.json`，报告中只使用重算值；此处披露以免两处数字不一致时被误读。
- **影响**：无——该字段不参与任何门槛判定（锚点门槛用的是候选池本身）。

## D12 收尾阶段的三项记录（人工审计块 / 坏字段打印 / 审计覆盖率）

- **人工精读块的位置与脆弱性**：§8.3 的人工精读结果以 `kind:"human_review_block"` 的**末行**追加在 `audit/labels.jsonl`（覆盖 324 条续跑中被读的 6 条分支的全文判定）。分析脚本 `analyze_feedback_recovery.py` 以 `"w"` 模式写该文件，**重跑分析会把它覆盖掉**；重跑后必须重新追加这一段（内容同时保存在本报告 §2 与收尾提交记录中）。本次收尾**没有**重跑分析，即当前 `audit/labels.jsonl` 是"自动抽样 + 人工块"的完整版本。
- **`analysis/tables.md` 中的坏字段**：B 表的 `tasks_with_any_exec_error = 40` 是 D11 所述坏字段的**原样透传**（自动生成的表不纠正它）。正确值 27 在 `analysis/summary.json.stage_b_recomputed.tasks_with_any_exec_error_corrected`；报告中一律用重算值。
- **人工审计覆盖率低于抽样规模**：自动抽样为 C 30 条分支 + B 20 条轨迹；实际人工精读 **C 6 条分支全文、B 3 条轨迹对照原始轨迹**（自动抽样的 1/5、3/20）。原因是收尾时间成本，且复核者即执行 agent（D6，无独立性）。**影响**：审计只支持"抽查未见反馈不可见/误判"这类否定性核对，不能支撑对分类或判定正确率的量化声明。

## D13 运行计划中一处错误的自我更正（`reparse_agrees_with_environment`）

- **事实**：`run_plan.md` §5 的 A 门槛段在运行期间写作 `reparse_agrees_with_environment=true`，与记录不符：`stage_a/summary.json.gate.reparse_agrees_with_environment=false`（环境计到 1 个 `call_form` 错误步，重解析器在同一条轨迹上为 0 个解析错误步）。
- **影响**：**无门槛影响**——自动门槛用的是 `parse_error_step_rate`（重解析口径 0.000）与 `length_truncated_without_action_rate`（0.024），两者各自都在 ≤0.10 阈值内，5 项自动检查全过；该布尔只表示两种口径是否逐条一致，"不一致但都合格"本身不是失败。
- **处置**：收尾时已在 `run_plan.md` 就地改正并加注，留此条以免"记录与计划不符"被当作事后篡改。

## D9 运行环境事实（非偏离，但影响可复现范围）

- 模型与 GPU：本机 RTX 4090D；沙箱 SandboxFusion 经 SSH 隧道在**另一台机器（Mac）**上运行，因此沙箱往返延迟计入墙钟但不计入 `python_seconds`（后者取沙箱自报 `execution_time`）。
- tokenizer 加载时 transformers 提示 regex 模式问题（建议 `fix_mistral_regex=True`）；**未改动**，因为训练与既往评测均用同一 tokenizer 配置，改了反而破坏可比性。记录于此以免被误认为新引入的偏差。
- 仓库基线提交 `206750e`，分支 `codex/feedback-recovery`，本轮代码在工作树（未提交、未合入 main），哈希逐文件记入 `protocol.lock.json`。
