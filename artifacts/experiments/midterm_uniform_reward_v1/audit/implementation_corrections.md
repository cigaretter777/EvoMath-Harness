# 统一工具奖励预实验：实现修正与冷启动审计

实验标识：`midterm_uniform_reward_v1`  
冻结日期：2026-10-22  
状态：冷启动门（阶段 C §7.3）已通过；提示格式、采样口径、vLLM 前缀缓存三项问题已定位并修复；最终代码的 2 步真实 GRPO 冒烟通过（128/128 条动作可解析）；完整检查点保存/恢复与适配器重载均已验证。阶段 D 冻结与成本报告已提交（`audit/stage_d_freeze_and_cost.md`），其存储章节已按评审四点修正（GiB 口径、后端先删后写、逐次保存风险窗口、保存前空间检查）；正式训练尚未开始，等待用户确认。

## 1. 运行时与不可变输入

| 项目 | 值 |
|---|---|
| 基础模型 | `/root/autodl-tmp/Adaptive-Solver-main-git/artifacts/models/qwen3_1_7b_sft_tir_merged` |
| 上游后端 | `verl-agent`，manifest SHA `20bd331bdbc9026a5668e11362178e10ab7400c8` |
| 工具沙箱 | SandboxFusion，经 SSH 转发到 `127.0.0.1:8080`，`/v1/ping` 返回 `pong` |
| 训练数据 | `data/processed/midterm_math_v1/train.parquet`，200 题 |
| 开发集 | `data/processed/midterm_math_v1/dev.parquet`，50 题 |
| 评测集 | `data/processed/midterm_math_v1/test.parquet`，200 题 |
| 任务池 | `artifacts/task_pools/math_midterm_v1_rl_train.jsonl`，SHA-256 `635f9571b056f395fd574b9fd1e4d6e1eb4b28efbf3d38f38c410aed19396d46` |
| 轮次表 | `artifacts/task_pools/math_midterm_v1_rl_schedule_50x8.parquet`，400 行，SHA-256 `3154e578308910fa837bb965955bc867f5923f3e52a838de6a4cc34f53b17ca0` |
| 冻结清单 | `artifacts/experiments/midterm_uniform_reward_v1/manifests/frozen_inputs.json` |

数据切分审计：`{"ok": true, "verified_rows": 450, "failures": []}`，训练/开发/测试两两不重叠，450 条参考答案均可自校验。

## 2. 奖励口径与实现

统一预算：`max_steps=6`、`max_tool_calls=4`、`max_tool_seconds=60`、`max_tool_call_seconds=30`、单次工具观察最多 8000 字符。`max_tool_seconds` 是 Python 与 SymPy 两类工具**共享**的累计执行时间预算；只有最终答案正确的轨迹才在 R2 中扣减工具次数与时间成本，非法动作惩罚不受正确性门控。

| 版本 | 定义 | 实现位置 |
|---|---|---|
| `r0-v2-unified-tool-time` | `I(τ)` | `src/adaptive_math/reward/functions.py` |
| `r1-v2-unified-tool-time` | `I(τ) − 0.10·min(N_invalid,3)` | 同上 |
| `r2-v2-unified-tool-time` | `I(τ)·[1 − 0.15·N_tool/4 − 0.10·T_tool/60] − 0.10·min(N_invalid,3)` | 同上 |

所有版本最后 `clip(·,−1,1)`。时间成本只在 `time_weight>0` 时出现；旧版本（无 `time_weight`）序列化与配置哈希保持不变。R2 的 `token_cost` 组件保留但权重为 0，正式实验不使用 token 成本。

工具计时规则：

- Python：优先使用 SandboxFusion 返回的 `execution_time`；超时按请求额度计入；沙箱不可用不计入工具时间、不计为非法动作，记为 `infrastructure_error`。
- SymPy：在独立 worker 进程两侧用 `perf_counter()` 计时；安全拒绝属于“未执行”，不计时。
- 解析失败、参数校验失败属于非法动作，不产生工具执行时间。

上游 `apply_invalid_action_penalty` 默认会对非法动作再次惩罚；三组配置统一设置 `actor_rollout_ref.actor.use_invalid_action_penalty=False`，避免与 R1/R2 的显式惩罚重复。

## 3. 实际优势计算实现

上游 `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage` 的实际行为：

- 以 `uid` 为组，`env.rollout.n=4` 时每个 `uid` 恰好对应 4 条轨迹；
- 组内均值使用 `torch.mean`；标准差使用 `torch.std` 的样本标准差（自由度 `n−1`）；
- 分母为 `std + 1e-6`；零方差组分子恒为 0，因此优势为 0；
- 单样本组（本实验不会出现）使用均值 0、标准差 1。

**发现的口径偏差**：上游默认 `compute_mean_std_cross_steps=True`。多回合收集器把每条轨迹的每个回合都作为独立行送入优势计算，默认行为会把 4 条轨迹的均值/标准差按“回合行数”加权，长轨迹被重复计入；这与开题公式“同一题目 4 条轨迹为一组”不一致，也与我们离线 `reward_bridge.group_advantages` 的口径不一致。

**统一修正**：`apply_trajectory_level_grpo_advantage()` 在 `run_grpo.py` 启动时给该函数注入 `compute_mean_std_cross_steps=False`，使每个 `(uid, traj_uid)` 只贡献一次终局奖励。修正对 R0/R1/R2 三组完全相同，不改变优化器、损失函数或组大小；仅把组内归一化口径对齐到开题公式。`backend_adapter.json` 会记录该补丁是否生效。相关测试：`tests/unit/training/test_advantage_normalization.py`。

## 4. 组展开与审计字段

上游 `TrajectoryCollector.multi_turn_loop` 对 8 个 prompt 槽位执行 `repeat(env.rollout.n=4, interleave=True)`，得到 32 个环境；每个环境一题，**没有环境侧的二次乘 4**。`uid` 每 4 行重新生成，因此每轮 8 个组、每组 4 条轨迹。配置同时固定：

- `actor_rollout_ref.rollout.n=1`（verl 侧不再重复采样）；
- `env.rollout.n=4`（环境组大小）；
- `data.train_batch_size=8`（每轮 8 个槽位）；
- `data.shuffle=false`（轮次表按行读取）；
- `trainer.total_training_steps=50`；
- `trainer.save_freq=10`、`trainer.max_actor_ckpt_to_keep=1`；
- `trainer.logger=["console"]`（不使用 wandb，避免外部登录和网络依赖）。

五个完整 FSDP 检查点（每个约 7.9 GB）无法在单组训练中同时保留。正式配置只保留最新完整检查点，并通过 `apply_adapter_archive()` 在每次保存后把 `global_step_N/actor/lora_adapter`（约 69.8 MB）复制到 `training/<group>/adapters/step_NN`，附 SHA-256 清单。第 10/20/30/40 轮保留可独立加载的 LoRA 适配器，第 50 轮同时保留完整检查点作为唯一主结果。跨三组的磁盘预算与保存前检查见 `audit/stage_d_freeze_and_cost.md` §5（不使用系统盘暂存）。

环境信息中携带 `task_id`、`prompt_slot_id`（`g{global_update}:s{slot_index}`）、`group_id`（`g{global_update}:{task_id}#{occurrence}`）、`sample_index`、`global_update`、`policy_version`、`generated_tokens`。可选的 `ADAPTIVE_MATH_AUDIT_LOG` 会把真实 reset/step 记录追加为 JSONL，供冒烟阶段保存批次展开样例；默认关闭。

### 4.1 训练提示的对话角色修正

第一次真实 2 步冒烟（`smoke/grpo_2step_flat_prompt_invalid/`，已保留但不作为结果）在第 1 步 32 条轨迹中出现 **23 条非法动作**，仅 8 次工具调用；而冷启动 80 条轨迹只有 8 条非法动作、78 次工具调用。检查上游 `agent_system/multi_turn_rollout/rollout_loop.py` 后确认根因：`preprocess_single_sample` 把环境返回的完整观察文本强行包装成单条 `user` 消息，丢失了 SFT 数据实际使用的 `system` / `user` / `assistant` / `tool` 角色。冷启动门走的是 `TransformersModelClient`，它直接用原始消息列表调用 chat template，因此没有这个偏移。

修复只涉及提示格式，不改变环境、奖励或优化器：`MathRolloutManager.build_chat_obs()` 返回与文本观察并行的结构化消息；`apply_structured_chat_patch()` 让上游 collector 在 `obs['chat']` 存在时用这些消息渲染 chat template，否则保持上游原有行为。用真实 tokenizer 和真实 collector 验证：修正后的 `raw_prompt_ids` 与 `tokenizer.apply_chat_template([system,user], add_generation_prompt=True)` 完全一致。相关测试：`tests/unit/training/test_structured_chat_patch.py`、`tests/unit/training/test_rollout_audit_fields.py`。

### 4.2 训练采样口径修正（top-p / top-k）

§4.1 的角色修正生效后，第三次冒烟 `smoke/grpo_2step/` 第 1 步仍有 **22/32 条非法动作**（有效轨迹中去重后同量级），而冷启动 80 条轨迹的首回合非法动作只有 **2/80 = 2.5%**（总计 8 次非法动作，7 次 `invalid_json`、1 次 `invalid_envelope`）。抽查 `rollout_dump/1.jsonl` 的原始输出可见：模型先按 SFT 先验正确写出 `<think>Execute the supplied calculation.</think>` 和 `<tool_call>{"name":"python",...` 的代码前缀，随后在 JSON/代码中间崩坏为泰语、阿拉伯语、韩语、俄语、emoji 与控制字符混杂的文本，并出现未闭合标签。这不是解析器或提示格式问题（`input` 解码后以 `system` 开头，说明结构化角色已生效），而是采样进入了低概率尾部后级联退化。

根因是两套采样器的实际参数不一致：

| 阶段 | 实现 | `temperature` | `top_p` | `top_k` |
|---|---|---:|---:|---:|
| 冷启动门（已验证） | `TransformersModelClient` → HF `generate`，仅显式传 `temperature=1.0` | 1.0 | 0.95 | 20 |
| 训练 rollout（前三次冒烟） | vLLM `SamplingParams`，由 GRPO 配置显式指定 | 1.0 | 1.0 | −1 |

冷启动客户端只覆盖了 `temperature`；`top_p=0.95`、`top_k=20` 来自合并模型自带的 `generation_config.json`，因此冷启动的实际解码规则是“温度 1.0 + 核采样 0.95 + top-20”。训练配置却关闭了两种截断（`top_k=-1`、`top_p=1.0`），在 2048 token 的长代码生成中频繁命中低概率尾部。设计 §7.1 要求“复用已验证配置并记录 temperature、top-p/top-k”，§7.3 要求冷启动“按训练采样规则”生成；两者必须一致。现在已把四个 GRPO 配置（冒烟 + R0/R1/R2）统一改为 `temperature=1.0, top_p=0.95, top_k=20`，与冷启动门的有效规则完全一致。

**采样口径对齐后退化仍未消失**：`smoke/grpo_2step_sampling_aligned/` 首回合仍只有 10/32 条可解析，并在第 3 回合预处理时因 `sequence_length=4515 > max_length=4096` 终止；safetensors 权重同步的对照运行结果完全相同（`smoke/grpo_2step_safetensors/`，日志中 `loaded_params: 392` 仅含 LoRA 张量），排除了 dummy 权重同步路径。随后用同一批第 0 轮题目做了四组生成栈对照（8 题 × 4 个副本，`temperature=1.0, top_p=0.95, top_k=20`，seed 20261022，提示 token 完全相同）：

| 生成栈 | 提示组织 | 关键配置 | 可解析 |
|---|---|---|---:|
| HF `generate` | 8 题 × 4 次独立采样 | `torch.manual_seed` | **32/32** |
| vLLM 0.11.0 | 4 题，同一请求 `n=2` | 默认前缀缓存 | **8/8** |
| vLLM 0.11.0 | 8 题 × 4 个**重复提示条目** | 默认前缀缓存 | **11/32** |
| vLLM 0.11.0 | 8 题 × 4 个重复提示条目 | `enable_prefix_caching=False` | **30/32** |

唯一与退化共变的变量是 vLLM V1 前缀缓存。训练侧 `env.rollout.n=4` 会把同一题目的提示在**同一批次内重复 4 次**（`repeat(..., interleave=True)`），而单请求 `n=2` 不产生重复提示条目；关闭前缀缓存后，可解析率回到与冷启动同量级（30/32；另 2 条是 SFT 固有的多余引号格式错误，冷启动中同样出现）。因此根因既不是采样参数，也不是提示格式；采样口径修正仍是设计 §7.1/§7.3 的一致性要求，必须保留，但它单独不足以消除退化。

训练栈无法直接覆盖该参数：上游 `vllm_rollout_spmd.py` 在 `LLM(...)` 里硬编码 `enable_prefix_caching=True`，又在后面展开 `engine_kwargs`；配置再提供同名键时会先触发 `TypeError: got multiple values for keyword argument`。新增的 `apply_vllm_prefix_cache_patch()` 在启动时把该行改为 `enable_prefix_caching=engine_kwargs.pop("enable_prefix_caching", True)`，只有显式配置该项的运行才改变行为；`backend_adapter.json` 记录补丁状态与解析后的值。三组正式配置和冒烟配置统一加入 `+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=False`。

真实训练栈验证（`smoke/grpo_2step_prefixcache_off_v2/`，2 个全局更新）：64/64 条动作可解析，两次更新的 32 条轨迹均在 2 回合内以 `<final>` 结束；`response_length` 均值 142/90、最大 677/255，`clip_ratio=0`；优势范围 `[-1.5, 1.5]`，梯度范数 0.151/0.202；检查点与 `adapters/step_02` 正常保存。最终代码（含 §4.3 的上下文守卫与 tokenizer 注入）在 `smoke/grpo_2step_prefixcache_off_v3/` 再跑一次：128/128 条动作可解析，`prompt_tokens` 471–1426，无上下文溢出，指标与 v2 的第 0 轮完全一致。关闭前缀缓存会让 8 组 × 4 条重复提示的预填充重复计算，但每回合提示约 0.5–1.4k token，生成耗时 36.2 s / 29.2 s，明显快于退化时的 120–302 s。

三次作废冒烟目录均原样保留，仅作诊断证据，不作为结果、也不得用其检查点初始化正式训练：

- `smoke/grpo_2step_flat_prompt_invalid/`：角色未修正（9/32 首回合可解析）；
- `smoke/grpo_2step_driver_patch_only/`：仅驱动进程打补丁，Ray 进程未生效（9/32）；
- `smoke/grpo_2step/`：TaskRunner 补丁生效但前缀缓存未关闭（23/64 首回合可解析，跨两次更新）。

`smoke/grpo_2step_sampling_aligned/` 与 `smoke/grpo_2step_safetensors/` 同样保留为“仅对齐采样、未关闭前缀缓存”的证据（首回合 10/32 可解析，第 3 回合上下文溢出）。`smoke/grpo_2step_prefixcache_off/` 保留了一次失败的启动日志：Hydra 覆盖已生效但上游重复传参导致 `TypeError`，该目录没有任何 rollout。

### 4.3 上下文溢出的显式终止

设计 §7 要求“上下文上限、每轮提示上限、聊天模板和历史截断规则在冒烟时一并固定……禁止静默截断问题、系统协议或随意丢弃历史；上下文溢出的处理及标志必须一致且可审计”。本实现的规则：

- 每轮提示上限固定为 `data.max_prompt_length=4096`，单次生成上限 2048；`data.truncation` 保持上游默认 `error`，不做静默截断。
- `MathRolloutManager` 注入与 collector 完全相同的 tokenizer 和 chat template，在每次 `env.step` 追加动作/观察后计算下一轮提示的 token 数。超过 4096 时：
  - 若轨迹尚未终止，置 `TerminationReason.CONTEXT_OVERFLOW`，按“没有最终答案”走与 `max_steps` 相同的奖励路径（R0 为 0；R1/R2 保留已累计的非法动作惩罚；R2 工具成本仍受正确性门控）。
  - 若该回合已经产生 `<final>`，保留 `final` 终止原因与奖励，仅记录溢出标志。
  - 该行在后续固定宽度批次中的提示恢复为初始 `system`/`user` 上下文；完整的动作与观察仍在轨迹事件和审计日志中。这样 collector 在下一回合预处理时不会整批崩溃，后续生成本来也不参与训练（`active_masks=False`）。
- `step` 信息新增 `prompt_tokens` 与 `context_overflow`，分析脚本可把“上下文溢出”与“生成长度截断”“观察截断”分列统计。
- 初始提示本身超过上限时在 `reset` 抛出明确错误，而不是让 collector 报 `sequence_length=...`。
- 单元测试：`tests/unit/training/test_verl_math_environment_manager.py::test_context_overflow_*`。

该守卫在正式运行中才可能触发；v3 冒烟的 128 条动作中 `prompt_tokens` 均值为 0.6–1.4k，未触发。由于它是冒烟之后新增的代码，最终代码的冒烟（v3）已重新执行并通过，满足“同一份代码先冒烟后正式”的要求。

## 5. 冷启动门（20 题 × 4 条 = 80 条）

运行目录：`artifacts/experiments/midterm_uniform_reward_v1/smoke/coldstart_dev_20x4/`  
模型：原始 SFT；采样：`temperature=1.0`、`top_p=0.95`、`top_k=20`（后两者继承自模型 `generation_config.json`）、`max_new_tokens=2048`；奖励记录使用 R2。

| 指标 | 结果 |
|---|---:|
| 轨迹数 / 组数 | 80 / 20 |
| 最终答案正确 | 24（30.0%） |
| 错误 / 验证器无法解析 | 54 / 2 |
| 有最终答案 | 79（98.75%） |
| 使用工具的轨迹 | 78（97.5%） |
| 工具调用总数 | 78 |
| 工具结果：成功 / 运行失败 / 超时 | 50 / 26 / 2 |
| 非法动作总数 | 8 |
| 解析错误：`invalid_json` / `invalid_envelope` | 7 / 1 |
| 工具观察被截断 | 1（8000 字符上限） |
| 单回合生成达到 2048 token 上限 | 1 |
| 基础设施错误 | 1（沙箱不可用，未计非法动作） |
| `max_steps` 终止 | 1（该轨迹 5 次非法动作，R2 = −0.30） |
| 有效奖励组（非零方差） | 12 / 20 |
| 全零组 / 全一组 / 混合组 | 8 / 0 / 12 |
| 平均奖励 / 标准差 | 0.2811 / 0.4474 |
| 总耗时 | 609.9 s |

说明：

- “全一组”为 0 是 R2 成本扣减的预期结果：正确轨迹奖励约 0.962，不会恰好等于 1.0；组是否有学习信号应看 `std>0`（12 组）。
- 78/80 条轨迹发生真实工具执行，50 条成功、26 条运行失败、2 条 30 秒超时，成功/失败/超时路径都产生了完整事件日志。
- 7 条 `invalid_json` 中 6 条是同一模型格式：`<final>{"answer":"..."}</final>` 的 JSON 后多了一个反引号；另 1 条是生成达到 2048 token 上限、动作未闭合。按冻结协议，反引号属于模型侧格式错误，由 R1/R2 的非法动作惩罚处理，未增加解析容忍规则；该现象三组一致，不影响组间公平性。
- 未观察到上下文溢出终止；验证器异常为 0。

离线重算：对全部 80 条终局轨迹用 `scripts/analysis/recompute_rewards.py --max-generated-tokens 12288` 重建 `Trajectory`/`VerifierResult` 并调用与训练相同的 `reward_for_trajectory`，总奖励与各分项 **80/80 完全一致，0 处不匹配**（`reward_recompute.json`）。

结论：冷启动门通过。存在正确且使用工具的轨迹，组内有有效奖励差异，协议与工具路径可用；没有出现“80 条全部零工具执行”或“没有正确且使用工具的轨迹”的暂停条件。

## 6. 阶段 C 检查完成情况

- [x] 从原始 SFT 另起临时目录完成 2 个真实 GRPO 全局更新（最终代码：`smoke/grpo_2step_prefixcache_off_v3/`）：128/128 条动作可解析、梯度有限、优势范围 `[-1.5, 1.5]`、真实批次 8 组 × 4 条且无重复展开、`generated_tokens` 与训练掩码一致（每回合 `generated_tokens` 由 `attention_mask` 的响应段求和传入）、`prompt_tokens` 已记录。
- [x] 真实批次展开审计样例保存在 `smoke/grpo_2step_prefixcache_off_v3/rollout_audit.jsonl`（reset/step 各 32 条/回合，含原始 `action_text`、组/槽位、`prompt_tokens` 与 `context_overflow`）。
- [x] 修正前后对比见 `audit/smoke_sampling_correction.md`。
- [x] 检查点保存：`checkpoints/global_step_2`（7.9 GB）与 `adapters/step_02`（69.8 MB，SHA-256 清单）均已生成。
- [x] 适配器重载：用 PEFT 加载 `adapters/step_02`，392 个 LoRA 张量、17,432,576 参数全部有限且非零（`max_abs=0.0221`），在 `math:bd8f844e577125925d85` 上生成 113 token 的合法 `<tool_call>`。
- [x] FSDP 完整检查点重载：`smoke/grpo_resume_from_step2/` 以 `trainer.resume_mode=resume_path` 从 `global_step_2` 恢复，`global_update=2`，在轮次表第 17–24 槽位（8 × 4 条轨迹）上完成第 3 个全局更新；首回合 30/32、次回合 30/32 动作可解析，第 3 回合全部以 `<final>` 结束。`grad_norm=0.187`，优势范围 `[-1.5, 0.783]`，`timing_s/step=380.5`。注意：即使 `save_freq=999`，verl 仍在最终步写出 `checkpoints/global_step_3` 与 `adapters/step_03`，正式训练 50 轮同样会在第 50 轮落盘（符合预期，主结果即第 50 轮完整检查点）。该次运行中 1/32 条轨迹达到 2048 token 生成上限（`clip_ratio=0.045`），属于生成长度上限，不是上下文溢出。
- [ ] 阶段 D：冻结最终配置/数据/模型/代码哈希，提交三组 50 轮的时间与成本估算，等待确认后再启动正式训练。冻结清单、测试结果、实测速率、时间/存储估算与执行顺序已写入 `audit/stage_d_freeze_and_cost.md`；`frozen_inputs.json` 已于 2026-10-22T04:20:51Z 重新生成，八个配置副本与源文件逐字节一致；三组正式配置已补上 `trainer.rollout_data_dir` 以保留逐轮生成，供阶段 F 复算，并显式设置 `trainer.resume_mode=disable`，保证每组从原始 SFT 启动。**当前等待用户确认，正式训练尚未启动。** 已复核 GPU 资源：当前只有 1 张 RTX 4090 D（24,564 MiB，复查时空闲），因此三组采用串行；串行仅表示执行顺序，三组仍各自从同一原始 SFT 独立初始化，使用独立优化器、训练状态与输出目录，不传递权重。

### 6.1 磁盘清理记录

为给最终冒烟与正式训练腾出空间，删除了两个**已作废冒烟**的完整 FSDP 检查点目录，保留其 `adapters/`、`rollout_dump/`、`rollout_audit.jsonl`、`train.log` 与 `backend_adapter.json`：

- `smoke/grpo_2step/checkpoints/global_step_2`（7.9 GB，前缀缓存未关闭的作废运行）；
- `smoke/grpo_2step_prefixcache_off_v2/checkpoints/global_step_2`（7.9 GB，前缀缓存修正的中间验证运行；适配器已归档且 SHA-256 已记录）。

在完整检查点重载验证通过后，又删除了两个**冒烟**完整检查点，保留其 `adapters/`、`rollout_dump/`、`rollout_audit.jsonl`、`train.log` 与 `backend_adapter.json`：

- `smoke/grpo_2step_prefixcache_off_v3/checkpoints/global_step_2`（7.9 GB，最终冒烟的检查点；已由重载实验证明可恢复，适配器与 SHA-256 保留）；
- `smoke/grpo_resume_from_step2/checkpoints/global_step_3`（7.9 GB，重载诊断运行写出的检查点；其 `adapters/step_03` 与日志保留）。

三份冒烟检查点合计释放约 23.7 GB，磁盘可用空间回到约 22 GB（`160G` 盘，已用 `139G`）。正式组（`training/r0|r1|r2`）的旧检查点目录未被触碰，任何正式检查点均未删除。

阶段 D 冻结后，又清理了两个**可再生缓存**（非实验数据，均可从上游重新生成）：`/root/autodl-tmp/hf-cache`（6.3 GB，Qwen3 基础模型与旧数据集缓存；正式训练从本地合并权重加载，不使用该缓存）与 `/root/autodl-tmp/pip-cache`（1.5 GB）。

随后经用户按确切路径授权，删除了旧项目冒烟的**中间**检查点 `Adaptive-Solver-main-git/artifacts/runs/grpo_qwen3_1_7b_smoke_r0/checkpoints/global_step_1`（7.9 GB）。该旧 run 的最终轮 `global_step_2`、`resolved_config.yaml`、`backend_adapter.json` 与导出适配器均保留，`latest_checkpointed_iteration.txt` 仍为 `2`；旧项目的 R0/R2/R4 正式检查点未动。

清理后数据盘可用 **35.90 GiB**（`160G` 盘已用 `125G`，78%；字节数 38,548,889,600）。按 GiB 逐项核算：三份第 50 轮完整检查点 23.53 GiB、15 份适配器 0.97 GiB、日志/dump/审计 0.04 GiB，训练新增合计 24.54 GiB，写完后剩余 **11.35 GiB**（含 3 GiB 安全余量后仍余 8.35 GiB）。pinned 后端的 `max_actor_ckpt_to_keep=1` 是**先删除上一份、再写入新一份**，同组检查点不共存，因此不存在“第 50 轮与第 40 轮共存”的 31.6 GiB 峰值；31.6 GB 只适用于“先保存后删除”的假设口径。风险窗口存在于每次保存，故正式运行前按 `free_now + rotated >= expected_new + reserve` 检查。三组检查点直接写数据盘，无需暂存；详见 `audit/stage_d_freeze_and_cost.md` §5 与 §7。

第一次冒烟尝试在 `fit()` 初始化日志时因上游默认 `trainer.logger=['console','wandb']` 且本机未配置 wandb API key 而退出；失败发生在任何 rollout/更新之前，未产生检查点。已将三组正式配置与冒烟配置统一改为 `trainer.logger=["console"]` 后重跑，不涉及训练算法或数据口径。

## 7. 可复现命令

```bash
PYTHONPATH=src /root/autodl-tmp/conda-envs/adaptive-math/bin/python \
  scripts/analysis/analyze_coldstart.py \
  --run-dir artifacts/experiments/midterm_uniform_reward_v1/smoke/coldstart_dev_20x4

PYTHONPATH=src /root/autodl-tmp/conda-envs/adaptive-math/bin/python \
  scripts/analysis/recompute_rewards.py \
  --trajectories artifacts/experiments/midterm_uniform_reward_v1/smoke/coldstart_dev_20x4/trajectories.jsonl \
  --max-generated-tokens 12288
```
