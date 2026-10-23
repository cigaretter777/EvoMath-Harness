# 冒烟采样与提示修正对照（grpo_2step 系列）

实验：`midterm_uniform_reward_v1`  
目的：记录 2 步真实 GRPO 冒烟从“首回合大量退化”到“全部可解析”的定位过程、对照证据与最终验收指标。所有作废目录均原样保留，仅作诊断证据，不作为结果，也不得用于初始化正式训练。

## 1. 修正前后的真实训练栈

每格为“可解析动作 / 该回合行数”。行数按 `env.rollout.n=4`、每轮 8 个槽位共计 32 条轨迹；`grpo_2step` 跨 2 个更新，故为 64。

| 运行目录 | 提示角色 | 采样参数 | 前缀缓存 | 回合 1 | 回合 2 | 回合 3 | 回合 4 | 结局 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `grpo_2step_flat_prompt_invalid` | 单条 `user`（上游默认） | 全尾 | 开 | 9/32 | 17/32 | 22/32 | 23/32 | 完成 2 步，作废 |
| `grpo_2step_driver_patch_only` | 结构化 | 全尾 | 开 | 9/32 | 17/32 | 22/32 | 23/32 | 驱动进程补丁未跨 Ray 边界，作废 |
| `grpo_2step` | 结构化（TaskRunner 内生效） | 全尾 | 开 | 23/64 | 49/64 | 59/64 | 60/64 | 完成 2 步，作废 |
| `grpo_2step_sampling_aligned` | 结构化 | `top_p=0.95, top_k=20` | 开 | 10/32 | 24/32 | — | — | 第 3 回合预处理崩溃：`4515 > 4096` |
| `grpo_2step_safetensors` | 结构化 | `top_p=0.95, top_k=20` | 开 | 10/32 | 24/32 | — | — | 同上；权重加载路径不是原因 |
| `grpo_2step_prefixcache_off` | 结构化 | `top_p=0.95, top_k=20` | 关 | — | — | — | — | 启动失败：上游重复传参 `TypeError`，无 rollout |
| `grpo_2step_prefixcache_off_v2` | 结构化 | `top_p=0.95, top_k=20` | 关 | 64/64 | 64/64 | — | — | **2 步通过**（中间验证代码） |
| `grpo_2step_prefixcache_off_v3` | 结构化 | `top_p=0.95, top_k=20` | 关 | 64/64 | 64/64 | — | — | **最终代码 2 步通过** |

“全尾”指 `top_p=1.0, top_k=-1`。`grpo_2step_flat_prompt_invalid` 与 `driver_patch_only` 的有效行数随回合下降，是因为已结束的轨迹不再产生新动作，分母固定为 32。

## 2. 定位：HF 与 vLLM 在相同提示/参数下的分歧

四组探针使用 8 道第 0 轮训练题（`bd8f844e…`、`84640504…`、`539958f7…`、`f3032159…`、`2c5273d6…`、`80901232…`、`73d45250…`、`7cf1e085…`），提示 token、`temperature=1.0`、`top_p=0.95`、`top_k=20`、seed `20261022`、`max_tokens=2048` 完全一致。

| 生成栈 | 提示组织 | 前缀缓存 | 可解析 | 失败分类 |
|---|---|---:|---:|---|
| HF `generate`（`torch.manual_seed`） | 8 题 × 4 次独立采样 | 不适用 | **32/32** | — |
| vLLM 0.11.0 | 4 题 × 同请求 `n=2` | 开 | **8/8** | — |
| vLLM 0.11.0 | 8 题 × 4 个重复提示条目，`n=1` | 开 | **11/32** | `invalid_envelope` 11、`multiple_actions` 8、`invalid_json` 2 |
| vLLM 0.11.0 | 同上 | 关 | **30/32** | `invalid_json` 2（SFT 固有的多余引号） |

退化样本的共同形态：先正确写出 `</think>` 与 `<tool_call>{"name":"python","arguments":{"code":"…`，随后在 JSON/代码中间崩坏为多语言、emoji、控制字符混杂的文本，或连续输出多个动作；与提示格式无关（输入 token 以 `<|im_start|>system` 开头）。

结论：根因是 vLLM V1 前缀缓存与**同批次重复提示**的组合。训练侧 `multi_turn_loop` 用 `repeat(env.rollout.n=4, interleave=True)` 把同一题目的提示在同一批次重复 4 次；单请求 `n=2` 不产生重复提示条目，因此 8/8 正常。关闭前缀缓存后，剩余 2 条失败与冷启动的格式错误同源（冷启动 80 条中 6 条同类），不构成新的退化。

## 3. 代码修正

1. **结构化对话角色**（§4.1）：`build_chat_obs()` + `apply_structured_chat_patch()`，恢复 SFT 的 `system`/`user`/`assistant`/`tool` 角色。
2. **采样口径**（§4.2）：四个 GRPO 配置统一为 `temperature=1.0, top_p=0.95, top_k=20`，与冷启动门一致。该修正单独不足以消除退化，但为设计 §7.1/§7.3 的一致性要求。
3. **关闭前缀缓存**：`apply_vllm_prefix_cache_patch()` 把上游硬编码的 `enable_prefix_caching=True` 改为 `engine_kwargs.pop("enable_prefix_caching", True)`，配置通过 `+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=False` 生效；`backend_adapter.json` 记录补丁状态与解析值。
4. **上下文溢出守卫**（§4.3）：按 collector 同一 tokenizer/template 预计算提示长度，超限时显式终止并记录 `prompt_tokens`/`context_overflow`，不静默截断；v3 的 128 条动作未触发（`prompt_tokens` 471–1426，上限 4096）。

## 4. 最终冒烟（`grpo_2step_prefixcache_off_v3`）

最终代码从原始 SFT 执行 2 个真实全局更新，数据为轮次表前 16 个槽位（2 × 8），每槽 4 条轨迹。

| 指标 | 更新 1 | 更新 2 |
|---|---:|---:|
| 可解析动作 / 总动作 | 64/64 | 64/64 |
| 上下文溢出 | 0 | 0 |
| `score/mean` | 0.219 | 0.188 |
| `advantages` 范围 | `[-0.866, 1.500]` | `[-1.500, 1.500]` |
| `grad_norm` | 0.150 | 0.195 |
| `entropy_loss` | 0.399 | 0.336 |
| `response_length` 均值 / 最大 | 142.3 / 677 | 99.5 / 387 |
| `response_length/clip_ratio` | 0 | 0 |
| `prompt_length` 均值 / 最大 | 643.1 / 1406 | 512.0 / 746 |
| `episode/success_rate` | 0.219 | 0.188 |
| `timing_s/gen` | 36.2 | 29.2 |
| `timing_s/update_actor` | 115.7 | 95.7 |
| `timing_s/step` | 181.0 | 191.1 |

批次结构（审计日志）：每回合 32 条 = 8 个槽位 × 4 条，`group_id` 为 `g{update}:{task_id}#{occurrence}`，无环境侧二次展开；每次更新的 32 条轨迹都在 2 回合内以 `<final>` 结束。rollout dump 共 64 行/更新（32 条首回合 + 32 条次回合），128/128 条输出可解析。

产物：

- `checkpoints/global_step_2/`：完整 FSDP 检查点，7.9 GB；
- `adapters/step_02/adapter_model.safetensors`：69,782,352 字节，SHA-256 `fdf25e7f…c1aec3`；
- 适配器重载（PEFT + HF）：392 个 LoRA 张量、17,432,576 参数，全部有限且非零，`max_abs=0.0221`；在 `math:bd8f844e577125925d85` 上生成 113 token 的合法工具调用；
- 完整检查点恢复：见 `smoke/grpo_resume_from_step2/`（从 `global_step_2` 恢复并执行第 3 个更新，不保存新检查点）。

## 5. 复现命令

```bash
PYTHONPATH=src /root/autodl-tmp/conda-envs/adaptive-math/bin/python \
  scripts/train/run_grpo.py \
  --config configs/grpo/smoke_midterm_unified_r2_2step_prefixcache_off_v3.yaml
```

审计日志由 `ADAPTIVE_MATH_AUDIT_LOG=<run>/rollout_audit.jsonl` 开启（默认关闭；仅冒烟与诊断使用）。
