# 无效检查点清理与 r0 重启准备（2026-10-22T08:15Z）

用户授权：
- “原来的检查点没用就删除，省着占地方”
- “与毕设无关的训练结果可以删除，腾出数据盘的空间”

本次只删除两次中止运行中**无法用于正式结果**的权重载荷，不删除任何正式运行结果、原始数据、SFT 检查点、adapter 归档、日志、审计或 rollout dump。

## 归档

| 原路径 | 归档路径 |
|---|---|
| `training/r0`（KeyError 中止，step 19） | `training/r0_aborted_20261022T075708Z_watchdog_keyerror` |
| `watchdog/status.json`、`events.jsonl`、`watchdog.out`、`chain.pid` | `watchdog/attempt_02_r0_watchdog_keyerror_20261022T075708Z/` |

归档后新建空 `training/r0`。归档保留：`adapters/step_10`（69,782,352 B + 清单）、`rollout_audit.jsonl`、`train.log`、`rollout_dump/`、`resolved_config.yaml` 等；另保留了 `checkpoints/global_step_10/data.pt`（1,947 B 骨架）作为轮换证据。

## 删除（不可恢复）

| 路径 | 字节数 | SHA-256 | 理由 |
|---|---:|---|---|
| `training/r0_aborted_20261022T075708Z_watchdog_keyerror/checkpoints/global_step_20/actor/model_world_size_1_rank_0.pt` | 6,392,133,611 | `05132736ad3426c15d911b7ee9851999a8261040946c5912e8ee614b2a8280ce` | step-20 保存在 SIGTERM 时截断；缺 `data.pt`，不能加载，非有效检查点 |
| `training/r0_aborted_20261022T055629Z_sandbox_lost/checkpoints/global_step_10`（整目录） | 8,422,827,512 | 模型分片 `90c709221f6c6e0bf95ed5c59d2a2706662dc6ce77a20d58de66bf1faa025882`（8,197,332,351 B） | 来自沙箱中断的无效运行；正式组必须从同一 SFT 重新初始化，不得用该权重续训；adapter 归档与日志已保留 |

删除前已逐项核对：两个 adapter 归档均为 69,782,352 B，`archive_manifest.json` 均存在；无效运行的部分产物、审计和 dump 均保留。

## 清理前后磁盘

- 清理前可用：23,557,152,768 B（21.94 GiB）
- 清理后可用：38,372,147,200 B（35.74 GiB）
- 预计三组各 8.6 GB 峰值后仍余约 10 GiB；满足看门狗保存前不等式与 3 GiB 保留线。

## 重启前对看门狗的补充加固（不属于冻结源码树）

在原有修复（`disk_check` 全字段返回、300 s 轮换宽限、骨架目录过滤、审计尾行容忍、adapter 清单容错）之外，新增：
1. `probe_sandbox` 对 `/run_code` 返回非对象或 `run_result=null` 做类型检查，避免看门狗崩溃；
2. `run_group` 启动前 `mkdir -p` 输出目录，目录不存在时给出明确预检错误而非 `FileNotFoundError`；
3. 每组正常退出后执行 `ray stop --force`，避免单卡上残留 Ray worker 影响下一独立组。

以上均只改 `scripts/experiments/watchdog_midterm_training.py`；`verify_frozen()` 复检必须仍为 `[]`。
