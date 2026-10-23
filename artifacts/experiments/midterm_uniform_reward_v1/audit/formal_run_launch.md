# 正式训练启动记录

- 正式启动时刻（UTC）：2026-10-22T04:48:37Z（第二次启动，见文末“首次启动中止记录”）
- 实验：`midterm_uniform_reward_v1`
- 执行顺序：r0 → r1 → r2（**仅执行顺序，不传递权重**）
- 启动方式：`scripts/experiments/launch_midterm_formal.sh` → `scripts/experiments/watchdog_midterm_training.py`
- 看门狗 PID：51758（`watchdog/chain.pid`）；r0 训练子进程 PID 51774（`watchdog/events.jsonl` 的 `group_start`）
- 每组命令：
  ```bash
  PYTHONPATH=src ADAPTIVE_MATH_SANDBOX_URL=http://localhost:8080 \
  ADAPTIVE_MATH_AUDIT_LOG=artifacts/experiments/midterm_uniform_reward_v1/training/<group>/rollout_audit.jsonl \
  /root/autodl-tmp/conda-envs/adaptive-math/bin/python scripts/train/run_grpo.py \
    --config configs/grpo/midterm_unified_<group>.yaml --manifest third_party/manifest.json
  ```

## 启动前核对（看门狗 `--check-only`，退出码 0）

| 项 | 结果 |
|---|---|
| `frozen_inputs.json` 配置 8 项、数据 8 项、源码树 3 棵、SFT 10 文件 | 全部 SHA-256/字节数一致 |
| GPU | 1 × RTX 4090 D，24,564 MiB；启动前占用 1 MiB、利用率 0%、35 °C、无计算进程 |
| 沙箱 | `/v1/ping` = `pong`；`print(6*7)` 返回 `42` |
| 数据盘 | 可用约 35.90 GiB（`df -B1` 启动时为 38.55×10⁹ B 量级） |
| `training/r0|r1|r2` | 全部为空目录，`resume_mode=disable` |

## 看门狗规则（自动执行，不改变训练代码）

1. 每次启动前复检冻结哈希、GPU 空闲、沙箱 `pong+42`、输出目录为空、磁盘不等式。
2. 运行中每 10 s 检查进程、磁盘、GPU；每 30 s 实测沙箱两次（ping + 执行）。
3. 每次检查点保存后核对：`global_step_N` 完整、`adapters/step_NN` 清单与 SHA-256、该组更新 0..N−1 均为 32 条 reset 且 32 个唯一 `(slot,sample)` 对。
4. 保存前磁盘不等式：`free_now + rotated >= 8.6 GB + 3 GiB`；不满足立即停止。
5. 硬停止条件：非零退出、NaN/Inf 指标、同一组两次 CUDA OOM、连续 3 次沙箱探测失败、30 min 无日志输出、单组 8 h、全链 20 h。
6. 停止时向训练进程组发送 SIGTERM，45 s 后 SIGKILL，并执行 `ray stop --force`；不自动重启、不删除任何已有产物。
7. 全链完成条件：三组均存在 `global_step_50`（≥7.5 GB）、`adapters/step_50`、`latest_checkpointed_iteration.txt=50`、50 个 rollout dump、50×32 条 reset 审计。

看门狗自身由 `setsid` 脱离会话，并设置 `PR_SET_PDEATHSIG`：看门狗若被杀死，训练子进程会收到 SIGTERM。

## 首次启动中止记录（保留证据，不计入结果）

- 首次启动 2026-10-22T04:43:30Z，完成 r0 第 1 个全局更新（`critic/rewards/mean=0.375`、`grad_norm=0.162`、`timing_s/step=157.9`），随后发现看门狗审计边界检查会把训练器**预取的下一个 update** 误判为异常。
- 04:48:13Z 主动停止看门狗与 r0；未产生任何检查点，未进入第 2 步之后的训练。
- 部分产物原样保留在 `training/r0_aborted_20261022T044813Z_watchdog_audit_boundary/`；该次看门狗日志原样保留在 `watchdog/attempt_00_aborted_20261022T044813Z/`。
- 修正内容：`validate_audit` 只校验已完成的 updates `0..step-1`，允许预取中的 update 部分出现；不改变任何训练配置、数据或奖励。
- 04:48:37Z 从同一原始 SFT、同一冻结配置重新启动 r0，`training/r0` 为空目录，未复用中止运行的任何权重或状态。

## r0 运行中断记录（沙箱通道丢失，不计入结果）

- 05:29:27Z：r0 第 10 步检查点保存并通过看门狗核验（`save_observed.problems=[]`）：
  - `global_step_10` 8,422,827,512 B（7.844 GiB），`latest_checkpointed_iteration.txt=10`；
  - `adapters/step_10` 69,784,450 B，`archive_manifest.json` 已记录 `adapter_model.safetensors` 的 SHA-256；
  - 审计 updates 0..9 各 32 条 reset、32 个唯一 `(slot,sample)` 对。
- 第 1–17 步奖励正常波动（0.375、0.312、…、0.366、0.312、0.338），步时 145–243 s，工具调用数约 1.0/条。
- 05:55:13Z、05:55:49Z、05:56:24Z 连续 3 次沙箱探测超时（`TimeoutError`）；05:56:24Z 看门狗按规则停止 r0，05:56:29Z 全链停止（exit 3），未自动重启。
- 停止时训练进行到第 18 步（最后一行 `critic/rewards/mean=0.094`，已受沙箱不可用影响）；第 20 步检查点未生成。GPU 已回到 1 MiB，训练与 Ray 进程均已退出。
- 05:57Z 复核 `http://127.0.0.1:8080/v1/ping` 与 `/run_code` 均为 `Connection refused`，判断为 SSH 远程端口转发断开，而非训练侧故障。
- 结论：r0 第 1–18 步为不完整运行，不能用于对照；`training/r0` 原样保留（含 step 10 检查点）作为中断证据。待沙箱恢复后，将 `training/r0` 归档为 `r0_aborted_<UTC>_sandbox_lost`，从同一 SFT、同一冻结配置重新启动 r0；r1、r2 尚未开始。

## 沙箱恢复后重启记录（2026-10-22T06:39:45Z）

- 06:37Z 复核端口 8080：`/v1/ping` = `pong`，`print(6*7)` 返回 `42`；4 路并发执行全部成功，通道恢复。
- 06:39Z 将中断的 `training/r0` 原样归档为 `training/r0_aborted_20261022T055629Z_sandbox_lost`（8.0 GB，含 `global_step_10` 全量检查点），新建空 `training/r0`；上次看门狗文件归档到 `watchdog/attempt_01_r0_sandbox_lost_20261022T055629Z/`。
- 06:39:45Z 重新启动看门狗，`chain_id=20261022T063945Z`，看门狗 PID 61534，r0 子进程 PID 61551；从同一 SFT、同一冻结配置、空输出目录开始，未复用中止运行的任何权重或状态。
- 启动前核对通过：冻结哈希一致、GPU 1 MiB / 0% / 37 °C、磁盘 27.974 GiB、沙箱 `pong+42`。
- 累计消耗：首次中止约 0.08 h + 第二次中断运行 1.13 h ≈ 1.2 GPU 小时（新链的 20 h 上限重新计时，累计仍受 20 h 总量约束）。
- 磁盘说明：保留中止 r0 的全量检查点使三组完成后的预计剩余由 11.35 GiB 降至约 4.4 GiB；仍高于 3 GiB 保留线，且 r2 每次保存前 `free + rotated` ≈ 12.26 GiB ≥ 11.01 GiB，训练可安全完成。若需为后续评估留出更多空间，建议在明确授权后仅删除中止目录内的 `checkpoints/global_step_10`，保留 adapter、日志、审计与 rollout dump。

## r0 第二次重启中断（看门狗 KeyError，不计入结果）

- 07:57:03Z 看门狗记录 `stopping_group: watchdog unwind`，07:57:08Z 记录 `watchdog_error: KeyError: 'free_gib'`；训练在 step 19→20 的检查点轮换期间被 SIGTERM 终止。
- 现场：`global_step_10/data.pt` 仅 1,947 B（旧检查点已被轮换删除，只剩骨架）；`global_step_20/actor/model_world_size_1_rank_0.pt` 6,392,133,611 B，明显小于完整检查点 8,422,827,514 B，且缺 `data.pt`，不能加载；没有 `adapters/step_20`。GPU 已回到 1 MiB，训练与 Ray 进程均已退出。
- 根因：`disk_check()` 在检测到两个 checkpoint 目录短暂共存（轮换窗口）时提前返回，dict 缺少 `free_gib`/`rotated_gib`；监控循环随后执行 `status.update(free_gib=...)` 触发 KeyError，看门狗异常退出并连带终止训练。该轮换窗口本身是后端 delete-before-save 的正常现象。
- 修复（仅 `scripts/experiments/watchdog_midterm_training.py`，不在冻结源码树内）：`disk_check()` 在任何分支都返回完整字段；短暂共存时以最大 checkpoint 作为 `rotated` 并标记 `rotation=true`，允许最多 `ROTATION_GRACE_SECONDS=300 s` 的轮换；超过 300 s 或出现 3 个以上目录才判定异常停止。修复已通过 `py_compile` 和轮换场景单元测试。
- 结论：该轮 r0 第 1–19 步无效；`training/r0` 已归档为 `r0_aborted_20261022T075708Z_watchdog_keyerror`。其中 6.39 GB 的不完整 step-20 文件无法使用，且占用空间使三组无法同时完成；保留 step_10 adapter、日志、审计、rollout dump 和目录清单，从同一 SFT 重启 r0。r1、r2 尚未开始。

## 清理与第三次重启（2026-10-22T08:15:40Z）

- 用户明确授权删除无效旧检查点与无关训练结果。本次仅删除两次中止运行的无效权重载荷，详见 `audit/cleanup_and_restart_20261022T075708Z.md`：
  - 截断的 `global_step_20/actor/model_world_size_1_rank_0.pt`（6,392,133,611 B，SHA-256 `0513…80ce`，缺 `data.pt`，不能加载）；
  - 沙箱中断运行的 `..._sandbox_lost/checkpoints/global_step_10`（8,422,827,512 B，模型分片 SHA-256 `90c7…5882`；其 adapter 归档、审计、日志、dump 全部保留）。
- 删除后可用空间 23,557,152,768 B → 38,372,147,200 B（35.74 GiB），满足三组峰值与 3 GiB 保留线。
- 看门狗补充加固：沙箱返回非对象/空结果时安全返回 False；输出目录不存在时自动创建后再做空目录检查；每组正常退出后执行 `ray stop --force`。此前修复（全字段 `disk_check`、300 s 轮换宽限、骨架目录过滤、审计尾行容忍、adapter 清单容错、日志跨读取缓冲）全部保留。
- 08:15:27Z `--check-only` 通过：冻结哈希一致（配置 8、组 3）、GPU 1 MiB / 0% / 37 °C、沙箱 `pong+42`、磁盘 35.737 GiB、`r0|r1|r2` 为空；退出码 0。
- 08:15:40Z 第三次启动，`chain_id=20261022T081540Z`，看门狗 PID 70874，r0 子进程 PID 70890；从同一原始 SFT、同一冻结配置、空输出目录开始，未复用任何中止运行的权重或状态。
- 重点监控：step 10 与 step 20 保存必须出现完整 `global_step_N` 与 `adapters/step_NN`，且看门狗不得因轮换骨架误停。

## r0 后停机计划（用户确认，2026-10-22T09:45Z）

- 用户计划：r0 完整跑完后终止训练并重启电脑，之后回来再从同一 SFT 启动 r1、r2。
- 必须满足：
  1. r0 跑到 step 50 且通过 `verify_group_complete` 后才可停机/重启；提前重启会使 r0 作废。
  2. r1 仍从同一原始 SFT 独立初始化，不得从 r0 权重续训；r0 检查点只作为该组结果保留。
  3. 重启后必须保持与 r0 相同的沙箱后端和执行条件。若从 Mac 沙箱换成云端沙箱，r0 与 r1/r2 不可直接比较，需三组全部在云端重跑。
- 已启动 stop-after-r0 监控：`scripts/experiments/stop_after_group.py` 监视 `group_complete`（r0），事件出现后 2 s 向看门狗 PID 发送 SIGTERM；看门狗按自身信号处理顺序停止，r0 已核验产物不受影响。若 r1 已被自动启动，重启后将其部分目录归档并新建空 `training/r1`。
- 已给看门狗增加 `--groups` 参数（默认 `r0,r1,r2`，必须按规范顺序）；重启后可用 `WATCH_GROUPS=r1,r2 bash scripts/experiments/launch_midterm_formal.sh` 只跑 r1、r2。改动均在非冻结的 `scripts/experiments`，`verify_frozen()` 返回 `[]`。
- 当前 r0 到 step 21，沙箱正常。

## r1、r2 重启（2026-10-22T13:24:23Z）

- r0 已于 12:38:53Z 完成并核验，链在 r1 启动前按计划停止；r1 的 3 秒空跑目录已归档为 `training/r1_aborted_20261022T123856Z_stopped_after_r0`。
- 13:23:37Z 首次 r1/r2 预检失败：`localhost:8080` 连接被拒（SSH 隧道刚重建）。等待后 13:24:13Z 复查通过：`/v1/ping=pong`、`print(6*7)=42`；当前 8080 监听者为 `sshd`（PID 96031），与 r0 使用的是同一条 SSH 隧道/同一沙箱后端。
- 13:24:23Z 启动新链 `chain_id=20261022T132423Z`，看门狗 PID 96106，r1 子进程 PID 96122；启动参数 `WATCH_GROUPS=r1,r2`，只跑 r1、r2，不重跑 r0。
- r1 的 `resolved_config.yaml` 核对：`model.path=/root/autodl-tmp/Adaptive-Solver-main-git/artifacts/models/qwen3_1_7b_sft_tir_merged`、`resume_mode=disable`、`resume_from_path=null`、`output_dir=.../training/r1`、`experiment_name=midterm_unified_r1`。确认从同一原始 SFT 初始化，未使用 r0 权重。
- r1 step 1 完成：`rewards/mean=0.375`、`grad_norm=0.150`、`timing_s/step=161.0 s`；训练健康。r1 结束后看门狗会自动启动 r2。
- **当前 8080 仍为 SSH 隧道，训练期间不得把 8080 翻到云端或另一后端**；否则 r2 与 r0/r1 的执行条件不一致，会破坏对照。若必须换后端，三组需全部重跑。

## 三组训练完成（2026-10-22T19:34:56Z）

- r0：08:15:45Z 开始，12:38:53Z 完成；r1：13:24:27Z 开始，16:33:08Z 完成；r2：16:33:09Z 开始，19:34:56Z 完成。
- 19:34:56Z 看门狗写入 `watchdog/CHAIN_COMPLETE.json`（`chain_id=20261022T132423Z`，`groups=[r1,r2]`）；r0 由前一链独立核验。19:39Z 对三组重新调用 `verify_group_complete`，均返回 `[]`。
- 三组最终产物一致：`checkpoints/global_step_50` 8,422,823,292 B、`latest_checkpointed_iteration.txt=50`、50 个 rollout dump、50×32 条 reset 审计；`adapters/step_50` 归档 69,784,450 B。
- 最终 adapter SHA-256：r0 `ef904bf700c116d714f1f6f987de6005c2818f10e921cf96965076070ec895c0`；r1 `e6b8914e163e59b9e68271b174eb1fc47bb12eff74ad18a28493c71aa1d07dcf`；r2 `12744a0aede8feeb102913dea143a6d760ce4adbfd8de744a7eadb482a7e01e3`。
- 训练结束后 GPU 1 MiB / 0%，无 `run_grpo`、看门狗或 Ray 训练进程；磁盘可用 11.094 GiB。
- 有效正式运行步时均值：r0 314.4 s、r1 225.0 s、r2 216.9 s；合计约 10.5 GPU 小时。加上已中止尝试约 1.2 GPU 小时，累计约 11.7 GPU 小时（仍低于 20 h 上限；若按更保守口径计约 13 GPU 小时）。
- 训练侧诊断（只来自训练 rollout，不是测试准确率）：奖励均值 r0 0.2826 / r1 0.2953 / r2 0.2761；成功率 r0 29.8% / r1 31.1% / r2 29.9%；每题工具调用约 0.98–1.00；响应截断率 0.55%–1.18%。完整逐 10 步统计见 `audit/training_metrics_summary_20261023.json`。

## 统一评测：首次运行暂停与修复后重跑

- 19:40:00Z 启动首次正式统一评测，输出目录 `evaluation/formal_unified_tool_eval_20261023`，组顺序 SFT → R0 → R2 → R1，沙箱 `http://127.0.0.1:8080`。
- 19:48:20Z 在 SFT 第 60 题 `math:45e5f7656e796fc9615c` 暂停：验证器 `internal_error`（`comparator raised TypeError`），重试一次仍复现；前 59 条 SFT 轨迹有效。该行与暂停标记原样保留，未混入后续结果。
- 根因：`x \in (0,36)` 被 latex2sympy 解析为 `Eq(x, Interval.open(0,36))`，旧 `_relation_signature` 无条件做 `Symbol - Interval`，抛出 `TypeError`；元组等式 `(p,q)=(1,1)` 同样会失败。详见 `audit/verifier_bug_20261023/README.md`。
- 仅修改 `src/adaptive_math/verifier/symbolic.py` 的结构比较路径（及对应回归测试）；未改训练代码、数据、奖励配置、模型或评测协议。修复前后源文件 SHA-256 为 `f4f21ac3…2194` → `1de77561…fd7`，补丁与备份保存在 `audit/verifier_bug_20261023/`。
- 20:00Z 从头启动新评测 `evaluation/formal_unified_tool_eval_20261023_v2`，四组全部使用修复后的同一验证器；旧目录仅作证据，不恢复、不拼接。`verifier_patch.json` 随新目录保存补丁说明。
- 训练侧审计同时发现 9 条 `internal_error`（r0 2、r1 4、r2 3），其中 2 条在修复后应判正确（均在 r1）。训练不回溯重跑；该训练期验证器假阴性作为限制写入正式报告。
- v2 在 20:00:13Z 于 SFT 第 5 题 `math:045a84b20397b11ecc6a` 遇到一次沙箱不可用（`sandbox_unavailable_calls=1`，工具调用未执行、未计非法动作）；原行保留为无效，20:00:4x 三次 `pong+42` 探针通过后以 `--resume` 重跑该题，20:01 该题在新的有效行中判对，随后继续。该事件记为基础设施重试，不挑选两次结果，分析只取最新有效行。
- 20:18:39Z 在 SFT 第 184 题 `math:dd48d65e326606be4a9c` 再次遇到同类沙箱瞬断（`sandbox_unavailable_calls=1`，工具调用未执行，原行保留为无效，评测以 exit 3 暂停）。20:23:2xZ 三次 `/v1/ping` + `/run_code` 探针全部返回 `pong+42`，后端未更换；随后以 `--resume` 在同一 v2 目录恢复，SFT 完成 200 题（112 正确，56.0%）。两次瞬断均未更换后端、未修改协议、未选择性删除无效行。
- 22:00:53Z 在 R1 第 38 题 `math:2cd7311ca81f38a94922` 第三次遇到同类沙箱瞬断（`sandbox_unavailable_calls=1`，工具调用未执行，原行无效，评测暂停）。22:03Z 三次探针恢复 `pong+42` 后，在同一 v2 目录再次 `--resume`。三次瞬断分别发生在 SFT 第 5、184 题和 R1 第 38 题；均由同一后端恢复，失败行不计入分析，不构成组间差异。评测期间累计 3 次基础设施暂停，已全部记录。
- 22:20:34Z 在 R1 第 122 题 `math:9907725e8fc559178031` 第四次遇到沙箱瞬断（`sandbox_unavailable_calls=1`，原行无效，评测暂停）；22:23Z 三次探针恢复后在同一目录 `--resume`。四次暂停均未更换后端、未修改协议、未删除任何行。
- 另需记录：R1 第 34–36 题（`math:26fd91ac7be838ec3f8c`、`math:2892076449fc22ef1afd`、`math:2c584c5199712c434c39`）在 21:57–21:59Z 连续出现 3 次 30 秒工具超时（`error_code=timeout`、`timing_missing=true`，每次计 31.0 秒），随后 22:00:53Z 沙箱完全失联。三次超时执行的代码均为秒级 SymPy 表达式，判断为沙箱侧停滞而非模型计算成本；评测器按既定规则将超时记为有效轨迹（仅 `sandbox_unavailable_calls` 判无效），因此这 3 条轨迹保留在主表口径内，报告另做敏感性分析并明确标注。
- 22:25:47Z 在 R1 第 140 题 `math:abc81a277e4c2a5015cd` 第五次沙箱瞬断；22:28Z 探针恢复后再次 `--resume`。检查发现 8080 监听者已由 r0/r1 启动时的 sshd PID 96031 变为 sshd PID 135060，说明是本机到沙箱的 SSH 转发通道重建，而不是训练/评测代码或模型侧故障。五次暂停均发生在通道重建窗口，探针全部通过；后端地址和端口未变。

## 最终分析与报告（2026-10-23）

- v2 评测于 2026-10-22T22:42:22Z 完成，`evaluation_summary.json` 记录 `all_complete=true`、`full_run=true`；四组各 200 条有效轨迹，组顺序 `sft,r0,r2,r1`。
- 离线分析于 22:43:43Z 完成，产物在 `evaluation/formal_unified_tool_eval_20261023_v2/analysis/`：`analysis.json`、`diagnostics.json`、`main_table.csv/md`、`paired_comparisons.csv/md`、`reward_recompute_20.json`、`analysis_report.md`。
- 主表正确率：SFT 56.00%、R0 53.00%、R1 51.50%、R2 54.00%；三个预设比较 R0−SFT、R1−R0、R2−R1 的 Holm 校正 p 值为 0.328125、0.548828、0.453125，均不显著。
- 完整报告：`report/midterm_uniform_reward_v1_report.md`；报告校验和：`report/SHA256SUMS`；实验索引：`README.md`。
- 本报告完成后的状态是**全部训练、评测与分析结束**；不再启动新的训练或评测运行。验证器补丁、五次沙箱暂停、R1 三次超时和训练期 9 条 `internal_error` 已在报告中明确披露。
