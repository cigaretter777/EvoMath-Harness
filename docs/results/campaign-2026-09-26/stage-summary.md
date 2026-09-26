# 阶段性实验报告 — E0/E1 无训练基线战役(2026-09-26)

> 阶段目标:BC-GRPO 论文方案的第一个 GPU 阶段——在 RL 之前建立四条基线
> (Base 直接 / Base+工具 / 固定规则策略 / SFT),回答"工具本身是否有效、
> SFT 带来多少收益",并给 RL 阶段立正确的对比基准。
> 配套文档:`preregistration.md`(预注册+addendum A)、`report.md`(结果细节)、
> 本报告(过程叙事)。证据目录:`artifacts/results/campaign-20260926/`。

---

## 一、做了什么

### 1.1 离线准备(纯 CPU,零 GPU 成本)
1. **relabel audit**(`scripts/campaign-20260925/relabel_audit.py`):用修复后的
   verifier 重新打分全部存量评测臂(09-19 四线 + E2 双臂)。结果:每臂仅 4 个
   `invalid_prediction → incorrect`,**correct 数零变化**(27/22/25 全不变)。
   结论:09-24 的 verifier 修复没有改变此前任何战役的结论,历史数字可直接沿用。
2. **固定规则路由器设计**:确定性文本规则(方程标记→SymPy;数论/组合/大数标记→
   Python;否则直接回答),在 Omni-MATH 200 上分布 direct 76 / python 66 / sympy 58。
3. **预注册**:四臂定义、路由器规则、端点、配对检验方案、字节一致性交叉检查,
   全部先于结果提交(commit `6638762`)。

### 1.2 GPU 战役(单卡 4090,三槽串行队列 `thesis_e0_queue.sh`)
| 槽 | 臂 | 内容 | 结果 |
|---|---|---|---|
| 1 | E0a/E1 | model_eval 重跑 base+sft 直接评测(batch 1, greedy, cap 1024) | base 0/200(全 invalid)、sft 27/200 |
| 2 | E0b | Base + 双工具 agent 轨迹(200 题) | 13/200,工具调用 7 次/200 条 |
| 3 | E0c | 固定规则策略(124 题走工具通道,76 题复用 base 直接) | 合成 10/200 |

### 1.3 结果分析(`analyze_e0_baselines.py`)
- **字节一致性交叉检查:通过**——slot 1 重生成的 SFT 臂与 09-24 E2 champion
  臂 200/200 字节一致,pipeline 无漂移。
- 四个预注册配对比较(McNemar exact + 配对 bootstrap, seed 20260913):

| 比较 | Δ | p | CI95 |
|---|---|---|---|
| Base+工具 vs Base 直接 | +6.5% | 0.0002 | [3.5%, 10.0%] |
| 固定规则 vs Base 直接 | +5.0% | 0.002 | [2.0%, 8.0%] |
| 固定规则 vs Base+工具 | −1.5% | 0.58 | [−5.0%, 2.0%] |
| SFT vs Base 直接 | +13.5% | 1.5e-8 | [9.0%, 18.5%] |

---

## 二、遇到的问题与解决

### 问题 1:沙箱隧道中断(~2.5 小时)——无 GPU 浪费,自动恢复
**现象**:slot 1 完成后,沙箱探针(`127.0.0.1:8080`)开始拒连。沙箱是外部对端
经 SSH 反向隧道注入容器的,容器内没有任何进程能重建它。
**解决**:队列的 `wait_sandbox` 每 60s 重试探测,隧道由对端自动重建后队列
无缝续跑。代价只有墙钟时间,GPU 全程没有空转。

### 问题 2:GPU 利用率只有 39% —— 分片并行
**现象**:agent 轨迹是串行的(每题下一步依赖上一步),batch-1 greedy 的小模型
前向吃不饱 4090(利用率 39%、功率 155W/450W)。
**解决**:任务间相互独立,给 `rule_baseline.py` 加 `--shard-id/--shard-count`,
slot 2/3 改为 3 个并发进程各跑 1/3 任务,完成后由 `merge_rule_shards.py` 合并。
利用率到 100%,速度约 2 倍;生成路径完全不变(同 client、同 greedy、同沙箱),
结果与串行等价。剩余功率/显存空余是 1.7B 模型 batch-1 推理的固有特性
(计算已饱和),不是可继续榨取的浪费。

### 问题 3:任务集错配(本次最大问题,浪费 ~3.5 GPU 小时)
**现象**:分析阶段发现配对为空——直接臂评测在 `frozen_eval.parquet`
(task_id 形如 `omni_math:<hash>`),而 agent 臂跑在 RL 训练池 `rl_r0_200.jsonl`
(task_id 形如 `openr1_math_220k:<hash>`)。两批 200 题**零交集**
(task_id / source_hash / 题干文本三向全不重合),逐题配对在数学上不可能。
**根因**:预注册错误声称"两者是同一批题",而我开卡前没有用 CPU 验证 task_id
集合——仓库的配对纪律(manifest 哈希)正是为防这类事故,我跳过了它。
**解决**:
1. 预注册 addendum A 撤回错误声明,记录修正方案;
2. `rule_baseline.py` 增加 `--data/--task-ids-file`:agent 臂改为从
   frozen_eval.parquet 精确选取存量直接臂的 200 个 task_id(顺序一致);
3. 两个 agent 臂重跑在正确题集上;错池产物归档为 `*_openr1_pool`
   (RL 训练池上的工具行为诊断数据,不算配对战役的一部分,不浪费);
4. **机械防复发**(用户明确要求"不允许这种资源浪费再出现"):
   - `verify_eval_alignment.sh` 门禁:纯 CPU 数秒,三向集合对账
     (task-ids 文件 ↔ 存量臂预测 ↔ parquet),任一不一致立即失败;
   - 门禁硬接入队列启动路径(不通过即 `FAILED` 退出,不花 GPU);
   - 监控自动重启路径同样先过门禁;
   - 写入长期记忆:一切 GPU 运行开卡前必须 CPU 验证身份对齐
     (任务集/sha/解码),凭假设开跑是红线。

### 问题 4:队列进程管理(容器重启、误杀)
**现象**:容器重启会杀掉所有前台进程;一次 `pkill -f` 因模式匹配到自己的
shell 命令行而自杀;旧队列实例在 kill 后仍残留一次 relaunch。
**解决**:队列设计为幂等——完成探测(COMPLETE/summary.json)、partial 目录
归档后重跑、有界重试(4 次)、基于日志生长的卡死检测(不只看进程存活)、
status.json 心跳;进程清理改用 PID 定向 kill,不再用模式匹配。

---

## 三、实验产出

**代码(全部已提交,当前分支 `fix/20260926-parser-tolerance-verifier-timeouts`):**
- `scripts/campaign-20260925/relabel_audit.py` — verifier 修复后的全臂重打分
- `scripts/campaign-20260926/rule_baseline.py` — 固定规则路由器 + 双模式
  (rule / all-tools)+ 分片 + parquet 任务选择
- `scripts/campaign-20260926/merge_rule_shards.py` — 分片证据合并
- `scripts/campaign-20260926/analyze_e0_baselines.py` — 四臂统计 + 配对检验 +
  字节一致性检查
- `scripts/campaign-20260926/thesis_e0_queue.sh` — 三槽自愈队列(幂等、分片、
  门禁、沙箱探针、卡死检测)
- `scripts/campaign-20260926/verify_eval_alignment.sh` — 开卡前身份对齐门禁

**文档:**
- `docs/results/campaign-2026-09-26/preregistration.md`(+ addendum A)
- `docs/results/campaign-2026-09-26/report.md`(结果细节)
- 本报告

**数据(artifacts,不入 git):**
- `artifacts/eval/thesis_e0_base_direct_b1/` — E0a/E1 双臂预测 + manifest + task_ids.txt
- `artifacts/rollout_health/thesis_e0_base_tool/` — E0b 轨迹(200 条,含分片证据)
- `artifacts/rollout_health/thesis_e0_rule_strategy/` — E0c 轨迹 + 路由表
- `artifacts/results/campaign-20260926/summary.json` — 全部分析数字
- `artifacts/audit/relabel-20260925/summary.json` — relabel 结果
- `artifacts/rollout_health/thesis_e0_{base_tool,rule_strategy}_openr1_pool/` —
  错池运行归档(训练池诊断用)

**提交记录:** `6638762`(基线战役)→ `0622277`(分析脚本)→ `383d7cd`(分片)→
`fdf9276`(任务集修正)→ `db54b1a`(对齐门禁)→ `9d0fba1`(结果+门禁接入)

**核心结论(给 RL 阶段的三条基线事实):**
1. SFT 是准确率与协议遵从的基本盘(27/200);Base 在直接口径下为零,但抽查显示
   它"会推理、不会交卷"(全部 1024 截断、无信封;34/200 文本中曾出现答案串)。
2. 工具通道在当前策略下几乎未被使用(E0b 仅 7 次调用/200 条)——E0b/E0c 的
   显著增益归因于多轮循环格式而非工具计算,报告如实声明。
3. RL 从 SFT adapter 起步;smoke 第一信号是"工具调用是否出现",第一防线是
   协议遵从不退化(F3 奖励反转已修复)。

---

## 四、GPU 消耗账目(诚实记账)

| 用途 | 时长 |
|---|---|
| slot 1 base/sft 直接(正确) | ~2.5h |
| slot 2/3 错池运行(浪费) | ~3.5h |
| slot 2/3 正确题集重跑(3 分片) | ~2.7h |
| **合计** | **~8.7h** |

浪费部分已通过身份对齐门禁机械堵死,后续战役不再承担此类成本。

## 五、下一步

1. BC-GRPO 算法实现(纯 CPU):dual-λ 预算更新、双相对优势(合并/分别标准化
   两变体)、预算状态入上下文;
2. RL smoke 夜(短程 1-2h × 3 方法:标准 GRPO / 固定成本 / BC-GRPO),预注册先行;
3. smoke 通过 → 正式训练 → 消融 → 预算水平 → 冻结评测 → 统计分析。
