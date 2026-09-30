# 简历口径（2026-09-29 复核版）

本文把"能写进简历的话"逐条对上工件，每一条都能用仓库里的文档与
`artifacts/` 下的 JSON 复核（个别文档/脚本尚未提交，已在对应条目标注）。本轮
复核没有新增任何 GPU 跑批；§1.1 中 SFT / r0 两个 agent 臂来自 2026-09-27
campaign（§4），其余数字都来自已存下来的生成结果与轨迹。

三条硬约束，先写在前面：

1. **严格协议是严格协议，宽容标尺是标尺。** 两者都报，但分开报。只报严格版会
   低估方法，只报宽容版会篡改协议。
2. **不显著就是不显著。** r0 在宽松标尺下与 base 的差异 p=0.21；r0/r2 与 SFT 的
   差异从未显著。这些必须留在"不能写"清单里。
3. **每个数字都指向一个能重跑的脚本。** 数字与工件 sha256 一一对应。

---

## 1. 可以写的四条

### 1.1 主线：冻结 Omni-MATH-200 上的严格准确率阶梯（直接 + Agent Runtime）

> 在冻结的 Omni-MATH-200（200 题，严格端到端协议）下：Base 直接作答 0.0%（0/200）、
> Base 接入多轮 Agent Runtime 6.5%（13/200）、SFT 直接作答 13.5%（27/200）、
> SFT 接入 Agent Runtime 14.0%（28/200）；相对 Base 的配对提升 +6.5 / +13.5 个
> 百分点（精确 McNemar p = 2.4×10⁻⁴ / 1.5×10⁻⁸），SFT+Runtime 较 Base+Runtime 再
> +7.5 个百分点（95% CI [+3.0, +12.5]，p = 0.0041，预注册 H1）；首轮 GRPO
> （纯结果 / 工具加权 / r0+Runtime）11.0% / 12.5% / 13.5%，均未显著超过 SFT。

证据：`docs/results/mvp-20260929-arm-table.md` §1、§3；
`docs/results/campaign-2026-09-27/stage-summary.md` §2–§3（尚未提交）；
`artifacts/results/mvp-20260929/arm-metrics.json`
（sha256 `f2290fae2c2eaff2a69bac7e8765fa639601c4da0a4ac6525d787c2ac091f871`）、
`artifacts/results/campaign-2026-09-27/arm-metrics.json`
（sha256 `a781206565b102e13f46279ee4e0ce52ef8fa76c4d3e5453103c959665ebc403`）、
`artifacts/results/campaign-2026-09-27/paired-agent-arms.json`
（sha256 `a9dcc730693825d2d2e4b07cd5685a5dfb38950b3a789710bba397b06db9d6dd`）。

### 1.2 增量（本轮新增，最值得替换旧表述的一条）：6.5% 有一半是协议造成的

> Agent Runtime 的 6.5% 中，22 道题的唯一缺陷是把 JSON 的 `answer` 写成了数值
> 而非字符串（如 `<final>{"answer":71}</final>`），被动作 schema 以
> `invalid_schema` 拒绝；按字面数字做机械修复（不引入任何新信息、不做搜索）后
> 为 35/200 = 17.5%，配对提升 +17.5 个百分点（p=5.8×10⁻¹¹）。对照组说明这不是
> 通用现象：三个直接作答臂在同一修复下均提升 0 行。

证据：`docs/results/mvp-20260929-arm-table.md` §6；
`artifacts/results/mvp-20260929/tolerance-rescore.json`
（sha256 `891ffb80befae03294380ce60b36779951c50e2baeb94af4799afdd82872b002`）、
`artifacts/results/mvp-20260929/paired-protocols.json`
（sha256 `7f2e8ba2e5d51b99a03b340527777f6a4b5ed2c9b08c984efea39314a84a5481`）；
脚本 `scripts/analysis/tolerance_rescore.py`、`scripts/analysis/paired_protocols.py`。

**为什么这条比 6.5% 强**：它把"方法有效"和"评测协议压缩了结果"两件事同时说清，
并且给了一个不可能造假的对照（直接作答臂提升 0 行）。

### 1.3 协议缺口：不是不会用工具，是信封丢掉了

> 复解析存储轨迹显示，Base 的 agent 臂共产生 422 次工具调用尝试，严格信封只执行
> 了 7 次；其中 393 次距可执行只差一个机械修复，193/200 题至少有这样一次尝试。

证据：`docs/results/protocol-gap-2026-09-29.md` §1–§4；
`artifacts/results/mvp-20260929/protocol-gap.json`
（sha256 `90c5d2dfd0fa7306f5969539f6875c00fe43fd97dd3c9392d2018ec28758262c`）。

可与 1.2 合并成一句：**"模型在格式上失败，而不是在数学上失败；本轮把这件事量化
到了行。"**

### 1.4 可复现性（必须用修正后的这句）

> 同一生成配置下重跑，200/200 条输出逐字节一致；跨 batch 配置则只有 3/200 条
> 逐字节一致、130/200 条判定一致，因此所有数字都取自单一冻结配置。

证据：`docs/results/mvp-20260929-arm-table.md` §4。
**不要**写"隔周复现 200/200 字节一致"——那是同配置跨两天，不是跨配置。

---

## 2. 不能写的

| 说法 | 为什么不成立 |
|---|---|
| r0 / r2（GRPO）超过 SFT | 11.0% / 12.5% vs 13.5%，从未显著；r0 在宽松标尺下与 base 的差异 p=0.21（15 升 8 降） |
| "RL 介于两者之间" | 同上，差异是几道题的噪声 |
| "agent 体现了工具执行带来的收益" | 严格信封只执行了 7 次调用；收益来自多轮循环与格式，不是执行 |
| "隔周 200/200 字节一致"（不加限定） | 见 1.4，跨配置只有 3/200 |
| "SFT 学会了用 sympy 工具" | sympy 示范的产出率门禁未通过（32/200），三个臂都在用没训练过的工具；`docs/results/sympy-demo-yield-gate.md` |
| r2 是成本感知奖励 | `configs/reward/r2.yaml` 是工具加权 shaping；仓库自己的标签是 "tool-weighted" |

---

## 3. 严格性交代（面试被追问时的答案）

* **同一套标尺**：解析器、提取器、提示词三处哈希在所有已存臂上逐字节一致
  （`extractor.py` = `029570bd…`，`prompts.py` = `9db23ccf…`）。
* **宽松标尺会翻转排序，必须并列披露**：lenient 重打分下 `base_agent` 40/200 >
  `sft_agent` 28/200（−6.0 pt，p = 0.0227；coerce 下 35 vs 28，p = 0.248 不显著）。
  因此所有结论都要写明"严格口径"；宽松行是对已记录文本的重打分上限，不是臂应得
  的分数（`docs/results/campaign-2026-09-27/stage-summary.md` §3 H4）。
* **验证器有一处漂移，已披露**：r0/r2 的判定来自 git `b4ca1604`
  （`verifier_source_sha256 b690cc7d…`），此后有三个提交改动过
  `verifier/service.py`，HEAD 为 `3cfc5e61…`。
  重跑复核发现 r0 4 行、r2 1 行判定不可复现，全部发生在两个"非正确"状态之间
  （`incorrect` vs `invalid_prediction`），**没有任何一行跨越 correct 边界**，因此
  不影响任何准确率；若有一行跨越，复核脚本会直接拒绝为该臂打分。
  详见 `docs/results/mvp-20260929-arm-table.md` §6.4。
* **身份门禁**：复核脚本先校验任务集 sha256、拒绝重复题号、要求严格档复现全部已记录
  状态；轨迹臂额外用运行时自己的解析器复读最后一轮，必须与轨迹记录的最终答案一致
  （324 条全部一致），否则拒绝打分。
* **所有复核对齐都在 CPU 上完成，未开卡。**

---

## 4. 空格子已补齐（2026-09-27 campaign）

原表唯一空着的格子——**SFT / r0 在 agent 模式下的收尾与工具列**——已填上
（预注册 §8b；两臂 × 200 题，单卡 1 h 32 m，在 2–3 GPU h 预算内）：

* **假设得到回答：SFT 确实缩小了协议缺口。** 合法收尾率 `base_agent` 13.5% →
  `sft_agent` 79.5% / `r0_agent` 94.0%。
* **但不是靠工具执行。** 严格信封的执行调用 7 → 0，尝试 422 → 13 / 11
  （有可执行尝试的题 193/200 → 1 / 3）；工具通道在循环里保持静默。H2 的工具半边
  按预注册判定为不成立。
* **步数同时下降**（5.76 → 2.17 / 1.51）：新臂作答，而不是打转。

证据：`docs/results/campaign-2026-09-27/stage-summary.md` §2–§3（尚未提交）、
`artifacts/results/campaign-2026-09-27/protocol-gap.json`
（sha256 `399d2ebc73b84d4283dff2850aae93e7a07ba18f5a372557f1b6f0b17bb88cf2`）。
原"先开这一枪"的建议已执行完毕；本表不再留需要开卡的格子。
