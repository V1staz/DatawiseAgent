# 合理预期结果与当前差距分析

更新时间：2026-06-09

## 核心主线

本项目不是简单给 DatawiseAgent 加 prompt / harness / verifier，而是延续原文最重要的思想：弱模型可以通过 notebook-style 多轮规划、执行、调试和反馈，用更多结构化交互时间弥补模型能力差距。我们的进一步问题是：

> 时间应该花在哪里、由谁来花、什么时候停止？

因此当前故事分为三层：

1. **任务适配**：不同 benchmark 的失败模式不同。InfiAgentBench 需要 answer-aware harness；DataModeling 需要 protocol-aware harness。
2. **时间换性能**：额外时间必须转化为结构化检查和修正，而不是无脑多跑几轮。
3. **预算控制与模型分工**：light / full harness、sub-agent、verifier-gated fallback 用于控制推理预算。

## 当前已观察结果

### InfiAgentBench weak model small set

实验对象：`qwen3.5-flash`，47 道 `small_error_set`，temperature=0。

| setting | ABQ | PASQ | UASQ | hard ABQ | avg runtime | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline raw | 0.00% | 0.00% | 0.00% | 0.00% | 30.24s | raw 长日志无闭式答案 |
| baseline + original reformat | 48.94% | 63.83% | 67.05% | 44.00% | 30.24s | 47/47 |
| harness_light | 59.57% | 65.25% | 69.32% | 60.00% | 47.03s | 47/47 |
| harness_full | 63.83% | 69.50% | 72.73% | 64.00% | 92.79s | 47/47 dedup |

已观察结论：

- light harness 明显提升，且成本可控。
- full harness 分数更高，但 runtime 明显增加，并出现 verifier blocked。
- task 733 的 FinalAnswerBlock JSON 进入 Python code cell 问题已修复。
- 当前只跑了 known-error small set，不能外推为 full 257 结论。

### 强模型与已有参考

已有 InfiAgentBench 强模型参考：

- `qwen3.5-35b-a3b` full baseline：ABQ 86.38%，PASQ 90.16%，UASQ 90.35%。
- 强模型 medium harness：ABQ 88.51% -> 91.95%。
- 强模型 hard harness：ABQ 81.82% -> 78.41%，说明 hard 题 full harness 可能因分支选择、口径漂移或 verifier 过度干预退化。

### DataModeling 初步观察

已有对比：

| run | complete | performance | avg time |
| --- | ---: | ---: | ---: |
| qwen25 | 68/74 | 0.4290 | 760.25s |
| qwen35b-a3b-full | 73/74 | 0.5318 | 288.60s |

可用于故事线的观察：

- 强 / thinking 风格模型并不一定更慢，少走弯路可能显著降低总时间。
- DataModeling 的主要改进方向不是 answer format，而是 protocol correctness：submission 文件、列名、行数、metric、performance parser、评估接口。

## 合理预期目标

以下都是 expected / 待验证，不是已完成实验结论。

| 档位 | InfiAgentBench full 257 预期 | DataModeling 预期 | 解释 |
| --- | --- | --- | --- |
| Conservative | weak + light 在 full 257 上稳定优于 weak baseline，ABQ 提升约 2--4 个百分点 | protocol checker 减少无效 submission / performance 缺失 | 证明方向有效，但收益主要来自格式和协议修复 |
| Target | weak + light 提升约 5--8 个百分点，hard 提升更明显；full 进一步提升但 runtime 高 | complete rate 明显提升，部分任务 performance 小幅上升 | 支撑“结构化时间换性能”主结论 |
| Optimistic | weak + full 或 gated harness 提升约 8--12 个百分点，并缩小 weak-to-strong gap | protocol-aware harness 将错误从 silent failure 转为可恢复失败 | 支撑后续 sub-agent / verifier-gated routing |

## 当前差距

| 目标项 | 当前状态 | 证据 | 差距 | 下一步 |
| --- | --- | --- | --- | --- |
| full 257 weak baseline vs harness_light | 未完成 | 只有 47 题 small set | 不能证明 full 分布成立 | 先跑 qwen3.5-flash full 257 baseline / light |
| harness_full 全量收益 | 未完成 | small set full 最高但 92.79s/题 | 成本高，可能 hard 退化 | light 稳定后再跑 full |
| 最小消融 | 未完成 | no_finalizer / no_skills / no_oracle_or_verifier 尚未跑 | 不能归因各模块贡献 | 只跑 47 题 small set 消融 |
| weak-to-strong gap | 待验证 | strong full baseline 已有，weak full 未跑 | small set gap 不能外推 | 生成 full gap 表 |
| DataModeling protocol-aware harness | 初步观察 | qwen35b 比 qwen25 更高更快；有 tweet sentiment 协议失败案例 | 缺系统 protocol failure 统计 | 做 failure taxonomy + case study |
| sub-agent / 难度路由 | 未完成 | 目前只有 light/full 两档 | 还没有模型分工实验 | 先做离线设计，再跑 gated fallback smoke |

## 主要困难与应对

| 困难 | 现象 | 风险 | 应对策略 | 是否影响 PPT 结论 |
| --- | --- | --- | --- | --- |
| full harness runtime 高 | small set full 92.79s/题 | full 257 成本和时间高 | 优先 light；full 只用于 hard / verifier fail | 不影响，反而支撑预算控制 |
| verifier 过严 | full 有 4 个 verifier_blocked | 答案可解析但被阻断 | 区分 hard fail / soft warning | 不影响趋势，但影响系统成熟度 |
| hard 多分支选错 | strong hard harness 曾退化 | 多想不等于选对 | 加独立复算 verifier 和 branch selector | 支撑“时间要花对地方” |
| final block 进入 code cell | task 733 曾卡死 | 无限 debug / execution error | 已加 code-cell guard 和 timeout | 已解决，可作为迭代案例 |
| weak model 统计语义仍错 | task 124 仍失败 | light/finalizer 无法修深层语义 | 针对统计口径做 oracle / verifier | 不影响 light 的格式收益 |
| DataModeling 慢且协议复杂 | submission / metric / performance 易错 | 运行成本高，失败隐蔽 | 先做 protocol smoke 和 case study | 需要谨慎表述 |
| qwen2.5 API 不可用 | 后续是 model-substituted setting | 不能严格复现原论文模型 | 明确使用 qwen3.5 系列替代 | 不影响方法故事 |
| small set 偏差 | known-error set 不代表 full | 可能高估或低估收益 | 必须补 full 257 | 当前结论只能说趋势 |
| memory 噪声 | round2 memory 收益不稳 | 负迁移 | memory gated retrieval | 不影响主线，作为后续 |

## 下一步优先级

1. 跑 `qwen3.5-flash` full 257：baseline + original reformat vs harness_light。
2. 若 light 结果稳定，再跑 harness_full，重点看 hard 和 runtime。
3. 跑 47 题最小消融：`no_finalizer`、`no_skills`、`no_oracle_or_verifier`。
4. 生成 weak-to-strong gap 表。
5. DataModeling 做 protocol failure case study：submission、metric、performance、jaccard/text mismatch。
6. 最后再考虑 difficulty-aware sub-agent / verifier-gated fallback。

## PPT 可用总结

- 当前 small set 结果证明：弱模型可以通过额外结构化交互获得明显提升。
- light harness 是目前最有性价比的方向。
- full harness 说明更多时间还能换性能，但边际成本高，必须预算控制。
- 后续目标是把 DatawiseAgent 的“时间换性能”推进为“任务感知、预算可控、模型分工”的系统设计。

