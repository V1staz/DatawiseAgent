# InfiAgentBench 弱模型增强 PPT Notes

## 1. 研究动机

DatawiseAgent 原始能力偏向“能执行代码、能调试 notebook”。InfiAgentBench 的闭卷数据分析题还需要更稳定的题目理解、字段约束、统计口径、执行校验和最终格式控制。

本项目关注的问题是：结构化 harness 是否能让更便宜、更小的弱模型在 InfiAgentBench 上接近强模型表现。

## 2. 为什么测弱模型

强模型 `qwen3.5-35b-a3b` 在已有结果中已经达到较高水平：

- full Q-Acc：0.8638
- Prop Sub-Q：0.9016
- Sub-Q：0.9035

继续只测强模型，增益空间有限，也难证明工程模块的边际价值。弱模型更容易暴露题目理解、格式、统计协议和执行校验问题，因此更适合验证 harness 是否真正补能力。

## 3. 为什么 harness 可能对弱模型更有用

弱模型常见失败不是不会写 Python，而是：

- 没有先识别任务类型；
- 重复摸索 CSV 结构；
- 忘记题目约束；
- 统计口径错误；
- ML split / random_state / metric 不一致；
- final answer 格式不稳定；
- reformat 从长日志里抽错答案。

DataCard、Contract、Skill Checklist、Semantic Oracle、Verifier 和 FinalAnswerBlock 可以把这些隐性要求显式化，降低弱模型自由发挥空间。

## 4. 实验设置

待运行。

计划比较：

- strong baseline：已有 `qwen3.5-35b-a3b`
- weak baseline：弱模型原始 DatawiseAgent
- weak + harness：弱模型完整 harness
- weak + harness ablation：逐步关闭 DataCard / Contract / Skills / Oracle+Verifier / Finalizer / Memory / Search

候选弱模型：

- `qwen3.5-flash`
- `qwen3-30b-a3b`
- `qwen3-32b`
- 其他 API 可用的 Qwen3/Qwen3.5 小模型

## 5. Baseline vs Harness 结果

待填弱模型结果。

已有强模型参考：

| 设置 | Q-Acc | Prop Sub-Q | Sub-Q |
| --- | ---: | ---: | ---: |
| `qwen3.5-35b-a3b` full baseline | 0.8638 | 0.9016 | 0.9035 |
| `qwen3.5-35b-a3b` medium baseline | 0.8851 | 0.9171 | 0.9145 |
| `qwen3.5-35b-a3b` medium harness | 0.9195 | 0.9306 | 0.9342 |
| `qwen3.5-35b-a3b` hard baseline | 0.8182 | 0.8854 | 0.8916 |
| `qwen3.5-35b-a3b` hard harness | 0.7841 | 0.8475 | 0.8719 |

## 6. 消融结果

待填。

计划展示：

- `baseline`
- `harness_full`
- `no_datacard`
- `no_contract`
- `no_skills`
- `no_oracle_or_verifier`
- `no_finalizer`
- `no_memory`
- `no_search`

## 7. 按难度分析

已有现象：

- medium：harness 有正收益。
- hard：harness 退化，说明最终格式不是唯一瓶颈，分支选择和可执行语义验证更重要。
- easy：强模型 baseline 低于论文参考 72B，可能存在简单题格式/标签稳定性问题。

弱模型结果跑完后，应重点回答：

- weak+harness 是否在 medium 最明显提升；
- hard 是否仍退化；
- no_search/no_memory 是否能减少 hard 退化；
- finalizer 是否主要减少格式错误和 reformat missing。

## 8. 典型修复案例

待填。

优先找这些类型：

- final block 修复 reformat 漏填；
- correlation abs(r) / 符号保留；
- p-value alpha 判断；
- feature formula 回显与前 5 行检查；
- ML random_state / split / metric 修复。

## 9. 典型退化案例

待填。

已有 memory-learning 对比中出现过 hard/ML/feature engineering 退化，如 137、424、521。后续弱模型实验需要确认是否由 memory、search、contract 误解析或 verifier 过松/过严导致。

## 10. 结论

待弱模型实验后填写。

目标表述：

如果 weak+harness 接近 strong baseline，则说明结构化 harness 能用低成本模型弥补一部分数据分析 agent 的协议和格式缺陷。

如果 weak+harness 没接近 strong baseline，也可以说明当前 harness 更偏格式/约束收束，对 hard 题需要更强可执行验证和分支选择器。

## 11. 局限

- 后续弱模型实验是 model-substituted setting，不是严格复现论文 Qwen2.5-72B 设置。
- harness 技能大多还是 prompt directive/checklist，不是完整确定性技能算子。
- memory 必须区分自更新运行记忆和人工错题分析，不能把人工报告伪装成 agent memory。
- DataModeling 更慢，暂不作为本轮主线。
