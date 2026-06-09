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

已完成第一轮 small_error_set live run。

实验对象：

- 47 道已知错题 small set：`runs/infiagent_improve/small_error_set.json`
- weak model：`qwen3.5-flash`
- temperature：0
- provider：DashScope OpenAI-compatible
- baseline：原始 DatawiseAgent + original reformat
- harness：DataCard + Contract + Skills + Oracle/Verifier + FinalAnswerBlock

已完成：

- baseline raw
- baseline + original reformat
- harness_full + final block reformat

未完成：

- full 257
- 最小消融

原因：harness_full small set 用时明显更高，且 task 733 出现 final block 被放入 Python cell 执行导致的调试循环。

## 5. Baseline vs Harness 结果

qwen3.5-flash small_error_set 结果：

| 设置 | ABQ | PASQ | UASQ | hard ABQ | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline raw | 0.0000 | 0.0000 | 0.0000 | n/a | raw 长日志不能直接闭式评估 |
| baseline + original reformat | 0.5957 | 0.6986 | 0.7273 | 0.5200 | 47/47 |
| harness_full + final block | 0.6087 | 0.6812 | 0.7294 | 0.5417 | 46/47，733 卡死未写入 |

已有强模型参考 full baseline：

| 设置 | Q-Acc | Prop Sub-Q | Sub-Q |
| --- | ---: | ---: | ---: |
| `qwen3.5-35b-a3b` full baseline | 0.8638 | 0.9016 | 0.9035 |
| `qwen3.5-35b-a3b` medium baseline | 0.8851 | 0.9171 | 0.9145 |
| `qwen3.5-35b-a3b` medium harness | 0.9195 | 0.9306 | 0.9342 |
| `qwen3.5-35b-a3b` hard baseline | 0.8182 | 0.8854 | 0.8916 |
| `qwen3.5-35b-a3b` hard harness | 0.7841 | 0.8475 | 0.8719 |

弱模型差距：

- small set full ABQ gap：baseline 到强模型约 0.2681，harness 到强模型约 0.2551。
- small set hard ABQ gap：baseline 到强模型约 0.2982，harness 到强模型约 0.2765。
- 结论：harness 有小幅缩小 gap，但还远不能说 qwen3.5-flash 接近 qwen3.5-35b-a3b。

## 6. 消融结果

本轮未跑消融。

原因：

- small set harness_full 已出现 733 卡死；
- verifier_blocked 和 reformat_missing 需要先归因；
- full harness 平均 runtime 约 103.60，baseline 约 30.25，直接扩到 full 257 成本和时间都较高。

下一轮最小消融建议只跑：

- `no_finalizer`
- `no_oracle_or_verifier`
- `no_skills`

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

本轮 qwen3.5-flash small set 观察：

- easy：baseline 与 harness ABQ 都是 0.6667。
- medium：baseline 与 harness ABQ 都是 0.6923。
- hard：baseline 0.5200，harness 0.5417，小幅提升。
- hard 提升不是很大，且有 4 个 verifier_blocked 与 1 个 task 卡死，说明 hard 题仍受 finalizer/verifier 稳定性限制。

## 8. 典型修复案例

small set fixed 题：

- `125`：ML/statistical protocol，harness 保留 split、target、metric 证据。
- `219`：IQR outlier，harness skill 明确 outlier 方法和输出格式。
- `273`：correlation + Z-score outlier，harness 约束 Pearson 与 Z-score 阈值。
- `450`：weather 月均值，DataCard 帮助处理列名前导空格。
- `496`：STEM feature engineering，contract 明确公式与 year filter。
- `741`：ratio feature，final block 输出更稳定。

这些案例支持的讲法：harness 在“字段/公式/方法/格式明确”的题上确实能减少弱模型自由发挥。

## 9. 典型退化案例

small set regressed 题：

- `62`
- `124`
- `418`
- `451`

blocked / not-run：

- verifier_blocked：`310, 550, 734, 743`
- reformat_missing：`554`
- stuck/not_run：`733`

典型问题：

- 733：模型把 JSON final block 放到 Python code cell 里执行，反复报 `NameError: false is not defined`，导致 runner 卡死。
- 310/550/734/743：有些题存在可读答案，但 verifier 状态为 blocked，说明 verifier 与最终 eval 的关系还需要校准。
- 418：空响应 execution_error，类似 baseline 的空 response 问题。

## 10. 结论

本轮 small set 不能支持“弱模型已经接近强模型”的结论。

可支持的结论：

- 结构化 harness 对 qwen3.5-flash 有小幅正向作用：ABQ 从 0.5957 到 0.6087，hard ABQ 从 0.5200 到 0.5417。
- final block 路径真实接入，raw baseline 不能直接闭式评估，baseline 依赖 original reformat。
- harness 代价明显更高，平均 runtime 从 30.25 到 103.60。
- 当前最大工程瓶颈不是 API，而是 final block/代码执行边界、verifier 阻断策略和多分支成本。

可用于 PPT 的保守表述：

> 在 qwen3.5-flash small_error_set 上，harness 能小幅提升 ABQ 和 hard ABQ，但收益被 verifier/finalizer 不稳定与多分支成本抵消。弱模型增强路线可行，但当前版本还不足以替代强模型。

## 11. 局限

- 后续弱模型实验是 model-substituted setting，不是严格复现论文 Qwen2.5-72B 设置。
- harness 技能大多还是 prompt directive/checklist，不是完整确定性技能算子。
- memory 必须区分自更新运行记忆和人工错题分析，不能把人工报告伪装成 agent memory。
- DataModeling 更慢，暂不作为本轮主线。
- 本轮 harness_full 是 46/47 partial，task 733 未写入结果。
- small set 是已知错题集合，不代表全 257 题总体分布。

## 12. Lightfix 后的完整 small set 结果

修复 FinalAnswerBlock 被写入 Python code cell 的问题后，重新跑了 47/47 small_error_set：

| 设置 | ABQ | PASQ | UASQ | hard ABQ | 平均 runtime | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline raw | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 30.24s | raw 长日志无闭式答案 |
| baseline + original reformat | 0.4894 | 0.6383 | 0.6705 | 0.4400 | 30.24s | 公平 baseline |
| harness_light | 0.5957 | 0.6525 | 0.6932 | 0.6000 | 47.03s | DataCard + Contract + basic skills + Finalizer |
| harness_full | 0.6383 | 0.6950 | 0.7273 | 0.6400 | 92.79s | 准确率最高，但 verifier 成本和阻断仍明显 |

task 733 已解决：

- `harness_light`：success，约 30.75s
- `harness_full`：success，约 75.60s
- 不再出现 JSON `false` / `null` 被 Python cell 执行导致的循环。

新的保守结论：

- weak model + harness 能明显缩小 small set 上的 gap，但还不能证明可以替代强模型。
- `harness_light` 是更划算的下一步路线：比 baseline 多 10.63 个 ABQ 点，runtime 增量约 16.79s。
- `harness_full` 绝对分数更高，但相比 light 只多 4.26 个 ABQ 点，runtime 多约 45.76s，还带来 4 个 verifier_blocked。
- 下一轮不建议直接跑 full 257，应先做 verifier/finalizer 校准和 light harness 版本。

## 13. Light vs Full 的案例信息

`harness_light` fixed：

- `57, 125, 271, 273, 450, 513, 528, 647`

`harness_light` regressed：

- `62`：outlier protocol 失控，light 算成 107 个 outliers；full 修回。
- `451`：dict 格式用了 JSON 双引号和紧凑格式；full 修回 Python dict 字符串风格。
- `684`：normality / distribution 判断退化。

`harness_full` fixed：

- `57, 125, 133, 271, 273, 513, 528`

`harness_full` regressed：

- `408`：相关性数值从 baseline 正确的 `r=0.10, p=0.0102` 退化为 `r=-0.05, p=0.0001`，疑似 search/oracle/contract 干预导致口径变化。

`harness_full` verifier_blocked：

- `219, 310, 550, 734`

这些结果支持 PPT 里的主线：

> 结构化 harness 对弱模型有真实收益，但 full harness 的重校验不一定划算。更有前途的是轻量化 harness：保留 DataCard、Contract、基础 skill 和 Finalizer，减少 search/memory/heavy verifier 的干预。
