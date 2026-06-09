# InfiAgentBench 等效 Token 优化计划

## 1. 目标与指标

本阶段以 Qwen3-32B 全流程运行为 baseline，不再使用花费数值作为评价指标，而是使用 **等效 token**：

```text
equivalent_tokens = tokens_32b + 0.5 * tokens_14b + 0.25 * tokens_8b
```

三个核心指标：

| 指标 | 作用 |
| --- | --- |
| **ABQ** | 核心完成指标。InfiAgentBench 是 closed-form QA，整题只要一个答案名、数值或格式错，就应视为任务失败，所以 ABQ 比 PASQ/UASQ 更适合作为主指标。 |
| LLM 调用次数 | 代表对话轮数、服务端请求次数、排队开销和 notebook 循环长度。 |
| equivalent token | 主资源指标，用 32B 口径统一比较 32B/14B/8B 的资源消耗。 |

优化目标：

> 在 ABQ 不明显低于 Qwen3-32B baseline、调用次数不明显增加的前提下，最大化降低 equivalent token。

## 2. Baseline

已有 Qwen3-32B 全量结果：

| 运行 | ABQ | PASQ | UASQ |
| --- | ---: | ---: | ---: |
| Qwen3-32B full latest | 81.74% | 86.21% | 86.43% |

已有多阶段 token 估算：

| 难度 | 题数 | 调用/题 | token/题 | 总调用 | equivalent token |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy | 82 | 6.0 | 15,000 | 492 | 1.230M |
| medium | 87 | 5.9 | 21,000 | 513 | 1.827M |
| hard | 88 | 7.0 | 33,000 | 616 | 2.904M |
| 总计 | 257 | 6.31 | 23,195 | 1,621 | 5.961M |

baseline 全部使用 32B，因此 raw token 和 equivalent token 相同。

## 3. 总方案：Resource-Aware DatawiseAgent

整体方案从两个角度展开：

1. **小模型使用规则**：不同系统角色、不同任务难度使用不同模型或确定性逻辑。
2. **减少模型对话发生**：不是每个阶段都问模型，不是每个失败都重跑完整对话，而是能确定性完成就确定性完成，能一次完成就一次完成，只有验证失败才触发短上下文修复。

这两个方向组成一个总方案，而不是两个独立方法。小模型路由解决“每次调用用多大的模型”，对话减少解决“是否需要发生这次调用”。

### 3.1 小模型使用规则

不同角色对模型能力的需求不同：

| 角色 / 阶段 | 实际任务 | 推荐处理 |
| --- | --- | --- |
| Schema Profile | 读取 CSV、列名、dtype、缺失值、数值范围 | deterministic |
| Contract Parser | 抽取答案名、列名、格式、小数位、禁止操作 | rule-first，歧义时 8B/14B |
| Task Router | 判断概念、难度、风险等级 | rule-first，歧义时 8B/14B |
| Planning | 拆步骤、决定处理顺序 | easy 用模板/8B，medium 用 14B，hard 用 32B |
| Code Generation | 写 pandas/scipy/sklearn 代码 | easy 用 14B，medium/hard 用 32B |
| Debug Execution | 修运行错误、列名错误、导入错误 | easy 用 8B/14B，medium 用 14B，hard 用 32B |
| Post-Debug Filter | 清理错误 cell、保留可运行路径 | deterministic |
| Verifier | 检查列、样本数、方法协议、final block | deterministic |
| Final Formatting | 渲染 `@answer[value]` | deterministic |

按难度的保守预算：

| 难度 | 调用结构 | 调用/题 | raw token/题 |
| --- | --- | ---: | ---: |
| easy | 14B 主体 + 8B debug/router + deterministic format | 4 | 3,000 |
| medium | 32B code + 14B planning/debug + 8B router/filter | 5 | 8,000 |
| hard | 32B planning/code/debug + 14B contract/router + 8B filter | 4 | 20,000 |

分模型 token 预估：

| 模型 | raw token | 权重 | equivalent token |
| --- | ---: | ---: | ---: |
| Qwen3-32B | 1.703M | 1.00 | 1.703M |
| Qwen3-14B | 0.670M | 0.50 | 0.335M |
| Qwen3-8B | 0.330M | 0.25 | 0.082M |
| 总计 | 2.702M | - | 2.120M |

保守预期：

| 指标 | 32B baseline | 小模型规则后 |
| --- | ---: | ---: |
| ABQ | 81.74% | 80.5%-82.5% |
| 调用次数 | 1,621 | 约 1,115 |
| equivalent token | 5.961M | 约 2.120M |
| equivalent token 降幅 | - | 约 64.4% |

### 3.2 减少模型对话发生

这里不把固定统计模板做成一个狭义独立方案，而是做成更 general 的 **Conversation Reduction Layer**。它的目标是减少模型对话本身的发生次数。

核心规则：

| 机制 | 含义 | 节省来源 |
| --- | --- | --- |
| Deterministic artifacts | schema profile、contract、skill checklist、format checker 都用程序生成 | 执行前少问模型 |
| Contract-to-operation path | 若题目约束能编译成明确操作计划，就直接执行 pandas/scipy/sklearn 流程 | 部分题 0 次 LLM |
| One-shot execution path | 对低风险题，一次模型调用直接产出 code + final block，不走 plan/update 多轮循环 | 少发生中间对话 |
| Verifier-gated retry | 只有 verifier/semantic check 失败才重试，且重试只给错误点和必要上下文 | 避免完整重跑 |
| Deterministic finalization | final block 到 `@answer[value]` 完全由程序渲染 | 去掉 reformat 对话 |
| Artifact reuse | 同一 CSV 的 schema、列名映射、常见任务规则跨题复用 | 减少上下文探索 |

`Contract-to-operation path` 比狭义固定公式执行器更一般。它不是只覆盖固定统计公式，而是把题目先解析成操作计划：

```text
required_columns
filters
feature_definitions
groupby
operations
statistical_methods
ml_protocol
format_requirements
```

只要计划足够明确，就由确定性执行器完成；计划不明确、方法冲突或 verifier 不可验证时，再进入模型执行。这样它既能覆盖 summary/correlation/outlier/normality，也可以逐步覆盖简单特征工程、简单 ML 协议和重复出现的数据分析模式。

推荐执行策略：

| 风险等级 | 默认路径 | 失败后 |
| --- | --- | --- |
| low | deterministic operation path 或 one-shot 14B | verifier fail 后 32B targeted retry |
| medium | one-shot 14B/32B，根据 task type 决定 | verifier fail 后 32B targeted retry |
| high | 32B guided execution | 只针对失败检查做 repair，不盲目多分支 |

保守预期：

| 指标 | 32B baseline | 只做小模型规则 | 小模型规则 + 对话减少 |
| --- | ---: | ---: | ---: |
| ABQ | 81.74% | 80.5%-82.5% | 81.0%-83.5% |
| 调用次数 | 1,621 | 约 1,115 | 约 700-900 |
| raw token | 5.961M | 2.702M | 约 1.35M-1.90M |
| equivalent token | 5.961M | 2.120M | 约 1.05M-1.60M |
| equivalent token 降幅 | - | 约 64.4% | 约 73%-82% |

ABQ 不一定下降的原因：

- 确定性 finalization 会减少格式错误。
- contract-to-operation path 不会额外清洗或自由发挥。
- verifier-gated retry 能阻止明显不合格答案进入评估。

调用次数不明显反弹的原因：

- 一部分题直接 0 次 LLM。
- 低风险题走 one-shot。
- retry 是 targeted repair，而不是重启完整 notebook 对话。

## 4. 实现计划

最小实现顺序：

1. 增加 `token_meter`。
   - 每次 LLM 调用记录 `model`、`stage`、`total_tokens`。
   - 每题聚合 `raw_tokens_by_model`、`equivalent_tokens`、`llm_calls`。

2. 完成 stage-level model routing。
   - `model_role_map.yaml` 配置 difficulty + stage 到模型。
   - Post-Debug Filter、Final Formatting、Verifier 全部确定性化。

3. 增加 conversation reduction policy。
   - `resource_policy={baseline,model_routing,conversation_reduction}`。
   - 执行前尝试 deterministic artifacts 和 contract-to-operation path。
   - low-risk 题走 one-shot。
   - verifier fail 才 targeted retry。

4. 跑实验。
   - 先跑 30 题 stratified smoke。
   - 再跑 257 题全量。
   - 记录 ABQ、调用次数、raw token、equivalent token、平均运行时间、错误类型。

## 5. 消融设计

| 实验 | 说明 | 目的 |
| --- | --- | --- |
| E0 32B baseline | 所有角色使用 Qwen3-32B | 当前参考点 |
| E1 model routing only | 只启用小模型使用规则 | 验证模型分工是否能降 equivalent token |
| E2 conversation reduction only | 仍以 32B 为主，但减少不必要对话 | 验证少对话本身是否有效 |
| E3 full resource-aware agent | 小模型规则 + 对话减少 | 主结果 |

每组报告：

- **ABQ** / PASQ / UASQ，其中 ABQ 是核心完成指标。
- 总调用次数。
- 分模型 raw token。
- equivalent token。
- equivalent token 相对 baseline 降幅。
- deterministic operation coverage。
- one-shot coverage。
- retry / escalation rate。
- 平均运行时间。
- 错误类型统计。

## 6. 推荐论文叙事

推荐把方法贡献写成：

> 我们把 DatawiseAgent 的资源效率问题形式化为三目标约束优化：在 ABQ 不明显下降、调用次数不显著增加的约束下，最小化按模型规模折算后的 equivalent tokens。整体方案从两个维度改造原始 notebook agent：第一，基于系统角色和任务难度进行小模型路由；第二，通过 deterministic artifacts、contract-to-operation path、one-shot execution 和 verifier-gated targeted retry 减少模型对话本身的发生。这样，LLM 不再是每个阶段默认调用的执行器，而是按任务风险和验证结果使用的受控资源。
