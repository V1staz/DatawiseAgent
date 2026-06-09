# DatawiseAgent 改进汇报 PPT 重构提纲

## 0. 当前问题

当前 `expected_results_beamer.tex` 的主要问题不是材料不够，而是结构层次不清：

1. 开头缺少概览页，听众一开始不知道这次汇报到底有几条改进线。
2. 中间缺少章节标题页，InfiAgentBench、DataModeling、时间换性能、资源优化混在一起。
3. 已观察结果、合理预期、未来计划没有足够视觉区分。
4. 三个方向的贡献关系没有讲清楚：谁负责哪一块、每一块解决什么、最后如何融合成一个系统。

建议把 PPT 改成：

> 复现结果暴露出三个改进方向：任务适配、结构化时间换性能、资源感知调度。我们分别验证它们的动机、初步结果和局限，最后把三条线融合成一个 Resource-Aware DatawiseAgent。

## 1. 开头概览

### Slide 1: 标题

标题建议：

> 从复现到改进：让 DatawiseAgent 的任务适配、长交互和资源预算可控

副标题：

> InfiAgentBench / DataModeling 上的失败模式、结构化 harness 与资源感知调度

### Slide 2: 总览页

这一页必须明确三条线：

| 方向 | 核心问题 | 当前材料状态 |
| --- | --- | --- |
| 1. 任务适配 | 不同 benchmark 的失败模式不同，不能靠统一 prompt 解决 | InfiAgentBench 较完整；DataModeling 目前主要是 protocol-aware 动机和初步观察 |
| 2. 结构化时间换性能 | 原作中的 notebook-style 长交互如何变成可验证、可停止的过程 | 当前 PPT 材料最完整，有 small set 结果和 light/full 对比 |
| 3. 资源感知调度 | 强模型、小模型、确定性模块如何分工，减少等效 token 和无效对话 | 当前是方案设计和预期，缺 token meter 与 E0-E3 实验 |

Takeaway:

> 三条线不是并列堆模块，而是逐层递进：先识别任务错法，再把额外交互结构化，最后把结构化后的流程做资源调度。

### Slide 3: 证据等级说明

用三种标签：

- `Observed`: 已跑出来的结果。
- `Diagnosed`: 从错误案例和日志得到的分析。
- `Planned`: 方案设计或合理预期，尚未全量验证。

这页很重要，因为当前 PPT 里 small set、强模型参考、资源计划混在一起，容易被听众误解为都已经验证。

## 2. 第一部分：任务适配

章节标题页：

> Part I. Task-Specific Adaptation  
> 不同任务错法不同，harness 应该跟着任务走

这一部分对应你说的第一点。当前 PPT 里有，但不是主贡献材料；需要诚实标注：InfiAgentBench 较完整，DataModeling 目前偏动机和初步观察。

### 2.1 Baseline 和失败分析

#### InfiAgentBench baseline

可用材料：

| 设置 | ABQ | PASQ | UASQ |
| --- | ---: | ---: | ---: |
| Qwen3-32B full latest | 81.74% | 86.21% | 86.43% |
| Qwen3.5-35B-A3B baseline | 86.38% | 90.16% | 90.35% |

典型失败：

- 答案格式、小数位、引号、dict/list 格式错误。
- 预处理顺序、缺失值处理错误。
- correlation 的符号、p-value、abs(r) 判断错误。
- outlier 方法协议错误。
- feature engineering 公式理解错误。
- final answer 从长日志抽取失败。

#### DataModeling baseline

可用材料：

| run | complete | performance | avg time |
| --- | ---: | ---: | ---: |
| qwen25 | 68/74 | 0.4290 | 760.25s |
| qwen35b-a3b-full | 73/74 | 0.5318 | 288.60s |

典型失败：

- submission / performance 文件没生成。
- 输出或评估格式不匹配。
- metric 类型错配。
- jaccard 文本指标收到数值列。

注意：这一部分不能说 DataModeling harness 已经系统提分；目前只能说 protocol failure 比单纯模型能力更突出。

### 2.2 方案设计

#### InfiAgentBench: answer-aware harness

| 模块 | 作用 |
| --- | --- |
| DataCard | 固定 CSV 结构，减少重复摸索列名和 dtype |
| Contract | 抽取答案名、列名、过滤、方法、小数位 |
| Skill checklist | 针对 correlation、outlier、ML、format 等题型给检查清单 |
| Verifier | 检查列、样本数、p-value、abs(r)、final block |
| Final Block | 机器可读答案块，避免从长日志自由抽答案 |

#### DataModeling: protocol-aware harness

| 模块 | 作用 |
| --- | --- |
| Submission checker | 检查 submission 文件是否生成、列是否匹配 |
| Metric checker | 检查 metric 类型、方向、输入列类型 |
| Performance parser | 统一读取 performance/result 文件 |
| Evaluation adapter | 避免 jaccard/text/numeric 等协议错配 |

DataModeling 部分建议用“设计方向 + 失败案例”，不要写成已完成系统实验。

### 2.3 实验结果和思考

InfiAgentBench 可讲 observed 结果：

- weak small set 中 `harness_light` 明显优于 baseline。
- task 733 的 final block/code cell 混淆问题被修复。
- light harness 更有性价比。

DataModeling 可讲 diagnosed 结果：

- 强模型用时更短，说明少走弯路比单次调用成本更重要。
- 当前更像 protocol correctness 问题，不是简单提升模型能力。

章节小结：

> 任务适配告诉我们：不同 benchmark 不应该共享一套泛化 prompt。InfiAgentBench 需要答案口径和格式收束；DataModeling 需要评估协议和 submission 闭环。

## 3. 第二部分：结构化时间换性能

章节标题页：

> Part II. Structured Time-for-Performance  
> 多花时间不是多问几轮，而是把时间变成可验证过程

这一部分是当前 PPT 最完整的主线。

### 3.1 Baseline 和问题

原作现象：

- DatawiseAgent 用 planning、execution、debugging、repair 的 notebook-style 长交互提升鲁棒性。

复现后问题：

- 长交互不一定更好。
- full harness 在 hard split 上可能退化。
- 多分支搜索可能选错。
- verifier 可能过度阻断。
- final block 如果混进 code cell，会触发无效 debug loop。

### 3.2 方案设计

把“时间”拆成几个可验证环节：

```text
Question
  -> DataCard
  -> Contract
  -> Skill checklist
  -> Notebook execution
  -> Verifier
  -> Final Block
  -> Deterministic reformat
```

关键设计：

- 执行前：固定数据结构和答案契约。
- 执行中：用 skill checklist 限制统计方法和预处理顺序。
- 执行后：verifier 检查过程和 final block。
- 重试时：只在 fail 后 targeted retry，不默认 full 多分支。

### 3.3 实验结果

small_error_set observed 结果：

| Setting | ABQ | Hard ABQ | Avg runtime |
| --- | ---: | ---: | ---: |
| baseline + original reformat | 48.94% | 44.00% | 30.24s |
| harness_light | 59.57% | 60.00% | 47.03s |
| harness_full | 63.83% | 64.00% | 92.79s |

可以讲：

- light 比 baseline 多 10.63 ABQ 点，runtime 增量约 16.79s。
- full 比 light 再多 4.26 ABQ 点，但 runtime 多约 45.76s。
- 说明结构化时间确实有效，但 full-everywhere 边际收益低。

强模型参考：

| Setting | Split | ABQ |
| --- | --- | ---: |
| baseline | medium | 88.51% |
| harness rerender | medium | 91.95% |
| baseline | hard | 81.82% |
| harness rerender | hard | 78.41% |

可以讲：

- medium 上 harness 有帮助。
- hard 上 heavy harness 可能退化。
- 这支持“时间要花对地方，而不是越多越好”。

### 3.4 思考

这一部分应该形成明确结论：

> 原作的长交互机制不能简单理解为“多跑几轮”。我们的改进是把额外时间转成 DataCard、Contract、Verifier 和 Final Block，使每一轮交互有明确的输入、检查和停止条件。

## 4. 第三部分：资源感知调度

章节标题页：

> Part III. Resource-Aware Scheduling  
> 既然知道时间该花在哪里，就可以决定由谁来花、是否要花

这一部分是你的优化方向，但在整体优化中只占一部分，所以不应该过度展开。建议控制在 3-4 页。

### 4.1 Baseline

Qwen3-32B baseline：

| 指标 | 数值 |
| --- | ---: |
| ABQ | 81.74% |
| 调用次数 | 约 1,621 |
| equivalent token | 5.961M |

等效 token 口径：

```text
EqTok = T_32B + 0.5 T_14B + 0.25 T_8B
```

### 4.2 方案设计

资源调度有两个维度：

#### 小模型使用规则

| 阶段 | 推荐处理 |
| --- | --- |
| Schema / Contract / Verifier / Finalizer | deterministic |
| Easy planning / routing | 8B / 14B / template |
| Medium planning/debug | 14B |
| Medium/hard code generation | 32B |
| Hard reasoning / fallback | 32B |

#### 减少模型对话发生

| 机制 | 作用 |
| --- | --- |
| deterministic artifacts | schema、contract、finalizer 不问模型 |
| contract-to-operation path | 明确操作计划直接执行 |
| one-shot execution | 低风险题一次模型调用完成 |
| verifier-gated retry | 失败才短上下文修复 |
| artifact reuse | 同一 CSV 的 schema 和列映射复用 |

这比狭义 solver 更 general：它不是只覆盖固定统计公式，而是减少不必要对话。

### 4.3 预期结果

| 方案 | ABQ | 调用次数 | EqTok |
| --- | ---: | ---: | ---: |
| 32B baseline | 81.74% | 1,621 | 5.961M |
| 小模型规则 | 80.5%-82.5% | 约 1,115 | 约 2.120M |
| 小模型规则 + 对话减少 | 81.0%-83.5% | 约 700-900 | 约 1.05M-1.60M |

必须标注：这是 expected / target，还没有 token_meter 和 E0-E3 全量验证。

### 4.4 思考

这一部分应该和前两部分接上：

> 资源调度不是一开始就为了省 token，而是在任务适配和结构化交互之后自然出现的：当我们知道哪些环节可以确定性完成、哪些环节需要强推理，就可以把简单明确的子任务拆给小模型 instruct，把复杂子任务留给强模型 thinking。

这里要补你想讲的关键发现：

- 强模型不一定让总时间变慢，因为它可能减少 debug/repair 轮数。
- 小模型处理简单子任务会增加通信，但如果只处理低风险、短上下文任务，通信开销可控。
- 最终目标不是单纯省资源，而是：
  - ABQ 提升或保持；
  - 总处理时间接近或更短；
  - equivalent token 显著下降。

## 5. 终极融合方案

章节标题页：

> Final Integration  
> 从三个局部改进到一个统一系统

这一部分是你要求补上的“讲完三点后的终极融合方案和发现”。

### 5.1 融合架构

建议用一张总图：

```text
Question + Data
  -> Task/Protocol Router
      -> InfiAgentBench answer-aware path
      -> DataModeling protocol-aware path
  -> DataCard / Contract / Protocol Spec
  -> Risk Estimator
      low risk: deterministic / one-shot small model
      medium risk: 14B + verifier
      high risk: 32B thinking + targeted retry
  -> Execution
  -> Verifier / Protocol Checker
  -> Final Block / Submission
  -> EqTok + Runtime + Error taxonomy report
```

这个总图能把三条线合起来：

- 任务适配：Task/Protocol Router 决定走 answer-aware 还是 protocol-aware。
- 时间换性能：DataCard、Contract、Verifier、Final Block 让长交互可验证。
- 资源调度：Risk Estimator 决定使用 deterministic、small model 还是 32B。

### 5.2 融合后的核心发现

可以收束为四个发现：

1. **失败模式不是统一的。**
   - InfiAgentBench 主要错在答案口径、统计协议、最终格式。
   - DataModeling 主要错在 submission、metric 和 evaluation 协议。

2. **长交互有效，但必须结构化。**
   - light harness 证明额外结构化步骤能显著提升 small set。
   - full harness 证明无控制增加检查和分支会带来高 runtime，甚至 hard 退化。

3. **强模型不一定更慢，小模型不一定更划算。**
   - 强模型可能少走弯路，总时间更短。
   - 小模型只有在简单、明确、短上下文子任务上才适合。

4. **最终系统应该做风险调度，而不是固定模型或固定轮数。**
   - low risk: deterministic / small instruct。
   - medium risk: small model + verifier。
   - high risk: 32B thinking + targeted fallback。

### 5.3 最终一句话

建议 PPT 最后一页改成：

> DatawiseAgent 的改进不是“多加 prompt”或“多跑几轮”，而是把数据分析任务变成一个可路由、可验证、可预算的执行过程：任务决定检查项，风险决定模型，验证决定是否继续。

## 6. 推荐 PPT 脉络

建议最终页序：

| 页码 | 内容 |
| ---: | --- |
| 1 | 标题 |
| 2 | 三条改进线总览 |
| 3 | 证据等级说明：Observed / Diagnosed / Planned |
| 4 | Part I 标题：Task-Specific Adaptation |
| 5 | InfiAgentBench 和 DataModeling 的 baseline / 失败模式对比 |
| 6 | InfiAgentBench answer-aware harness |
| 7 | DataModeling protocol-aware harness 动机 |
| 8 | Part I 小结 |
| 9 | Part II 标题：Structured Time-for-Performance |
| 10 | 原作长交互现象与复现后的问题 |
| 11 | DataCard -> Contract -> Execution -> Verifier -> Final Block |
| 12 | small set light/full 结果 |
| 13 | hard 退化与 task 733 案例 |
| 14 | Part II 小结：时间要花对地方 |
| 15 | Part III 标题：Resource-Aware Scheduling |
| 16 | EqTok baseline |
| 17 | 小模型使用规则 |
| 18 | 减少模型对话发生 |
| 19 | 预期结果和 E0-E3 消融 |
| 20 | Final Integration 标题 |
| 21 | 融合架构图 |
| 22 | 四个核心发现 |
| 23 | 当前差距与下一步 |
| 24 | 最后一页：任务决定检查项，风险决定模型，验证决定是否继续 |

如果时间不够，可以压缩到 18 页：

- 合并 DataModeling 和 Part I 小结；
- 合并 task 733 和 hard 退化；
- 合并资源预期和消融计划；
- 合并融合架构和核心发现。

