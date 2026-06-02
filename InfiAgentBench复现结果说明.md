# InfiAgentBench 复现与 Harness 更新说明

这份说明给组内同步本次 InfiAgentBench 复现和 harness 改造的结果。重点是：哪些结果可信、代码做了什么、分数怎么看、后续还需要补哪里。

## 1. 本次提交范围

### 包含

1. **InfiAgentBench 关键复现实验结果**
   - Qwen3.5-35B-A3B：easy / medium / hard 分难度原始 jsonl、reformat jsonl、eval json。
   - Qwen3.5-35B-A3B：full 汇总 reformat jsonl 与 eval json。
   - 论文原使用的 Qwen2.5-72B-Instruct 结果：easy / medium / hard / full 的 reformat jsonl 与 eval json，用于横向对比。

2. **Harness 改造后的代表性结果**
   - medium / hard 的 harness rerender 结果与评分。
   - 两轮 full memory-learning 实验：
     - round1：只写入自更新记忆，不检索注入；
     - round2：读取 round1 沉淀的自更新记忆，不继续写入。

3. **代码更新**
   - 新增/整理 harness 模块。
   - 更新 InfiAgentBench 运行、reformat、eval、summary 脚本。
   - 修复 Jupyter/runtime 会话清理问题。
   - 补回归测试。

### 不包含

以下内容没有提交，避免仓库膨胀或混入意义不大的中间实验：

- first5、smoke、targeted、generalization、小规模 round3/round4 等局部消融结果；
- `run_logs/`、`harness_artifacts/` 这类运行日志和中间缓存；
- agent 自更新的 memory store。它属于运行时学习状态，不作为可复制交付物转移；
- 临时根目录 `InfiAgentBench/` 下的重复 eval 输出。

## 2. 代码改造概览

### 2.1 Harness 主链路

新增 `datawiseagent/harness/`，把原来散在 prompt 和 notebook 文本里的流程约束拆成几个明确模块：

| 模块 | 作用 |
| --- | --- |
| `datacard.py` | 把题目、数据字段、闭卷约束、指标类型整理成结构化 DataCard。 |
| `contract.py` | 定义题目解答 contract，约束输出、字段、证据和最终答案格式。 |
| `rules.py` | 管理静态规则，和模型自更新记忆分离。 |
| `finalizer.py` / `verifier.py` | 统一最终答案块，减少格式错、漏答、答案落在过程文本里的问题。 |
| `oracles.py` | 做结果一致性检查，尤其是闭卷题和多子问题。 |
| `search.py` | 为 medium/hard 提供多分支候选、选择和回退接口。 |
| `recovery.py` | 运行失败时记录错误、恢复上下文，避免只在日志里沉没。 |
| `memory.py` + `self_memory_*` | 自更新记忆的 schema、沉淀和检索，和人工规则/报告分离。 |
| `trace.py` | 记录 reason/action/observation 风格的结构化过程。 |

这次不是单纯加长 prompt，而是把“题目理解—约束检查—候选生成—答案收束—失败记忆”做成可测、可开关的 harness 组件。

#### 2.1.1 和预设主线的对应关系

`汇总.pdf` 里给出的改进主线是：把 DatawiseAgent 从“能执行、能调试”的 notebook agent，往“能识别任务、能理解约束、能校验过程、能保证格式”的数据分析 agent 推。下面按这个主线说明这轮做到哪里，哪些还没做。

| 预设方向 | 本轮做到的程度 | 还没做 / 没纳入本轮 |
| --- | --- | --- |
| 任务类型识别与技能路由 | 已做基础版。`contract.py` 从题面、constraints、format 中抽取概念、方法、目标答案；`skills.py` 按 concept / keyword 路由到统计、分布、相关性、异常值、预处理、特征工程、机器学习、最终格式等 skill protocol。支持多标签，不是单一路由。 | 还不是强分类器，也没有训练/标注一个专门的 task router。没有做“人工选择技能模块 vs 模型自动选择技能模块”的正式对照实验。 |
| 技能模块库 | 做了 protocol 级技能库。每类技能有 required checks 和规则卡，例如 std 的 ddof、相关性正负号、IQR/z-score、预处理顺序、ML split/random_state/feature/metric、final format。 | 还不是完整可执行技能库：大部分技能仍通过 prompt directive、rule、oracle 约束 agent 行为，没有把每一类题都封装成独立确定性 Python 算子。 |
| 数据结构预分析 | 已做 DataCard。`datacard.py` 会整理列名、类型、缺失、数值范围、日期/类别/ID 候选等信息，并作为后续 contract、oracle、prompt 的输入。 | 当前 DataCard 主要服务约束和列检查，还没有系统压缩 token，也没有做数据卡片质量对最终分数的单独消融。 |
| 答案契约 | 已做。`contract.py` 解析目标答案名、类型、rounding、格式模板、筛选/方法/字段约束；后续 final block 和 reformat 都围绕 contract 收束。 | 对复杂自然语言约束仍是启发式解析，不保证所有筛选条件、统计口径和禁止操作都能被完整抽出。 |
| 执行过程校验 | 做了基础校验。`verifier.py` 检查 final block、answer names、format targets、类型和 verified；`oracles.py` 检查 std ddof、年份边界、ML split/features/metric、required columns 等高频错误点。 | 还没有做到每一步 dataframe shape、缺失值、分母、筛选样本数、日期解析等都强制自动验证。也没有把列名错误、预处理顺序错误、统计判断错误全部做成独立计数指标。 |
| 最终答案块与 reformat 解耦 | 已做得比较完整。主 agent 输出 `FinalAnswerBlock`；`reformat.py` 支持优先从 final block deterministic 抽取 `@answer[value]`，避免从长日志里猜答案。 | 如果模型没有生成可信 final block，当前主要靠 fallback / rerender 修复；还没形成“缺 final block 就自动回到执行阶段补算”的闭环。 |
| 安全重试与降级 | 做了接口和部分机制。`recovery.py` 记录恢复事件；runner 支持 retry/final-block rerender；Jupyter/session cleanup 也做了异常兜底。 | 还不是完整自动回溯系统。遇到空输出、超时、样本数为 0、语义校验失败时，不一定能自动定位到上一正确状态并换策略继续。 |
| 搜索 / ToT / LATS 化 hard 题 | 做了轻量多分支。`search.py` 和 runner 支持 deterministic branch、alternative branch、oracle guarded branch，并用 validator / oracle / runtime 信息选分支。 | 还没做真正的 MCTS、价值函数、自反思 rollout 或可执行 branch verifier。hard 题下降说明“生成多个候选”还不等于“选对候选”。 |
| 失败经验沉淀 | 已做自更新记忆链路。validator/runtime/oracle/final-block evidence 可以写入 self memory；round2 能检索 round1 记忆。人工规则和自更新记忆是分开的。 | 记忆检索还偏粗，没有“是否值得注入”的门控；收益不稳定。也没有把自更新 memory store 作为可复制产物提交。 |
| 实验设计与指标 | 做了 Qwen3.5-35B-A3B 全量、论文 72B 对比、medium/hard harness、两轮 memory-learning full；eval 增加了 harness metrics 字段。 | 没完整跑 `数据卡片 / 契约 / 技能路由 / 执行校验 / 最终格式 / 全部模块` 的逐层消融。平均 token、平均调用次数、平均运行时间、错误类型计数也没有形成统一主表。 |

所以这轮的定位更准确地说是：**把预设主线中的关键工程骨架搭起来，并用 medium/hard 与两轮 full memory 实验验证链路能跑；但还没有完成严格的模块级消融、难度自适应 harness、完整可执行技能库和强 hard 分支搜索。**

### 2.2 记忆与规则分离

当前有两类知识来源：

- **静态规则**：来自工程设计和人工总结，放在规则模块里，可复制、可审查；
- **自更新记忆**：来自 agent 自己在运行中犯错、修复、复盘得到的经验，只在运行时写入和检索。

自更新记忆不会接收人工报告、人工 label 或直接规则灌入，避免把我们的分析文档伪装成 agent 自己学到的东西。

### 2.3 评测脚本

主要更新在：

- `evaluation/run_harness_infiagent.py`
- `evaluation/summarize_harness_metrics.py`
- `evaluation/InfiAgentBench/scripts/reformat.py`
- `evaluation/InfiAgentBench/scripts/eval_closed_form.py`
- `evaluation/eval_infiagent_bench.py`

用途分别是 harness 运行、结果汇总、统一 reformat、闭卷题评分和 benchmark 层评测入口。

## 3. 结果文件索引

### 3.1 Qwen3.5-35B-A3B 基线

| 类型 | 路径 |
| --- | --- |
| easy 原始结果 | `evaluation/experimental_results/InfiAgent-Bench/results_qwen35b_a3b_easy.jsonl` |
| medium 原始结果 | `evaluation/experimental_results/InfiAgent-Bench/results_qwen35b_a3b_medium.jsonl` |
| hard 原始结果 | `evaluation/experimental_results/InfiAgent-Bench/results_qwen35b_a3b_hard.jsonl` |
| full reformat | `evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_qwen35b_a3b_full.jsonl` |
| full eval | `evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_qwen35b_a3b_full_evaluation_analysis.json` |

### 3.2 论文 Qwen2.5-72B-Instruct 对比结果

| 类型 | 路径 |
| --- | --- |
| full reformat | `evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_paper_qwen25_72b_full.jsonl` |
| full eval | `evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_paper_qwen25_72b_full_evaluation_analysis.json` |
| 分难度 reformat | `evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_paper_qwen25_72b_{easy,medium,hard}.jsonl` |
| 分难度 eval | `evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_paper_qwen25_72b_{easy,medium,hard}_evaluation_analysis.json` |

### 3.3 Harness 代表性结果

| 类型 | 路径 |
| --- | --- |
| medium harness 原始结果 | `evaluation/experimental_results/InfiAgent-Bench/results_harness_fullfix_qwen35b_medium_final.jsonl` |
| hard harness 原始结果 | `evaluation/experimental_results/InfiAgent-Bench/results_harness_fullfix_qwen35b_hard_final.jsonl` |
| medium reformat | `evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_harness_round2_rerender_qwen35b_medium.jsonl` |
| hard reformat | `evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_harness_round2_rerender_qwen35b_hard.jsonl` |
| medium / hard eval | `evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_harness_round2_rerender_qwen35b_{medium,hard}_evaluation_analysis.json` |

这里的 raw 结果来自 fullfix harness 运行；reformat/eval 使用后续 final-block rerender 版本，目的是固定最终答案抽取方式。

### 3.4 两轮记忆实验

| 类型 | 路径 |
| --- | --- |
| round1 write-only 原始结果 | `evaluation/experimental_results/InfiAgent-Bench/results_harness_memory_learning_r1_writeonly_qwen35b_full_20260602.jsonl` |
| round2 read-memory 原始结果 | `evaluation/experimental_results/InfiAgent-Bench/results_harness_memory_learning_r2_readmem_qwen35b_full_20260602.jsonl` |
| round1 / round2 reformat | `evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_harness_memory_learning_r{1,2}_*.jsonl` |
| 两轮 summary | `evaluation/experimental_results/InfiAgent-Bench/harness_memory_learning_r{1,2}_*_summary.json` |
| 两轮对比 | `evaluation/experimental_results/InfiAgent-Bench/harness_memory_learning_two_round_comparison_20260602.json` |

## 4. 主要分数

指标说明：

- **Q-Acc**：按题计分，题目完全答对才算对。
- **Proportional Sub-Q**：多子问按比例计分。
- **Sub-Q**：所有子问展开后统计准确率。

### 4.1 Qwen3.5-35B-A3B 与论文 72B 对比

| 设置 | 难度 | Q-Acc | Proportional Sub-Q | Sub-Q |
| --- | --- | ---: | ---: | ---: |
| 论文 Qwen2.5-72B | full | 81.71% | 87.27% | 85.09% |
| Qwen3.5-35B-A3B | full | 86.38% | 90.16% | 90.35% |
| 论文 Qwen2.5-72B | easy | 95.12% | 96.34% | 96.04% |
| Qwen3.5-35B-A3B | easy | 89.02% | 90.24% | 91.09% |
| 论文 Qwen2.5-72B | medium | 81.61% | 86.73% | 86.18% |
| Qwen3.5-35B-A3B | medium | 88.51% | 91.71% | 91.45% |
| 论文 Qwen2.5-72B | hard | 69.32% | 79.36% | 78.82% |
| Qwen3.5-35B-A3B | hard | 81.82% | 88.54% | 89.16% |

结论比较直接：Qwen3.5-35B-A3B 的 full 分数比论文 72B 高，主要优势来自 medium 和 hard；easy 反而低于论文结果，说明 easy 上存在一些格式、标签或简单题稳定性问题，不是模型越新就全维度更高。

### 4.2 Harness 代表性效果

| 设置 | 难度 | Q-Acc | Proportional Sub-Q | Sub-Q |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-35B-A3B 基线 | medium | 88.51% | 91.71% | 91.45% |
| Harness rerender | medium | 91.95% | 93.06% | 93.42% |
| Qwen3.5-35B-A3B 基线 | hard | 81.82% | 88.54% | 89.16% |
| Harness rerender | hard | 78.41% | 84.75% | 87.19% |

medium 的 harness 提升明显，说明格式收束、约束检查、多子问整理确实有用。hard 没有同步提升，主要问题不是“没有多想”，而是分支选择和计算验证还不够强：有些题能生成接近正确的思路，但最终答案选择、边界条件或单位处理仍会掉分。

### 4.3 两轮记忆实验

| 设置 | Q-Acc | Proportional Sub-Q | Sub-Q | 说明 |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-35B-A3B 基线 full | 86.38% | 90.16% | 90.35% | 无 harness 记忆注入 |
| Round1 write-only full | 87.16% | 88.93% | 90.35% | 只写入自更新记忆，不检索 |
| Round2 read-memory full | 86.77% | 88.60% | 90.79% | 读取 round1 记忆，不继续写入 |

round2 中 257 道题都检索到了记忆，平均每题 3 条。记忆链路本身是通的，但整体收益很小，甚至部分指标回落。当前结论是：记忆模块已经可运行，但检索粒度偏粗，缺少“什么时候不用记忆”的门控，容易把相似但不完全同型的经验带进来。

## 5. 结果怎么看

### 5.1 已经有效的部分

1. **格式和答案收束**
   - medium 提升最明显。
   - 多子问、闭卷题、最终答案字段的稳定性变好。

2. **运行链路可观测**
   - 现在能看到 contract、rule、trace、memory 命中情况。
   - 比之前只看 notebook 输出更容易定位错在题目理解、计算、标签还是最终格式。

3. **记忆机制不是假实现**
   - round1 能沉淀；round2 能检索；summary 能统计命中。
   - 但“能用”和“带来稳定增益”是两回事，目前只能说链路成立，收益还不够。

### 5.2 主要问题

1. **hard 题仍缺少可靠分支选择器**
   - hard 题常见问题不是没有搜索，而是搜索后没有选对。
   - 需要更强的可执行验证、数值一致性检查和反事实比较。

2. **没有按题目难度动态调 harness 厚度**
   - 当前 harness 偏统一处理。
   - 简单题可能被过度处理，hard 题又还不够强。
   - 后续应做轻/中/重三档策略：easy 轻收束，medium contract+oracle，hard 搜索+验证+回退。

3. **记忆检索需要门控**
   - 现在平均每题 3 条命中，覆盖很高，但不一定都相关。
   - 后续要做 memory usefulness 判断：题型、指标、字段、错误模式都匹配时才注入。

4. **部分标签/指标边界还需要维护**
   - easy 低于论文 72B，说明仍有简单题不稳定。
   - 应建立 label conflict registry，把评分器常见边界单独沉淀。

## 6. 当前结论

- Qwen3.5-35B-A3B 复现的 full 分数高于论文 Qwen2.5-72B，主要靠 medium 和 hard。
- Harness 对 medium 有明确增益，对 hard 暂时没有形成稳定提升。
- 自更新记忆链路已经跑通，但还不是稳定提分模块；下一步重点不是继续堆记忆数量，而是提高检索门控和使用判断。
- 代码层面已经把 harness、记忆、规则、评测脚本拆开，后续可以在同一链路上继续做分支选择、难度路由和可执行 oracle。

## 7. 建议后续优先级

1. **难度自适应 harness**
   - easy：只做格式收束和最终答案校验。
   - medium：启用 DataCard、Contract、Oracle。
   - hard：启用搜索、多候选、可执行验证、回退。

2. **hard 分支选择器**
   - 不只比较语言解释，要比较中间变量、单位、计数边界和最终数值。
   - 对多子问题目做逐子问验证，不让一个错误分支污染整题。

3. **记忆检索门控**
   - 低相关记忆不注入。
   - 记忆命中后先生成“使用理由”，理由不充分则跳过。

4. **评分边界案例表**
   - 把常见误判、标签冲突、输出字段问题整理为可审查规则。
   - 注意仍然和 agent 自更新记忆分开。
