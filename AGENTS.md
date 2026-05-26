# AGENTS.md

## 项目目标

本项目在已经复现 DatawiseAgent 三类任务的基础上，优先针对 **InfiAgentBench / InfiAgent-DABench** 做科研级改进。

核心目标不是简单提高 prompt，而是把原始 DatawiseAgent 从“能执行、能调试”的 notebook agent，升级为：

> **任务感知、约束结构化、过程可校验、答案格式可验证的数据分析智能体。**

重点解决 InfiAgentBench 中暴露出的几类错误：

- 最终答案格式、小数位、引号、字典 / 列表格式错误
- 预处理顺序、子集口径、缺失值处理错误
- 相关性、显著性、正态性、异常值判断错误
- 特征工程公式理解错误
- 机器学习 split、random_state、特征列、metric 协议不一致
- reformat 阶段从长日志中漏抽、错抽、输出占位符
- 代码能运行但统计语义错误

## 工作范围

优先处理：

```text
evaluation/InfiAgentBench/
evaluation/InfiAgentBench/data/
evaluation/InfiAgentBench/scripts/eval_closed_form.py
evaluation/experimental_results/InfiAgent-Bench/
```

如果实际目录不同，先用 `find` / `ls` / `grep` 探索，不要凭空假设。

本阶段不优先改 MatplotBench 和 DataModeling。它们只作为背景，不作为主要开发对象。

## 核心改进方向

### 1. 任务类型识别与技能路由

每道题先做多标签任务识别，而不是让模型自由发挥。

可能标签：

- `summary_statistics`
- `distribution_analysis`
- `correlation_analysis`
- `outlier_detection`
- `comprehensive_preprocessing`
- `feature_engineering`
- `machine_learning`
- `final_formatting`

注意：一题可以有多个标签。例如：

```text
correlation_analysis + feature_engineering
comprehensive_preprocessing + machine_learning
summary_statistics + final_formatting
```

任务识别结果应写入结构化文件，例如：

```json
{
  "task_id": 133,
  "labels": ["comprehensive_preprocessing", "summary_statistics"],
  "reason": "The question requires dropping rows with missing Cabin before imputing Age and computing a statistic.",
  "required_skills": ["preprocessing_skill", "summary_statistics_skill", "format_skill"]
}
```

### 2. 数据结构预分析

不要让 LLM 每题都临时探索表格结构。应固定运行数据结构探测代码，生成 `schema_profile`。

至少包含：

- 文件名
- 行列数
- 列名
- 数据类型
- 缺失值数量和比例
- 数值列范围、均值、标准差
- 类别列唯一值数量和少量样例
- 日期列候选
- ID 列候选
- 可能的目标列或关键列
- 列名规范化映射

输出建议保存为：

```text
runs/infiagent_improve/schema_profiles/{task_id}.json
```

### 3. 答案契约

在写代码前，先把题目解析成 `answer_contract`。

必须尽量包含：

```json
{
  "task_id": 0,
  "answer_names": [],
  "required_columns": [],
  "filters": [],
  "groupby": [],
  "operations": [],
  "preprocessing_steps": [],
  "feature_definitions": [],
  "statistical_methods": [],
  "ml_protocol": {},
  "format_requirements": {
    "decimal_places": null,
    "quote_string": null,
    "container_type": null,
    "answer_pattern": "@answer_name[value]"
  },
  "forbidden_actions": [],
  "verification_checks": []
}
```

如果题目没有明确说明，不要擅自添加额外数据清洗、调参、特征或模型改进。

### 4. 执行过程校验

代码能运行不等于答案正确。每个关键步骤后都要记录校验信息。

常见校验：

- 筛选后样本数是否为 0
- 关键列是否存在
- 日期解析是否成功
- 缺失值处理顺序是否符合题目
- 分母是否合理
- groupby 后类别是否完整
- correlation 是否按 `abs(r)` 比较，同时保留正负号
- p-value 是否和 alpha 阈值对应
- outlier 方法是否与题目一致
- 特征公式是否先回显，再计算，再验证前几行
- ML 的 split、random_state、feature names、metric 是否符合题目
- 输出小数位、引号、dict/list 格式是否符合题目

校验报告建议保存为：

```text
runs/infiagent_improve/verifier_reports/{task_id}.json
```

### 5. 最终答案块与 reformat 解耦

主 agent 不应让 reformat 阶段从完整 notebook 日志中自由抽取答案。主 agent 应在末尾输出机器可读 final block：

```json
{
  "task_id": 0,
  "answers": [
    {
      "name": "answer_name",
      "value": "value_as_string",
      "raw_value": 0.12345,
      "format_checked": true,
      "source_variable": "computed_result"
    }
  ],
  "verification": {
    "passed": true,
    "warnings": []
  }
}
```

reformat 阶段只从 final block 转成：

```text
@answer_name[value]
```

如果 final block 缺失、含占位符、格式不合法，应回到执行阶段补算，不能继续输出。

### 6. 错误分类与实验报告

每次实验都应统计错误类型，而不仅是 ABQ。

建议错误类型：

- `format_error`
- `column_selection_error`
- `filtering_error`
- `aggregation_error`
- `preprocessing_order_error`
- `feature_definition_error`
- `statistical_method_error`
- `significance_judgment_error`
- `outlier_protocol_error`
- `ml_protocol_error`
- `reformat_error`
- `silent_failure`
- `unknown`

报告至少包含：

- ABQ / PASQ / UASQ
- easy / medium / hard 分数
- 各概念分数
- 各错误类型数量
- 修复前后对比
- token / LLM 调用次数 / 运行时间
- 典型案例分析

## 实验设计

必须做消融实验，建议顺序：

1. 原始 DatawiseAgent
2. `+ schema_profile`
3. `+ answer_contract`
4. `+ task_skill_router`
5. `+ execution_verifier`
6. `+ final_block_and_format_checker`
7. 全部模块

可选对照：

- 人工真实标签选择 skill，即 `oracle skill routing`
- 模型自动选择 skill，即 `automatic skill routing`

这样可以回答两个问题：

1. skill 模块本身是否有效？
2. 模型自动选择 skill 是否可靠？

## 开发原则

1. 不要修改 benchmark 原始数据。
2. 不要使用标准答案参与作答；标准答案只允许在离线错误分析和评估阶段使用。
3. 不要为了单个题硬编码答案。
4. 不要让模型擅自调参或引入题目没有要求的新方法。
5. 不要只追求代码执行成功，要追求统计口径正确。
6. 所有新增模块必须有可复现实验、日志和测试。
7. 所有输出文件放在 `runs/` 或新的实验目录，不要污染原始评估结果。
8. 若路径不确定，先探索再执行；不要写死不存在的路径。
9. 若运行时间过长，优先小样本 smoke test，再跑全量。
10. 若出现静默失败、空输出、占位符、格式不合法，必须触发重试、降级或停止报告。

## 建议新增目录

可根据现有项目结构调整，但建议保持清晰：

```text
src/infiagent_improve/
  __init__.py
  schema_profile.py
  answer_contract.py
  task_router.py
  skill_registry.py
  verifier.py
  final_block.py
  error_taxonomy.py

src/infiagent_improve/skills/
  summary_statistics.py
  distribution_analysis.py
  correlation_analysis.py
  outlier_detection.py
  preprocessing.py
  feature_engineering.py
  machine_learning.py
  final_formatting.py

scripts/
  run_infiagent_improve.py
  eval_infiagent_improve.py
  analyze_infiagent_errors.py
  compare_infiagent_ablation.py

runs/infiagent_improve/
  schema_profiles/
  answer_contracts/
  verifier_reports/
  final_blocks/
  predictions/
  ablations/
```

## 代码风格

- Python 代码要有清晰函数边界。
- 关键函数必须可单独测试。
- 不要把大型逻辑塞进一个 notebook cell。
- 结构化中间结果优先保存为 JSON / JSONL。
- 对 pandas 操作要显式处理缺失值、数据类型、日期解析和列名空格。
- 所有随机过程必须设置 `random_state`。
- 机器学习任务不得擅自调参，除非题目明确要求。
- 最终答案必须通过格式检查器后再输出。

## 推荐实现顺序

第一阶段：最小可用版本。

1. 读取一道题及对应 CSV。
2. 生成 `schema_profile`。
3. 生成 `answer_contract`。
4. 执行原始 DatawiseAgent 或简化执行流程。
5. 输出 final block。
6. 格式检查后生成 `@answer[value]`。

第二阶段：技能路由。

1. 实现多标签任务识别。
2. 实现 skill registry。
3. 先接入格式、相关性、预处理、特征工程四个高价值 skill。
4. 做人工标签路由与自动路由对比。

第三阶段：校验与错误分析。

1. 加 verifier。
2. 加失败重试和降级。
3. 跑 257 题全量评估。
4. 输出错误分类报告。
5. 做 ablation 表格和典型案例。

## 最终交付

至少交付：

- 可运行代码
- 运行命令
- 评估结果
- 消融实验表
- 错误类型统计表
- 典型案例分析
- 一份简短报告，说明方法、实验和结论

推荐结论表述：

> InfiAgentBench 的瓶颈不是模型不会写代码，而是模型缺乏稳定的任务路由、约束解析、过程校验和答案格式控制。因此，本项目把自由探索式 notebook agent 改造成任务感知、约束驱动、可验证的数据分析 agent。
