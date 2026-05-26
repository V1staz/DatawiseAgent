# InfiAgentBench Error-Aware Data Analysis Skill

## 适用场景

当任务满足以下条件时，使用本 skill：

- 目标 benchmark 是 InfiAgentBench / InfiAgent-DABench
- 输入是 CSV 表格 + 自然语言数据分析问题
- 需要输出闭式答案，通常为 `@answer_name[value]`
- 问题涉及统计、分布、相关性、异常值、预处理、特征工程或机器学习
- 需要减少 DatawiseAgent 中“代码能跑但答案错”的问题

本 skill 的目标不是简单写更强 prompt，而是建立：

> **任务识别 → 数据结构预分析 → 答案契约 → 技能执行 → 过程校验 → 最终格式验证**

的受控流程。

## 总流程

### Step 0. 读取题目和数据

读取以下信息：

- task id
- difficulty
- concept / tags
- question
- answer format
- uploaded CSV path
- any explicit constraints

禁止读取标准答案参与作答。标准答案只允许评估阶段使用。

### Step 1. 生成数据结构卡片

运行固定代码，生成 `schema_profile`。

必须检查：

- `shape`
- `columns`
- `dtypes`
- `missing_count`
- `missing_rate`
- numeric columns and summary
- categorical columns and examples
- datetime candidates
- id-like columns
- normalized column names

推荐 JSON 格式：

```json
{
  "file": "data.csv",
  "shape": [100, 8],
  "columns": ["col1", "col2"],
  "dtypes": {"col1": "float64"},
  "missing": {"col1": {"count": 0, "rate": 0.0}},
  "numeric_summary": {},
  "categorical_summary": {},
  "datetime_candidates": [],
  "id_candidates": [],
  "column_name_map": {}
}
```

### Step 2. 多标签任务识别

根据题目判断任务类型。不要只选一个标签。

可用标签：

```text
summary_statistics
distribution_analysis
correlation_analysis
outlier_detection
comprehensive_preprocessing
feature_engineering
machine_learning
final_formatting
```

任务识别输出：

```json
{
  "labels": ["correlation_analysis", "final_formatting"],
  "skills": ["correlation_skill", "format_skill"],
  "confidence": 0.86,
  "reason": "The question asks for the strongest correlation and requires a formatted numeric answer."
}
```

如果不确定，保留多个可能标签，并让 verifier 更严格。

### Step 3. 生成答案契约

把自然语言问题解析成结构化约束。

输出：

```json
{
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

重点：

- 保留题目要求的小数位。
- 明确是否需要引号。
- 明确是否返回数值、字符串、列表、字典、日期或特征名。
- 如果题目要求“创建特征名”，不要返回特征值。
- 如果题目要求某个统计方法，不要自行替换方法。

### Step 4. 调用任务技能

根据标签调用对应技能。可以多个 skill 连用。

---

## 技能 1：描述统计

适用：

- mean / median / std / min / max / count
- group mean / yearly average / monthly average
- top-k / argmax / argmin

流程：

1. 确认统计列。
2. 确认聚合粒度：全表、按年、按国家、按类别、按过滤子集。
3. 确认是否排除缺失值。
4. 执行统计。
5. 校验样本数和分母。
6. 按题目小数位输出。

常见错误：

- 算错粒度
- 选错列
- 忘记过滤条件
- 把全部字段列表当成 argmax 答案

---

## 技能 2：分布分析

适用：

- skewness
- normality
- histogram / distribution
- Shapiro / normal test
- high / low distribution comparison

流程：

1. 确认分析对象列。
2. 确认是否先分组或过滤。
3. 确认缺失值处理。
4. 执行分布统计或检验。
5. 输出统计量、p-value、yes/no 判断。
6. 校验输出格式。

常见错误：

- 对错误维度计算 skewness
- 过度依赖检验但没有按题目阈值判断
- reformat 阶段漏掉 p-value

---

## 技能 3：相关性分析

适用：

- Pearson correlation
- Spearman correlation
- strongest correlation
- linear / nonlinear / none
- p-value / significance

流程：

1. 确认变量集合。
2. 清理非数值和缺失值。
3. 如果题目问 strongest correlation，默认按 `abs(r)` 比较，同时保留正负号。
4. 输出 correlation coefficient。
5. 如果需要显著性，输出 p-value 与 alpha。
6. 明确 yes/no 或 relationship type。
7. verifier 复算一次关键结果。

常见错误：

- 只看最大正相关，漏掉强负相关
- 把无关系说成 nonlinear
- p-value 四舍五入错误
- 没按子集分别计算

---

## 技能 4：异常值检测

适用：

- IQR outlier
- Z-score outlier
- outlier count
- outlier list
- remove outliers then compute statistic

流程：

1. 从题目确认方法：IQR / Z-score / other。
2. 不允许模型自行换方法。
3. 明确阈值。
4. 输出 outlier mask 的数量和 id 列表。
5. 若需剔除异常值，记录剔除前后 shape。
6. 校验 outlier 列表是否完整，没有截断。

常见错误：

- IQR / Z-score 方法混用
- 阈值不一致
- 只输出列表前半段
- outlier list 为空但没有检查原因

---

## 技能 5：综合预处理

适用：

- missing value
- drop / impute
- log transform
- normalization
- encoding
- save processed file

流程：

1. 把题目中的预处理顺序写成 checklist。
2. 每一步执行后输出 shape、缺失值和关键列统计。
3. 不允许自行调换顺序。
4. 处理 `?`、空字符串、非数值列时必须显式说明。
5. 如果题目要求保存文件，检查路径和文件存在性。
6. 最后运行 verification cell。

常见错误：

- 先插补再删除，而题目要求先删除再插补
- 子集口径错误
- log transform 对 0 或缺失处理不一致
- 保存路径错误

---

## 技能 6：特征工程

适用：

- create new feature
- ratio / price range / price change
- STEM indicator
- IsAlone
- log GDP per capita
- abandonment rate

流程：

1. 先用代码注释回显公式。
2. 明确分子、分母、单位和列集合。
3. 创建新列。
4. 打印前 5 行验证。
5. 计算题目要求的统计量或返回特征名。
6. verifier 检查新列是否存在、公式是否可复算。

常见错误：

- 把“返回特征名”误解成“返回某个数值”
- 分母选错
- High-Low / Low-High 方向错
- 自然对数和 log10 混用
- 字段集合漏选

---

## 技能 7：机器学习协议

适用：

- train/test split
- classification / regression
- accuracy / MSE / R2
- model fitting

流程：

1. 只使用题目允许的数据和特征。
2. 固定题目指定的 split、random_state、stratify、class_weight。
3. 若题目未要求，不要额外调参。
4. 打印 `X.shape`、`y.shape`、feature names。
5. 训练并输出指定 metric。
6. 不要用 test label 或答案文件泄露信息。

常见错误：

- 自由添加调参
- split 不一致
- feature columns 不一致
- metric 计算对象错
- 输出精度差异

---

## 技能 8：最终格式检查

适用所有题。

检查：

- 是否包含所有 `@answer_name[value]`
- answer name 是否完全一致
- 小数位是否符合题目
- 字符串是否需要引号
- dict / list 空格和分隔符是否符合标准
- 日期是否完整
- 是否有占位符，如 `r_value`、`significance`、`TODO`
- 是否出现多个互相冲突的最终答案

最终输出前必须先生成 final block：

```json
{
  "answers": [
    {
      "name": "answer_name",
      "value": "value_as_string",
      "raw_value": null,
      "format_checked": true
    }
  ],
  "verification": {
    "passed": true,
    "warnings": []
  }
}
```

然后再转成：

```text
@answer_name[value]
```

---

## Verifier 规则

每道题至少执行以下校验：

```json
{
  "schema_check": "pass/warn/fail",
  "contract_check": "pass/warn/fail",
  "execution_check": "pass/warn/fail",
  "statistical_check": "pass/warn/fail",
  "format_check": "pass/warn/fail",
  "warnings": [],
  "must_retry": false
}
```

触发重试的条件：

- 关键列不存在
- 筛选后样本数为 0 且题目未要求 nan
- final block 缺失
- 输出占位符
- 格式检查失败
- p-value / coefficient / metric 没有写入最终答案
- 保存文件不存在
- outlier list 明显截断
- ML feature names 与契约不一致

## 任务路由 prompt 模板

```text
你是 InfiAgentBench 的任务路由器。请根据题目、格式要求和 schema_profile，给出多标签任务类型。

可选标签：
summary_statistics, distribution_analysis, correlation_analysis, outlier_detection,
comprehensive_preprocessing, feature_engineering, machine_learning, final_formatting

要求：
1. 可以多选。
2. 不要只根据概念标签判断，要根据题目实际要求判断。
3. 输出 JSON。
4. 给出需要调用的 skills。
```

## 答案契约 prompt 模板

```text
你是数据分析任务的约束解析器。请把自然语言问题转成 answer_contract。

必须解析：
- answer names
- required columns
- filters
- groupby
- operations
- preprocessing steps and order
- feature definitions
- statistical methods
- ML protocol
- final format requirements
- forbidden actions
- verification checks

只输出 JSON，不要输出解释。
```

## 校验 prompt 模板

```text
你是数据分析结果校验器。请根据 question、answer_contract、schema_profile、execution_log、final_block 判断结果是否可信。

重点检查：
- 是否使用正确列
- 是否满足筛选条件
- 是否按正确顺序预处理
- 统计方法是否符合题目
- correlation 是否按 abs(r) 排序
- p-value 与阈值是否一致
- 特征公式是否正确
- ML 协议是否一致
- final answer 格式是否严格正确

输出 JSON：
{
  "passed": true/false,
  "warnings": [],
  "errors": [],
  "must_retry": true/false,
  "retry_suggestion": ""
}
```

## reformat prompt 模板

```text
你是 InfiAgentBench 最终答案格式器。

只允许读取 final_block，不允许从长日志中猜答案。
把 final_block 转成 @answer_name[value] 格式。

规则：
1. 不要改动数值精度。
2. 不要补充解释。
3. 不要输出没有出现在 final_block 中的答案。
4. 如果 final_block 缺失、占位符、format_checked=false，输出 ERROR_NEED_RETRY。
```

## 推荐评估

必须比较：

- baseline DatawiseAgent
- baseline + schema_profile
- baseline + answer_contract
- baseline + task_router
- baseline + verifier
- baseline + final_format_checker
- full system

指标：

- ABQ
- PASQ
- UASQ
- hard ABQ
- format error count
- preprocessing order error count
- feature definition error count
- statistical method error count
- reformat error count
- runtime
- LLM calls
- token cost

## 研究表述

推荐把方法称为：

```text
Task-Aware Constraint-Verified DatawiseAgent
```

中文可称：

```text
任务感知的约束校验型 DatawiseAgent
```

核心论点：

```text
InfiAgentBench 的主要瓶颈不是模型不会写代码，而是自由探索式 notebook agent 缺少稳定的任务路由、约束解析、执行校验和最终格式控制。本方法通过技能路由、答案契约、过程校验和 final block，把数据分析 agent 从“能运行”推进到“可验证”。
```
