# InfiAgentBench 错题与改进分析

## 1. 任务与指标

参考根目录论文精读记录与本地数据，InfiAgentBench / InfiAgent-DABench 是数据分析类 benchmark。本地数据位于 `evaluation/InfiAgentBench/data/`，实际共有 257 题：

- easy: 82
- medium: 87
- hard: 88

每道题给定一个 CSV 表格、自然语言问题、约束条件、概念标签和固定答案格式。任务目标是让 agent 通过 notebook 工作流读取数据、执行分析，并输出 `@answer_name[value]` 格式的闭式答案。

本地评估脚本为 `evaluation/InfiAgentBench/scripts/eval_closed_form.py`，核心指标包括：

- ABQ / Accuracy by Question: 一道题所有子答案全对才算该题正确。
- PASQ / Accuracy Proportional by Sub-Question: 每题内部按子问题比例给分，再对题目平均。
- UASQ / Accuracy by Sub-Question: 所有子答案摊平后计算准确率。

## 2. 当前模型准确率

分析对象：

- 结果文件：`evaluation/experimental_results/InfiAgent-Bench/reformat/results_datawise-Qwen2.5-32B-Instruct-temperature-0-args-7-6-8.jsonl`
- 评估输出：`evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_datawise-Qwen2.5-32B-Instruct-temperature-0-args-7-6-8_evaluation_analysis.json`

总体结果：

| 指标 | 数值 |
|---|---:|
| ABQ / Accuracy by Question | 81.71% |
| PASQ / Accuracy Proportional by Sub-Question | 86.25% |
| UASQ / Accuracy by Sub-Question | 87.06% |
| 正确题数 | 210 / 257 |
| 错误题数 | 47 / 257 |

按难度：

| 难度 | 题数 | 正确题数 | ABQ |
|---|---:|---:|---:|
| easy | 82 | 73 | 89.02% |
| medium | 87 | 74 | 85.06% |
| hard | 88 | 63 | 71.59% |

按概念看，错误率最高的是：

| 概念 | 错题数 / 总题数 | 错误率 |
|---|---:|---:|
| Comprehensive Data Preprocessing | 13 / 45 | 28.9% |
| Machine Learning | 5 / 19 | 26.3% |
| Feature Engineering | 12 / 50 | 24.0% |
| Correlation Analysis | 16 / 72 | 22.2% |
| Distribution Analysis | 11 / 64 | 17.2% |
| Summary Statistics | 13 / 90 | 14.4% |
| Outlier Detection | 5 / 35 | 14.3% |

结论：模型在简单统计题上表现较稳，但在综合预处理、特征工程、机器学习协议、相关性判断这几类题上明显更容易出错。hard 题准确率降到 71.59%，说明长链条、多步骤、需要严格遵守约束的题是主要失分来源。

## 3. 逐题错因概览

| ID | 难度 | 概念 | 大致错因 |
|---:|---|---|---|
| 7 | hard | Machine Learning | 训练流程基本正确，但精度输出 0.79，标准为 0.78；属于随机划分、模型参数或四舍五入细节差异。 |
| 23 | hard | Machine Learning, Summary Statistics | 回归任务中错误地构造训练特征或目标，MSE 输出为 0.00，明显没有按题目要求预测 March 2020 employment level。 |
| 55 | easy | Summary Statistics | 统计口径错，把均值算到错误粒度或错误列上；标准是全国家全年 cases 平均值。 |
| 56 | easy | Distribution Analysis, Summary Statistics | 最大 deaths 的国家和年份定位错，可能用了错误 deaths 列或没有按单年最大值筛选。 |
| 57 | easy | Correlation Analysis | cases 与 deaths 相关系数算错，疑似选错列或先做了不该做的聚合。 |
| 59 | easy | Distribution Analysis, Summary Statistics, Feature Engineering | Americas 筛选口径错，可能把区域 / 国家字段理解错，导致最高平均 cases 国家错误。 |
| 62 | medium | Outlier Detection, Distribution Analysis | IQR 后对均值的计算口径不一致，可能没有正确保留或剔除 outlier。 |
| 117 | medium | Correlation Analysis | 只比较了常见正相关变量，忽略 Happiness Rank 与 score 的强负相关。 |
| 124 | hard | Summary Statistics, Correlation Analysis | 显著性检验判断错，把不同疫苗组差异判断为 yes，而标准答案为 no。 |
| 125 | hard | Correlation Analysis, Machine Learning | R2 为 0.6060，标准 0.6059；主要是精度 / 四舍五入误差。 |
| 133 | hard | Comprehensive Data Preprocessing | 先算 Age 中位数再删 Cabin 缺失行，顺序错；题目要求删 Cabin 缺失后再对 Age 插补。 |
| 137 | hard | Feature Engineering, Machine Learning | `IsAlone` 特征与模型训练协议不匹配，模型分数 0.80 而标准为 0.61；可能使用了不同 split / class_weight。 |
| 142 | hard | Correlation Analysis | 把近似无关系判断成 nonlinear，并给出显著 p 值；关系类型解释和相关性计算都偏离标准。 |
| 219 | medium | Outlier Detection | 数值本质正确，但格式为 9.00，标准为 9.0；严格字符串格式导致误判。 |
| 252 | medium | Distribution Analysis | skewness 计算对象错，把国家维度或年份维度理解反了。 |
| 271 | hard | Comprehensive Data Preprocessing | 预处理链条中的 log transform 错，可能对 0 / 缺失 / 填充值处理不一致。 |
| 273 | hard | Correlation Analysis, Outlier Detection | outlier list 输出为空，说明 Z-score 检测或格式化阶段失败。 |
| 275 | hard | Comprehensive Data Preprocessing, Feature Engineering, Machine Learning | 多步骤综合题中前置预处理或新特征公式错，导致 new feature mean 偏差。 |
| 297 | hard | Summary Statistics, Comprehensive Data Preprocessing | tree 缺失 / 非缺失分组口径错，两个均值都偏离标准。 |
| 298 | medium | Distribution Analysis | 正态性判断错，把标准答案 yes 判断为 No；可能过度依赖检验或阈值。 |
| 300 | hard | Correlation Analysis, Comprehensive Data Preprocessing | Pearson r 0.53 vs 0.54，属于边界四舍五入误差。 |
| 310 | hard | Correlation Analysis | 数值变量间 strongest correlation 选对方向但系数偏差，疑似缺失值处理或变量选择不同。 |
| 359 | easy | Distribution Analysis | wind speed skewness 算成 1.00，标准 0.83；可能使用了清洗后数据或错误列。 |
| 408 | medium | Correlation Analysis | fare 与 age 的关系类型错，把 nonlinear 判断为 none。 |
| 418 | medium | Outlier Detection | trading volume outlier 数量漏检，IQR / Z-score 方法或阈值与标准不一致。 |
| 431 | hard | Correlation Analysis, Comprehensive Data Preprocessing | high damage 子集相关系数与 p 值错，可能 damage 分组或 storm duration 构造错。 |
| 450 | easy | Summary Statistics | 月均风速数值正确，但 dict 格式空格 / 小数格式与标准不完全一致。 |
| 451 | easy | Comprehensive Data Preprocessing | 缺失值字典数值正确，但格式空格与标准不一致。 |
| 452 | hard | Correlation Analysis, Feature Engineering | p-value 0.676 vs 0.6756，主要是小数位输出不符合格式要求。 |
| 492 | easy | Summary Statistics | 问的是 2010 年最高字段，模型返回了全部字段列表，未执行 argmax。 |
| 496 | hard | Feature Engineering, Summary Statistics | STEM 特征定义或字段集合不一致，mean_STEM 偏差。 |
| 513 | medium | Correlation Analysis, Distribution Analysis | 酒店星级分组和 correlation 子集划分错，两个分组系数都偏离。 |
| 528 | medium | Outlier Detection, Comprehensive Data Preprocessing | outlier id 列表不完整，只输出前半段；可能被输出长度或中间截断影响。 |
| 529 | hard | Correlation Analysis, Feature Engineering | SibSp 与 Parch 关系类型解释错，把 nonlinear 说成 linear。 |
| 550 | hard | Comprehensive Data Preprocessing, Distribution Analysis | 内容正确但字符串带引号要求没有满足，严格格式导致错。 |
| 554 | easy | Summary Statistics, Distribution Analysis | 过滤条件下没有记录，标准为 nan；模型后续又改成了 22.86，属于自我纠偏方向错误。 |
| 572 | hard | Summary Statistics, Correlation Analysis | 日期格式被截断为年月，标准要求完整日期。 |
| 589 | medium | Feature Engineering | abandonment rate 公式错，分母或字段选择错误。 |
| 647 | hard | Feature Engineering, Distribution Analysis | Price Range 特征计算严重偏差，可能没有按 High-Low 或列单位处理错误。 |
| 662 | hard | Feature Engineering, Summary Statistics | Price Change 统计量轻微偏差，属于边界四舍五入 / 数据清洗口径差异。 |
| 684 | medium | Distribution Analysis | Shapiro p-value 未按格式输出，reformat 后缺失。 |
| 722 | hard | Summary Statistics, Comprehensive Data Preprocessing | 最高 horsepower 车辆对应 model year 识别错，可能没有正确处理 horsepower 中的缺失 / 非数值。 |
| 730 | medium | Correlation Analysis | p-value 0.2910 vs 0.2909，属于精度格式误差。 |
| 733 | hard | Feature Engineering | log10 GDP per capita 特征均值和标准差偏差，可能使用自然对数或先做了错误清洗。 |
| 734 | hard | Correlation Analysis, Comprehensive Data Preprocessing | 未按 continent 分别输出全部相关系数，reformat 输出了占位符 `r_value` / `significance`。 |
| 741 | medium | Feature Engineering, Comprehensive Data Preprocessing | 题目要创建 ratio 特征，标准答案字段是特征名 `ratio`，模型输出成了某个数值。 |
| 743 | hard | Comprehensive Data Preprocessing, Feature Engineering | normalization 输出范围和文件路径不符合标准，可能用了错误归一化尺度或保存路径。 |

## 4. 错题类型归因

### 4.1 严格格式与小数精度问题

典型题：7, 125, 219, 300, 450, 451, 452, 550, 662, 730。

这类题的计算结果接近甚至本质正确，但评测脚本只做非常严格的字符串 / 数值比较。比如 `9.0` 和 `9.00`、`0.6756` 和 `0.676`、dict 内空格差异都会被算错。

改进方向：

- 在最终答案前增加格式校验器，根据题目 `format` 自动检查小数位、引号、dict 空格、列表分隔符。
- 对 reformat 阶段加规则模板，避免 LLM 二次格式化时改动数值精度。
- 对数值答案统一保留题目要求的小数位，不要让模型自由决定。

### 4.2 预处理顺序和数据子集口径错误

典型题：133, 271, 275, 297, 431, 528, 554, 722, 743。

这些题的错误来自数据处理链条细节，例如先删缺失再插补，还是先插补再删；按 tree null 分组还是按 notnull 分组；是否正确处理 `?`、空值、非数值列；是否保存到指定路径。

改进方向：

- 把约束条件解析成 checklist，在执行前明确每一步顺序。
- 对每个中间表输出 shape、缺失值数量、关键统计量，防止预处理顺序跑偏。
- 在 final answer 前重新运行一个独立 verification cell，检查关键中间量是否来自正确子集。

### 4.3 统计方法和显著性判断错误

典型题：117, 124, 142, 252, 298, 310, 408, 418, 513, 529。

这类题通常不是代码完全失败，而是分析判断错。例如只看正相关而漏掉绝对值最大的负相关，把无关系解释成非线性，把正常性检验或 outlier 检测阈值用错。

改进方向：

- correlation 题统一按 `abs(r)` 排序，同时保留正负号解释。
- 显著性题固定输出检验方法、统计量、p-value、alpha 阈值和 yes/no 判断。
- outlier 题明确 IQR / Z-score 方法，不允许模型自行换方法。

### 4.4 特征工程定义理解错误

典型题：59, 137, 496, 589, 647, 733, 741。

模型容易把“新特征是什么”与“新特征的某个数值是多少”混在一起，也容易漏掉字段集合定义，例如 STEM 包含哪些列、abandonment rate 的分母是什么、Price Range 是否就是 High-Low。

改进方向：

- 先把新特征公式以代码注释形式写出来，再执行计算。
- 对需要返回特征名的题，不要自动返回第一个样本值。
- 对特征工程题增加“公式回显 + 前 5 行验证 + 聚合值”三步。

### 4.5 机器学习实验协议不一致

典型题：7, 23, 137。

机器学习题对 split、random_state、特征列、模型参数、是否 stratify、是否 class_weight 很敏感。模型一旦自由添加改进项，就会偏离标准答案。

改进方向：

- 严格禁止额外调参，除非题目明确要求。
- 统一固定 `random_state`、split 比例、特征列和 metric。
- 训练前打印 X/y shape 和 feature names，训练后只输出题目指定 metric。

### 4.6 reformat 阶段引入或放大错误

典型题：684, 734。

原始 notebook 里可能算过部分结果，但 reformat 阶段需要把长回答压成 `@name[value]`。如果 reformat 模型没读到明确最终答案，就会漏填或输出占位符。

改进方向：

- 让主 agent 在回答末尾先写一段 machine-readable final block。
- reformat 不应从完整长日志中抽取，而应优先读取 final block。
- 如果答案缺失或仍是占位符，自动回到 notebook 执行阶段补算。

## 5. 对 DatawiseAgent 方法的解释

这些错题反过来说明论文中 notebook-centric 与 FST multi-stage 架构的价值：

- notebook-centric 让每一步计算可追踪，便于定位错在数据读取、预处理、统计检验还是最终格式。
- incremental execution 能把长链条任务拆成可检查的小步骤。
- self-debugging 主要解决代码报错，但对“代码能跑但语义错”的问题帮助有限。
- post-filtering 能减少错误轨迹污染，但如果最终答案提取仍依赖 LLM，格式错误仍会进入评测。

因此，InfiAgentBench 的主要改进空间不是单纯换更强模型，而是加强“约束解析、执行校验、最终格式检查”这三个环节。

## 6. 汇报可用结论

可放 PPT 的简短结论：

- Qwen2.5-32B 在 InfiAgentBench 上 ABQ 为 81.71%，easy / medium 表现较稳，hard 明显下降。
- 错误主要集中在综合预处理、特征工程、机器学习协议和相关性判断。
- 大量错题不是代码无法执行，而是执行口径、题目约束或最终格式不一致。
- 改进方向应从三方面入手：约束结构化、执行过程校验、最终答案格式验证。

