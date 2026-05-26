# InfiAgentBench 改进版 DatawiseAgent

## 方法动机

InfiAgentBench 的主要失败点不是代码无法执行，而是自由探索式 notebook agent 容易遗漏题目约束、使用错误统计口径，并在 reformat 阶段从长日志中错抽答案。本仓库新增的改进版采用可插拔流程：

```text
schema_profile -> question_contract -> skill_plan -> notebook execution -> verifier_report -> final_block reformat
```

目标是把 DatawiseAgent 增强为任务感知、约束结构化、过程可校验、答案格式可验证的 InfiAgentBench 专用流程，同时保留原始 baseline 入口。

## 仓库入口

- 原始 InfiAgentBench 运行入口：`evaluation/eval_infiagent_bench.py`
- 原始 reformat 脚本：`evaluation/InfiAgentBench/scripts/reformat.py`
- 原始评估脚本：`evaluation/InfiAgentBench/scripts/eval_closed_form.py`
- 问题文件：`evaluation/InfiAgentBench/data/da-dev-questions.jsonl`
- 标签文件：`evaluation/InfiAgentBench/data/da-dev-labels.jsonl`
- 表格目录：`evaluation/InfiAgentBench/data/da-dev-tables/`
- 原始实验结果目录：`evaluation/experimental_results/InfiAgent-Bench/`

新增模块都放在 `agent_extensions/infiagentbench/`，新增脚本放在 `scripts/`，默认输出到 `runs/infiagent_improve/`。

## 模块结构

- `schema_profile.py`：固定读取 CSV，生成 `schema_profile.json`，包含 shape、columns、dtypes、missing、numeric summary、unique counts、datetime/category/id candidates、normalized column map。
- `question_contract.py`：规则优先解析题目，生成 `question_contract.json`，包含 answer_names、required_columns、filters、groupby、statistics、methods、rounding、format、forbidden_actions、task_types。
- `skills.py` / `skill_router.py`：根据 task_types 输出 `skill_plan.json`，每个 skill 包含 checklist 和 verification_rules。
- `verifier.py`：独立检查执行日志和 final block，输出 `verifier_report.json`，覆盖列、筛选、预处理顺序、相关性 abs(r)、p-value、异常值、特征公式、ML 协议、最终格式。
- `final_block.py`：只从 `FINAL_BLOCK` JSON 转换为 `@answer_name[value]`，缺失、占位符或格式错误时返回 `ERROR_NEED_RETRY`。
- `error_taxonomy.py`：统一错误类型统计。
- `prompt_builder.py`：为改进版运行脚本构造可插拔 prompt，不修改原始 DatawiseAgent 主流程。

## 运行方式

生成单题 dry-run 工件，不调用 DatawiseAgent 服务：

```bash
python3 scripts/run_infiagent_improve.py \
  --dry-run \
  --task-id 5 \
  --ablation full \
  --note smoke
```

在已有 DatawiseAgent 服务运行时执行单题完整流程：

```bash
python3 scripts/run_infiagent_improve.py \
  --task-id 5 \
  --ablation full \
  --note full-single
```

对已有 baseline 响应做 final block reformat：

```bash
python3 scripts/reformat_infiagent_final_block.py \
  --responses-file runs/infiagent_improve/full-single/full/predictions/results_full-single_full.jsonl \
  --output-file runs/infiagent_improve/full-single/full/predictions/results_full-single_full_reformat.jsonl
```

评估改进版输出并统计错误类型：

```bash
python3 scripts/eval_infiagent_improve.py \
  --responses-file runs/infiagent_improve/full-single/full/predictions/results_full-single_full.jsonl \
  --output runs/infiagent_improve/full-single/full/eval.json
```

比较多个消融实验：

```bash
python3 scripts/compare_infiagent_ablation.py \
  --eval-files runs/infiagent_improve/*/*/eval.json \
  --output runs/infiagent_improve/ablations/ablation_compare.json
```

抽取典型错误案例：

```bash
python3 scripts/analyze_infiagent_errors.py \
  --eval-report runs/infiagent_improve/full-single/full/eval.json \
  --output runs/infiagent_improve/full-single/full/error_analysis.json
```

## 消融选项

`scripts/run_infiagent_improve.py --ablation` 支持：

- `baseline`：原始 prompt，不加新模块
- `schema`：增加 schema profiler
- `contract`：增加 schema profiler 和 contract parser
- `router`：增加 skill router
- `skill`：`router` 的别名，增加 skill router
- `verifier`：增加 execution verifier
- `final_formatter`：增加 final block 和格式检查
- `full`：启用全部模块

审计和小集 dry-run 可使用：

```bash
python3 scripts/audit_infiagent_dry_run.py \
  --task-ids-file runs/infiagent_improve/small_error_set.json \
  --note small-error-dry-run
```

## 输出文件

每题默认输出到：

```text
runs/infiagent_improve/{note}/{ablation}/tasks/{task_id}/
  schema_profile.json
  question_contract.json
  skill_plan.json
  agent_prompt.json
  final_block.json
  verifier_report.json
```

预测文件输出到：

```text
runs/infiagent_improve/{note}/{ablation}/predictions/results_{note}_{ablation}.jsonl
```

## 与原始 DatawiseAgent 的区别

原始流程依赖 agent 在 notebook 中自由探索数据，再由 LLM reformat 从长日志中抽取答案。新增流程在执行前固定生成 schema 和 contract，在执行时通过 skill checklist 约束关键步骤，在执行后通过 verifier 和 final block 检查答案格式。原始文件不删除、不替换；改进流程通过 `scripts/run_infiagent_improve.py` 可插拔启用。
