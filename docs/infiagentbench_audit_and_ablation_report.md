# InfiAgentBench 改进版审计与消融报告

## 审计结论

当前改进版已经以可插拔方式接入 InfiAgentBench 运行链路，入口是 `scripts/run_infiagent_improve.py`。原始 DatawiseAgent 主流程未被修改，原始入口 `evaluation/eval_infiagent_bench.py`、`evaluation/InfiAgentBench/scripts/reformat.py` 和 `evaluation/InfiAgentBench/scripts/eval_closed_form.py` 仍可独立运行。

代码审计确认：

- `schema_profile`：由 `run_infiagent_improve.py` 在启用 schema 及其后续模块时调用 `profile_csv()`，并写入每题 `schema_profile.json`。
- `question_contract`：由 `parse_question_contract()` 解析 question / constraints / format，并写入 `question_contract.json`。
- `skill_router`：`skill` 与 `router` 消融都会调用 `route_skills()`，并写入 `skill_plan.json`。
- `final_block`：不是只写在 prompt 里。启用 `final_formatter` 或 `full` 时，运行脚本会从 raw response 提取 `FINAL_BLOCK`，调用 `final_block_to_response()`，并把最终 `response` 改成 `@answer[value]` 或 `ERROR_NEED_RETRY`。`scripts/reformat_infiagent_final_block.py` 也只读取 final block，不从长日志猜答案。
- `verifier`：启用 `verifier` 或 `full` 时会写入 `verifier_report.json`。已修正为当 `must_retry=true` 时把最终 response 阻断为 `ERROR_NEED_RETRY`，避免 verifier fail 后仍假装成功。
- `baseline`：`baseline` 消融只构造原始 prompt，不生成 schema/contract/skill/final/verifier 工件，不影响原始 baseline 运行方式。
- 数据泄露：运行链路只读取 questions 文件和 CSV 表格；标准答案只在 `scripts/eval_infiagent_improve.py` 的评估阶段读取，不进入 prompt、contract、skill、verifier 或 final block 生成。

限制：

- 当前 verifier 主要是独立日志和结构检查，尚未对所有统计结果做完全复算。
- final block 依赖 agent 按 prompt 输出 `FINAL_BLOCK`，没有侵入 DatawiseAgent 内部 notebook 执行循环。
- contract parser 是规则优先版本，复杂自然语言仍可能需要后续 LLM fallback 或人工 oracle routing 对照。

## 已知错题小集

已创建小集：

```text
runs/infiagent_improve/small_error_set.json
```

共 47 题：

```text
7, 23, 55, 56, 57, 59, 62, 117, 124, 125, 133, 137, 142, 219, 252, 271, 273, 275, 297, 298, 300, 310, 359, 408, 418, 431, 450, 451, 452, 492, 496, 513, 528, 529, 550, 554, 572, 589, 647, 662, 684, 722, 730, 733, 734, 741, 743
```

原始错误类型分布：

| error_type | count |
| --- | --- |
| format_error | 10 |
| statistical_method_error | 8 |
| feature_definition_error | 6 |
| filtering_error | 5 |
| outlier_protocol_error | 4 |
| preprocessing_order_error | 4 |
| ml_protocol_error | 3 |
| aggregation_error | 2 |
| significance_judgment_error | 2 |
| reformat_error | 2 |
| column_selection_error | 1 |

## Dry-Run 消融

运行命令：

```bash
python3 scripts/audit_infiagent_dry_run.py \
  --task-ids-file runs/infiagent_improve/small_error_set.json \
  --note small-error-dry-run \
  --output runs/infiagent_improve/small_error_dry_run_audit.json
```

结果：7 个消融、47 道题全部完成 dry-run。预期工件缺失数均为 0。

| ablation | missing_expected_total | schema_profile.json | question_contract.json | skill_plan.json | verifier_report.json | final_block.json | agent_prompt.json |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| schema | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| contract | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| skill | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| verifier | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| final_formatter | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| full | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Dry-run 输出目录：

```text
runs/infiagent_improve/small-error-dry-run/
```

## 小集真实实验

未运行真实小集实验。原因：本地 `127.0.0.1:8000` DatawiseAgent 服务不可用，端口检查结果为 unavailable。没有编造 ABQ / PASQ / UASQ。

服务可用后可直接运行：

```bash
for ablation in baseline schema contract skill verifier final_formatter full; do
  python3 scripts/run_infiagent_improve.py \
    --task-ids-file runs/infiagent_improve/small_error_set.json \
    --ablation "$ablation" \
    --note small-error-live

  python3 scripts/eval_infiagent_improve.py \
    --responses-file "runs/infiagent_improve/small-error-live/${ablation}/predictions/results_small-error-live_${ablation}.jsonl" \
    --output "runs/infiagent_improve/small-error-live/${ablation}/eval.json"
done

python3 scripts/compare_infiagent_ablation.py \
  --eval-files runs/infiagent_improve/small-error-live/*/eval.json \
  --output runs/infiagent_improve/small-error-live/ablation_compare.json

python3 scripts/attribute_infiagent_fixes.py \
  --small-error-set runs/infiagent_improve/small_error_set.json \
  --baseline-eval runs/infiagent_improve/small-error-live/baseline/eval.json \
  --full-eval runs/infiagent_improve/small-error-live/full/eval.json \
  --output runs/infiagent_improve/small-error-live/fix_attribution.json
```

## 修复归因

已生成归因表：

```text
runs/infiagent_improve/small_error_fix_attribution.json
```

由于真实 baseline/full 小集实验未运行，当前每题状态均为 `not_run`，不能声称任何错题已经被修复。预期模块归因如下：

| task_id | original_error_type | module |
| --- | --- | --- |
| 7 | ml_protocol_error | contract/skill/verifier |
| 23 | ml_protocol_error | contract/skill/verifier |
| 55 | aggregation_error | contract/skill |
| 56 | column_selection_error | schema/contract |
| 57 | statistical_method_error | contract/skill/verifier |
| 59 | filtering_error | contract/verifier |
| 62 | outlier_protocol_error | contract/skill/verifier |
| 117 | statistical_method_error | contract/skill/verifier |
| 124 | significance_judgment_error | contract/skill/verifier |
| 125 | format_error | final_formatter |
| 133 | preprocessing_order_error | contract/skill/verifier |
| 137 | ml_protocol_error | contract/skill/verifier |
| 142 | statistical_method_error | contract/skill/verifier |
| 219 | format_error | final_formatter |
| 252 | statistical_method_error | contract/skill/verifier |
| 271 | preprocessing_order_error | contract/skill/verifier |
| 273 | outlier_protocol_error | contract/skill/verifier |
| 275 | feature_definition_error | contract/skill/verifier |
| 297 | filtering_error | contract/verifier |
| 298 | significance_judgment_error | contract/skill/verifier |
| 300 | format_error | final_formatter |
| 310 | statistical_method_error | contract/skill/verifier |
| 359 | statistical_method_error | contract/skill/verifier |
| 408 | statistical_method_error | contract/skill/verifier |
| 418 | outlier_protocol_error | contract/skill/verifier |
| 431 | filtering_error | contract/verifier |
| 450 | format_error | final_formatter |
| 451 | format_error | final_formatter |
| 452 | format_error | final_formatter |
| 492 | aggregation_error | contract/skill |
| 496 | feature_definition_error | contract/skill/verifier |
| 513 | filtering_error | contract/verifier |
| 528 | outlier_protocol_error | contract/skill/verifier |
| 529 | statistical_method_error | contract/skill/verifier |
| 550 | format_error | final_formatter |
| 554 | filtering_error | contract/verifier |
| 572 | format_error | final_formatter |
| 589 | feature_definition_error | contract/skill/verifier |
| 647 | feature_definition_error | contract/skill/verifier |
| 662 | format_error | final_formatter |
| 684 | reformat_error | final_formatter |
| 722 | preprocessing_order_error | contract/skill/verifier |
| 730 | format_error | final_formatter |
| 733 | feature_definition_error | contract/skill/verifier |
| 734 | reformat_error | final_formatter |
| 741 | feature_definition_error | contract/skill/verifier |
| 743 | preprocessing_order_error | contract/skill/verifier |

## 已通过测试

```bash
python3 -m pytest tests/infiagentbench -q
# 6 passed

python3 -m py_compile \
  agent_extensions/infiagentbench/*.py \
  scripts/run_infiagent_improve.py \
  scripts/reformat_infiagent_final_block.py \
  scripts/eval_infiagent_improve.py \
  scripts/compare_infiagent_ablation.py \
  scripts/analyze_infiagent_errors.py \
  scripts/audit_infiagent_dry_run.py \
  scripts/attribute_infiagent_fixes.py
# passed
```

## 后续最值得继续改的 3 点

1. 加入独立复算型 verifier：对相关性、p-value、IQR/Z-score、ML split/metric 做结构化复算，而不是只依赖执行日志。
2. 增强 contract parser：对复杂筛选、groupby、日期、保存文件路径和“返回特征名而不是数值”加入更精确规则，并增加 LLM fallback。
3. 跑真实 small_error_set 后做针对性闭环：只对 still_failed 的错误类型扩 verifier/skill，不直接上 257 全量。

