# InfiAgentBench Result Inventory

更新时间：2026-06-09

本文件只清点已有结果，不包含新实验。完整机器可读总表：`evaluation/experimental_results/InfiAgent-Bench/infiagentbench_result_inventory.csv`。

## 指标命名统一

- **ABQ**：Answer by Question，按整题计分；旧文档里的 `Q-Acc` 等同于 ABQ。
- **PASQ**：Proportional Answer by Sub-Question；旧文档里的 `Proportional Sub-Q`。
- **UASQ**：Unweighted / unfolded Answer by Sub-Question；旧文档里的 `Sub-Q`。
- 本清单所有表格统一使用 `ABQ / PASQ / UASQ`。

## 可信度分级

- **high**：eval 完整，路径清楚，任务数量与难度分组一致，设置可解释。
- **medium**：结果可读，但有 small set、summary-only、raw/reformat 不完整或行数轻微不一致等限制。
- **low**：missing_eval、旧版本被覆盖、partial、格式混乱或只能当计划/预期。

## 一句话结论

- 论文 Qwen2.5-72B 和 Qwen3.5-35B-A3B baseline 是最稳的第一组对照。
- Harness 在 **medium** 上有可靠提升：Qwen3.5-35B-A3B medium ABQ `88.51% -> 91.95%`。
- Harness 在 **hard** 上没有可靠提升，反而下降：ABQ `81.82% -> 78.41%`。
- qwen3.5-flash small_error_set 显示 weak + harness 有趋势收益，但它只是 47 题 known-error 小集。
- Resource-aware / equivalent token 目前主要是计划；只有 Qwen3-32B baseline 是 observed/reference，路由节省是 expected/target。

## 1. 论文参考结果

| run_name | model | difficulty | task_count | ABQ | PASQ | UASQ | reliability | eval_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_qwen25_72b_full | Qwen2.5-72B | full | 257 | 81.71% | 87.27% | 85.09% | high | evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_paper_qwen25_72b_full_evaluation_analysis.json |
| paper_qwen25_72b_easy | Qwen2.5-72B | easy | 82 | 95.12% | 96.34% | 96.04% | high | evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_paper_qwen25_72b_easy_evaluation_analysis.json |
| paper_qwen25_72b_medium | Qwen2.5-72B | medium | 87 | 81.61% | 86.73% | 86.18% | high | evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_paper_qwen25_72b_medium_evaluation_analysis.json |
| paper_qwen25_72b_hard | Qwen2.5-72B | hard | 88 | 69.32% | 79.36% | 78.82% | high | evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_paper_qwen25_72b_hard_evaluation_analysis.json |

建议：可直接汇报。它是论文参考或 paper baseline 复现结果。

## 2. 新模型 baseline

### Qwen3.5-35B-A3B

| run_name | difficulty | task_count | ABQ | PASQ | UASQ | reliability | eval_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35b_a3b_full | full | 257 | 86.38% | 90.16% | 90.35% | high | evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_qwen35b_a3b_full_evaluation_analysis.json |
| qwen35b_a3b_easy | easy | 82 | 89.02% | 90.24% | 91.09% | high | evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_qwen35b_a3b_easy_evaluation_analysis.json |
| qwen35b_a3b_medium | medium | 87 | 88.51% | 91.71% | 91.45% | high | evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_qwen35b_a3b_medium_evaluation_analysis.json |
| qwen35b_a3b_hard | hard | 88 | 81.82% | 88.54% | 89.16% | high | evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_qwen35b_a3b_hard_evaluation_analysis.json |

建议：可直接汇报。与 paper Qwen2.5-72B 对比时，full/medium/hard 更强，easy 更低。

### Qwen3-32B / qwen32b

| run_name | difficulty | task_count | ABQ | PASQ | UASQ | reliability | presentation_tier | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen32b_full | full | 257 | 81.74% | 86.21% | 86.43% | medium | cautious_report | Qwen3-32B full latest；cost plan 使用此结果作 resource-aware baseline；未找到匹配 full raw/reformat 文件。 |
| qwen32b_easy | easy | 82 | 91.46% | 92.07% | 92.08% | high | cautious_report | Qwen32B 分难度结果。 |
| qwen32b_medium | medium | 86 | 83.72% | 86.19% | 86.00% | medium | internal_only | Qwen32B 分难度结果。 reformat 行数 86 与预期 87 不一致。 |
| qwen32b_hard | hard | 64 | 66.13% | 78.49% | 82.99% | medium | internal_only | Qwen32B 分难度结果。 reformat 行数 64 与预期 88 不一致。 |

建议：full latest 可作为 resource-aware cost plan 的 observed baseline 谨慎引用；medium/hard 分难度存在行数不一致，内部使用更合适。

## 3. Harness 结果

### Rerender / final-block harness

| run_name | model | setting | difficulty | task_count | ABQ | PASQ | UASQ | reliability | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| harness_round2_rerender_qwen35b_medium | Qwen3.5-35B-A3B | harness fullfix raw + round2 rerender final-block | medium | 87 | 91.95% | 93.06% | 93.42% | high | raw 来自 fullfix harness；reformat/eval 使用 round2 rerender/final-block 版本；avg_branch_count=1.023; format_error=0; reformat_missing=0. 相对 qwen35b medium baseline 有提升。 |
| harness_round2_rerender_qwen35b_hard | Qwen3.5-35B-A3B | harness fullfix raw + round2 rerender final-block | hard | 88 | 78.41% | 84.75% | 87.19% | high | raw 来自 fullfix harness；reformat/eval 使用 round2 rerender/final-block 版本；avg_branch_count=2.9659; format_error=0; reformat_missing=0. 相对 qwen35b hard baseline 下降，主要作负例/风险说明。 |

说明：这组 raw 来自 `results_harness_fullfix_qwen35b_*_final.jsonl`，reformat/eval 使用 `results_reformat_harness_round2_rerender_qwen35b_*`。不要和 memory learning 或 weak small set 混在一起。

### Memory learning

| run_name | setting | difficulty | task_count | ABQ | PASQ | UASQ | runtime | reliability | presentation_tier | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| harness_memory_r1_writeonly_qwen35b_full | Round1 write-only | full | 257 | 87.16% | 88.93% | 90.35% | 68.0676 | medium | internal_only | 全量 memory-learning 链路可运行；format_error=4; reformat_missing=4; validator_failed=18; 平均每题约 68.0676s。收益很小/指标混合，不建议作为主结果。 |
| harness_memory_r2_readmem_qwen35b_full | Round2 read-memory | full | 257 | 86.77% | 88.60% | 90.79% | 64.4631 | medium | internal_only | 全量 memory-learning 链路可运行；format_error=2; reformat_missing=2; validator_failed=11; 平均每题约 64.4631s。收益很小/指标混合，不建议作为主结果。 |

建议：不作为主 PPT 结果。可以在问答或附录里说“链路可跑，但收益小且有负迁移风险”。

## 4. Weak model small set

### 最新 lightfix 47/47 版本

| run_name | setting | difficulty | task_count | ABQ | PASQ | UASQ | runtime | reliability | presentation_tier | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weak_qwen35_flash_small_lightfix_baseline_raw | baseline_raw | small_error_set | 47 | 0.00% | 0.00% | 0.00% | 30.2419 | low | do_not_show | qwen3.5-flash small_error_set 47题 lightfix；status={'success': 47}; format_error=47; reformat_missing=47; verifier_blocked=0; stuck=0. |
| weak_qwen35_flash_small_lightfix_baseline_reformat | baseline_reformat | small_error_set | 47 | 48.94% | 63.83% | 67.05% | 30.2419 | medium | cautious_report | qwen3.5-flash small_error_set 47题 lightfix；status={'success': 47}; format_error=0; reformat_missing=0; verifier_blocked=0; stuck=0. |
| weak_qwen35_flash_small_lightfix_harness_light | harness_light | small_error_set | 47 | 59.57% | 65.25% | 69.32% | 47.0341 | medium | cautious_report | qwen3.5-flash small_error_set 47题 lightfix；status={'success': 44, 'reformat_missing': 2, 'execution_error': 1}; format_error=1; reformat_missing=3; verifier_blocked=0; stuck=1. |
| weak_qwen35_flash_small_lightfix_harness_full | harness_full | small_error_set | 47 | 63.83% | 69.50% | 72.73% | 92.7897 | medium | cautious_report | qwen3.5-flash small_error_set 47题 lightfix；status={'success': 41, 'verifier_blocked': 4, 'reformat_missing': 1, 'execution_error': 1}; format_error=0; reformat_missing=2; verifier_blocked=4; stuck=1. |

固定/退化信息来自 `weak_model_comparison/small-qwen35-flash-lightfix-20260609/baseline_light_full_per_task.md`：

- `harness_light` fixed：57, 125, 271, 273, 450, 513, 528, 647。
- `harness_light` regressed：62, 451, 684。
- `harness_full` fixed：57, 125, 133, 271, 273, 513, 528。
- `harness_full` regressed：408。
- `harness_full` verifier_blocked：219, 310, 550, 734。
- `harness_full` stuck / execution_error：647。

建议：可谨慎汇报，必须写明这是 `small_error_set` 47 题，不是 full 257。

### 旧 small set / superseded

| run_name | setting | difficulty | task_count | ABQ | PASQ | UASQ | runtime | reliability | presentation_tier | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weak_qwen35_flash_small_v2_baseline_raw | baseline_raw | small_error_set_partial | 47 | 0.00% | 0.00% | 0.00% | 30.2469 | low | internal_only | 旧 small set 结果，已被 lightfix 47/47 完整版本覆盖；evaluated_count=44; task_status={'success': 44, 'execution_error': 3}. 与 lightfix baseline 指标冲突，不应混用。 |
| weak_qwen35_flash_small_v2_baseline_original_reformat | baseline_original_reformat | small_error_set_partial | 47 | 59.57% | 69.86% | 72.73% | 30.2469 | low | internal_only | 旧 small set 结果，已被 lightfix 47/47 完整版本覆盖；evaluated_count=47; task_status={'success': 44, 'execution_error': 3}. 与 lightfix baseline 指标冲突，不应混用。 |
| weak_qwen35_flash_small_v2_harness_full_final_block | harness_full_final_block | small_error_set_partial | 46 | 60.87% | 68.12% | 72.94% | 103.604 | low | internal_only | 旧 small set 结果，已被 lightfix 47/47 完整版本覆盖；evaluated_count=46; task_status={'success': 39, 'verifier_blocked': 4, 'execution_error': 2, 'reformat_missing': 1}. 与 lightfix baseline 指标冲突，不应混用。 |

建议：不建议展示。旧 v2 与 lightfix baseline 指标冲突，且 harness_full 只有 46 个 response；只适合解释实验迭代过程。

## 5. Cost / resource-aware

| run_name | model | setting | difficulty | task_count | ABQ | PASQ | UASQ | reliability | presentation_tier | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resource_qwen3_32b_full_baseline | Qwen3-32B | resource baseline observed | full | 257 | 81.74% | 86.21% | 86.43% | medium | cautious_report | observed/reference in cost plan; equivalent token 5.961M; eval file also exists as results_reformat_qwen32b_full_latest_evaluation_analysis.json but matching raw/reformat full path was not found. |
| resource_model_routing_expected | mixed 32B/14B/8B | model routing only expected | full | 257 | 80.5%-82.5% |  |  | low | cautious_report_as_plan | expected/target only; equivalent token about 2.120M; not an observed result. |
| resource_routing_conversation_reduction_expected | mixed 32B/14B/8B + deterministic | model routing + conversation reduction expected | full | 257 | 81.0%-83.5% |  |  | low | cautious_report_as_plan | expected/target only; equivalent token about 1.05M-1.60M; not an observed result. |

建议：

- `Qwen3-32B full baseline` 是 observed/reference，可作为成本计划基准谨慎引用。
- `model routing` 和 `routing + conversation reduction` 是 expected / target，不是结果。

## 6. missing_eval / 不建议展示

| run_name | model | difficulty | task_count | reliability | presentation_tier | reformat_path | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| results_datawise-Qwen2.5-14B-Instruct-temperature-0-args-7-6-8 | datawise-Qwen2.5-14B-Instruct-temperature-0-args-7-6-8 | full | 257 | low | do_not_show | evaluation/experimental_results/InfiAgent-Bench/reformat/results_datawise-Qwen2.5-14B-Instruct-temperature-0-args-7-6-8.jsonl | Found reformat jsonl but no matching eval_outputs JSON in requested locations; do not present as result. |
| results_datawise-Qwen2.5-7B-Instruct-temperature-0-args-7-6-8 | datawise-Qwen2.5-7B-Instruct-temperature-0-args-7-6-8 | full | 257 | low | do_not_show | evaluation/experimental_results/InfiAgent-Bench/reformat/results_datawise-Qwen2.5-7B-Instruct-temperature-0-args-7-6-8.jsonl | Found reformat jsonl but no matching eval_outputs JSON in requested locations; do not present as result. |
| results_reformat_datawise-gpt-4o-temperature-0-args-7-6-8 | datawise-gpt-4o-temperature-0-args-7-6-8 | full | 257 | low | do_not_show | evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_datawise-gpt-4o-temperature-0-args-7-6-8.jsonl | Found reformat jsonl but no matching eval_outputs JSON in requested locations; do not present as result. |
| results_reformat_datawise-qwen25-70B-temperature-0-args-7-6-8 | datawise-qwen25-70B-temperature-0-args-7-6-8 | full | 257 | low | do_not_show | evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_datawise-qwen25-70B-temperature-0-args-7-6-8.jsonl | Found reformat jsonl but no matching eval_outputs JSON in requested locations; do not present as result. |

这些文件有 raw/reformat，但没有在指定 `eval_outputs` 下找到对应 eval JSON。不要把它们当指标展示。

## 7. 冲突和注意事项

1. **weak small set baseline 冲突**：旧 `small-qwen35-flash-20260609` 中 baseline original reformat ABQ 是 `59.57%`，但最新 lightfix 47/47 版本 baseline reformat ABQ 是 `48.94%`。应使用 lightfix 版本，旧版本标为 superseded。
2. **旧 harness_full small set 不完整**：旧版本只有 46 个 response，task 733 曾因为 FinalAnswerBlock JSON 进入 Python code cell 卡死；不要与最新 47/47 lightfix 混用。
3. **Qwen32B 分难度行数不一致**：medium reformat 86 条，hard reformat 64 条，不适合直接和 87/88 题完整难度集做结论。
4. **Qwen3-32B full latest raw/reformat 路径不完整**：eval 文件和 cost plan 数值存在，但未找到匹配 full raw/reformat 文件，因此作为 cost reference 谨慎引用。
5. **memory learning 指标混合**：Round1 ABQ 略高于 baseline，但 PASQ 下降；Round2 UASQ 略升但 ABQ/PASQ 不稳定。不要把 memory 讲成稳定提升。

## 8. 特别判断

### InfiAgentBench harness 到底有没有可靠提升？

有，但要限定范围：**medium 难度上可靠提升**。Qwen3.5-35B-A3B medium baseline ABQ `88.51%`，harness rerender ABQ `91.95%`。hard 上没有可靠提升，ABQ 从 `81.82%` 降到 `78.41%`。

### 提升主要出现在 medium / hard / small error set 哪一类？

- 最可靠提升：medium。
- 小集趋势：weak model small_error_set，baseline reformat `48.94%`，light `59.57%`，full `63.83%`。
- hard：全量强模型 hard harness 下降；weak small set hard ABQ 上升，但只是 known-error 小集，不能外推。

### hard 题为什么有些结果下降？

hard 题问题更像分支选择和语义校验不足：多候选不一定能选对，oracle/contract/search 可能引入过滤口径、列选择、边界条件或单位处理漂移。small set task 408 是典型例子，full harness 相对 baseline 退化。

### memory learning 是否值得讲？

不建议作为主结果。它证明链路能跑，round2 每题都有 memory hits，但收益很小且有回落。适合作为内部观察或后续“memory gate”动机。

### weak model small set 是否值得放进正式 PPT？

值得，但要放在“趋势/动机”页，不要和 full 257 主结果并列。推荐使用最新 lightfix 47/47 版本，并明确 `known-error small set`。

### cost reduction 应该作为结果还是预期？

Qwen3-32B baseline 是 observed/reference；equivalent token 路由节省是 expected/target。PPT 中必须写 `not yet validated`。
