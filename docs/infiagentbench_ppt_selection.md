# InfiAgentBench PPT Result Selection

更新时间：2026-06-09

本文件只回答“哪些结果适合放进 PPT、怎么讲”。完整清单见：

- `docs/infiagentbench_result_inventory.md`
- `evaluation/experimental_results/InfiAgent-Bench/infiagentbench_result_inventory.csv`

## 可直接汇报结果清单

| run_name | model | setting | difficulty | task_count | ABQ | PASQ | UASQ | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_qwen25_72b_full | Qwen2.5-72B | paper baseline / reformat | full | 257 | 81.71% | 87.27% | 85.09% | 论文/参考 baseline；Q-Acc 在本表统一记为 ABQ。 |
| paper_qwen25_72b_easy | Qwen2.5-72B | paper baseline / reformat | easy | 82 | 95.12% | 96.34% | 96.04% | 论文/参考 baseline；Q-Acc 在本表统一记为 ABQ。 |
| paper_qwen25_72b_medium | Qwen2.5-72B | paper baseline / reformat | medium | 87 | 81.61% | 86.73% | 86.18% | 论文/参考 baseline；Q-Acc 在本表统一记为 ABQ。 |
| paper_qwen25_72b_hard | Qwen2.5-72B | paper baseline / reformat | hard | 88 | 69.32% | 79.36% | 78.82% | 论文/参考 baseline；Q-Acc 在本表统一记为 ABQ。 |
| qwen35b_a3b_full | Qwen3.5-35B-A3B | baseline + reformat | full | 257 | 86.38% | 90.16% | 90.35% | 新模型 baseline；full raw 未在根目录找到，但 reformat/eval 完整。 |
| qwen35b_a3b_easy | Qwen3.5-35B-A3B | baseline + reformat | easy | 82 | 89.02% | 90.24% | 91.09% | 新模型 baseline；raw/reformat/eval 路径清楚。 |
| qwen35b_a3b_medium | Qwen3.5-35B-A3B | baseline + reformat | medium | 87 | 88.51% | 91.71% | 91.45% | 新模型 baseline；raw/reformat/eval 路径清楚。 |
| qwen35b_a3b_hard | Qwen3.5-35B-A3B | baseline + reformat | hard | 88 | 81.82% | 88.54% | 89.16% | 新模型 baseline；raw/reformat/eval 路径清楚。 |
| harness_round2_rerender_qwen35b_medium | Qwen3.5-35B-A3B | harness fullfix raw + round2 rerender final-block | medium | 87 | 91.95% | 93.06% | 93.42% | raw 来自 fullfix harness；reformat/eval 使用 round2 rerender/final-block 版本；avg_branch_count=1.023; format_error=0; reformat_missing=0. 相对 qwen35b medium baseline 有提升。 |
| harness_round2_rerender_qwen35b_hard | Qwen3.5-35B-A3B | harness fullfix raw + round2 rerender final-block | hard | 88 | 78.41% | 84.75% | 87.19% | raw 来自 fullfix harness；reformat/eval 使用 round2 rerender/final-block 版本；avg_branch_count=2.9659; format_error=0; reformat_missing=0. 相对 qwen35b hard baseline 下降，主要作负例/风险说明。 |

这些结果有完整 eval、路径清楚、设置清楚。适合作为 PPT 第一部分的主表。

## 可谨慎汇报结果清单

| run_name | model | setting | difficulty | task_count | ABQ | PASQ | UASQ | runtime | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen32b_full | Qwen3-32B / qwen32b | baseline + reformat | full | 257 | 81.74% | 86.21% | 86.43% |  | Qwen3-32B full latest；cost plan 使用此结果作 resource-aware baseline；未找到匹配 full raw/reformat 文件。 |
| qwen32b_easy | Qwen3-32B / qwen32b | baseline + reformat | easy | 82 | 91.46% | 92.07% | 92.08% |  | Qwen32B 分难度结果。 |
| weak_qwen35_flash_small_lightfix_baseline_reformat | qwen3.5-flash | baseline_reformat | small_error_set | 47 | 48.94% | 63.83% | 67.05% | 30.2419 | qwen3.5-flash small_error_set 47题 lightfix；status={'success': 47}; format_error=0; reformat_missing=0; verifier_blocked=0; stuck=0. |
| weak_qwen35_flash_small_lightfix_harness_light | qwen3.5-flash | harness_light | small_error_set | 47 | 59.57% | 65.25% | 69.32% | 47.0341 | qwen3.5-flash small_error_set 47题 lightfix；status={'success': 44, 'reformat_missing': 2, 'execution_error': 1}; format_error=1; reformat_missing=3; verifier_blocked=0; stuck=1. |
| weak_qwen35_flash_small_lightfix_harness_full | qwen3.5-flash | harness_full | small_error_set | 47 | 63.83% | 69.50% | 72.73% | 92.7897 | qwen3.5-flash small_error_set 47题 lightfix；status={'success': 41, 'verifier_blocked': 4, 'reformat_missing': 1, 'execution_error': 1}; format_error=0; reformat_missing=2; verifier_blocked=4; stuck=1. |
| resource_qwen3_32b_full_baseline | Qwen3-32B | resource baseline observed | full | 257 | 81.74% | 86.21% | 86.43% |  | observed/reference in cost plan; equivalent token 5.961M; eval file also exists as results_reformat_qwen32b_full_latest_evaluation_analysis.json but matching raw/reformat full path was not found. |
| resource_model_routing_expected | mixed 32B/14B/8B | model routing only expected | full | 257 | 80.5%-82.5% |  |  |  | expected/target only; equivalent token about 2.120M; not an observed result. |
| resource_routing_conversation_reduction_expected | mixed 32B/14B/8B + deterministic | model routing + conversation reduction expected | full | 257 | 81.0%-83.5% |  |  |  | expected/target only; equivalent token about 1.05M-1.60M; not an observed result. |

这些结果适合当趋势、动机或计划，不要写成已完成主结论。

## 不建议展示结果清单

| run_name | model | setting | difficulty | task_count | ABQ | PASQ | UASQ | presentation_tier | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen32b_medium | Qwen3-32B / qwen32b | baseline + reformat | medium | 86 | 83.72% | 86.19% | 86.00% | internal_only | Qwen32B 分难度结果。 reformat 行数 86 与预期 87 不一致。 |
| qwen32b_hard | Qwen3-32B / qwen32b | baseline + reformat | hard | 64 | 66.13% | 78.49% | 82.99% | internal_only | Qwen32B 分难度结果。 reformat 行数 64 与预期 88 不一致。 |
| datawise_qwen25_32b_full | Qwen2.5-32B-Instruct | baseline + reformat | full | 257 | 81.71% | 86.25% | 87.06% | internal_only | 额外旧模型 baseline；不在当前 PPT 主线中。 |
| harness_memory_r1_writeonly_qwen35b_full | Qwen3.5-35B-A3B | Round1 write-only | full | 257 | 87.16% | 88.93% | 90.35% | internal_only | 全量 memory-learning 链路可运行；format_error=4; reformat_missing=4; validator_failed=18; 平均每题约 68.0676s。收益很小/指标混合，不建议作为主结果。 |
| harness_memory_r2_readmem_qwen35b_full | Qwen3.5-35B-A3B | Round2 read-memory | full | 257 | 86.77% | 88.60% | 90.79% | internal_only | 全量 memory-learning 链路可运行；format_error=2; reformat_missing=2; validator_failed=11; 平均每题约 64.4631s。收益很小/指标混合，不建议作为主结果。 |
| weak_qwen35_flash_small_v2_baseline_raw | qwen3.5-flash | baseline_raw | small_error_set_partial | 47 | 0.00% | 0.00% | 0.00% | internal_only | 旧 small set 结果，已被 lightfix 47/47 完整版本覆盖；evaluated_count=44; task_status={'success': 44, 'execution_error': 3}. 与 lightfix baseline 指标冲突，不应混用。 |
| weak_qwen35_flash_small_v2_baseline_original_reformat | qwen3.5-flash | baseline_original_reformat | small_error_set_partial | 47 | 59.57% | 69.86% | 72.73% | internal_only | 旧 small set 结果，已被 lightfix 47/47 完整版本覆盖；evaluated_count=47; task_status={'success': 44, 'execution_error': 3}. 与 lightfix baseline 指标冲突，不应混用。 |
| weak_qwen35_flash_small_v2_harness_full_final_block | qwen3.5-flash | harness_full_final_block | small_error_set_partial | 46 | 60.87% | 68.12% | 72.94% | internal_only | 旧 small set 结果，已被 lightfix 47/47 完整版本覆盖；evaluated_count=46; task_status={'success': 39, 'verifier_blocked': 4, 'execution_error': 2, 'reformat_missing': 1}. 与 lightfix baseline 指标冲突，不应混用。 |
| weak_qwen35_flash_small_lightfix_baseline_raw | qwen3.5-flash | baseline_raw | small_error_set | 47 | 0.00% | 0.00% | 0.00% | do_not_show | qwen3.5-flash small_error_set 47题 lightfix；status={'success': 47}; format_error=47; reformat_missing=47; verifier_blocked=0; stuck=0. |
| results_datawise-Qwen2.5-14B-Instruct-temperature-0-args-7-6-8 | datawise-Qwen2.5-14B-Instruct-temperature-0-args-7-6-8 | reformat only / missing_eval | full | 257 |  |  |  | do_not_show | Found reformat jsonl but no matching eval_outputs JSON in requested locations; do not present as result. |
| results_datawise-Qwen2.5-7B-Instruct-temperature-0-args-7-6-8 | datawise-Qwen2.5-7B-Instruct-temperature-0-args-7-6-8 | reformat only / missing_eval | full | 257 |  |  |  | do_not_show | Found reformat jsonl but no matching eval_outputs JSON in requested locations; do not present as result. |
| results_reformat_datawise-gpt-4o-temperature-0-args-7-6-8 | datawise-gpt-4o-temperature-0-args-7-6-8 | reformat only / missing_eval | full | 257 |  |  |  | do_not_show | Found reformat jsonl but no matching eval_outputs JSON in requested locations; do not present as result. |
| results_reformat_datawise-qwen25-70B-temperature-0-args-7-6-8 | datawise-qwen25-70B-temperature-0-args-7-6-8 | reformat only / missing_eval | full | 257 |  |  |  | do_not_show | Found reformat jsonl but no matching eval_outputs JSON in requested locations; do not present as result. |

不建议展示原因主要有：missing_eval、旧版本被覆盖、partial、小集 raw 无 reformat、memory 收益混合、qwen32b 分难度行数不一致。

## 推荐 PPT 第一部分：3--5 页讲法

### 第 1 页：复现基准先摆清楚

**标题**：InfiAgentBench baseline：新模型比论文 72B 更强，但不是每个难度都强。

**核心表格**：

| Setting | full ABQ | easy ABQ | medium ABQ | hard ABQ |
| --- | ---: | ---: | ---: | ---: |
| Paper Qwen2.5-72B | 81.71% | 95.12% | 81.61% | 69.32% |
| Qwen3.5-35B-A3B | 86.38% | 89.02% | 88.51% | 81.82% |

**Takeaway**：Qwen3.5-35B-A3B full 更高，优势主要来自 medium/hard；easy 反而低，说明格式/简单题稳定性仍值得处理。

### 第 2 页：harness 不是全难度通吃

**标题**：Harness 对 medium 有用，但 hard 会退化。

**核心表格**：

| Setting | difficulty | ABQ | PASQ | UASQ |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-35B-A3B baseline | medium | 88.51% | 91.71% | 91.45% |
| Harness rerender | medium | 91.95% | 93.06% | 93.42% |
| Qwen3.5-35B-A3B baseline | hard | 81.82% | 88.54% | 89.16% |
| Harness rerender | hard | 78.41% | 84.75% | 87.19% |

**Takeaway**：medium 的格式收束/约束检查有效；hard 的问题是搜索后选对答案和复算口径，不是简单多跑几轮。

### 第 3 页：weak model small set 作为趋势，不当 full 结论

**标题**：弱模型 small_error_set：结构化 harness 有趋势收益，但成本上升。

**核心表格**：

| Setting | ABQ | Hard ABQ | Avg runtime |
| --- | ---: | ---: | ---: |
| qwen3.5-flash baseline + reformat | 48.94% | 44.00% | 30.24s |
| harness_light | 59.57% | 60.00% | 47.03s |
| harness_full | 63.83% | 64.00% | 92.79s |

**Takeaway**：light 性价比最好；full 分数最高但太慢且有 verifier_blocked。这页必须标 `47题 known-error small set, not full 257`。

### 第 4 页：DataModeling 只讲动机，不讲已系统提分

**标题**：DataModeling 的失败更像协议问题。

**核心内容**：列出 4 类失败：submission/performance 文件未生成、评估格式不匹配、metric 类型错配、jaccard 文本指标收到数值列。

**Takeaway**：这支持 protocol-aware harness 的动机，但还没有系统消融和稳定提分验证。

### 第 5 页：成本计划是下一步，不是已完成结果

**标题**：Resource-aware 是下一步：用 EqTok 衡量模型路由。

**核心表格**：

| Setting | ABQ | EqTok | status |
| --- | ---: | ---: | --- |
| Qwen3-32B baseline | 81.74% | 5.961M | observed/reference |
| model routing | 80.5%-82.5% | about 2.120M | expected |
| routing + conversation reduction | 81.0%-83.5% | 1.05M-1.60M | expected |

**Takeaway**：成本控制是预算调度计划，需要 token_meter 和 E0/E1/E2/E3 真实验证。

## 哪些结果不要放进 PPT 主线

- `weak_qwen35_flash_small_v2_*`：旧 small set，和 lightfix 指标冲突，被覆盖。
- `baseline_raw`：raw 长日志无闭式答案，ABQ 0，只说明需要 reformat，不适合当模型能力指标。
- `memory_learning`：可以说链路跑通，但不要说稳定提升。
- `Qwen32B medium/hard` 分难度：行数不一致，内部观察即可。
- `missing_eval`：没有 eval JSON，不展示分数。
- 任何 expected EqTok 节省：只能作为目标，不是结果。

## 建议的第一部分叙事

1. 先把指标和基准讲清楚：ABQ/PASQ/UASQ，paper Qwen2.5-72B vs Qwen3.5-35B-A3B。
2. 再讲复现发现：strong baseline 已经很强，但 harness 效果有难度差异。
3. 用 medium/hard 对比说明：不是“加 harness 一定涨”，关键是时间花在哪里。
4. 用 weak small set 作为趋势证据：弱模型确实受益，但 full 太慢，light 更适合继续做。
5. 最后自然引到资源计划：既然知道不同阶段收益不同，下一步才是 model routing 和 EqTok。
