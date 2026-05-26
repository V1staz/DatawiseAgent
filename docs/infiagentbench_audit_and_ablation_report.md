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

## Live service status

已检查服务启动说明和入口：

- `README.md` 只列出新增 InfiAgentBench 改进版入口。
- `docs/infiagentbench_improvement.md` 的 live run 说明假设已有 DatawiseAgent 服务。
- `evaluation/eval_infiagent_bench.py`、`evaluation/eval_data_modeling.py` 等脚本说明默认服务是 `localhost:8000`。
- `main.py` 是 FastAPI 入口，底部用 uvicorn 绑定 `127.0.0.1:8000`。

实际启动结果：

```bash
env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python main.py
```

服务端口可用，`GET http://127.0.0.1:8000/users` 返回 HTTP 200。直接使用系统 `python3 main.py` 会缺少 `python-dotenv`；使用 `.venv310` 可启动。需要清空代理环境变量，否则 OpenAI client 初始化会受到 SOCKS 代理依赖影响。

当前阻塞点不是 FastAPI 服务本身，而是模型 API 账号状态。批量 small set baseline 在 task 7 的第一次 LLM 调用处返回：

```text
BadRequestError 400, code=Arrearage
Access denied, please make sure your account is in good standing.
```

因此真实 47 题 small_error_set 没有跑完；没有编造完整 small set 或 7 组消融指标。`Arrearage` 是外部运行条件失败，不记为 InfiAgentBench 改进方法失败。

## Small error set true experiment

已完成一个真实 live smoke：task 219，属于 `small_error_set`，原始错误类型为 `format_error`。这只是 partial live smoke，用来验证调用链、final block 和 formatter，不作为 47 题总体 ABQ 结论。

运行命令：

```bash
env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python scripts/run_infiagent_improve.py \
  --task-id 219 --ablation baseline --note live-smoke

env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python scripts/run_infiagent_improve.py \
  --task-id 219 --ablation full --note live-smoke

.venv310/bin/python scripts/reformat_infiagent_final_block.py \
  --responses-file runs/infiagent_improve/live-smoke/full/predictions/results_live-smoke_full.jsonl \
  --output-file runs/infiagent_improve/live-smoke/full/predictions/results_live-smoke_full_reformatted.jsonl

.venv310/bin/python scripts/eval_infiagent_improve.py \
  --responses-file runs/infiagent_improve/live-smoke/baseline/predictions/results_live-smoke_baseline.jsonl \
  --output runs/infiagent_improve/live-smoke/baseline/eval.json

.venv310/bin/python scripts/eval_infiagent_improve.py \
  --responses-file runs/infiagent_improve/live-smoke/full/predictions/results_live-smoke_full_reformatted.jsonl \
  --output runs/infiagent_improve/live-smoke/full/eval_reformatted.json
```

单题结果，仅代表 task 219：

| run | tasks | ABQ | PASQ | UASQ | hard ABQ | format errors | reformat missing | avg runtime | avg log length | token estimate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline raw | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 1 | 74.6053 | 3813 | 1114 |
| full + final block reformat | 1 | 1.0 | 1.0 | 1.0 | 0.0 | 0 | 0 | 54.9975 | 5044 | 3253 |

注意：baseline 这里是 raw DatawiseAgent 输出，未经过原始 `evaluation/InfiAgentBench/scripts/reformat.py`。原始 reformat 也需要 LLM API；当前账号 `Arrearage` 状态下无法公正补跑 baseline reformat。

47 题 baseline 批量尝试命令：

```bash
env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python scripts/run_infiagent_improve.py \
  --task-ids-file runs/infiagent_improve/small_error_set.json \
  --ablation baseline \
  --note small-error-live \
  --output-dir runs/infiagent_improve \
  --write-incremental \
  --continue-on-error
```

该运行在 task 7 的 LLM 调用处因 `Arrearage` 被手动中止；未生成 predictions jsonl，只生成了：

```text
runs/infiagent_improve/small-error-live/baseline/tasks/7/agent_prompt.json
```

运行和评估结果使用 task-level status：

| status | 含义 |
| --- | --- |
| success | 执行、格式化和评估均通过 |
| model_api_error | 模型 API、账号、额度、连接或本地服务调用失败 |
| execution_error | 本地文件、schema、artifact 或执行链路异常 |
| verifier_blocked | verifier 要求重试，最终 response 被阻断为 `ERROR_NEED_RETRY` |
| reformat_missing | 缺少可解析 final block 或 `@answer[...]` |
| format_error | 能解析答案名，但不满足题目格式 |
| evaluation_wrong | 格式可解析，但值与 gold 不匹配 |

## Baseline vs full comparison

逐题对比表已生成，脚本支持公平比较三种结果：

- `baseline raw`
- `baseline + original reformat`
- `full + final block reformat`

```text
runs/infiagent_improve/live-smoke/baseline_vs_full_comparison.json
```

当前只有 task 219 有真实 baseline/full 可比结果，其余 46 题是 `not_run`。

| task_id | known_error_type | baseline_raw_answer | baseline_reformat_answer | full_final_block_answer | gold_answer | baseline_correct | full_correct | status | failure_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 219 | format_error | raw log computed `(HA2)121`, `326`, `9.026365`, `9.002765` but no strict `@answer[...]` block | not_run | `{"site_identifiers":"(HA2)121,326","outlier_values":"9.03,9.0"}` | `{"site_identifiers":"(HA2)121,326","outlier_values":"9.03,9.0"}` | false | true | fixed |  |

修复归因：task 219 的 baseline 已经算出正确 outlier，但没有严格 `@site_identifiers[...] @outlier_values[...]`，且数值未按两位小数输出。`full` 通过 final block 提取，并在 formatter 中修正了列表外层括号和逗号空格，最终通过评估。

## Ablation results

完整 7 组消融未运行，原因是模型 API 返回 `Arrearage`。当前只有 dry-run 全通过，以及 task 219 的 baseline/full live smoke。

| ablation | true small_error_set status | reason |
| --- | --- | --- |
| baseline | not_run | task 7 LLM call returned `Arrearage` before first prediction |
| schema | not_run | skipped after baseline API failure |
| contract | not_run | skipped after baseline API failure |
| router | not_run | skipped after baseline API failure |
| verifier | not_run | skipped after baseline API failure |
| final_formatter | not_run | skipped after baseline API failure |
| full | not_run | skipped after baseline API failure |

服务/账号恢复后，按以下顺序继续，不要直接跑 257 全量。运行脚本支持 `--resume`、`--skip-existing`、`--max-tasks`、`--start-from-task-id`、`--stop-on-error` / `--continue-on-error`，每题失败会写入 `task_status`、`error_type`、`traceback` 和 `tasks/{task_id}/error.json`，默认建议继续后续题。

```bash
for ablation in baseline full; do
  env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
    http_proxy= https_proxy= all_proxy= \
    NO_PROXY=127.0.0.1,localhost \
    .venv310/bin/python scripts/run_infiagent_improve.py \
    --task-ids-file runs/infiagent_improve/small_error_set.json \
    --ablation "$ablation" \
    --note small-error-live \
    --output-dir runs/infiagent_improve \
    --write-incremental \
    --resume \
    --continue-on-error
done
```

baseline raw 和 full 跑通后，先补 baseline 原始 reformat：

```bash
cd evaluation
../.venv310/bin/python InfiAgentBench/scripts/reformat.py \
  --model "$REFORMAT_MODEL" \
  --responses_file_path ../runs/infiagent_improve/small-error-live/baseline/predictions/results_small-error-live_baseline.jsonl \
  --output_file_path ../runs/infiagent_improve/small-error-live/baseline/predictions/results_small-error-live_baseline_original_reformat.jsonl
```

再分别评估三种结果：

```bash
for file in \
  runs/infiagent_improve/small-error-live/baseline/predictions/results_small-error-live_baseline.jsonl \
  runs/infiagent_improve/small-error-live/baseline/predictions/results_small-error-live_baseline_original_reformat.jsonl \
  runs/infiagent_improve/small-error-live/full/predictions/results_small-error-live_full.jsonl
do
  .venv310/bin/python scripts/eval_infiagent_improve.py \
    --responses-file "$file" \
    --output "${file%.jsonl}_eval.json"
done
```

full 如果需要从 raw response 重新生成 final block reformat：

```bash
.venv310/bin/python scripts/reformat_infiagent_final_block.py \
  --responses-file runs/infiagent_improve/small-error-live/full/predictions/results_small-error-live_full.jsonl \
  --output-file runs/infiagent_improve/small-error-live/full/predictions/results_small-error-live_full_final_block_reformat.jsonl
```

生成公平对比表：

```bash
.venv310/bin/python scripts/attribute_infiagent_fixes.py \
  --small-error-set runs/infiagent_improve/small_error_set.json \
  --baseline-raw-eval runs/infiagent_improve/small-error-live/baseline/predictions/results_small-error-live_baseline_eval.json \
  --baseline-raw-predictions runs/infiagent_improve/small-error-live/baseline/predictions/results_small-error-live_baseline.jsonl \
  --baseline-reformat-eval runs/infiagent_improve/small-error-live/baseline/predictions/results_small-error-live_baseline_original_reformat_eval.json \
  --baseline-reformat-predictions runs/infiagent_improve/small-error-live/baseline/predictions/results_small-error-live_baseline_original_reformat.jsonl \
  --full-eval runs/infiagent_improve/small-error-live/full/predictions/results_small-error-live_full_final_block_reformat_eval.json \
  --full-predictions runs/infiagent_improve/small-error-live/full/predictions/results_small-error-live_full_final_block_reformat.jsonl \
  --output runs/infiagent_improve/small-error-live/baseline_full_comparison.json
```

baseline/full 跑通后再补 7 组消融。

## Fixed cases

Partial live smoke 中已修复：

```text
219
```

这个修复只证明 final block + final formatter 在一个已知 format_error 题上有效，不代表 47 题 small_error_set 已整体改善。

## Regressed cases

未发现退化题。原因：只有 task 219 有可比 live 结果，其余未运行。

## Verifier blocked cases

未发现 verifier blocked 题。task 219 的 `verifier_report.json` 为 `status=pass`、`must_retry=false`。

## Remaining bottlenecks

- 模型 API 账号状态阻塞真实 small set 和消融实验。
- 原始 baseline reformat 依赖 LLM，账号不可用时无法与 full final block reformat 做完全公平对照。
- verifier 仍以 contract/log/final block 结构校验为主，尚未对所有统计题做独立复算。
- strict format 规则很脆弱；本次 live smoke 已暴露并修复 `@answer[[...]]` 与逗号空格问题。

## Next improvement priorities

1. 先恢复模型 API，再只跑 `baseline` 和 `full` 的 47 题 small_error_set，确认真实修复/退化题号。
2. 对 `full` 的 still_failed 题做复算型 verifier，而不是继续扩大 prompt。
3. 将 baseline 原始 reformat 与 final block reformat 分开统计，明确区分“计算错”和“抽答案/格式错”。

## 已通过测试

```bash
.venv310/bin/python -m pytest tests/infiagentbench -q
# 7 passed

.venv310/bin/python -m py_compile \
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
