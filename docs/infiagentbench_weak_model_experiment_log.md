# InfiAgentBench 弱模型增强实验日志

## 2026-06-09：审计与实验控制准备

### 实验目的

为后续弱模型实验建立可复用的运行、消融、评估和汇报链路。当前不运行 InfiAgentBench 任务。

### 模型

本轮未运行模型。后续候选：

- `qwen3.5-flash`
- `qwen3-30b-a3b`
- `qwen3-32b`
- `qwen3-vl-32b-instruct`，仅在接口可用且文本 agent 兼容时考虑
- `qwen3-vl-32b-thinking`，仅在接口可用且成本可控时考虑

强模型参考：已有 `qwen3.5-35b-a3b` baseline 结果。

### 命令

本轮只做静态审计和代码准备，未启动服务，未跑任务。

后续 small set baseline 示例：

```bash
python evaluation/run_harness_infiagent.py \
  --note weak-qwen35-flash-small-baseline \
  --ablation baseline \
  --model-name qwen3.5-flash \
  --temperature 0 \
  --question-id 7,23,55,56,57,59,62,117,124,125 \
  --continue-on-error
```

后续 small set harness 示例：

```bash
python evaluation/run_harness_infiagent.py \
  --note weak-qwen35-flash-small-harness \
  --ablation harness_full \
  --model-name qwen3.5-flash \
  --temperature 0 \
  --question-id 7,23,55,56,57,59,62,117,124,125 \
  --continue-on-error
```

后续对比表生成示例：

```bash
python evaluation/build_weak_model_comparison.py \
  --weak-model qwen3.5-flash \
  --weak-baseline evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_weak-qwen35-flash-small-baseline.jsonl \
  --weak-harness evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_weak-qwen35-flash-small-harness.jsonl \
  --strong-reference evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_qwen35b_a3b_full.jsonl
```

### 数据范围

本轮未运行。后续先跑 small set，再跑 full 257。

### 改动模块

- `evaluation/run_harness_infiagent.py`
  - 增加 `--ablation`
  - 增加 `--model-name`、`--temperature`、`--max-tokens`、`--provider`
  - 增加 `--max-tasks`、`--start-from-task-id`、`--skip-existing`、`--resume`、`--continue-on-error`、`--stop-on-error`
  - 每条结果记录 `task_status` 和 `experiment_metadata`
- `datawiseagent/harness/controller.py`
  - 增加 `disabled_capabilities`
  - 将 finalizer 从 verifier 中解耦，支持 `no_oracle_or_verifier` 保留 final block、`no_finalizer` 关闭 final block
- `evaluation/InfiAgentBench/scripts/eval_closed_form.py`
  - 增加 task status、verifier blocked、runtime、memory hit、artifact count 汇总
- `evaluation/summarize_harness_metrics.py`
  - 增加同类 summary 字段
- `evaluation/build_weak_model_comparison.py`
  - 新增评估后对比表生成脚本

### 结果

未运行新实验。

已有参考结果：

- `qwen3.5-35b-a3b` full baseline Q-Acc：0.8638
- harness medium Q-Acc：0.9195，高于 medium baseline 0.8851
- harness hard Q-Acc：0.7841，低于 hard baseline 0.8182

### 失败案例

本轮无新运行失败。已有结果显示 hard harness 可能因 branch selection / semantic verification 不够强而退化。

### 解释

当前 harness 对格式收束和 medium 题有效，但 hard 题需要更强的可执行验证和分支选择。弱模型实验应优先验证结构化 harness 是否能缩小 weak-to-strong gap，而不是继续证明强模型本身强。

### PPT 可用结论

- 已有工程骨架覆盖 DataCard、Contract、Skill、Oracle、Verifier、FinalBlock、Search、Memory。
- 现有结果表明 harness 的收益具有难度差异：medium 增益，hard 退化。
- 弱模型实验的科学问题更清楚：结构化 harness 是否能补足弱模型在题目理解、格式和协议执行上的短板。

### 下一步

1. 选一个弱模型先跑 small set baseline / harness。
2. 如果 API 稳定，再跑 full 257。
3. 对 full 结果做消融：`no_datacard`、`no_contract`、`no_skills`、`no_oracle_or_verifier`、`no_finalizer`、`no_memory`、`no_search`。

## 2026-06-09：qwen3.5-flash small_error_set live run

### 实验目的

验证弱模型 `qwen3.5-flash` 在 InfiAgentBench 已知错题小集上，原始 DatawiseAgent baseline 与完整 harness 的真实差异。小集来自 `runs/infiagent_improve/small_error_set.json`，共 47 题。

本轮只跑 small set 的两组：

- `baseline`
- `harness_full`

未跑 full 257，未跑最小消融。原因是 `harness_full` 在 small set 上已经明显更慢，且 task 733 发生执行调试循环卡死，需要先分析小集结果。

### 环境与服务

- Python：`.venv310`
- 服务：`env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= NO_PROXY=127.0.0.1,localhost .venv310/bin/python main.py`
- API/model：DashScope OpenAI-compatible，`qwen3.5-flash`
- temperature：0
- 服务检查：`GET /users` 返回 200

### 实际命令

baseline:

```bash
env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python evaluation/run_harness_infiagent.py \
  --note weak-qwen35-flash-small-baseline-v2-20260609 \
  --ablation baseline \
  --model-name qwen3.5-flash \
  --provider dashscope \
  --temperature 0 \
  --question-id "$IDS" \
  --continue-on-error --skip-existing
```

baseline original reformat:

```bash
env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python evaluation/InfiAgentBench/scripts/reformat.py \
  --url_file evaluation/InfiAgentBench/scripts/url.txt \
  --api_key_file evaluation/InfiAgentBench/scripts/api_key.txt \
  --questions_file_path evaluation/InfiAgentBench/data/da-dev-questions.jsonl \
  --responses_file_path evaluation/experimental_results/InfiAgent-Bench/results_weak-qwen35-flash-small-baseline-v2-20260609.jsonl \
  --output_file_path evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_weak-qwen35-flash-small-baseline-v2-20260609.jsonl \
  --model qwen3.5-flash \
  --max_resp 2048 \
  --max_retries 3 \
  --request_timeout 120
```

harness:

```bash
env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python evaluation/run_harness_infiagent.py \
  --note weak-qwen35-flash-small-harness-full-20260609 \
  --ablation harness_full \
  --model-name qwen3.5-flash \
  --provider dashscope \
  --temperature 0 \
  --question-id "$IDS" \
  --continue-on-error --skip-existing
```

task 733 卡在反复把 JSON final block 放入 Python cell 执行，报 `NameError: name 'false' is not defined`。停止该 runner 后，从 734 继续：

```bash
env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python evaluation/run_harness_infiagent.py \
  --note weak-qwen35-flash-small-harness-full-20260609 \
  --ablation harness_full \
  --model-name qwen3.5-flash \
  --provider dashscope \
  --temperature 0 \
  --question-id 734,741,743 \
  --continue-on-error --skip-existing
```

### 结果文件

- baseline raw：`evaluation/experimental_results/InfiAgent-Bench/results_weak-qwen35-flash-small-baseline-v2-20260609.jsonl`
- baseline reformat：`evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_weak-qwen35-flash-small-baseline-v2-20260609.jsonl`
- harness full：`evaluation/experimental_results/InfiAgent-Bench/results_weak-qwen35-flash-small-harness-full-20260609.jsonl`
- 对比目录：`evaluation/experimental_results/InfiAgent-Bench/weak_model_comparison/small-qwen35-flash-20260609/`

### 指标

| setting | ABQ | PASQ | UASQ | hard ABQ | response/eval | format error | reformat missing | verifier blocked | avg runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline raw | 0.0000 | 0.0000 | 0.0000 | n/a | 47/44 | 47 | 47 | 0 | 30.25 |
| baseline + original reformat | 0.5957 | 0.6986 | 0.7273 | 0.5200 | 47/47 | 0 | 0 | 0 | 30.25 |
| harness_full + final block | 0.6087 | 0.6812 | 0.7294 | 0.5417 | 46/46 | 3 | 3 | 4 | 103.60 |

解释：

- baseline raw 为 0%，因为 raw notebook 长日志没有 closed-form `@answer[...]`，不能直接评估。
- baseline + original reformat 是公平的原始 DatawiseAgent 闭式答案结果。
- harness_full 比 baseline ABQ 高 1.30 个百分点，UASQ 高 0.21 个百分点，但 PASQ 低 1.74 个百分点。
- harness_full 未完成 task 733，因此当前是 partial small-set 结果，不是 47/47 完整 full-harness 结果。

### 逐题变化

在 47 题 small set 上：

- fixed：6 题，`125, 219, 273, 450, 496, 741`
- regressed：4 题，`62, 124, 418, 451`
- both_correct：22 题
- unchanged_wrong：10 题
- verifier_blocked：4 题，`310, 550, 734, 743`
- not_run/stuck：1 题，`733`

### 观察

- harness 对 feature engineering、ML protocol、部分 outlier/correlation 题有正向作用。
- final block 路径真实生效，例如 task 7 写入 `@prediction_accuracy[0.78]`，不依赖长日志 reformat。
- verifier 过严或 final block schema 不稳会把可解析答案标为 blocked，例如 310、550、734、743。
- 733 暴露了一个关键工程问题：FinalAnswerBlock 指令虽然要求 JSON，但模型仍可能把 JSON 放进 Python code cell，导致执行循环。这个不是 API 失败，也不是余额/限速问题，是 harness finalization 执行控制问题。
- weak+harness 只轻微缩小了与 `qwen3.5-35b-a3b` 的差距。小集 full ABQ gap 从 0.2681 降到 0.2551，hard gap 从 0.2982 降到 0.2765。

### 当前结论

在 qwen3.5-flash small_error_set 上，harness_full 有小幅 ABQ/hard ABQ 提升，但提升不足以认为弱模型已经接近强模型。主要瓶颈从“没有结构化答案”转移到：

- final block 生成和执行通道不稳定；
- verifier 阻断与真实评估正确性不完全一致；
- 多分支 harness 成本高，平均 runtime 从 30.25 增加到 103.60；
- task 733 这类 JSON/Python cell 混淆会导致长时间卡死。

### 下一步

1. 不急着跑 full 257，先修复/限制 final block 只能作为 markdown 或外部解析，不允许被 notebook 当 Python 执行。
2. 调整 verifier_blocked 的处理策略，区分“格式有答案但校验失败”和“无有效答案”。
3. 再跑同一 small set 的 `no_finalizer`、`no_oracle_or_verifier`、`no_skills` 最小消融，看 6 个 fixed 与 4 个 regressed 分别来自哪个模块。

## 2026-06-09：qwen3.5-flash small_error_set lightfix run

### 实验目的

修复 task 733 的 FinalAnswerBlock JSON 被写入 Python code cell 后，重新跑 47 题 small set 的三组真实实验：

- `baseline`
- `harness_light`
- `harness_full`

本轮仍不跑 full 257，也不跑额外大消融。重点验证稳定性、runtime 和轻量 harness 的性价比。

### 稳定性修复

- FinalAnswerBlock / JSON 只作为 markdown final artifact，不作为可执行 Python cell。
- 对包含 `final_answer_block`、`format_targets`、`validator_report_id` 且带有 JSON `false` / `true` / `null` 风格内容的 fenced block 做 code-cell guard。
- `harness_light` 强制启用 finalizer，但关闭 search、memory、heavy oracle、heavy verifier。
- runner 使用 `--request-timeout 300 --continue-on-error --skip-existing`，避免单题无限阻塞。

### 实际命令

服务启动：

```bash
env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python main.py
```

baseline：

```bash
env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python evaluation/run_harness_infiagent.py \
  --note weak-qwen35-flash-small-baseline-lightfix-20260609 \
  --ablation baseline \
  --model-name qwen3.5-flash \
  --provider dashscope \
  --temperature 0 \
  --question-id "$IDS" \
  --request-timeout 300 \
  --continue-on-error --skip-existing
```

baseline original reformat：

```bash
env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python evaluation/InfiAgentBench/scripts/reformat.py \
  --url_file evaluation/InfiAgentBench/scripts/url.txt \
  --api_key_file evaluation/InfiAgentBench/scripts/api_key.txt \
  --questions_file_path evaluation/InfiAgentBench/data/da-dev-questions.jsonl \
  --responses_file_path evaluation/experimental_results/InfiAgent-Bench/results_weak-qwen35-flash-small-baseline-lightfix-20260609.jsonl \
  --output_file_path evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_weak-qwen35-flash-small-baseline-lightfix-20260609.jsonl \
  --model qwen3.5-flash \
  --max_resp 2048 \
  --max_retries 3 \
  --request_timeout 120
```

harness light：

```bash
env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python evaluation/run_harness_infiagent.py \
  --note weak-qwen35-flash-small-harness-light-lightfix-20260609 \
  --ablation harness_light \
  --model-name qwen3.5-flash \
  --provider dashscope \
  --temperature 0 \
  --question-id "$IDS" \
  --request-timeout 300 \
  --continue-on-error --skip-existing
```

harness full：

```bash
env HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= NO_PROXY=127.0.0.1,localhost \
  .venv310/bin/python evaluation/run_harness_infiagent.py \
  --note weak-qwen35-flash-small-harness-full-lightfix-20260609 \
  --ablation harness_full \
  --model-name qwen3.5-flash \
  --provider dashscope \
  --temperature 0 \
  --question-id "$IDS" \
  --request-timeout 300 \
  --continue-on-error --skip-existing
```

### 结果文件

- baseline raw：`evaluation/experimental_results/InfiAgent-Bench/results_weak-qwen35-flash-small-baseline-lightfix-20260609.jsonl`
- baseline reformat：`evaluation/experimental_results/InfiAgent-Bench/reformat/results_reformat_weak-qwen35-flash-small-baseline-lightfix-20260609.jsonl`
- harness light：`evaluation/experimental_results/InfiAgent-Bench/results_weak-qwen35-flash-small-harness-light-lightfix-20260609.jsonl`
- harness full raw append：`evaluation/experimental_results/InfiAgent-Bench/results_weak-qwen35-flash-small-harness-full-lightfix-20260609.jsonl`
- harness full dedup eval file：`evaluation/experimental_results/InfiAgent-Bench/results_weak-qwen35-flash-small-harness-full-lightfix-20260609.dedup.jsonl`
- 对比目录：`evaluation/experimental_results/InfiAgent-Bench/weak_model_comparison/small-qwen35-flash-lightfix-20260609/`

说明：本轮曾发现两个同 note `harness_full` runner 同时写同一 jsonl，导致 raw append 文件有重复行。已停止较早的重复进程，并保留 raw 文件；评估使用按 task_id 最后一次结果生成的 dedup 文件，47/47 无缺失。

### 指标

| setting | ABQ | PASQ | UASQ | easy ABQ | medium ABQ | hard ABQ | format error | reformat missing | verifier blocked | stuck | avg runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline raw | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 47 | 47 | 0 | 0 | 30.24 |
| baseline + original reformat | 0.4894 | 0.6383 | 0.6705 | 0.5556 | 0.5385 | 0.4400 | 0 | 0 | 0 | 0 | 30.24 |
| harness_light + final block | 0.5957 | 0.6525 | 0.6932 | 0.6667 | 0.5385 | 0.6000 | 1 | 3 | 0 | 1 | 47.03 |
| harness_full + final block | 0.6383 | 0.6950 | 0.7273 | 0.6667 | 0.6154 | 0.6400 | 0 | 2 | 4 | 1 | 92.79 |

### 逐题变化

相对 baseline + original reformat：

- `harness_light` fixed：`57, 125, 271, 273, 450, 513, 528, 647`
- `harness_light` regressed：`62, 451, 684`
- `harness_full` fixed：`57, 125, 133, 271, 273, 513, 528`
- `harness_full` regressed：`408`
- `harness_full` verifier_blocked：`219, 310, 550, 734`
- `harness_full` stuck / execution_error：`647`

注意：task 647 虽然最终答案可被 eval 判为正确，但 `task_status` 是 `execution_error`，因此稳定性统计里仍记为 stuck/execution_error。

### task 733 结果

- `baseline`：success
- `harness_light`：success，runtime 30.75s
- `harness_full`：success，runtime 75.60s
- final block reformat：`@has_nan_values_in_new_feature[False] @new_feature_mean[3.54] @new_feature_std[0.54]`

结论：733 的 JSON/Python cell 卡死问题已在本轮 small set 中解决。

### 指定退化题归因

- `62`：baseline 正确，light 错误，full 正确。light 关闭 heavy oracle/verifier 后把 outlier 个数算成 107，说明 outlier protocol 需要 full harness 的方法约束或校验。
- `124`：baseline、light、full 都错误。full 重新计算出 p-value 0.0043，但仍判断 significant；更像统计口径/题意解析错误，不是 finalizer 问题。
- `418`：本轮 baseline、light、full 都正确，不再是退化。
- `451`：baseline 正确，light 错误，full 正确。light 用 JSON dict 双引号和紧凑格式，label 需要 Python dict 字符串风格；这是 finalizer 格式控制问题。
- `408`：本轮 full 唯一相对 baseline 的真实退化，full 输出 `r=-0.05, p=0.0001`，gold 是 `r=0.10, p=0.0102`。疑似 full 的 search/oracle/contract 干预导致过滤或列选择口径变化。

### 结论

本轮 47/47 完整 small set 支持：

- `harness_full` 准确率最高：ABQ 0.6383，比 baseline + reformat 高 14.89 个百分点。
- `harness_light` 性价比更好：ABQ 0.5957，比 baseline 高 10.63 个百分点，但平均 runtime 只从 30.24s 增至 47.03s。
- `harness_full` 从 light 再提升 4.26 个百分点，但 runtime 从 47.03s 增至 92.79s，且有 4 个 verifier_blocked。
- 当前不应跑 full 257；下一步应优先轻量化 harness，并校准 verifier/finalizer。
