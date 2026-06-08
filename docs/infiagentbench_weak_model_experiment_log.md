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
