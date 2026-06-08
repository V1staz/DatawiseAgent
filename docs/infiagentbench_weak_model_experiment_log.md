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
