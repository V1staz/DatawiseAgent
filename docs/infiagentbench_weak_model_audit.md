# InfiAgentBench 弱模型增强实验审计记录

更新时间：2026-06-09

## 本轮目标

验证弱模型在结构化 harness 加持下，是否能接近强模型 `qwen3.5-35b-a3b` 在 InfiAgentBench 上的表现。当前阶段只审计和补实验控制能力，不运行任务，不优先处理 DataModeling，也不做多模型难度路由。

## 当前可用模块

主线模块位于 `datawiseagent/harness/`，仓库中同时存在 `DatawiseAgent/harness/` 的大小写兼容副本。

| 模块 | 当前作用 | 接入状态 | 备注 |
| --- | --- | --- | --- |
| `datacard.py` | 读取 CSV 并生成列名、shape、dtype、missing、unique、range、date/category/id hints | 已接入 `HarnessController.prepare()` | 等价于 schema profiler 的主线实现 |
| `contract.py` | 规则化解析题目、constraints、format，抽取 answer names、rounding、required columns、methods、preprocessing steps | 已接入 | 仍是启发式，不是强语义解析器 |
| `skills.py` + `default_rules.jsonl` | 根据 concepts/keywords 路由 skill protocol 和持久规则卡 | 已接入 | 主要是 prompt directives + checklist，不是确定性执行算子 |
| `oracles.py` | 从 contract/datacard 派生语义检查，例如 std ddof、year boundary、ML split/features/metric | 已接入 verify 阶段 | 只覆盖高频错误，hard 题分支选择仍不稳 |
| `verifier.py` | 验证 FinalAnswerBlock、answer names、format targets、类型、rounding/range | 已接入 postprocess | 会影响 branch score，但当前不是强制重新执行闭环 |
| `finalizer.py` | 抽取 FinalAnswerBlock，渲染 deterministic `@answer[value]` | 已接入 runner 和 reformat 脚本 | final block 优先于 LLM reformat |
| `search.py` | hard-only 多分支 prompt/选择 | 已接入 | 当前 hard 结果下降，说明 selector 还不可靠 |
| `memory.py` / `self_memory_*` | 从运行时错误和 validator/oracle 结果沉淀、检索自更新记忆 | 已接入 memory/full 模式 | 不应注入人工报告或 gold answer |
| `recovery.py` / `trace.py` | 记录恢复策略和结构化过程 | 已接入 | 主要是记录/提示，不是完整 rollback 系统 |

旧扩展模块 `agent_extensions/infiagentbench/` 仍存在，包含 schema/profile/contract/router/verifier/final block 等可插拔工具。当前 fetch 后的主实验链路已经转向 `datawiseagent/harness`，本轮不重复维护一套新系统。

## 当前脚本入口

| 用途 | 脚本 |
| --- | --- |
| 原始 baseline 入口 | `evaluation/eval_infiagent_bench.py` |
| harness 入口 | `evaluation/run_harness_infiagent.py` |
| reformat | `evaluation/InfiAgentBench/scripts/reformat.py` |
| eval | `evaluation/InfiAgentBench/scripts/eval_closed_form.py` |
| harness summary | `evaluation/summarize_harness_metrics.py` |
| 弱模型对比表生成 | `evaluation/build_weak_model_comparison.py` |

`evaluation/run_harness_infiagent.py --harness-mode baseline` 走原始 DatawiseAgent prompt，不生成 data card/contract/final block。非 baseline 模式由 `HarnessController` 包装 prompt、artifact、final block 和 validation。

## 当前结果文件

已有强模型参考结果：

| 设置 | 路径 | Q-Acc | Prop Sub-Q | Sub-Q |
| --- | --- | ---: | ---: | ---: |
| `qwen3.5-35b-a3b` full baseline | `evaluation/experimental_results/InfiAgent-Bench/eval_outputs/results_reformat_qwen35b_a3b_full_evaluation_analysis.json` | 0.8638 | 0.9016 | 0.9035 |
| `qwen3.5-35b-a3b` easy baseline | `.../results_reformat_qwen35b_a3b_easy_evaluation_analysis.json` | 0.8902 | 0.9024 | 0.9109 |
| `qwen3.5-35b-a3b` medium baseline | `.../results_reformat_qwen35b_a3b_medium_evaluation_analysis.json` | 0.8851 | 0.9171 | 0.9145 |
| `qwen3.5-35b-a3b` hard baseline | `.../results_reformat_qwen35b_a3b_hard_evaluation_analysis.json` | 0.8182 | 0.8854 | 0.8916 |
| harness medium | `.../results_reformat_harness_round2_rerender_qwen35b_medium_evaluation_analysis.json` | 0.9195 | 0.9306 | 0.9342 |
| harness hard | `.../results_reformat_harness_round2_rerender_qwen35b_hard_evaluation_analysis.json` | 0.7841 | 0.8475 | 0.8719 |

论文 Qwen2.5-72B 结果只作为参考，不作为后续实验的严格同模型复现设置。后续弱模型实验属于 model-substituted setting。

## 已确认的现象

- `qwen3.5-35b-a3b` full 高于论文 Qwen2.5-72B 参考结果，主要提升来自 medium/hard。
- harness 对 medium 有明显收益：Q-Acc 0.8851 -> 0.9195。
- harness 对 hard 下降：Q-Acc 0.8182 -> 0.7841。
- hard harness 的 format/reformat/final block 指标并非主要问题，下降更可能来自分支选择、计算口径、边界条件或 ML protocol。
- memory-learning full 链路能跑，但收益不稳定，有 improved 和 regressed 两类 flip。

## 半成品风险

- skill 仍以 prompt directive/checklist 为主，不是可执行技能库。
- oracle 覆盖面有限，尤其缺少 correlation abs(r)/p-value、outlier 方法、preprocessing 顺序、feature formula 前 5 行等更系统的检查。
- branch score 仍偏格式/schema 与轻量语义证据，hard 题选择错误会导致退化。
- memory 默认在 full 中可用，必须在消融中显式记录是否启用，并确认来源不是人工报告。
- `DatawiseAgent/harness` 与 `datawiseagent/harness` 双份路径需要避免漂移。

## 本轮已补的实验能力

- `evaluation/run_harness_infiagent.py` 支持 `--ablation`：
  - `baseline`
  - `harness_full`
  - `no_datacard`
  - `no_contract`
  - `no_skills`
  - `no_oracle_or_verifier`
  - `no_finalizer`
  - `no_memory`
  - `no_search`
- 支持 `--model-name`、`--temperature`、`--max-tokens`、`--provider`，不再需要为每个弱模型手写配置文件。
- 每条结果写入 `experiment_metadata` 和 `task_status`。
- summary/eval 增加 task status、memory hit、artifact count、runtime 等指标。
- 新增 `evaluation/build_weak_model_comparison.py`，用于评估后生成：
  - `weak_baseline_vs_harness_per_task.csv`
  - `ablation_summary.csv`
  - `weak_to_strong_gap.csv`

## 本轮不改的内容

- 不改原始 DatawiseAgent 主流程。
- 不新建第二套 schema/contract/skill 系统。
- 不优先做 DataModeling。
- 不做难度路由 / 多模型 sub-agent。
- 不把 gold answer、eval 输出或人工错题答案注入 prompt/memory。

## 下一步实验建议

1. 先用 `qwen3.5-flash` 或 `qwen3-30b-a3b` 跑 small set：
   - weak baseline
   - weak harness_full
   - strong reference 从已有 `qwen3.5-35b-a3b` 结果抽取
2. small set 跑通后再做 full 257。
3. 消融按 easy/medium/hard 分开统计，重点判断 weak+harness 是否缩小 strong gap。
