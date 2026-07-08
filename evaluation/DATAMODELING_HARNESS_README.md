# DataModeling Agent Harness

正式模型能力评测使用 `evaluation/run_agent_harness_datamodeling.py`。这不是代替模型建模的 AutoML runner，而是围绕原 FastAPI/DatawiseAgent/Jupyter 执行链路加的一层 harness：LLM 仍然负责读题、写代码、训练模型、生成 `./input/final_submission.csv`；harness 负责把任务约束说清楚、阻止 schema/row/id 等低级错误、记录过程证据，并在 validator/quality gate 失败时给出有限 repair feedback。

设计边界：

- **允许**：读取题面、train/test/sample submission、label-blind contract/profile、本题 validator 报错。
- **不允许**：读取 hidden answers/GT、由 harness 直接训练并提交结果、把 offline recipe 输出伪装成模型输出。
- **可比较模型能力的结果**：`harness.mode == "llm_in_the_loop_agent_harness"` 且 `deterministic_recipe_used == false`。
- **新增质量闸门**：结构合法 CSV 不再自动等于完成；模型需要同时写 `./input/submission_manifest.json`，说明 fallback、validation/proxy、所选方法和预测分布。缺 manifest、fallback 直接当 final、常数低信息预测等会触发 repair 或失败。

## Run the official agent harness

示例：

```bash
CUSTOM_CONFIG=configs/setting_model_qwen3_vl_8b_thinking.yaml python main.py
python evaluation/run_agent_harness_datamodeling.py \
  --model-name qwen3-vl-8b-thinking \
  --num-workers 2 \
  --score
```

每题会写入：

- `contract.json` / `profile.json` / `route.json`：label-blind 任务约束、数据画像与任务族路由；
- `guidance_AGENTS.md` / `guidance_SKILLS.md` / `guidance_MCP_TOOLS.md` / `guidance_manifest.json`：分层 guidance 归档，避免把所有 prompt 规则堆在代码字符串里；
- `initial_prompt.txt` / `repair_prompt_attempt_*.txt`：给模型的引导与修复提示；
- `chat_response_attempt_*.json`：原始模型回复；
- `submission_attempt_*.csv` / `final_submission.csv`：每轮验证的 submission 快照；
- `validation_attempt_*.json` / `quality_attempt_*.json` / `runtime.json`：validator、quality gate 与计时证据。

`results.jsonl` 的 `harness.mode` 应为 `llm_in_the_loop_agent_harness`，且 `deterministic_recipe_used=false`。

正式 agent validator 使用严格策略：概率/范围越界会触发 repair；完全复制 `sample_submission.csv` 的 target 列会被视为 `sample_submission_copy` 失败，而不是“跑通”。quality gate 进一步检查 manifest、fallback 误用、预测多样性和公开验证证据；它不读取答案，也不证明 leaderboard 质量，只拦截“空跑通”和低信息 final。

## qwen3-vl 实验闸门

按当前计划，先跑 `qwen3-vl-8b-thinking`。只有当 8B 的 DataModeling normalized performance 达到记录的 qwen2.5-72B baseline（默认阈值 `0.4290`）且 full run 基本可评分，才启动 `qwen3-vl-32b-thinking`：

```bash
export SILICONFLOW_API_KEY=...
evaluation/DataModeling/scripts/run_qwen3vl_quality_gate.sh
```

脚本只从环境变量读取 SiliconFlow key，不写入仓库。可调参数：

- `SILICONFLOW_BASE_URL`（默认 `https://api.siliconflow.cn/v1`）
- `DWA_DM_8B_GATE_THRESHOLD`（默认 `0.4290`）
- `DWA_DM_WORKERS`（默认 `2`）
- `DWA_DM_MIN_SCORE_OK`（默认 `70`）

## Offline deterministic baseline / oracle components

`evaluation/run_harness_datamodeling.py` 是离线 deterministic baseline/oracle runner，不用于比较 qwen3-vl 等 LLM 模型能力。它复用同一套 contract/profile/validator，但会用 Python recipe 直接生成 submission；因此 `--model-name` 只是结果标签。代码中对应 `OfflineDataModelingBaselineRunner`，旧的 `DataModelingHarnessRunner` 名称仅作为兼容 alias。

## Why this is separate from InfiAgentBench harness

InfiAgentBench is closed-form data analysis: the hard part is exact computation and final answer formatting. DataModeling is Kaggle-style prediction: the hard part is metric-aware modeling, validation, leakage control, and submission schema correctness. This harness therefore focuses on AutoML-like engineering rather than final-answer reformatting.

## Run the offline baseline

Fast smoke on one task（仅用于本地 validator / recipe sanity check）：

```bash
python evaluation/run_harness_datamodeling.py \
  --model-name datamodeling-harness-smoke \
  --tasks titanic \
  --budget-seconds 300 \
  --max-candidates 3 \
  --candidate-allowlist dummy,logistic,lightgbm \
  --score
```

Full heavy run:

```bash
python evaluation/run_harness_datamodeling.py \
  --model-name small-model-dm-harness-full \
  --budget-seconds 3600 \
  --max-candidates 6 \
  --n-splits 3 \
  --score
```

Output defaults to:

```text
evaluation/experimental_results/Datamodeling-DSBench/<model-name>/
```

## Run the LLM-in-the-loop agent harness

This is the official harness path for model evaluation. The LLM still uses
DatawiseAgent/FastAPI/Jupyter to inspect data, write code, fit models, and save
`./input/final_submission.csv`; the harness only constrains the task, validates
the result, runs a label-blind quality gate, and asks for bounded repairs.

```bash
python evaluation/run_agent_harness_datamodeling.py \
  --model-name qwen3-vl-8b-thinking-agent-harness-full \
  --num-workers 2 \
  --max-repair-attempts 2 \
  --chat-timeout-seconds 1200 \
  --early-stop-min-seconds 60 \
  --early-stop-poll-seconds 10 \
  --score
```

`--chat-timeout-seconds` is a harness guardrail rather than a modeling shortcut:
if a chat request overruns, the runner validates any `final_submission.csv`
already produced and can issue a repair prompt. The prompt also tells the LLM to
keep code cells small, avoid `n_jobs=-1` on large data, and finish only once a
quality-backed final submission plus `submission_manifest.json` have been
created.

The runner also polls the workspace during long chats. After
`--early-stop-min-seconds`, if the agent-created `./input/final_submission.csv`
already passes both strict validator and quality gate, the runner stops the
session and records the result instead of letting the agent keep optimizing or
looping.

After a chat returns or times out, if the LLM created an explicit candidate such
as `./input/candidate_submission.csv` but forgot to copy it to the canonical
path, the runner may adopt that agent-created file only after it passes the same
strict validator and sample-copy rejection check. `fallback_submission.csv` is
still treated as a recovery artifact: adopting it can fix file placement, but
the quality gate will reject fallback-as-final unless the manifest honestly
marks it as recovery, and recovery is not counted as quality final. The runner
does not adopt candidate files while the LLM is still running, and it never
generates predictions itself.

## Agent artifacts

- `results.jsonl`: DSBench-compatible result rows, including harness metadata and optional official score.
- `performances/<task>/result.txt`: optional official score output when `--score` is enabled.
- `artifacts/<id>_<task>/contract.json`: metric, targets, submission schema, task kinds, risk flags.
- `artifacts/<id>_<task>/profile.json`: train/test schema and target profile.
- `artifacts/<id>_<task>/route.json`: selected task family, skill branch, quality checks, forbidden shortcuts.
- `artifacts/<id>_<task>/guidance_*`: layered guidance surfaces copied for reproducibility.
- `artifacts/<id>_<task>/initial_prompt.txt`: LLM-facing task guidance.
- `artifacts/<id>_<task>/chat_response_attempt_*.json`: raw chat responses.
- `artifacts/<id>_<task>/submission_attempt_*.csv`: attempted outputs before/after repairs.
- `artifacts/<id>_<task>/validation_attempt_*.json`: strict validator reports.
- `artifacts/<id>_<task>/quality_attempt_*.json`: manifest/fallback/diversity quality-gate reports.
- `agent_harness_summary.json`: aggregate agent-run status.

Offline baseline artifacts are similar but include recipe-specific files such as `leaderboard.json`, `submission_validation.json`, and `harness_summary.json`; those are oracle/baseline evidence, not LLM-capability evidence.

## Label-blind policy

During modeling the harness reads only:

- task description files;
- train/test/sample submission schemas and data;
- local eval script text for metric hints.

It does **not** read `data/answers/` or `save_performance/GT` until optional post-run scoring. Those paths must not be used by recipes, memory retrieval, or small-model prompts.

## Offline recipe layers

- Generic tabular classification/regression: dummy, regularized linear, random forest, extra trees, LightGBM, XGBoost when installed.
- Text features: combined TF-IDF feature channel.
- RMSLE: log target and non-negative clipping.
- Probability/ranking metrics: `predict_proba`, ROC-AUC, normalized Gini, log-loss, and range validation.
- Categorical labels: string/bool class labels are encoded for training and decoded back for submission.
- Top-k label submissions: MPA@3-style tasks emit space-separated top-3 labels.
- Small sample/high dimensional risk: regularized logistic + feature selection.
- Special recipes: tweet selected-text string baseline, store-item demand seasonal average, COVID cumulative last-observation forecast, Game-of-Life stop-copy heuristic.

## Offline baseline current limits

- This is not AutoGluon/MLJAR; it uses only core dependencies to keep the environment controllable.
- It does not call an external small model for `ModelingPlan JSON`; `modeling_plan_prompt.txt` is an offline planning artifact only and is not consumed by the official agent harness.
- Candidate training is process-local; large tasks should be sharded at the task level rather than using many heavy candidates in one process.
