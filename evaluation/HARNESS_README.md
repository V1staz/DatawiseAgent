# DatawiseAgent Harness Runner

This harness wraps the existing FastAPI `/chat/` + DatawiseAgent FST flow without changing baseline behavior.

## Run

Start the backend first, then run from the repository root:

```bash
python evaluation/run_harness_infiagent.py \
  --note qwen35b_a3b_full_harness \
  --level hard \
  --model-config configs/setting_model_qwen35b_a3b.yaml \
  --harness-mode full \
  --search-policy hard-only \
  --num-workers 2
```

`--harness-mode` supports cumulative ablations:

- `baseline`
- `datacard`
- `contract`
- `skills`
- `verify`
- `memory`
- `search`
- `full`

Search policies:

- `off`
- `hard-only`
- `all`
- `validator-fail`

## Result fields

Each JSONL row includes the legacy fields plus:

- `final_block`
- `contract`
- `data_card_path`
- `trace_path`
- `validator_report_path`
- `search_summary`
- `memory_hits`
- `recovery_events`
- `runtime`

## Reformat from FinalAnswerBlock

For harness outputs, prefer deterministic final-block reformatting:

```bash
cd evaluation
python InfiAgentBench/scripts/reformat.py \
  --final_block_only \
  --responses_file_path experimental_results/InfiAgent-Bench/results_qwen35b_a3b_full_harness.jsonl \
  --output_file_path experimental_results/InfiAgent-Bench/reformat/results_reformat_qwen35b_a3b_full_harness.jsonl
```

Without `--final_block_only`, `reformat.py` still preserves the legacy LLM fallback for old baseline outputs; if a `FinalAnswerBlock` exists, it is always used first.

## Metrics

Closed-form eval now also emits harness metrics:

- `format_error_count`
- `reformat_missing_count`
- `final_block_missing_count`
- `validator_failed_count`
- `recovery_event_count`
- `average_branch_count`

A raw artifact summary can be produced with:

```bash
python evaluation/summarize_harness_metrics.py \
  --results evaluation/experimental_results/InfiAgent-Bench/results_qwen35b_a3b_full_harness.jsonl
```

## Persistent rule/memory/oracle harness layers

The current harness separates three kinds of guidance so prompt engineering does
not keep growing as a single monolithic prompt:

1. **Persistent rulebook** (`datawiseagent/harness/default_rules.jsonl`)
   - Curated reusable rule cards with `id`, `skill`, `layer`, `priority`,
     `severity`, routing concepts/methods/keywords, directive, and checks.
   - `RuleBook` selects only rules relevant to the current contract/Data Card and
     writes `rule_manifest.json` per question artifact.
   - Optional custom rules can be supplied with `--rulebook-path` without editing
     the runner.
   - This layer is the copyable/project-curated configuration surface.  It is
     not updated by the model during benchmark execution.

2. **Agent self-updated persistent memory** (`--memory-path` JSONL)
   - The memory subsystem is deliberately split:
     - `self_memory_sedimentation.py` writes local lessons from harness/agent
       artifacts: validator reports, semantic oracle reports, runtime errors,
       and FinalAnswerBlock evidence.
     - `self_memory_retrieval.py` retrieves only local self-updated memories by
       concept/method/column overlap.
     - `memory.py` remains a compatibility facade for existing harness callers.
   - Every newly written self-memory record is marked with
     `memory_kind="self_updated"`, `provenance="agent_harness_artifact"`,
     `transferable=false`, and
     `copy_policy="local_self_update_only_not_for_rulebook_or_project_transfer"`.
   - Human analysis reports, copied/transferred memories, curated rule cards, and
     benchmark-label-bearing artifacts are blocked from memory retrieval.  The
     word "retrieval" in this harness therefore means retrieval from the
     model/harness self-updated memory store, not retrieval from the curated
     rulebook.
   - Retrieval is hierarchical and non-label: concept/method/column overlap,
     scope level (`domain`, `task`, `episode`, `question`), severity, and
     priority. Question-local memories are stored for audit but not injected by
     default to avoid benchmark-ID overfitting.

3. **Semantic oracle plan** (`oracle_plan.json` / `semantic_report_branch_*.json`)
   - Derived from the contract and Data Card, not labels.
   - Checks method policies such as std `ddof`, inclusive/exclusive year
     boundary wording, ML split/feature/metric evidence, engineered-feature
     baseline construction, and required source-column evidence.
   - Branch selection now combines JSON/schema validation, semantic oracle score,
     runtime success, and branch cost.

Search is still bounded for cost, but the branches now have explicit hierarchy:
`deterministic_skill_branch` is the root, while `alternative_interpretation_branch`
and `oracle_guarded_backtrack_branch` are sibling/backtracking nodes with oracle
focus metadata. This keeps the harness compatible with existing evaluation while
making ToT/LATS-style expansion auditable and extensible.
