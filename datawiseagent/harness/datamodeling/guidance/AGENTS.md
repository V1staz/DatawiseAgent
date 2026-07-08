# DataModeling Agent Operating Contract

This contract governs DataModeling benchmark sessions run through the agent harness. It is deliberately layered: code enforces the hard boundaries, this file defines the lifecycle, and task-specific skills are selected by the router.

## Non-negotiable boundaries

1. Use only public task files uploaded under `./input/`: `train.csv`, `test.csv`, `sample_submission.csv`, and public task metadata embedded in the prompt.
2. Never inspect hidden answers, score files, leaderboard artifacts, or repository paths outside the uploaded session data.
3. The LLM agent owns modeling. The harness may validate, route, and request repair; it must not create model predictions for the agent.
4. Structural validity is not enough. A final submission must be backed by a short manifest explaining the selected modeling attempt.

## Submission lifecycle

```
inspect public data
  -> build a safe fallback at ./input/fallback_submission.csv
  -> train/evaluate at least one public-data model or task-specific heuristic
  -> compare against fallback on an internal split or cross-validation proxy
  -> write selected improved result to ./input/final_submission.csv
  -> write ./input/submission_manifest.json
  -> read back final CSV and manifest
```

The fallback is a recovery artifact, not the default final answer. Do not copy it to `final_submission.csv` unless every improved attempt failed and the manifest marks the result as `fallback_recovery`; the harness may score that as low-quality.

## Manifest contract

Write `./input/submission_manifest.json` with at least:

```json
{
  "stage": "improved",
  "modeling_method": "short method name",
  "selected_reason": "why this beats fallback on public-data validation",
  "validation_metric_name": "metric or proxy name",
  "metric_direction": "maximize or minimize",
  "validation_metric_value": 0.0,
  "baseline_metric_value": 0.0,
  "trained_rows": 0,
  "prediction_summary": {
    "target": {"nunique": 2, "min": 0.1, "max": 0.9}
  }
}
```

Use `stage="improved"` only when a model or defensible task heuristic produced the selected final file. Use `stage="fallback_recovery"` for emergency fallback. Use `stage="failed"` if no valid final can be created.

## Finalization policy

Finish only after the final CSV and manifest both exist and have been read back. The final response should mention the selected method, validation check, and exact path `./input/final_submission.csv`. Do not keep optimizing indefinitely after a quality-backed final exists.
