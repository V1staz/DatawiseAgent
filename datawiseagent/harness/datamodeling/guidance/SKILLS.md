# DataModeling Skill Tree

The router selects a small branch of this tree for each task. Use the selected branch first; do not paste every skill into the working code.

## tabular_binary_classification
- Use stratified train/validation split when feasible.
- Output probabilities for AUC, log-loss, average precision, normalized gini, or probability sample submissions.
- Start with preprocessing pipeline: impute numeric/categorical, one-hot encode categoricals with unknown handling, fit logistic/HistGradientBoosting/LightGBM when available.
- Reject constant probability predictions unless validation proves the baseline is already optimal.

## tabular_multiclass_classification
- Preserve label strings for class-label submissions.
- Use `predict_proba` only when the sample submission expects class probabilities or top-k strings.
- For MAP@3/top-k tasks, generate three space-separated class labels ordered by probability.
- Check that unseen categories cannot crash prediction.

## tabular_regression
- Compare simple median/mean fallback against at least one fitted regressor.
- Respect metric transforms: RMSLE/SMAPE require non-negative predictions; clip only after predicting.
- For skewed targets, try log1p target modeling only when the metric supports it and invert with expm1.

## text_modeling
- Use TF-IDF/HashingVectorizer or concise text-derived features before linear models.
- For multi-output text targets, use one model per target or a multi-output wrapper.
- Avoid numeric constants for text/span outputs unless the task explicitly asks for numeric text IDs.

## time_series_or_grouped
- Avoid random leakage when the task is chronological or grouped.
- Use time/group-aware validation if obvious date/group columns exist.
- Include recent-value, lag, rolling, or group aggregate features only from public train data.

## multi_output
- Validate every target column separately.
- Do not row-normalize independent binary probabilities.
- Report per-target prediction diversity in the manifest.

## simulation_or_special
- Prefer task-specific deterministic logic only when it is derived from public task rules and train/test files.
- If a learned model is used, still validate output shape/range against the sample submission.
