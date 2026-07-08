# Harness Tool Surfaces

These are conceptual harness tools exposed through files and validation artifacts. They constrain the agent without replacing the agent's modeling work.

## contract.json
Label-blind task contract: metric, target columns, expected row count, risk flags, and local resplit schema. Treat it as authoritative over original Kaggle text when row counts disagree.

## profile.json
Public-data schema profile: sampled dtypes, missingness, unique counts, target summaries, and warnings. Use it to choose preprocessing and model family before writing code.

## route.json
Task-family route selected by the harness. It names required skills, quality checks, and forbidden shortcuts. If it conflicts with a quick intuition, inspect data and explain why before deviating.

## submission validator
Enforces exact columns, row count, ID/order, target dtype, missing values, metric range, non-negative requirements, and sample-copy rejection.

## quality gate
Label-blind finalization gate. It checks manifest presence, fallback misuse, prediction diversity, NaN/constant risks for correlation/classification metrics, and public validation evidence. Passing it does not prove leaderboard quality; failing it means the answer is probably a fallback or incomplete run.
