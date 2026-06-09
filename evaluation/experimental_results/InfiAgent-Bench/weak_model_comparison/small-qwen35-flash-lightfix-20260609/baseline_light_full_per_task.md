# qwen3.5-flash small_error_set comparison

## Metrics

| setting | ABQ | PASQ | UASQ | easy | medium | hard | format_error | reformat_missing | verifier_blocked | stuck | avg_runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_raw | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 | 47 | 47 | 0 | 0 | 30.24 |
| baseline_reformat | 0.4894 | 0.6383 | 0.6705 | 0.5556 | 0.5385 | 0.44 | 0 | 0 | 0 | 0 | 30.24 |
| harness_light | 0.5957 | 0.6525 | 0.6932 | 0.6667 | 0.5385 | 0.6 | 1 | 3 | 0 | 1 | 47.03 |
| harness_full | 0.6383 | 0.6950 | 0.7273 | 0.6667 | 0.6154 | 0.64 | 0 | 2 | 4 | 1 | 92.79 |

## Full vs baseline status

{"both_correct": 21, "fixed": 7, "regressed": 1, "stuck": 1, "unchanged_wrong": 13, "verifier_blocked": 4}

## Per-task table

| task_id | difficulty | concepts | known_error_type | baseline_correct | light_correct | full_correct | full_status | task_status_full | suspected_module | failure_reason |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | hard | Machine Learning | ml_protocol_error | True | True | True | both_correct | success |  |  |
| 23 | hard | Machine Learning;Summary Statistics | ml_protocol_error | True | True | True | both_correct | success |  |  |
| 55 | easy | Summary Statistics | aggregation_error | True | True | True | both_correct | success |  |  |
| 56 | easy | Distribution Analysis;Summary Statistics | column_selection_error | True | True | True | both_correct | success |  |  |
| 57 | easy | Correlation Analysis | statistical_method_error | False | True | True | fixed | success | contract/skills/finalizer |  |
| 59 | easy | Distribution Analysis;Summary Statistics;Feature Engineering | filtering_error | True | True | True | both_correct | success |  |  |
| 62 | medium | Outlier Detection;Distribution Analysis | outlier_protocol_error | True | False | True | both_correct | success |  |  |
| 117 | medium | Correlation Analysis | statistical_method_error | True | True | True | both_correct | success |  |  |
| 124 | hard | Summary Statistics;Correlation Analysis | significance_judgment_error | False | False | False | unchanged_wrong | success |  | both baseline and harness_full failed closed-form eval |
| 125 | hard | Correlation Analysis;Machine Learning | format_error | False | True | True | fixed | success | contract/skills/finalizer |  |
| 133 | hard | Comprehensive Data Preprocessing | preprocessing_order_error | False | False | True | fixed | success | contract/skills/finalizer |  |
| 137 | hard | Feature Engineering;Machine Learning | ml_protocol_error | True | True | True | both_correct | success |  |  |
| 142 | hard | Correlation Analysis | statistical_method_error | True | True | True | both_correct | success |  |  |
| 219 | medium | Outlier Detection | format_error | False | False | False | verifier_blocked | verifier_blocked | verifier | harness_full blocked by verifier before final success state |
| 252 | medium | Distribution Analysis | statistical_method_error | False | False | False | unchanged_wrong | success |  | both baseline and harness_full failed closed-form eval |
| 271 | hard | Comprehensive Data Preprocessing | preprocessing_order_error | False | True | True | fixed | success | contract/skills/finalizer |  |
| 273 | hard | Correlation Analysis;Outlier Detection | outlier_protocol_error | False | True | True | fixed | success | contract/skills/finalizer |  |
| 275 | hard | Comprehensive Data Preprocessing;Feature Engineering;Machine Learning | feature_definition_error | True | True | True | both_correct | success |  |  |
| 297 | hard | Summary Statistics;Comprehensive Data Preprocessing | filtering_error | False | False | False | unchanged_wrong | success |  | both baseline and harness_full failed closed-form eval |
| 298 | medium | Distribution Analysis | significance_judgment_error | False | False | False | unchanged_wrong | success |  | both baseline and harness_full failed closed-form eval |
| 300 | hard | Correlation Analysis;Comprehensive Data Preprocessing | format_error | False | False | False | unchanged_wrong | success |  | both baseline and harness_full failed closed-form eval |
| 310 | hard | Correlation Analysis | statistical_method_error | True | True | True | verifier_blocked | verifier_blocked | verifier | harness_full blocked by verifier before final success state |
| 359 | easy | Distribution Analysis | statistical_method_error | False | False | False | unchanged_wrong | success |  | both baseline and harness_full failed closed-form eval |
| 408 | medium | Correlation Analysis | statistical_method_error | True | True | False | regressed | success | harness_full_oracle/verifier/finalizer | baseline correct but harness_full answer failed closed-form eval |
| 418 | medium | Outlier Detection | outlier_protocol_error | True | True | True | both_correct | success |  |  |
| 431 | hard | Correlation Analysis;Comprehensive Data Preprocessing | filtering_error | False | False | False | unchanged_wrong | success |  | both baseline and harness_full failed closed-form eval |
| 450 | easy | Summary Statistics | format_error | False | True | False | unchanged_wrong | success |  | both baseline and harness_full failed closed-form eval |
| 451 | easy | Comprehensive Data Preprocessing | format_error | True | False | True | both_correct | success |  |  |
| 452 | hard | Correlation Analysis;Feature Engineering | format_error | True | True | True | both_correct | success |  |  |
| 492 | easy | Summary Statistics | aggregation_error | True | True | True | both_correct | success |  |  |
| 496 | hard | Feature Engineering;Summary Statistics | feature_definition_error | True | True | True | both_correct | success |  |  |
| 513 | medium | Correlation Analysis;Distribution Analysis | filtering_error | False | True | True | fixed | success | contract/skills/finalizer |  |
| 528 | medium | Outlier Detection;Comprehensive Data Preprocessing | outlier_protocol_error | False | True | True | fixed | success | contract/skills/finalizer |  |
| 529 | hard | Correlation Analysis;Feature Engineering | statistical_method_error | True | True | True | both_correct | success |  |  |
| 550 | hard | Comprehensive Data Preprocessing;Distribution Analysis | format_error | False | False | False | verifier_blocked | verifier_blocked | verifier | harness_full blocked by verifier before final success state |
| 554 | easy | Summary Statistics;Distribution Analysis | filtering_error | False | False | False | unchanged_wrong | reformat_missing |  | both baseline and harness_full failed closed-form eval |
| 572 | hard | Summary Statistics;Correlation Analysis | format_error | False | False | False | unchanged_wrong | success |  | both baseline and harness_full failed closed-form eval |
| 589 | medium | Feature Engineering | feature_definition_error | True | True | True | both_correct | success |  |  |
| 647 | hard | Feature Engineering;Distribution Analysis | feature_definition_error | False | True | True | stuck | execution_error | execution | execution/model error |
| 662 | hard | Feature Engineering;Summary Statistics | format_error | False | False | False | unchanged_wrong | success |  | both baseline and harness_full failed closed-form eval |
| 684 | medium | Distribution Analysis | reformat_error | True | False | True | both_correct | success |  |  |
| 722 | hard | Summary Statistics;Comprehensive Data Preprocessing | preprocessing_order_error | True | True | True | both_correct | success |  |  |
| 730 | medium | Correlation Analysis | format_error | True | True | True | both_correct | success |  |  |
| 733 | hard | Feature Engineering | feature_definition_error | True | True | True | both_correct | success |  |  |
| 734 | hard | Correlation Analysis;Comprehensive Data Preprocessing | reformat_error | False | False | False | verifier_blocked | verifier_blocked | verifier | harness_full blocked by verifier before final success state |
| 741 | medium | Feature Engineering;Comprehensive Data Preprocessing | feature_definition_error | False | False | False | unchanged_wrong | success |  | both baseline and harness_full failed closed-form eval |
| 743 | hard | Comprehensive Data Preprocessing;Feature Engineering | preprocessing_order_error | False | False | False | unchanged_wrong | success |  | both baseline and harness_full failed closed-form eval |
