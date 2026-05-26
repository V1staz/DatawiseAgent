from __future__ import annotations

from typing import Any


SKILL_REGISTRY: dict[str, dict[str, Any]] = {
    "summary_statistics_skill": {
        "task_type": "summary_statistics",
        "checklist": [
            "Confirm the statistic column, aggregation scope, filters, and grouping before computing.",
            "Record the denominator or sample count used for each statistic.",
            "Do not add unrequested cleaning, features, or model changes.",
            "Round only at the final answer step unless the question explicitly says otherwise.",
        ],
        "verification_rules": [
            "required_columns_exist",
            "filter_sample_count_nonzero",
            "groupby_categories_recorded",
            "decimal_places_match_contract",
        ],
    },
    "distribution_analysis_skill": {
        "task_type": "distribution_analysis",
        "checklist": [
            "Confirm the analyzed column and any subgroup or filter before testing distribution.",
            "Drop or impute missing values only if the question requires it.",
            "If a statistical test is requested, report statistic, p-value, alpha, and decision.",
        ],
        "verification_rules": [
            "required_columns_exist",
            "normality_test_method_matches_contract",
            "p_value_alpha_if_required",
            "final_answer_contains_requested_values",
        ],
    },
    "correlation_skill": {
        "task_type": "correlation_analysis",
        "checklist": [
            "Confirm the variables and correlation method from the contract.",
            "Use only the required subset and handle missing values consistently with the question.",
            "For strongest correlation, compare by abs(r) while preserving the original sign.",
            "If significance is requested, report p-value and the alpha threshold decision.",
        ],
        "verification_rules": [
            "required_columns_exist",
            "correlation_abs_r_if_strongest",
            "correlation_sign_retained",
            "p_value_alpha_if_required",
            "decimal_places_match_contract",
        ],
    },
    "outlier_skill": {
        "task_type": "outlier_detection",
        "checklist": [
            "Use the outlier method and threshold specified in the contract.",
            "Do not switch between IQR, z-score, and other protocols unless requested.",
            "Record outlier count and, when requested, the full id/list without truncation.",
            "If outliers are removed, record before and after shape.",
        ],
        "verification_rules": [
            "outlier_method_matches_contract",
            "outlier_count_recorded",
            "outlier_list_not_truncated",
            "shape_before_after_if_removed",
        ],
    },
    "preprocessing_skill": {
        "task_type": "comprehensive_preprocessing",
        "checklist": [
            "Convert the contract preprocessing steps into an ordered checklist.",
            "Execute steps in the requested order without adding extra cleaning.",
            "After each step, record shape and missing-value counts for key columns.",
            "If the question asks to save a processed file, verify the saved path exists.",
        ],
        "verification_rules": [
            "preprocessing_order",
            "missing_values_shape_recorded",
            "filter_sample_count_nonzero",
            "saved_file_exists_if_required",
        ],
    },
    "feature_engineering_skill": {
        "task_type": "feature_engineering",
        "checklist": [
            "Echo the feature formula before computing it.",
            "Confirm numerator, denominator, units, and source columns.",
            "Create the new column exactly as named in the contract.",
            "Print the first 5 rows of source and derived columns for verification.",
        ],
        "verification_rules": [
            "feature_formula_echoed",
            "feature_source_columns_exist",
            "feature_column_created",
            "feature_first_five_rows_checked",
        ],
    },
    "machine_learning_protocol_skill": {
        "task_type": "machine_learning",
        "checklist": [
            "Use only the feature columns, target, split, random_state, and metric in the contract.",
            "Do not tune hyperparameters unless explicitly requested.",
            "Print X.shape, y.shape, and feature names after preprocessing.",
            "Evaluate on the correct split and report only the requested metric.",
        ],
        "verification_rules": [
            "ml_random_state",
            "ml_split",
            "ml_feature_names",
            "ml_metric",
            "no_unrequested_tuning",
        ],
    },
    "final_answer_extraction_skill": {
        "task_type": "final_formatting",
        "checklist": [
            "Create a machine-readable final block before any @answer output.",
            "Include answer name, value as string, raw_value when available, source_variable, and format_checked.",
            "Do not let reformat infer answers from the full notebook log.",
            "Fail with ERROR_NEED_RETRY if the final block is missing, placeholder-like, or invalid.",
        ],
        "verification_rules": [
            "final_block_present",
            "answer_names_match_contract",
            "no_placeholder_values",
            "decimal_places_match_contract",
            "quote_and_container_format_match_contract",
        ],
    },
}


TASK_TYPE_TO_SKILL = {
    value["task_type"]: name for name, value in SKILL_REGISTRY.items()
}

