"""Benchmark-grounded modeling protocol planning.

The protocol is a compact, label-blind plan that sits between coarse task
routing and prompt execution.  It preserves a shared data-science stage flow
while specializing the checks that differ across the DataModeling benchmark's
currently observed task families.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .contracts import DataModelingContract
from .data_availability import DataAvailabilityReport, inspect_data_availability
from .profiler import DataProfile
from .task_router import TaskRoute


COMMON_MODELING_STAGES: tuple[str, ...] = (
    "validate_public_data_availability",
    "inspect_schema_and_target_contract",
    "clean_missing_and_malformed_values",
    "eda_hypotheses_from_public_train_only",
    "choose_leakage_aware_split_policy",
    "build_feature_encoding_pipeline",
    "train_bounded_baseline_and_candidate_models",
    "compare_against_defensive_fallback",
    "finalize_submission_and_manifest",
)


DEFERRED_ABSENT_TASK_FAMILIES: tuple[str, ...] = (
    "image_multimodal_modeling",
    "graph_relational_modeling",
    "causal_uplift_survival_decision_modeling",
    "clustering_segmentation",
    "dimensionality_reduction_representation",
    "anomaly_detection",
)


@dataclass(slots=True)
class ModelingProtocolPlan:
    target_type: str
    data_structure: list[str]
    split_constraint: str
    business_objective: str
    common_stages: list[str]
    task_specific_stages: list[str]
    required_manifest_evidence: list[str]
    deferred_task_families: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_summary(self) -> str:
        payload = {
            "target_type": self.target_type,
            "data_structure": self.data_structure,
            "split_constraint": self.split_constraint,
            "business_objective": self.business_objective,
            "common_stages": self.common_stages,
            "task_specific_stages": self.task_specific_stages,
            "required_manifest_evidence": self.required_manifest_evidence,
            "warnings": self.warnings,
        }
        import json

        return json.dumps(payload, ensure_ascii=False, indent=2)


def build_protocol_plan(
    contract: DataModelingContract,
    profile: DataProfile,
    route: TaskRoute,
    availability: DataAvailabilityReport | None = None,
) -> ModelingProtocolPlan:
    """Build a staged modeling protocol from label-blind task evidence."""

    availability = availability or inspect_data_availability(contract)
    warnings = _protocol_warnings(contract, profile, route, availability)
    if not availability.usable_for_modeling:
        return ModelingProtocolPlan(
            target_type="data_unavailable_or_pointer",
            data_structure=_data_structure(contract, profile, availability),
            split_constraint="not_applicable_data_unavailable",
            business_objective="restore_or_skip_unusable_public_data_before_modeling",
            common_stages=list(COMMON_MODELING_STAGES[:2]),
            task_specific_stages=[
                "detect_git_lfs_pointer_or_missing_public_files",
                "avoid_text_or_tabular_misrouting_from_pointer_bytes",
                "surface_data_availability_blocker_in_artifacts",
            ],
            required_manifest_evidence=[
                "data_availability_status",
                "blocked_public_files",
            ],
            deferred_task_families=list(DEFERRED_ABSENT_TASK_FAMILIES),
            warnings=warnings,
        )

    target_type = _target_type(contract, profile, route)
    return ModelingProtocolPlan(
        target_type=target_type,
        data_structure=_data_structure(contract, profile, availability),
        split_constraint=_split_constraint(contract, profile, route, target_type),
        business_objective=_business_objective(contract, route, target_type),
        common_stages=list(COMMON_MODELING_STAGES),
        task_specific_stages=_task_specific_stages(contract, profile, route, target_type),
        required_manifest_evidence=_required_manifest_evidence(contract, route, target_type),
        deferred_task_families=list(DEFERRED_ABSENT_TASK_FAMILIES),
        warnings=warnings,
    )


def _target_type(contract: DataModelingContract, profile: DataProfile, route: TaskRoute) -> str:
    metric = contract.metric
    target_count = len(contract.submission.target_columns)
    if route.family == "text_modeling" or metric.prediction_kind == "text":
        return "text_modeling"
    if route.family == "simulation_or_special":
        return "special_simulation"
    if route.family == "time_series_or_grouped" or contract.datetime_columns or "time_series_forecasting" in contract.task_kinds:
        if target_count > 1 or "multi_target" in contract.task_kinds:
            return "time_series_multi_output_forecasting"
        return "time_series_forecasting"
    if metric.prediction_kind == "top_k_class":
        return "ranking_or_topk_multiclass"
    if target_count > 1:
        if metric.probability_output or metric.prediction_kind == "probability":
            return "multilabel_probability"
        if _all_targets_numeric(profile):
            return "multi_output_regression"
        return "multi_output_prediction"
    if metric.probability_output or metric.prediction_kind == "probability":
        return "binary_probability"
    if metric.prediction_kind == "class":
        return "multiclass_classification"
    if metric.prediction_kind == "regression":
        return "regression"
    return route.family


def _data_structure(contract: DataModelingContract, profile: DataProfile, availability: DataAvailabilityReport) -> list[str]:
    structures: list[str] = []
    if availability.lfs_pointer_files:
        structures.append("lfs_pointer")
    if contract.text_columns:
        structures.append("text")
    if contract.datetime_columns:
        structures.append("datetime")
    if getattr(contract, "group_columns", []):
        structures.append("grouped")
    if len(contract.submission.target_columns) > 1 or "multi_target" in contract.task_kinds:
        structures.append("multi_output")
    if _has_high_cardinality_categorical(profile):
        structures.append("high_cardinality_categorical")
    if contract.categorical_columns:
        structures.append("categorical")
    if contract.numeric_columns:
        structures.append("numeric")
    if not structures or any(item in structures for item in ("numeric", "categorical", "datetime")):
        structures.insert(0, "tabular")
    return _dedupe(structures)


def _split_constraint(contract: DataModelingContract, profile: DataProfile, route: TaskRoute, target_type: str) -> str:
    lower = " ".join([contract.name, contract.description_excerpt, *contract.risk_flags, *contract.recipe_hints]).lower()
    if target_type.startswith("time_series") or contract.datetime_columns or "time_series_or_grouped" == route.family:
        return "time_ordered_or_group_aware"
    if getattr(contract, "group_columns", []):
        return "group_aware"
    if target_type in {"ranking_or_topk_multiclass"}:
        return "query_or_group_aware_if_group_column_exists"
    if any(token in lower for token in ("group", "customer", "user", "store", "patient", "series", "forecast")):
        return "group_aware_if_repeated_entity_detected"
    if target_type in {"binary_probability", "multiclass_classification", "multilabel_probability"}:
        return "stratified"
    if profile.train_rows_sampled < 100:
        return "repeated_kfold_or_holdout_with_small_data_caution"
    return "random_or_kfold"


def _business_objective(contract: DataModelingContract, route: TaskRoute, target_type: str) -> str:
    metric = contract.metric
    if target_type in {"binary_probability", "multilabel_probability"}:
        return f"probability_ranking_quality_with_calibration_awareness:{metric.name}"
    if target_type == "ranking_or_topk_multiclass":
        return f"top_k_or_relative_order_quality:{metric.name}"
    if target_type in {"multiclass_classification", "multi_output_prediction"}:
        return f"class_coverage_and_error_reduction:{metric.name}"
    if target_type in {"regression", "multi_output_regression"}:
        return f"numeric_error_reduction_with_distribution_checks:{metric.name}"
    if target_type.startswith("time_series"):
        return f"future_generalization_without_temporal_leakage:{metric.name}"
    if target_type == "text_modeling":
        return f"text_signal_extraction_or_similarity:{metric.name}"
    if route.family == "simulation_or_special":
        return "task_rule_faithfulness_over_generic_ml"
    return f"metric_aligned_public_validation:{metric.name}"


def _task_specific_stages(contract: DataModelingContract, profile: DataProfile, route: TaskRoute, target_type: str) -> list[str]:
    stages: list[str] = []
    if target_type in {"binary_probability", "multilabel_probability"}:
        stages.extend([
            "check_class_balance_and_minority_recall_proxy",
            "emit_probabilities_not_hard_labels",
            "calibrate_or_rank_probability_scores",
        ])
    if target_type == "multiclass_classification":
        stages.extend([
            "preserve_label_space_and_dtype",
            "inspect_confusion_or_class_coverage_proxy",
        ])
    if target_type == "ranking_or_topk_multiclass":
        stages.extend([
            "construct_top_k_label_output_format",
            "avoid_collapsing_ranked_choices_to_single_label",
        ])
    if target_type in {"regression", "multi_output_regression"}:
        stages.extend([
            "inspect_target_distribution_skew_and_outliers",
            "choose_metric_safe_target_transform",
            "clip_or_inverse_transform_predictions_when_metric_requires",
        ])
    if target_type.startswith("time_series"):
        stages.extend([
            "sort_by_time_before_validation",
            "create_lag_rolling_calendar_features_without_future_leakage",
            "use_last_window_or_grouped_rolling_validation",
        ])
    if getattr(contract, "group_columns", []):
        stages.extend([
            "detect_group_columns_and_repeated_entities",
            "use_group_exclusive_validation_when_groups_repeat",
        ])
    if target_type == "text_modeling":
        stages.extend([
            "normalize_text_without_target_leakage",
            "select_lightweight_text_vectorizer_or_embedding",
            "handle_empty_or_long_text_outputs",
        ])
    if "multi_output" in target_type or len(contract.submission.target_columns) > 1:
        stages.extend([
            "validate_each_target_column_separately",
            "avoid_constant_columns_in_multi_output_submission",
        ])
    if _has_high_cardinality_categorical(profile):
        stages.append("use_leakage_safe_high_cardinality_encoding")
    return _dedupe(stages or ["metric_aligned_baseline_and_manifest"])


def _required_manifest_evidence(contract: DataModelingContract, route: TaskRoute, target_type: str) -> list[str]:
    evidence = [
        "stage",
        "modeling_method",
        "metric_direction",
        "validation_metric_value",
        "baseline_metric_value",
        "trained_rows",
        "selected_reason",
        "prediction_summary",
        "split_policy",
        "features_used",
        "leakage_controls",
    ]
    if target_type.startswith("time_series") or route.family == "time_series_or_grouped":
        evidence.extend(["validation_strategy", "time_or_group_columns", "future_leakage_checks"])
    elif getattr(contract, "group_columns", []):
        evidence.extend(["validation_strategy", "time_or_group_columns"])
    if target_type in {"binary_probability", "multilabel_probability"}:
        evidence.extend(["probability_range", "calibration_or_ranking_check"])
    if target_type == "text_modeling":
        evidence.extend(["text_columns_used", "text_preprocessing"])
    if "multi_output" in target_type or len(contract.submission.target_columns) > 1:
        evidence.append("per_target_prediction_summary")
    return _dedupe(evidence)


def _protocol_warnings(
    contract: DataModelingContract,
    profile: DataProfile,
    route: TaskRoute,
    availability: DataAvailabilityReport,
) -> list[str]:
    warnings = list(profile.warnings)
    if availability.lfs_pointer_files:
        warnings.append(f"git_lfs_pointer_files:{','.join(availability.lfs_pointer_files)}")
    if availability.missing_files:
        warnings.append(f"missing_public_files:{','.join(availability.missing_files)}")
    if route.family == "text_modeling" and availability.lfs_pointer_files:
        warnings.append("text_route_may_be_lfs_pointer_artifact")
    if len(contract.submission.target_columns) > 1 and route.family != "multi_output":
        warnings.append("multi_target_submission_requires_per_column_plan")
    return _dedupe(warnings)


def _all_targets_numeric(profile: DataProfile) -> bool:
    if not profile.target_summary:
        return False
    for summary in profile.target_summary.values():
        dtype = str(summary.get("dtype", "")).lower()
        if not any(token in dtype for token in ("int", "float", "uint")):
            return False
    return True


def _has_high_cardinality_categorical(profile: DataProfile) -> bool:
    for col in profile.columns:
        if col.role == "feature" and col.dtype == "object" and col.unique_count >= 50 and col.unique_fraction >= 0.2:
            return True
    return False


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
