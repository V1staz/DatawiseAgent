"""Task-family routing for DataModeling harness runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .contracts import DataModelingContract
from .data_availability import DataAvailabilityReport
from .profiler import DataProfile
from .skill_cards import skill_card_ids_for_route


@dataclass(slots=True)
class TaskRoute:
    family: str
    metric_family: str
    required_skills: list[str]
    skill_card_ids: list[str]
    quality_checks: list[str]
    forbidden_shortcuts: list[str]
    rationale: str
    scopes: list[str] = field(default_factory=list)
    facets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def route_task(contract: DataModelingContract, profile: DataProfile | None = None, availability: DataAvailabilityReport | None = None) -> TaskRoute:
    metric = contract.metric
    kinds = set(contract.task_kinds)
    lower = " ".join([contract.name, metric.name, metric.prediction_kind, *contract.risk_flags, *contract.recipe_hints]).lower()

    if availability is not None and not availability.usable_for_modeling:
        family = "data_unavailable_or_pointer"
    elif "cellular_automaton" in kinds or "conway" in lower or "game of life" in lower:
        family = "simulation_or_special"
    elif "multi_target" in kinds or len(contract.submission.target_columns) > 1:
        family = "multi_output"
    elif contract.text_columns or "text_" in " ".join(kinds) or metric.prediction_kind == "text":
        family = "text_modeling"
    elif "time_series_forecasting" in kinds or contract.datetime_columns or any(tok in lower for tok in ("forecast", "time series", "date")):
        family = "time_series_or_grouped"
    elif metric.prediction_kind == "probability" or metric.probability_output:
        family = "tabular_binary_classification"
    elif metric.prediction_kind in {"class", "top_k_class"}:
        family = "tabular_multiclass_classification"
    elif metric.prediction_kind == "regression":
        family = "tabular_regression"
    else:
        family = "tabular_regression"

    metric_family = _metric_family(metric.name, metric.prediction_kind)
    facets = _route_facets(contract, family, metric_family, kinds, lower)
    required_skills = _skills_for_family(family, metric_family, facets)
    skill_card_ids = skill_card_ids_for_route(family, metric_family, facets=facets)
    quality_checks = _quality_checks_for_family(family, metric_family)
    forbidden_shortcuts = [
        "hidden_answers_or_gt_files",
        "sample_submission_target_copy",
        "raw_object_columns_into_numeric_estimators",
        "final_submission_without_manifest",
    ]
    if family not in {"simulation_or_special", "data_unavailable_or_pointer"}:
        forbidden_shortcuts.append("constant_final_without_validation_evidence")
    if metric.probability_output:
        forbidden_shortcuts.append("hard_labels_for_probability_metric")
    if family == "data_unavailable_or_pointer":
        forbidden_shortcuts.extend(["modeling_on_lfs_pointer_bytes", "text_misroute_from_pointer_file"])

    scopes = ["all", family]
    if family.startswith("tabular"):
        scopes.append("tabular")
    if family == "multi_output":
        scopes.append("multi_output")
    if family == "text_modeling":
        scopes.append("text_modeling")
    if family == "data_unavailable_or_pointer":
        scopes.append("data_availability")
    scopes.extend(facet for facet in facets if facet not in scopes)

    rationale_parts = [f"metric={metric.name}/{metric.prediction_kind}"]
    if contract.text_columns:
        rationale_parts.append(f"text_columns={contract.text_columns[:4]}")
    if contract.datetime_columns:
        rationale_parts.append(f"datetime_columns={contract.datetime_columns[:4]}")
    if getattr(contract, "group_columns", []):
        rationale_parts.append(f"group_columns={contract.group_columns[:4]}")
    if len(contract.submission.target_columns) > 1:
        rationale_parts.append(f"submission_targets={len(contract.submission.target_columns)}")
    if profile and profile.warnings:
        rationale_parts.append(f"profile_warnings={profile.warnings[:3]}")
    if availability is not None and not availability.usable_for_modeling:
        rationale_parts.append(f"data_availability_blocked=lfs:{availability.lfs_pointer_files},missing:{availability.missing_files}")

    return TaskRoute(
        family=family,
        metric_family=metric_family,
        required_skills=required_skills,
        skill_card_ids=skill_card_ids,
        quality_checks=quality_checks,
        forbidden_shortcuts=forbidden_shortcuts,
        rationale="; ".join(rationale_parts),
        scopes=scopes,
        facets=facets,
    )


def _metric_family(name: str, prediction_kind: str) -> str:
    if name in {"roc_auc", "log_loss", "average_precision", "normalized_gini"} or prediction_kind == "probability":
        return "probability"
    if name in {"accuracy", "f1", "quadratic_weighted_kappa", "mpa_at_3"} or prediction_kind in {"class", "top_k_class"}:
        return "classification"
    if name in {"rmse", "mse", "mae", "median_absolute_error", "rmsle", "smape", "r2", "pearson"} or prediction_kind == "regression":
        return "regression"
    if prediction_kind == "text" or name == "jaccard":
        return "text"
    return "unknown"


def _route_facets(contract: DataModelingContract, family: str, metric_family: str, kinds: set[str], lower: str) -> list[str]:
    facets: list[str] = []
    if metric_family == "probability" and family != "tabular_binary_classification":
        facets.append("tabular_binary_classification")
    if (contract.datetime_columns or getattr(contract, "group_columns", []) or "time_series_forecasting" in kinds or any(tok in lower for tok in ("forecast", "time series", "date"))) and family != "time_series_or_grouped":
        facets.append("time_series_or_grouped")
    if (contract.text_columns or "text_" in " ".join(kinds)) and family != "text_modeling":
        facets.append("text_modeling")
    if len(contract.submission.target_columns) > 1 and family != "multi_output":
        facets.append("multi_output")
    return _dedupe(facets)


def _skills_for_family(family: str, metric_family: str, facets: list[str] | None = None) -> list[str]:
    common = ["inspect_schema", "safe_fallback", "manifested_finalization"]
    mapping = {
        "tabular_binary_classification": ["tabular_binary_classification", "probability_calibration", "pipeline_preprocessing"],
        "tabular_multiclass_classification": ["tabular_multiclass_classification", "label_preservation", "pipeline_preprocessing"],
        "tabular_regression": ["tabular_regression", "target_transform_check", "pipeline_preprocessing"],
        "text_modeling": ["text_modeling", "vectorization", "multi_target_if_needed"],
        "time_series_or_grouped": ["time_series_or_grouped", "leakage_aware_validation", "group_features"],
        "multi_output": ["multi_output", "per_target_validation", "independent_predictions"],
        "simulation_or_special": ["simulation_or_special", "task_rule_inspection"],
        "data_unavailable_or_pointer": ["data_availability_guard", "restore_public_data_before_modeling"],
    }
    skills = common + mapping.get(family, [metric_family])
    for facet in facets or []:
        skills.extend(mapping.get(facet, []))
    return _dedupe(skills)


def _quality_checks_for_family(family: str, metric_family: str) -> list[str]:
    checks = ["manifest_present", "final_not_sample_copy", "exact_schema", "non_missing_predictions"]
    if metric_family in {"probability", "regression"}:
        checks.extend(["prediction_diversity", "finite_numeric_targets"])
    if metric_family == "probability":
        checks.extend(["probabilities_in_range", "not_all_hard_labels_if_auc_like"])
    if family == "multi_output":
        checks.append("per_target_diversity")
    if family == "text_modeling":
        checks.append("text_outputs_not_numeric_constants")
    if family == "time_series_or_grouped":
        checks.append("leakage_aware_manifest")
    if family == "data_unavailable_or_pointer":
        checks.extend(["public_data_available", "not_modeling_on_pointer_bytes"])
    return checks


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
