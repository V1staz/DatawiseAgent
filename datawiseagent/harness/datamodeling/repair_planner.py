"""Targeted repair planning for DataModeling harness attempts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import DataModelingContract
from .quality import QualityReport
from .submission import SubmissionValidationResult
from .task_router import TaskRoute


@dataclass(frozen=True, slots=True)
class RepairAction:
    id: str
    priority: int
    problem: str
    required_steps: tuple[str, ...]
    success_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_summary(self) -> str:
        steps = "; ".join(self.required_steps)
        evidence = "; ".join(self.success_evidence)
        return f"- `{self.id}`: {self.problem}. Steps: {steps}. Evidence: {evidence}."


def plan_repair_actions(
    contract: DataModelingContract,
    validation: SubmissionValidationResult,
    *,
    quality: QualityReport | None = None,
    route: TaskRoute | None = None,
    protocol_plan: Any | None = None,
    limit: int = 3,
) -> list[RepairAction]:
    """Return bounded, label-blind repair actions for the next agent attempt."""

    actions: list[RepairAction] = []
    validation_errors = validation.errors or []
    quality_errors = quality.errors if quality is not None else []
    all_errors = [*validation_errors, *quality_errors]

    if not validation.passed:
        if any("submission_file_missing" in err for err in validation_errors):
            actions.append(_missing_final_action())
        if any("columns" in err or "row_count" in err or "id_" in err for err in validation_errors):
            actions.append(_schema_action(contract))
        if any("outside_metric_range" in err for err in validation_errors):
            actions.append(_range_action(contract))
        if any("sample_submission" in err for err in validation_errors):
            actions.append(_sample_copy_action())

    if any("probability_metric_predictions_look_like_hard_labels" in err for err in all_errors):
        actions.append(_probability_action())
    if any("multi_output_target_columns_constant" in err for err in all_errors):
        actions.append(_multi_output_action())
    if any("multiclass_predictions_low_class_coverage" in err for err in all_errors):
        actions.append(_multiclass_action())
    if any("text_modeling_constant_output" in err for err in all_errors):
        actions.append(_text_action())
    if any("time_group_manifest_missing_validation_strategy" in err for err in all_errors):
        actions.append(_time_group_action())
    if any(err.startswith("protocol_public_data_unavailable_for_modeling") for err in all_errors):
        actions.append(_public_data_action())
    if any(err.startswith("protocol_manifest_missing_required_evidence") for err in all_errors):
        actions.append(_protocol_manifest_action(protocol_plan))
    if any(err.startswith("protocol_probability_manifest_missing") for err in all_errors):
        actions.append(_protocol_probability_action())
    if any(err.startswith("protocol_time_manifest_missing") for err in all_errors):
        actions.append(_protocol_time_action())
    if any(err.startswith("protocol_text_manifest_missing") for err in all_errors):
        actions.append(_protocol_text_action())
    if any(err.startswith("protocol_multi_output_manifest_missing") for err in all_errors):
        actions.append(_protocol_multi_output_action())
    if any(err.startswith("stage_agent_") for err in all_errors):
        actions.append(_stage_agent_outputs_action())
    if any(err.startswith("verified_evidence_metric_conflict") for err in all_errors):
        actions.append(_verified_evidence_conflict_action())
    if any("manifest" in err for err in all_errors):
        actions.append(_manifest_action())
    if any("constant" in err for err in all_errors) and not any(action.id == "repair_nonconstant_modeling" for action in actions):
        actions.append(_nonconstant_action(route))
    if any("fallback" in err for err in all_errors):
        actions.append(_fallback_action())

    if not actions:
        actions.append(_route_default_action(route))

    deduped: dict[str, RepairAction] = {}
    for action in sorted(actions, key=lambda item: (item.priority, item.id)):
        deduped.setdefault(action.id, action)
    return list(deduped.values())[:limit]


def actions_to_dict(actions: list[RepairAction]) -> list[dict[str, Any]]:
    return [action.to_dict() for action in actions]


def _missing_final_action() -> RepairAction:
    return RepairAction(
        id="repair_missing_final_submission",
        priority=10,
        problem="The required ./input/final_submission.csv is missing.",
        required_steps=(
            "find any modeled candidate under ./input/",
            "copy only a modeled candidate to ./input/final_submission.csv",
            "if only fallback exists, rerun modeling before using final_submission.csv",
        ),
        success_evidence=("final_submission.csv exists", "validator reports exact schema and row count"),
    )


def _schema_action(contract: DataModelingContract) -> RepairAction:
    return RepairAction(
        id="repair_submission_schema",
        priority=20,
        problem="The CSV schema, row count, ID order, or columns do not match the contract.",
        required_steps=(
            f"rebuild output from sample_submission columns {contract.submission.columns}",
            "preserve test/sample ID order exactly",
            "write the corrected CSV to ./input/final_submission.csv",
        ),
        success_evidence=("columns match contract", "row count and ID order match contract"),
    )


def _range_action(contract: DataModelingContract) -> RepairAction:
    metric_range = contract.metric.bounded_range if contract.metric.bounded_range is not None else "metric/domain-valid range"
    return RepairAction(
        id="repair_metric_range",
        priority=25,
        problem="Predictions violate the metric/prediction range.",
        required_steps=(
            "clip or transform predictions only when mathematically appropriate",
            f"respect metric range {metric_range}",
            "rerun validation and update prediction_summary",
        ),
        success_evidence=("all target values are inside the allowed range",),
    )


def _sample_copy_action() -> RepairAction:
    return RepairAction(
        id="repair_sample_copy",
        priority=30,
        problem="The final targets copied sample_submission values.",
        required_steps=(
            "train or compute a public-data candidate instead of copying sample targets",
            "keep any recovery output in fallback_submission.csv",
            "rewrite final_submission.csv only after selecting a modeled candidate",
        ),
        success_evidence=("final targets differ from sample targets", "manifest stage/evidence supports final selection"),
    )


def _probability_action() -> RepairAction:
    return RepairAction(
        id="repair_probability_scores",
        priority=40,
        problem="Probability/ranking metric output looks like hard labels.",
        required_steps=(
            "train a classifier that exposes predict_proba or calibrated decision scores",
            "write continuous probabilities such as predict_proba[:, 1] for binary targets",
            "compare validation against majority/constant fallback",
        ),
        success_evidence=("target probabilities are not only 0/1", "manifest reports validation_metric_value > baseline_metric_value"),
    )


def _multi_output_action() -> RepairAction:
    return RepairAction(
        id="repair_multi_output_diversity",
        priority=45,
        problem="One or more multi-output target columns are constant.",
        required_steps=(
            "train per-target models or a multi-output estimator",
            "avoid copying one constant vector across target columns",
            "update manifest prediction_summary per target",
        ),
        success_evidence=("each target column has non-zero diversity where defensible", "manifest describes multi-output strategy"),
    )


def _multiclass_action() -> RepairAction:
    return RepairAction(
        id="repair_multiclass_coverage",
        priority=45,
        problem="Multiclass output covers too few classes.",
        required_steps=(
            "train a multiclass model or class-probability/ranking model",
            "map encoded predictions back to original labels",
            "compare against majority-class fallback",
        ),
        success_evidence=("prediction diversity includes more than one class when defensible", "manifest reports class coverage"),
    )


def _text_action() -> RepairAction:
    return RepairAction(
        id="repair_text_feature_modeling",
        priority=50,
        problem="Text route output is constant or ignored text evidence.",
        required_steps=(
            "use text columns via vectorizer, hashing, length/count features, or task-specific parsing",
            "combine text features with tabular features if available",
            "record text feature usage in the manifest",
        ),
        success_evidence=("predictions vary when text evidence varies", "manifest lists text features or vectorizer family"),
    )


def _time_group_action() -> RepairAction:
    return RepairAction(
        id="repair_time_group_validation",
        priority=50,
        problem="Time/group route lacks validation or feature evidence.",
        required_steps=(
            "identify date/group columns and avoid leakage-prone random validation",
            "use ordered/group split or explain the public-data proxy",
            "add validation_strategy, features_used, or risk_notes to manifest",
        ),
        success_evidence=("manifest includes validation_strategy/features_used/risk_notes", "final is not a bare global mean marked improved"),
    )


def _public_data_action() -> RepairAction:
    return RepairAction(
        id="repair_public_data_availability",
        priority=5,
        problem="The public train/test/sample files are missing or Git LFS pointer stubs.",
        required_steps=(
            "stop modeling on the pointer/missing public files",
            "restore real public CSV data before attempting an improved final",
            "record the blocker rather than claiming a modeled submission",
        ),
        success_evidence=("data_availability.json reports usable_for_modeling=true",),
    )


def _protocol_manifest_action(protocol_plan: Any | None) -> RepairAction:
    required = tuple((getattr(protocol_plan, "required_manifest_evidence", []) or [])[:8])
    return RepairAction(
        id="repair_protocol_manifest_evidence",
        priority=55,
        problem="The improved manifest does not include evidence required by the selected modeling protocol.",
        required_steps=(
            "re-read the modeling_protocol_plan block",
            "add concrete split_policy, features_used, leakage_controls, and protocol-specific evidence",
            "do not fabricate validation evidence; rerun the public-data validation/proxy if needed",
        ),
        success_evidence=required or ("protocol required_manifest_evidence is present",),
    )


def _protocol_probability_action() -> RepairAction:
    return RepairAction(
        id="repair_protocol_probability_evidence",
        priority=56,
        problem="Probability protocol evidence is missing from the manifest.",
        required_steps=(
            "record the probability range actually written to final_submission.csv",
            "record calibration_or_ranking_check from validation/proxy comparison",
            "ensure predictions remain continuous probabilities rather than hard labels",
        ),
        success_evidence=("manifest includes probability_range", "manifest includes calibration_or_ranking_check"),
    )


def _protocol_time_action() -> RepairAction:
    return RepairAction(
        id="repair_protocol_time_evidence",
        priority=56,
        problem="Time-series/group protocol evidence is missing from the manifest.",
        required_steps=(
            "identify time_or_group_columns",
            "record ordered/group validation_strategy",
            "record future_leakage_checks explaining how future rows were excluded from training features",
        ),
        success_evidence=("manifest includes validation_strategy", "manifest includes future_leakage_checks"),
    )


def _protocol_text_action() -> RepairAction:
    return RepairAction(
        id="repair_protocol_text_evidence",
        priority=57,
        problem="Text modeling protocol evidence is missing from the manifest.",
        required_steps=(
            "list text_columns_used",
            "record text_preprocessing and vectorizer/embedding strategy",
            "confirm text signal influenced the selected candidate",
        ),
        success_evidence=("manifest includes text_columns_used", "manifest includes text_preprocessing"),
    )


def _protocol_multi_output_action() -> RepairAction:
    return RepairAction(
        id="repair_protocol_multi_output_evidence",
        priority=57,
        problem="Multi-output protocol evidence is missing from the manifest.",
        required_steps=(
            "write per_target_prediction_summary for every target column",
            "record whether targets used per-target models or a multi-output estimator",
            "report any defensible constant target separately",
        ),
        success_evidence=("manifest includes per_target_prediction_summary",),
    )


def _stage_agent_outputs_action() -> RepairAction:
    return RepairAction(
        id="repair_stage_agent_outputs",
        priority=58,
        problem="The manifest does not summarize required stage sub-agent outputs.",
        required_steps=(
            "add stage_agent_outputs as an object in submission_manifest.json",
            "include an entry for each stage agent listed in stage_agent_plan.json",
            "summarize concrete outputs such as split_policy, features_used, validation result, and finalization checks",
        ),
        success_evidence=("manifest includes stage_agent_outputs for every required stage agent",),
    )


def _verified_evidence_conflict_action() -> RepairAction:
    return RepairAction(
        id="repair_verified_evidence_conflict",
        priority=59,
        problem="The manifest metric claims conflict with harness-generated public-data baseline/CV evidence.",
        required_steps=(
            "open verified_modeling_evidence.json or read the quality_gate modeling_evidence payload",
            "align validation_metric_value and baseline_metric_value with the harness-verified proxy when using that proxy",
            "if using a different public-data validation, set validation_source='agent_additional_validation' and include concrete split/metric details",
        ),
        success_evidence=(
            "manifest metrics match verified_modeling_evidence.json or validation_source='agent_additional_validation'",
            "quality gate no longer reports verified_evidence_metric_conflict",
        ),
    )


def _manifest_action() -> RepairAction:
    return RepairAction(
        id="repair_manifest_evidence",
        priority=60,
        problem="The manifest is missing, incomplete, or unsupported by validation evidence.",
        required_steps=(
            "write ./input/submission_manifest.json next to final_submission.csv",
            "use stage='improved' only when validation_metric_value beats baseline_metric_value",
            "include metric_direction, trained_rows, selected_reason, and prediction_summary",
        ),
        success_evidence=("manifest parses as JSON", "manifest_validation_beats_baseline check passes"),
    )


def _nonconstant_action(route: TaskRoute | None) -> RepairAction:
    family = route.family if route is not None else "unknown"
    return RepairAction(
        id="repair_nonconstant_modeling",
        priority=70,
        problem=f"The {family} final submission is low-information or constant.",
        required_steps=(
            "add public-data features/preprocessing and rerun a bounded model",
            "validate against a constant fallback",
            "only keep stage='improved' if validation beats fallback",
        ),
        success_evidence=("prediction diversity is non-constant", "manifest comparison supports improved stage"),
    )


def _fallback_action() -> RepairAction:
    return RepairAction(
        id="repair_fallback_misuse",
        priority=75,
        problem="Fallback output is being used as a quality final submission.",
        required_steps=(
            "keep fallback_submission.csv as recovery only",
            "rerun modeling for final_submission.csv or set fallback_recovery honestly",
            "do not claim improved stage for fallback-identical output",
        ),
        success_evidence=("final differs from fallback or manifest stage is fallback_recovery",),
    )


def _route_default_action(route: TaskRoute | None) -> RepairAction:
    family = route.family if route is not None else "unknown"
    return RepairAction(
        id="repair_route_default",
        priority=100,
        problem=f"The current attempt failed for route {family}; apply the selected route skills.",
        required_steps=(
            "re-read contract, route, and selected skill cards",
            "fix the reported validator/quality issues only",
            "write final_submission.csv and submission_manifest.json",
        ),
        success_evidence=("validator passes", "quality gate passes"),
    )
