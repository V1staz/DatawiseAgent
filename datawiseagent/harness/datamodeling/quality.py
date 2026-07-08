"""Label-blind quality gate for DataModeling final submissions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

import pandas as pd

from .contracts import DataModelingContract
from .manifest import fallback_path_for_submission, metric_improved, read_submission_manifest
from .submission import SubmissionValidationResult
from .task_router import TaskRoute


@dataclass(slots=True)
class PredictionColumnQuality:
    column: str
    nunique: int
    missing: int
    constant: bool
    numeric: bool
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QualityReport:
    passed: bool
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    route_family: str = "unknown"
    manifest: dict[str, Any] = field(default_factory=dict)
    prediction_columns: list[dict[str, Any]] = field(default_factory=list)
    verified_evidence: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QualityReport":
        return cls(
            passed=bool(payload.get("passed")),
            checks=list(payload.get("checks") or []),
            warnings=list(payload.get("warnings") or []),
            errors=list(payload.get("errors") or []),
            route_family=str(payload.get("route_family") or "unknown"),
            manifest=dict(payload.get("manifest") or {}),
            prediction_columns=list(payload.get("prediction_columns") or []),
            verified_evidence=dict(payload.get("verified_evidence") or {}),
            summary=dict(payload.get("summary") or {}),
        )


def evaluate_submission_quality(
    submission_path: str | Path,
    contract: DataModelingContract,
    route: TaskRoute,
    validation: SubmissionValidationResult,
    *,
    write_path: str | Path | None = None,
    require_manifest: bool = True,
    protocol_plan: Any | None = None,
    stage_agent_plan: Any | None = None,
    verified_evidence: Any | None = None,
) -> QualityReport:
    report = QualityReport(passed=False, route_family=route.family)
    if not validation.passed:
        report.errors.append("structural_validation_failed")
        return _write_report(report, write_path)

    submission = Path(submission_path)
    manifest = read_submission_manifest(submission)
    report.manifest = manifest.to_dict()
    if require_manifest and manifest.errors:
        report.errors.extend(manifest.errors)
    report.warnings.extend(manifest.warnings)

    try:
        data = pd.read_csv(submission)
    except Exception as exc:
        report.errors.append(f"submission_read_error:{type(exc).__name__}")
        return _write_report(report, write_path)

    fallback_path = fallback_path_for_submission(submission)
    if fallback_path.exists():
        try:
            fallback = pd.read_csv(fallback_path)
            if list(fallback.columns) == list(data.columns) and fallback.equals(data):
                stage = manifest.stage
                if stage != "fallback_recovery":
                    report.errors.append("final_identical_to_fallback_without_recovery_stage")
                else:
                    report.warnings.append("final_identical_to_fallback_recovery")
        except Exception as exc:
            report.warnings.append(f"fallback_compare_failed:{type(exc).__name__}")

    target_cols = contract.submission.target_columns
    column_reports: list[PredictionColumnQuality] = []
    for col in target_cols:
        if col not in data.columns:
            report.errors.append(f"quality_target_missing:{col}")
            continue
        series = data[col]
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric = bool(numeric_series.notna().all())
        nunique = int(series.nunique(dropna=True))
        col_report = PredictionColumnQuality(
            column=col,
            nunique=nunique,
            missing=int(series.isna().sum()),
            constant=nunique <= 1,
            numeric=numeric,
            min=_maybe_float(numeric_series.min()) if numeric else None,
            max=_maybe_float(numeric_series.max()) if numeric else None,
            mean=_maybe_float(numeric_series.mean()) if numeric else None,
            std=_maybe_float(numeric_series.std()) if numeric else None,
        )
        column_reports.append(col_report)
        if col_report.missing:
            report.errors.append(f"quality_missing_predictions:{col}")

    report.prediction_columns = [col.to_dict() for col in column_reports]
    all_constant = bool(column_reports) and all(col.constant for col in column_reports)
    any_constant = any(col.constant for col in column_reports)
    constant_columns = [col.column for col in column_reports if col.constant]
    manifest_improved = metric_improved(manifest.payload, contract.metric.direction) if manifest.payload else None
    stage = manifest.stage
    validation_metric_trust = "agent_attested" if manifest_improved is not None else "unavailable"

    if stage == "failed":
        report.errors.append("manifest_stage_failed")
    if stage == "fallback_recovery":
        report.errors.append("fallback_recovery_is_not_quality_final")
    if require_manifest and stage != "improved":
        report.errors.append(f"manifest_stage_not_improved:{stage or 'missing'}")

    if all_constant and route.family not in {"simulation_or_special"}:
        if manifest_improved is True:
            report.warnings.append("all_target_columns_constant_but_manifest_claims_validation_improvement")
        else:
            report.errors.append("all_target_columns_constant_without_validation_improvement")
    elif any_constant and route.family == "multi_output":
        report.warnings.append("some_target_columns_constant")

    if route.family == "multi_output" and constant_columns:
        report.errors.append(f"multi_output_target_columns_constant:{','.join(constant_columns)}")

    if route.family == "tabular_multiclass_classification" and constant_columns:
        report.errors.append(f"multiclass_predictions_low_class_coverage:{','.join(constant_columns)}")

    if route.family == "text_modeling" and constant_columns:
        report.errors.append(f"text_modeling_constant_output:{','.join(constant_columns)}")

    if route.family == "time_series_or_grouped" and stage == "improved":
        has_time_group_evidence = any(
            manifest.payload.get(key)
            for key in ("validation_strategy", "features_used", "risk_notes")
        )
        if not has_time_group_evidence:
            report.errors.append("time_group_manifest_missing_validation_strategy")

    if protocol_plan is not None:
        _apply_protocol_quality_checks(report, manifest.payload, protocol_plan, require_manifest=require_manifest)
        _apply_observable_manifest_checks(report, manifest.payload, column_reports, protocol_plan)
    if stage_agent_plan is not None:
        _apply_stage_agent_quality_checks(report, manifest.payload, stage_agent_plan, require_manifest=require_manifest)

    validation_metric_trust = _apply_verified_modeling_evidence(
        report,
        manifest.payload,
        verified_evidence,
        default_trust=validation_metric_trust,
    )

    if contract.metric.name in {"pearson", "r2", "roc_auc", "normalized_gini", "average_precision"} and all_constant:
        report.errors.append(f"constant_predictions_make_metric_uninformative:{contract.metric.name}")

    if contract.metric.probability_output:
        numeric_targets = [col for col in column_reports if col.numeric]
        hard_label_like = [
            col.column
            for col in numeric_targets
            if col.min in (0.0, 1.0) and col.max in (0.0, 1.0) and col.nunique <= 2
        ]
        if numeric_targets and len(hard_label_like) == len(numeric_targets):
            report.errors.append("probability_metric_predictions_look_like_hard_labels")

    if manifest_improved is True:
        report.checks.append("manifest_validation_beats_baseline")
        if validation_metric_trust == "harness_verified":
            report.checks.append("manifest_validation_metric_harness_verified")
        else:
            report.warnings.append("manifest_validation_metric_agent_attested")
    elif manifest_improved is False:
        report.errors.append("manifest_validation_not_better_than_baseline")
    else:
        if require_manifest and stage == "improved":
            report.errors.append("manifest_validation_comparison_unavailable")
        else:
            report.warnings.append("manifest_validation_comparison_unavailable")

    report.checks.extend([
        "structural_validation_passed",
        "prediction_columns_profiled",
    ])
    if manifest.exists and manifest.valid_json and not manifest.errors:
        report.checks.append("manifest_schema_present")
    if not report.errors:
        report.checks.append("quality_gate_passed")

    report.summary = {
        "route_family": route.family,
        "protocol_target_type": getattr(protocol_plan, "target_type", None),
        "protocol_split_constraint": getattr(protocol_plan, "split_constraint", None),
        "stage_agent_ids": getattr(stage_agent_plan, "required_agent_ids", None),
        "metric": contract.metric.name,
        "stage": stage,
        "all_target_columns_constant": all_constant,
        "any_target_column_constant": any_constant,
        "constant_target_columns": constant_columns,
        "manifest_improved": manifest_improved,
        "validation_metric_trust": validation_metric_trust,
    }
    report.passed = not report.errors
    return _write_report(report, write_path)


def quality_failure_category(report: QualityReport) -> str:
    if report.passed:
        return "none"
    if not report.errors:
        return "quality_failed"
    first = report.errors[0]
    if first.startswith("manifest_missing") or first.startswith("manifest_"):
        return "quality_manifest_error"
    if first.startswith("protocol_manifest_"):
        return "quality_protocol_evidence_missing"
    if first.startswith("protocol_public_data_"):
        return "quality_protocol_data_unavailable"
    if first.startswith("stage_agent_"):
        return "quality_stage_agent_evidence_missing"
    if first.startswith("verified_evidence_metric_conflict"):
        return "quality_verified_evidence_conflict"
    if "constant" in first or "fallback" in first or "hard_labels" in first or "low_class_coverage" in first:
        return "quality_low_information_submission"
    if first.startswith("time_group_"):
        return "quality_route_evidence_missing"
    if "structural" in first:
        return "quality_structural_dependency_failed"
    return f"quality_{first.split(':', 1)[0]}"


def _apply_verified_modeling_evidence(
    report: QualityReport,
    payload: dict[str, Any],
    verified_evidence: Any | None,
    *,
    default_trust: str,
) -> str:
    """Cross-check manifest metric claims against harness-generated evidence.

    The deterministic recipe proxy is not an official score, but it is a
    harness-observed public-data signal.  When it is available, the agent must
    either align its manifest metric fields to this signal or explicitly declare
    a separate public-data validation source.
    """

    if verified_evidence is None:
        return default_trust

    evidence = _verified_evidence_to_dict(verified_evidence)
    if not evidence:
        report.warnings.append("verified_modeling_evidence_unreadable")
        return default_trust

    report.verified_evidence["modeling_evidence"] = evidence
    status = str(evidence.get("status") or "")
    trust = str(evidence.get("validation_metric_trust") or "")
    if status != "ok" or trust != "harness_verified":
        report.warnings.append(f"verified_modeling_evidence_not_available:{status or 'unknown'}")
        return default_trust

    report.checks.append("verified_modeling_evidence_available")
    validation_source = str(payload.get("validation_source") or "")
    if validation_source == "agent_additional_validation":
        report.warnings.append("verified_modeling_evidence_bypassed_by_agent_additional_validation")
        return default_trust

    conflicts: list[str] = []
    manifest_value_present = "validation_metric_value" in payload
    manifest_value = _maybe_float(payload.get("validation_metric_value"))
    evidence_value = _maybe_float(evidence.get("best_candidate_metric_value"))
    if manifest_value is not None and evidence_value is not None and not _floats_close(manifest_value, evidence_value):
        conflicts.append("validation_metric_value")
        report.errors.append("verified_evidence_metric_conflict:validation_metric_value")

    manifest_baseline_present = "baseline_metric_value" in payload
    manifest_baseline = _maybe_float(payload.get("baseline_metric_value"))
    evidence_baseline = _maybe_float(evidence.get("baseline_metric_value"))
    if manifest_baseline is not None and evidence_baseline is not None and not _floats_close(manifest_baseline, evidence_baseline):
        conflicts.append("baseline_metric_value")
        report.errors.append("verified_evidence_metric_conflict:baseline_metric_value")

    manifest_metric = str(payload.get("validation_metric_name") or "")
    evidence_metric = str(evidence.get("metric_name") or "")
    if manifest_metric and evidence_metric and evidence_metric not in manifest_metric and manifest_metric not in evidence_metric:
        report.warnings.append(f"verified_evidence_metric_name_differs:{manifest_metric}!={evidence_metric}")

    if conflicts:
        report.verified_evidence["modeling_evidence_conflicts"] = conflicts
        return default_trust

    value_aligned = manifest_value_present and manifest_value is not None and evidence_value is not None
    baseline_aligned = manifest_baseline_present and manifest_baseline is not None and evidence_baseline is not None
    if value_aligned and baseline_aligned:
        report.checks.append("manifest_metrics_match_verified_evidence")
        return "harness_verified"
    report.warnings.append("verified_modeling_evidence_missing_manifest_metric_alignment")
    return default_trust


def _verified_evidence_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
            return dict(payload) if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _apply_protocol_quality_checks(report: QualityReport, payload: dict[str, Any], protocol_plan: Any, *, require_manifest: bool) -> None:
    target_type = str(getattr(protocol_plan, "target_type", "") or "")
    if target_type == "data_unavailable_or_pointer":
        report.errors.append("protocol_public_data_unavailable_for_modeling")
        return
    if not require_manifest or str(payload.get("stage") or "") != "improved":
        return

    required = list(getattr(protocol_plan, "required_manifest_evidence", []) or [])
    missing = [field for field in required if not _manifest_has_evidence(payload, field)]
    if missing:
        report.errors.append(f"protocol_manifest_missing_required_evidence:{','.join(missing)}")

    if target_type in {"binary_probability", "multilabel_probability"}:
        probability_fields = ("probability_range", "calibration_or_ranking_check")
        missing_probability = [field for field in probability_fields if not _manifest_has_evidence(payload, field)]
        if missing_probability:
            report.errors.append(f"protocol_probability_manifest_missing:{','.join(missing_probability)}")

    if target_type.startswith("time_series"):
        time_fields = ("validation_strategy", "time_or_group_columns", "future_leakage_checks")
        missing_time = [field for field in time_fields if not _manifest_has_evidence(payload, field)]
        if missing_time:
            report.errors.append(f"protocol_time_manifest_missing:{','.join(missing_time)}")

    if target_type == "text_modeling":
        text_fields = ("text_columns_used", "text_preprocessing")
        missing_text = [field for field in text_fields if not _manifest_has_evidence(payload, field)]
        if missing_text:
            report.errors.append(f"protocol_text_manifest_missing:{','.join(missing_text)}")

    if "multi_output" in target_type or "per_target_prediction_summary" in required:
        if not _manifest_has_evidence(payload, "per_target_prediction_summary"):
            report.errors.append("protocol_multi_output_manifest_missing:per_target_prediction_summary")


def _manifest_has_evidence(payload: dict[str, Any], field: str) -> bool:
    value = payload.get(field)
    if value in (None, "", [], {}):
        return False
    return True


def _apply_stage_agent_quality_checks(report: QualityReport, payload: dict[str, Any], stage_agent_plan: Any, *, require_manifest: bool) -> None:
    if not require_manifest or str(payload.get("stage") or "") != "improved":
        return
    manifest_key = str(getattr(stage_agent_plan, "required_manifest_key", "stage_agent_outputs") or "stage_agent_outputs")
    outputs = payload.get(manifest_key)
    if not outputs:
        report.errors.append(f"stage_agent_manifest_missing:{manifest_key}")
        return
    if not isinstance(outputs, dict):
        report.errors.append(f"stage_agent_manifest_not_object:{manifest_key}")
        return
    required_agent_ids = list(getattr(stage_agent_plan, "required_agent_ids", []) or [])
    missing_agents = [agent_id for agent_id in required_agent_ids if not outputs.get(agent_id)]
    if missing_agents:
        report.errors.append(f"stage_agent_outputs_missing:{','.join(missing_agents)}")


def _apply_observable_manifest_checks(
    report: QualityReport,
    payload: dict[str, Any],
    column_reports: list[PredictionColumnQuality],
    protocol_plan: Any,
) -> None:
    """Verify manifest claims that are directly observable from final CSV.

    This intentionally does not pretend to recompute arbitrary public-data CV
    metrics.  Instead it separates independently checkable submission evidence
    from agent-attested validation metrics.
    """

    if str(payload.get("stage") or "") != "improved":
        return
    target_type = str(getattr(protocol_plan, "target_type", "") or "")
    by_col = {col.column: col for col in column_reports}
    verified: dict[str, Any] = {
        "prediction_summary_matches_submission": None,
        "probability_range_matches_submission": None,
        "per_target_prediction_summary_matches_submission": None,
    }

    prediction_summary = payload.get("prediction_summary")
    if isinstance(prediction_summary, dict):
        mismatches = _prediction_summary_mismatches(prediction_summary, by_col)
        verified["prediction_summary_matches_submission"] = not mismatches
        if mismatches:
            report.errors.append(f"manifest_prediction_summary_mismatch:{','.join(mismatches)}")

    if target_type in {"binary_probability", "multilabel_probability"} and payload.get("probability_range") is not None:
        probability_match = _probability_range_matches(payload.get("probability_range"), column_reports)
        verified["probability_range_matches_submission"] = probability_match
        if not probability_match:
            report.errors.append("manifest_probability_range_mismatch")

    if "multi_output" in target_type and payload.get("per_target_prediction_summary") is not None:
        per_target_summary = payload.get("per_target_prediction_summary")
        if not isinstance(per_target_summary, dict):
            verified["per_target_prediction_summary_matches_submission"] = False
            report.errors.append("manifest_per_target_prediction_summary_not_object")
        else:
            missing = [col.column for col in column_reports if col.column not in per_target_summary]
            verified["per_target_prediction_summary_matches_submission"] = not missing
            if missing:
                report.errors.append(f"manifest_per_target_prediction_summary_missing:{','.join(missing)}")

    report.verified_evidence.update(verified)


def _prediction_summary_mismatches(summary: dict[str, Any], by_col: dict[str, PredictionColumnQuality]) -> list[str]:
    mismatches: list[str] = []
    for col, claims in summary.items():
        if col not in by_col or not isinstance(claims, dict):
            continue
        actual = by_col[col]
        if "nunique" in claims and _maybe_int(claims.get("nunique")) != actual.nunique:
            mismatches.append(f"{col}.nunique")
        for field in ("min", "max"):
            if field in claims:
                claimed = _maybe_float(claims.get(field))
                actual_value = getattr(actual, field)
                if claimed is not None and actual_value is not None and abs(claimed - actual_value) > 1e-9:
                    mismatches.append(f"{col}.{field}")
    return mismatches


def _probability_range_matches(value: Any, column_reports: list[PredictionColumnQuality]) -> bool:
    numeric = [col for col in column_reports if col.numeric]
    if not numeric:
        return False
    try:
        low, high = value
    except Exception:
        return False
    claimed_low = _maybe_float(low)
    claimed_high = _maybe_float(high)
    lows = [col.min for col in numeric if col.min is not None]
    highs = [col.max for col in numeric if col.max is not None]
    if not lows or not highs:
        return False
    actual_low = min(lows)
    actual_high = max(highs)
    if claimed_low is None or claimed_high is None:
        return False
    return abs(claimed_low - actual_low) <= 1e-9 and abs(claimed_high - actual_high) <= 1e-9


def _maybe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _floats_close(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= scale * 1e-9


def _write_report(report: QualityReport, write_path: str | Path | None) -> QualityReport:
    if write_path is not None:
        Path(write_path).write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def _maybe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None
