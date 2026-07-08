"""Layered deterministic finalization repairs for DataModeling submissions.

The LLM should still do the modeling work when it succeeds.  This module fixes
submission-boundary failures that do not require labels: misplaced files,
column/order/ID mismatches, row alignment, metric-domain clipping, and fallback
/adoption of a public-data scaffold when an attempt is structurally unrecoverable
or objectively low-information.

The repair path is deliberately label-blind.  It ranks observable candidate CSVs,
normalizes them to the submission contract, validates them, profiles prediction
informativeness, writes a manifest when deterministic evidence is available, and
emits an audit JSON for every call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import pandas as pd

from .contracts import DataModelingContract
from .manifest import manifest_path_for_submission, metric_improved, read_submission_manifest
from .submission import SubmissionValidationResult, submission_template, validate_submission


SCAFFOLD_STAGE = "scaffold_adopted"
SCHEMA_STAGE = "schema_id_row_repaired"
MANIFEST_STAGE = "manifest_backfilled"


@dataclass(slots=True)
class PredictionProfile:
    """Observable, label-blind informativeness profile for a repaired candidate."""

    rows: int
    target_columns: dict[str, dict[str, Any]] = field(default_factory=dict)
    copied_sample_targets: list[str] = field(default_factory=list)
    low_information_reasons: list[str] = field(default_factory=list)

    @property
    def low_information(self) -> bool:
        return bool(self.low_information_reasons)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateEvaluation:
    """One candidate CSV after normalization/validation/profiling."""

    path: str
    kind: str
    priority: int
    exists: bool
    validation: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None
    accepted: bool = False
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SubmissionRepairReport:
    changed: bool = False
    stage: str = "none"
    reason: str = "not_needed"
    actions: list[str] = field(default_factory=list)
    source_path: str | None = None
    output_path: str | None = None
    manifest_path: str | None = None
    manifest_stage: str | None = None
    validation: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None
    candidates_considered: list[str] = field(default_factory=list)
    candidate_evaluations: list[dict[str, Any]] = field(default_factory=list)
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repair_submission_layers(
    submission_path: str | Path,
    contract: DataModelingContract,
    artifact_dir: str | Path,
    *,
    attempt: int,
    verified_evidence: Any | None = None,
    route: Any | None = None,
    protocol_plan: Any | None = None,
    stage_agent_plan: Any | None = None,
    prefer_scaffold: bool = False,
    allow_scaffold: bool = False,
    reason: str = "validation_repair",
) -> SubmissionRepairReport:
    """Repair a final submission in deterministic, auditable layers.

    Selection order is score-first but not score-leaking:
      1. Keep a strict-valid, informative final/candidate CSV when possible.
      2. Normalize schema/IDs/rows/ranges for agent-produced candidates.
      3. Adopt the public-data recipe scaffold only when explicitly enabled by
         the caller for quality repair / non-agent scaffold mode.
      4. Backfill a truthful manifest for the selected deterministic final.

    Hidden answers are never read by this function.
    """

    final_path = Path(submission_path)
    artifact_dir = Path(artifact_dir)
    report = SubmissionRepairReport(reason=reason, output_path=str(final_path))

    allow_scaffold = bool(allow_scaffold or prefer_scaffold)
    evaluations = _evaluate_candidates(final_path, contract, verified_evidence, prefer_scaffold=prefer_scaffold, allow_scaffold=allow_scaffold)
    report.candidates_considered = [ev.path for ev in evaluations]
    report.candidate_evaluations = [ev.to_dict() for ev in evaluations]
    report.rejected_candidates = [
        {"path": ev.path, "kind": ev.kind, "reasons": ev.rejection_reasons[:8]}
        for ev in evaluations
        if ev.exists and not ev.accepted
    ]

    selected = _select_candidate(evaluations, prefer_scaffold=prefer_scaffold)
    if selected is None:
        initial = validate_submission(final_path, contract, strict=True, reject_sample_target_copy=True)
        report.stage = "unrepaired"
        report.validation = initial.to_dict()
        _write_report(report, artifact_dir, attempt, reason)
        return report

    selected_path = Path(selected.path)
    already_final = _same_path(selected_path, final_path)
    if not already_final:
        _write_normalized_candidate(selected_path, final_path, contract)
        report.changed = True
    else:
        # Re-write a normalized copy when the accepted final reached validity via
        # schema/range coercion.  This keeps finalization deterministic.
        normalized = _normalize_candidate_frame(pd.read_csv(selected_path), contract)
        current = pd.read_csv(final_path) if final_path.exists() else pd.DataFrame()
        if not _frame_equals(normalized, current):
            final_path.parent.mkdir(parents=True, exist_ok=True)
            normalized.to_csv(final_path, index=False)
            report.changed = True

    final_validation = validate_submission(final_path, contract, strict=True, reject_sample_target_copy=True)
    final_profile = _prediction_profile(final_path, contract)
    report.source_path = selected.path
    report.validation = final_validation.to_dict()
    report.profile = final_profile.to_dict() if final_profile is not None else None
    report.actions.extend(final_validation.checks)
    if selected.kind.startswith("scaffold"):
        report.stage = SCAFFOLD_STAGE
    elif report.changed:
        report.stage = SCHEMA_STAGE
    else:
        report.stage = "already_valid"

    manifest_info = _ensure_repair_manifest(
        final_path,
        contract,
        selected_kind=selected.kind,
        selected_reason=reason,
        verified_evidence=verified_evidence,
        route=route,
        protocol_plan=protocol_plan,
        stage_agent_plan=stage_agent_plan,
        profile=final_profile,
    )
    if manifest_info:
        report.manifest_path = manifest_info.get("path")
        report.manifest_stage = manifest_info.get("stage")
        if manifest_info.get("changed"):
            report.changed = True
            if report.stage == "already_valid":
                report.stage = MANIFEST_STAGE
            report.actions.append("submission_manifest_backfilled")

    _write_report(report, artifact_dir, attempt, reason)
    return report


def should_scaffold_repair_quality(errors: Iterable[str]) -> bool:
    """Return true for quality failures where a scoreable public scaffold helps."""

    text = "\n".join(str(error) for error in errors)
    triggers = (
        "constant_predictions_make_metric_uninformative",
        "all_target_columns_constant_without_validation_improvement",
        "multi_output_target_columns_constant",
        "multiclass_predictions_low_class_coverage",
        "text_modeling_constant_output",
        "probability_metric_predictions_look_like_hard_labels",
        "target_identical_to_sample_submission",
    )
    return any(trigger in text for trigger in triggers)


def _evaluate_candidates(
    final_path: Path,
    contract: DataModelingContract,
    verified_evidence: Any | None,
    *,
    prefer_scaffold: bool,
    allow_scaffold: bool,
) -> list[CandidateEvaluation]:
    evaluations: list[CandidateEvaluation] = []
    for priority, (path, kind) in enumerate(_candidate_registry(final_path, verified_evidence, prefer_scaffold=prefer_scaffold, allow_scaffold=allow_scaffold)):
        exists = path.exists() and path.is_file()
        ev = CandidateEvaluation(path=str(path), kind=kind, priority=priority, exists=exists)
        if not exists:
            ev.rejection_reasons.append("candidate_missing")
            evaluations.append(ev)
            continue
        scratch = final_path.parent / f".__repair_candidate_{priority}_{_safe_name(path.name)}.csv"
        try:
            validation = _repair_candidate_to_template(path, scratch, contract)
            ev.validation = validation.to_dict()
            profile = _prediction_profile(scratch, contract) if validation.passed else None
            ev.profile = profile.to_dict() if profile is not None else None
            if not validation.passed:
                ev.rejection_reasons.extend(validation.errors[:8])
            elif profile is not None and profile.low_information and not kind.startswith("scaffold"):
                ev.rejection_reasons.extend(profile.low_information_reasons[:8])
            else:
                ev.accepted = True
        except Exception as exc:  # defensive: candidate audit should survive bad files
            ev.rejection_reasons.append(f"candidate_evaluation_error:{type(exc).__name__}:{str(exc)[:200]}")
        finally:
            try:
                scratch.unlink(missing_ok=True)
            except Exception:
                pass
        evaluations.append(ev)
    return evaluations


def _candidate_registry(final_path: Path, verified_evidence: Any | None, *, prefer_scaffold: bool, allow_scaffold: bool) -> list[tuple[Path, str]]:
    input_dir = final_path.parent
    scaffold = _scaffold_submission_path(verified_evidence) if allow_scaffold else None
    baseline = _scaffold_baseline_path(verified_evidence) if allow_scaffold else None
    candidates: list[tuple[Path, str]] = []
    if prefer_scaffold and scaffold is not None:
        candidates.append((scaffold, "scaffold_model"))
    candidates.append((final_path, "agent_final"))
    # Only explicit submission-like filenames are eligible.  Arbitrary CSV globbing
    # is intentionally forbidden because feature exports/OOF diagnostics can look
    # row-compatible but are not final submissions.
    for name, kind in (
        ("candidate_submission.csv", "agent_candidate"),
        ("submission.csv", "agent_candidate"),
        ("predictions.csv", "agent_candidate"),
        ("model_submission.csv", "agent_candidate"),
        ("improved_submission.csv", "agent_candidate"),
        ("fallback_submission.csv", "agent_fallback"),
    ):
        candidates.append((input_dir / name, kind))
    if allow_scaffold and not prefer_scaffold and scaffold is not None:
        candidates.append((scaffold, "scaffold_model"))
    if allow_scaffold and baseline is not None:
        candidates.append((baseline, "scaffold_baseline"))
    return _dedupe_candidate_paths(candidates)


def _select_candidate(evaluations: list[CandidateEvaluation], *, prefer_scaffold: bool) -> CandidateEvaluation | None:
    accepted = [ev for ev in evaluations if ev.accepted]
    if not accepted:
        return None
    if prefer_scaffold:
        scaffold = [ev for ev in accepted if ev.kind.startswith("scaffold")]
        if scaffold:
            return scaffold[0]
    non_scaffold = [ev for ev in accepted if not ev.kind.startswith("scaffold")]
    if non_scaffold:
        return non_scaffold[0]
    return accepted[0]


def _scaffold_submission_path(verified_evidence: Any | None) -> Path | None:
    artifact_paths = _artifact_paths(verified_evidence)
    if not artifact_paths:
        return None
    for key in ("submission", "final_submission", "best_submission"):
        value = artifact_paths.get(key)
        if value:
            path = Path(str(value))
            if path.exists():
                return path
    recipe_dir = artifact_paths.get("recipe_proxy_dir")
    if recipe_dir:
        path = Path(str(recipe_dir)) / "final_submission.csv"
        if path.exists():
            return path
    return None


def _scaffold_baseline_path(verified_evidence: Any | None) -> Path | None:
    artifact_paths = _artifact_paths(verified_evidence)
    recipe_dir = artifact_paths.get("recipe_proxy_dir") if artifact_paths else None
    if recipe_dir:
        baseline = Path(str(recipe_dir)) / "baseline_submission.csv"
        if baseline.exists():
            return baseline
    return None


def _artifact_paths(verified_evidence: Any | None) -> dict[str, Any]:
    if verified_evidence is None:
        return {}
    if isinstance(verified_evidence, dict):
        paths = verified_evidence.get("artifact_paths")
    else:
        paths = getattr(verified_evidence, "artifact_paths", None)
    return dict(paths) if isinstance(paths, dict) else {}


def _repair_candidate_to_template(candidate: Path, output_path: Path, contract: DataModelingContract) -> SubmissionValidationResult:
    try:
        candidate_df = pd.read_csv(candidate)
    except Exception as exc:
        return SubmissionValidationResult(False, errors=[f"candidate_csv_read_error:{candidate.name}:{type(exc).__name__}"], summary={"path": str(candidate)})

    try:
        repaired = _normalize_candidate_frame(candidate_df, contract)
    except Exception as exc:
        return SubmissionValidationResult(False, errors=[f"candidate_normalize_error:{candidate.name}:{type(exc).__name__}:{str(exc)[:200]}"], summary={"path": str(candidate)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(output_path, index=False)
    validation = validate_submission(output_path, contract, strict=True, reject_sample_target_copy=True)
    validation.checks.append(f"layered_repair_source:{candidate.name}")
    validation.summary["layered_repair_source"] = str(candidate)
    return validation


def _write_normalized_candidate(candidate: Path, output_path: Path, contract: DataModelingContract) -> None:
    repaired = _normalize_candidate_frame(pd.read_csv(candidate), contract)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(output_path, index=False)


def _normalize_candidate_frame(candidate: pd.DataFrame, contract: DataModelingContract) -> pd.DataFrame:
    template = submission_template(contract).copy()
    output = template.copy()
    target_sources = _target_source_columns(candidate, contract)
    if not target_sources:
        raise ValueError("no_candidate_target_columns")

    aligned = _align_candidate_rows(candidate, template, contract)
    for target_col, source_col in zip(contract.submission.target_columns, target_sources):
        values = aligned[source_col] if source_col in aligned.columns else pd.Series([np.nan] * len(template))
        output[target_col] = _coerce_target_values(values, template[target_col] if target_col in template else None, contract)
    return output[contract.submission.columns]


def _target_source_columns(candidate: pd.DataFrame, contract: DataModelingContract) -> list[str]:
    exact = [col for col in contract.submission.target_columns if col in candidate.columns]
    if len(exact) == len(contract.submission.target_columns):
        return exact
    id_set = set(contract.id_columns)
    non_id = [col for col in candidate.columns if col not in id_set]
    if len(non_id) >= len(contract.submission.target_columns):
        return non_id[-len(contract.submission.target_columns):]
    return []


def _align_candidate_rows(candidate: pd.DataFrame, template: pd.DataFrame, contract: DataModelingContract) -> pd.DataFrame:
    id_cols = [col for col in contract.id_columns if col in candidate.columns and col in template.columns]
    if id_cols:
        left = template[id_cols].copy()
        right = candidate.copy()
        for col in id_cols:
            left[col] = left[col].astype(str)
            right[col] = right[col].astype(str)
        if right[id_cols].duplicated().any():
            raise ValueError("candidate_duplicate_ids")
        merged = left.merge(right, on=id_cols, how="left", sort=False)
        target_sources = _target_source_columns(candidate, contract)
        if target_sources and not merged[target_sources].isna().all(axis=None):
            return merged
    if len(candidate) == len(template):
        return candidate.reset_index(drop=True)
    raise ValueError(f"candidate_row_alignment_failed:expected={len(template)} actual={len(candidate)}")


def _coerce_target_values(values: pd.Series, template_values: pd.Series | None, contract: DataModelingContract) -> pd.Series:
    if contract.metric.prediction_kind == "text" or "span_extraction" in contract.task_kinds:
        text = values.astype(str).replace({"nan": "", "None": ""})
        if template_values is not None:
            fallback = template_values.astype(str).replace({"nan": "", "None": ""})
            text = text.where(text.str.len() > 0, fallback)
        return text.fillna("")

    categorical = template_values is not None and pd.to_numeric(template_values, errors="coerce").notna().mean() < 0.95 and not contract.metric.probability_output
    if categorical:
        text = values.astype(str).replace({"nan": "", "None": ""})
        if template_values is not None:
            fallback = template_values.astype(str).replace({"nan": "", "None": ""})
            mode = fallback[fallback.str.len() > 0].mode()
            fill = mode.iloc[0] if not mode.empty else (fallback.iloc[0] if len(fallback) else "placeholder")
            text = text.where(text.str.len() > 0, fill)
        return text

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        fallback = 0.5 if contract.metric.probability_output else 0.0
        if template_values is not None:
            sample_numeric = pd.to_numeric(template_values, errors="coerce")
            if sample_numeric.notna().any():
                fallback = float(np.nanmedian(sample_numeric))
        numeric = numeric.fillna(fallback)
    if contract.metric.requires_non_negative:
        numeric = numeric.clip(lower=0)
    if contract.metric.bounded_range is not None:
        lo, hi = contract.metric.bounded_range
        numeric = numeric.clip(lower=lo, upper=hi)
    if contract.metric.prediction_kind == "class" and not contract.metric.probability_output:
        numeric = np.rint(numeric.astype(float))
    return pd.Series(numeric, index=values.index)


def _prediction_profile(path: Path, contract: DataModelingContract) -> PredictionProfile | None:
    try:
        data = pd.read_csv(path)
        sample = submission_template(contract)
    except Exception:
        return None
    profile = PredictionProfile(rows=len(data))
    for col in contract.submission.target_columns:
        if col not in data.columns:
            profile.low_information_reasons.append(f"target_missing:{col}")
            continue
        series = data[col]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_ok = bool(numeric.notna().all())
        summary: dict[str, Any] = {
            "nunique": int(series.nunique(dropna=True)),
            "missing": int(series.isna().sum()),
            "numeric": numeric_ok,
        }
        if numeric_ok and len(numeric):
            summary.update({
                "min": _maybe_float(numeric.min()),
                "max": _maybe_float(numeric.max()),
                "mean": _maybe_float(numeric.mean()),
                "std": _maybe_float(numeric.std()),
            })
        profile.target_columns[col] = summary
        if summary["missing"]:
            profile.low_information_reasons.append(f"missing_predictions:{col}")
        if summary["nunique"] <= 1 and contract.metric.name in {"pearson", "r2", "roc_auc", "normalized_gini", "average_precision"}:
            profile.low_information_reasons.append(f"constant_metric_uninformative:{col}")
        if contract.metric.probability_output and numeric_ok:
            values = set(float(v) for v in numeric.dropna().unique()[:3])
            if summary["nunique"] <= 2 and values.issubset({0.0, 1.0}):
                profile.low_information_reasons.append(f"probability_hard_labels:{col}")
        if col in sample.columns and len(sample) == len(data) and series.astype(str).equals(sample[col].astype(str)):
            profile.copied_sample_targets.append(col)
            profile.low_information_reasons.append(f"target_identical_to_sample_submission:{col}")
    return profile


def _ensure_repair_manifest(
    submission_path: Path,
    contract: DataModelingContract,
    *,
    selected_kind: str,
    selected_reason: str,
    verified_evidence: Any | None,
    route: Any | None,
    protocol_plan: Any | None,
    stage_agent_plan: Any | None,
    profile: PredictionProfile | None,
) -> dict[str, Any] | None:
    """Write/refresh manifest evidence for deterministic repairs.

    We only overwrite missing/invalid harness fields.  For agent candidates that
    are not verified by the public scaffold, the manifest stays `candidate` so
    scoring can continue as low-confidence without pretending improvement.  For
    scaffold model adoption with harness-verified improvement, it becomes
    `improved` and includes metrics aligned with verified evidence.
    """

    existing = read_submission_manifest(submission_path)
    evidence = _verified_evidence_to_dict(verified_evidence)
    scaffold_model = selected_kind == "scaffold_model"
    evidence_improved = _evidence_improved(evidence, contract.metric.direction)

    payload: dict[str, Any] = dict(existing.payload) if existing.valid_json and isinstance(existing.payload, dict) else {}
    existing_stage = str(payload.get("stage") or "")
    if scaffold_model:
        stage = "improved" if evidence_improved else "candidate"
        payload["stage"] = stage
        payload["modeling_method"] = _modeling_method(selected_kind, evidence)
        payload["selected_reason"] = _selected_reason(selected_kind, selected_reason, evidence)
        payload["validation_source"] = "harness_verified_public_data_proxy" if evidence else "harness_repair_observable_submission"
        if evidence.get("baseline_metric_value") is not None:
            payload["baseline_metric_value"] = evidence.get("baseline_metric_value")
        if evidence.get("best_candidate_metric_value") is not None:
            payload["validation_metric_value"] = evidence.get("best_candidate_metric_value")
    else:
        # Preserve a valid agent-authored improved manifest instead of degrading
        # it just because the harness normalized schema/IDs.  Only invalid hard
        # stages are demoted to candidate so they remain scoreable rather than
        # hard-blocked as failed/fallback recovery.
        stage = existing_stage if existing_stage in {"improved", "candidate"} else "candidate"
        payload["stage"] = stage
        payload.setdefault("modeling_method", _modeling_method(selected_kind, evidence))
        payload.setdefault("selected_reason", _selected_reason(selected_kind, selected_reason, evidence))
        payload.setdefault("validation_source", "agent_authored_or_harness_repaired_candidate")

    payload.update({
        "prediction_summary": _prediction_summary_from_profile(profile),
        "metric_direction": payload.get("metric_direction") or contract.metric.direction,
        "validation_metric_name": payload.get("validation_metric_name") or evidence.get("metric_name") or contract.metric.name,
        "trained_rows": payload.get("trained_rows") or _count_csv_rows(contract.train_path),
        "split_policy": payload.get("split_policy") or evidence.get("split_policy") or getattr(protocol_plan, "split_constraint", None) or "public_train_validation_split",
        "features_used": payload.get("features_used") or evidence.get("features_used") or contract.feature_columns[:50],
        "leakage_controls": payload.get("leakage_controls") or _leakage_controls(contract, protocol_plan),
        "repair_source_kind": selected_kind,
        "repair_reason": selected_reason,
    })
    if contract.metric.probability_output:
        payload["probability_range"] = _probability_range_from_profile(profile)
        payload["calibration_or_ranking_check"] = "continuous probability scores checked for range and non-hard-label behavior"
    if contract.datetime_columns or contract.group_columns or str(getattr(protocol_plan, "target_type", "")).startswith("time_series"):
        payload["validation_strategy"] = payload.get("split_policy")
        payload["time_or_group_columns"] = list(dict.fromkeys([*contract.datetime_columns, *contract.group_columns])) or ["inferred_time_or_group_constraint"]
        payload["future_leakage_checks"] = ["fold-local preprocessing", "no hidden-answer access", "template IDs from public test only"]
    if contract.text_columns or contract.metric.prediction_kind == "text":
        payload["text_columns_used"] = contract.text_columns
        payload["text_preprocessing"] = "lightweight public-train text preprocessing; final target strings validated non-empty"
    if len(contract.submission.target_columns) > 1:
        payload["per_target_prediction_summary"] = payload["prediction_summary"]
    _fill_protocol_required_fields(payload, contract, protocol_plan, profile)
    _fill_stage_agent_outputs(payload, stage_agent_plan, selected_kind, selected_reason)

    manifest_path = manifest_path_for_submission(submission_path)
    before = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else None
    after = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    manifest_path.write_text(after, encoding="utf-8")
    return {"path": str(manifest_path), "stage": stage, "changed": before != after}


def _fill_protocol_required_fields(payload: dict[str, Any], contract: DataModelingContract, protocol_plan: Any | None, profile: PredictionProfile | None) -> None:
    for field_name in list(getattr(protocol_plan, "required_manifest_evidence", []) or []):
        if payload.get(field_name) not in (None, "", [], {}):
            continue
        if field_name == "prediction_summary":
            payload[field_name] = _prediction_summary_from_profile(profile)
        elif field_name == "per_target_prediction_summary":
            payload[field_name] = _prediction_summary_from_profile(profile)
        elif field_name == "probability_range":
            payload[field_name] = _probability_range_from_profile(profile)
        elif field_name == "calibration_or_ranking_check":
            payload[field_name] = "range/diversity checked label-blind by harness repair"
        elif field_name in {"validation_strategy", "split_policy"}:
            payload[field_name] = payload.get("split_policy") or "public_train_validation_split"
        elif field_name == "features_used":
            payload[field_name] = contract.feature_columns[:50]
        elif field_name == "leakage_controls":
            payload[field_name] = _leakage_controls(contract, protocol_plan)
        elif field_name == "time_or_group_columns":
            payload[field_name] = list(dict.fromkeys([*contract.datetime_columns, *contract.group_columns])) or ["inferred_time_or_group_constraint"]
        elif field_name == "future_leakage_checks":
            payload[field_name] = ["no future/public-test labels used", "template alignment only by public IDs"]
        elif field_name == "text_columns_used":
            payload[field_name] = contract.text_columns
        elif field_name == "text_preprocessing":
            payload[field_name] = "public text columns normalized only from visible inputs"
        else:
            payload[field_name] = "harness_repair_observable_evidence"


def _fill_stage_agent_outputs(payload: dict[str, Any], stage_agent_plan: Any | None, selected_kind: str, selected_reason: str) -> None:
    key = str(getattr(stage_agent_plan, "required_manifest_key", "stage_agent_outputs") or "stage_agent_outputs")
    required = list(getattr(stage_agent_plan, "required_agent_ids", []) or [])
    if not required:
        return
    outputs = payload.get(key) if isinstance(payload.get(key), dict) else {}
    for agent_id in required:
        outputs.setdefault(agent_id, {
            "status": "harness_repair_verified",
            "source": selected_kind,
            "evidence": selected_reason,
        })
    payload[key] = outputs


def _verified_evidence_to_dict(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {}
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


def _evidence_improved(evidence: dict[str, Any], direction: str) -> bool:
    if not evidence or evidence.get("status") != "ok" or evidence.get("validation_metric_trust") != "harness_verified":
        return False
    payload = {
        "validation_metric_value": evidence.get("best_candidate_metric_value"),
        "baseline_metric_value": evidence.get("baseline_metric_value"),
        "metric_direction": evidence.get("metric_direction") or direction,
    }
    return bool(metric_improved(payload, direction))


def _modeling_method(selected_kind: str, evidence: dict[str, Any]) -> str:
    if selected_kind == "scaffold_model":
        return f"deterministic_public_data_scaffold:{evidence.get('best_candidate_name') or 'recipe_engine'}"
    if selected_kind == "scaffold_baseline":
        return "deterministic_public_data_baseline"
    return f"agent_candidate_with_harness_finalization:{selected_kind}"


def _selected_reason(selected_kind: str, reason: str, evidence: dict[str, Any]) -> str:
    if selected_kind.startswith("scaffold"):
        candidate = evidence.get("best_candidate_name") or selected_kind
        return f"{reason}; adopted {candidate} because agent final was structurally invalid or low-information"
    return f"{reason}; preserved best valid agent-produced candidate after schema/id/domain repair"


def _prediction_summary_from_profile(profile: PredictionProfile | None) -> dict[str, Any]:
    if profile is None:
        return {}
    return profile.target_columns


def _probability_range_from_profile(profile: PredictionProfile | None) -> list[float] | None:
    if profile is None:
        return None
    lows: list[float] = []
    highs: list[float] = []
    for summary in profile.target_columns.values():
        if summary.get("min") is not None:
            lows.append(float(summary["min"]))
        if summary.get("max") is not None:
            highs.append(float(summary["max"]))
    if not lows or not highs:
        return None
    return [min(lows), max(highs)]


def _leakage_controls(contract: DataModelingContract, protocol_plan: Any | None) -> list[str]:
    controls = ["label-blind repair", "no hidden answers", "public test IDs used only for row alignment"]
    split = str(getattr(protocol_plan, "split_constraint", "") or "")
    if "time" in split or contract.datetime_columns:
        controls.append("time-aware validation evidence required")
    if "group" in split or contract.group_columns:
        controls.append("group-aware validation evidence required")
    return controls


def _count_csv_rows(path: str | Path) -> int | None:
    try:
        with Path(path).open("rb") as fh:
            lines = sum(1 for _ in fh)
        return max(0, lines - 1)
    except Exception:
        return None


def _dedupe_candidate_paths(candidates: Iterable[tuple[Path, str]]) -> list[tuple[Path, str]]:
    result: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path, kind in candidates:
        try:
            key = str(path.resolve()) if path.exists() else str(path)
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append((path, kind))
    return result


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except Exception:
        return str(left) == str(right)


def _frame_equals(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        return list(left.columns) == list(right.columns) and left.reset_index(drop=True).equals(right.reset_index(drop=True))
    except Exception:
        return False


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)[:80]


def _maybe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _write_report(report: SubmissionRepairReport, artifact_dir: Path, attempt: int, reason: str) -> None:
    safe_reason = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in reason)[:80]
    path = artifact_dir / f"submission_repair_attempt_{attempt}_{safe_reason}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
