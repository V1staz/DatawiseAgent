"""Harness-generated, label-blind modeling evidence for DataModeling tasks.

This module intentionally produces a small public-data proxy signal rather than
an official score.  It reuses the deterministic recipe/CV engine in an isolated
artifact directory so the LLM-facing harness can distinguish code-observed
baseline/candidate evidence from agent-authored manifest claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import traceback

from .data_availability import DataAvailabilityReport
from .profiler import DataProfile
from .protocol import ModelingProtocolPlan
from .recipes import RecipeConfig, RecipeEngine
from .task_router import TaskRoute


@dataclass(slots=True)
class VerifiedModelingEvidence:
    status: str
    metric_name: str
    metric_direction: str
    baseline_metric_value: float | None = None
    best_candidate_metric_value: float | None = None
    best_candidate_name: str | None = None
    split_policy: str | None = None
    features_used: list[str] = field(default_factory=list)
    candidate_count: int = 0
    validation_metric_trust: str = "unavailable"
    warnings: list[str] = field(default_factory=list)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_summary(self) -> str:
        payload = {
            "status": self.status,
            "metric_name": self.metric_name,
            "metric_direction": self.metric_direction,
            "baseline_metric_value": self.baseline_metric_value,
            "best_candidate_metric_value": self.best_candidate_metric_value,
            "best_candidate_name": self.best_candidate_name,
            "split_policy": self.split_policy,
            "features_used": self.features_used[:20],
            "candidate_count": self.candidate_count,
            "validation_metric_trust": self.validation_metric_trust,
            "preprocessing_scope": self.summary.get("preprocessing_scope"),
            "group_column": self.summary.get("group_column"),
            "group_count": self.summary.get("group_count"),
            "score_gap_vs_dummy": self.summary.get("score_gap_vs_dummy"),
            "performance_scaffold_artifacts": {
                key: value
                for key, value in self.artifact_paths.items()
                if key in {"candidate_leaderboard", "fold_metrics", "oof_predictions", "feature_pipeline_summary", "model_recommendations"}
            },
            "recommendations": self.summary.get("recommendations", [])[:5],
            "warnings": self.warnings[:8],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def build_verified_modeling_evidence(
    contract: Any,
    profile: DataProfile,
    route: TaskRoute,
    protocol_plan: ModelingProtocolPlan,
    availability: DataAvailabilityReport,
    artifact_dir: str | Path,
    *,
    budget_seconds: int = 90,
    max_candidates: int = 3,
    n_splits: int = 3,
) -> VerifiedModelingEvidence:
    """Run a small public-data model/CV proxy and write evidence artifacts."""

    target = Path(artifact_dir)
    target.mkdir(parents=True, exist_ok=True)
    evidence_path = target / "verified_modeling_evidence.json"

    if not availability.usable_for_modeling:
        evidence = VerifiedModelingEvidence(
            status="skipped_unusable_data",
            metric_name=contract.metric.name,
            metric_direction=contract.metric.direction,
            validation_metric_trust="unavailable",
            warnings=["public train/test/sample data is unavailable or Git LFS pointer-backed"],
            artifact_paths={"verified_modeling_evidence": str(evidence_path)},
            summary={"route_family": route.family, "protocol_target_type": protocol_plan.target_type},
        )
        _write_evidence(evidence, evidence_path)
        return evidence

    if protocol_plan.target_type in {"data_unavailable_or_pointer", "special_simulation"}:
        evidence = VerifiedModelingEvidence(
            status="skipped_unsupported",
            metric_name=contract.metric.name,
            metric_direction=contract.metric.direction,
            validation_metric_trust="unavailable",
            warnings=[f"verified evidence skipped for target_type={protocol_plan.target_type}"],
            artifact_paths={"verified_modeling_evidence": str(evidence_path)},
            summary={"route_family": route.family, "protocol_target_type": protocol_plan.target_type},
        )
        _write_evidence(evidence, evidence_path)
        return evidence

    recipe_dir = target / "recipe_proxy"
    try:
        result = RecipeEngine(
            RecipeConfig(
                budget_seconds=budget_seconds,
                max_candidates=max_candidates,
                n_splits=n_splits,
            )
        ).run(contract, recipe_dir)
        evidence = _evidence_from_recipe_result(
            contract,
            profile,
            route,
            protocol_plan,
            result.to_dict(),
            evidence_path=evidence_path,
            recipe_dir=recipe_dir,
        )
    except Exception as exc:
        (target / "verified_modeling_evidence_exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        evidence = VerifiedModelingEvidence(
            status="failed",
            metric_name=contract.metric.name,
            metric_direction=contract.metric.direction,
            validation_metric_trust="unavailable",
            warnings=[f"verified_evidence_failed:{type(exc).__name__}:{str(exc)[:200]}"],
            artifact_paths={
                "verified_modeling_evidence": str(evidence_path),
                "exception": str(target / "verified_modeling_evidence_exception.txt"),
            },
            summary={"route_family": route.family, "protocol_target_type": protocol_plan.target_type},
        )
    _write_evidence(evidence, evidence_path)
    return evidence


def _evidence_from_recipe_result(
    contract: Any,
    profile: DataProfile,
    route: TaskRoute,
    protocol_plan: ModelingProtocolPlan,
    result: dict[str, Any],
    *,
    evidence_path: Path,
    recipe_dir: Path,
) -> VerifiedModelingEvidence:
    leaderboard = list(result.get("leaderboard") or [])
    best_name = str(result.get("selected_candidate") or "") or None
    baseline = _find_candidate_metric(leaderboard, "dummy")
    best = _find_candidate_metric(leaderboard, best_name) if best_name else None
    if best is None:
        best = _best_candidate_metric(leaderboard)
    feature_meta_path = (result.get("artifacts") or {}).get("feature_meta")
    feature_meta = _read_json(feature_meta_path) if feature_meta_path else {}
    recommendations_path = (result.get("artifacts") or {}).get("model_recommendations")
    recommendations = _read_json(recommendations_path) if recommendations_path else {}
    features_used = _features_used(feature_meta)
    warnings = [str(item) for item in result.get("warnings") or []]
    for item in leaderboard:
        warnings.extend(str(w) for w in item.get("warnings") or [])
    status = "ok" if best is not None else "failed"
    return VerifiedModelingEvidence(
        status=status,
        metric_name=contract.metric.name,
        metric_direction=contract.metric.direction,
        baseline_metric_value=baseline,
        best_candidate_metric_value=best,
        best_candidate_name=best_name,
        split_policy=_split_policy_from_protocol(protocol_plan),
        features_used=features_used,
        candidate_count=len([item for item in leaderboard if item.get("cv_result")]),
        validation_metric_trust="harness_verified" if status == "ok" else "unavailable",
        warnings=warnings[:25],
        artifact_paths={
            "verified_modeling_evidence": str(evidence_path),
            "recipe_proxy_dir": str(recipe_dir),
            **{str(k): str(v) for k, v in (result.get("artifacts") or {}).items()},
        },
        summary={
            "route_family": route.family,
            "route_facets": route.facets,
            "protocol_target_type": protocol_plan.target_type,
            "profile_rows_sampled": profile.train_rows_sampled,
            "selected_candidate": best_name,
            "preprocessing_scope": feature_meta.get("preprocessing_scope"),
            "fold_local_transformers": bool(feature_meta.get("fold_local_transformers")),
            "group_column": feature_meta.get("group_column"),
            "group_count": feature_meta.get("group_count"),
            "recommendations": recommendations.get("recommendations", []),
            "score_gap_vs_dummy": recommendations.get("score_gap_vs_dummy"),
            "top_candidate": recommendations.get("top_candidate"),
        },
    )


def _find_candidate_metric(leaderboard: list[dict[str, Any]], candidate_name: str | None) -> float | None:
    if not candidate_name:
        return None
    for item in leaderboard:
        name = str(item.get("name") or "")
        if name == candidate_name or candidate_name.startswith(name):
            cv = item.get("cv_result") or {}
            return _maybe_float(cv.get("mean_raw_score"))
    return None


def _best_candidate_metric(leaderboard: list[dict[str, Any]]) -> float | None:
    best_item: dict[str, Any] | None = None
    best_selection: float | None = None
    for item in leaderboard:
        cv = item.get("cv_result") or {}
        selection = _maybe_float(cv.get("mean_selection_score"))
        raw = _maybe_float(cv.get("mean_raw_score"))
        if selection is None or raw is None:
            continue
        if best_selection is None or selection > best_selection:
            best_selection = selection
            best_item = item
    if best_item is None:
        return None
    return _maybe_float((best_item.get("cv_result") or {}).get("mean_raw_score"))


def _features_used(feature_meta: dict[str, Any]) -> list[str]:
    features: list[str] = []
    for key in ("numeric_columns", "categorical_columns", "text_columns"):
        value = feature_meta.get(key)
        if isinstance(value, list):
            features.extend(str(item) for item in value)
    if not features and feature_meta.get("original_feature_count"):
        features.append(f"{feature_meta.get('original_feature_count')}_contract_features")
    return features


def _split_policy_from_protocol(protocol_plan: ModelingProtocolPlan) -> str:
    split = protocol_plan.split_constraint
    if split == "stratified":
        return "stratified_cv_when_class_counts_allow_else_kfold"
    if split == "time_ordered_or_group_aware":
        return "time_series_split_when_task_kind_allows_else_group_or_kfold_proxy"
    return split


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_evidence(evidence: VerifiedModelingEvidence, path: Path) -> None:
    path.write_text(json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
