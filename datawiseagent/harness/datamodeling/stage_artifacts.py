"""Artifact-backed stage evidence for DataModeling harness runs.

These artifacts convert the internal stage-role plan from prompt-only manifest
requirements into concrete, harness-readable evidence.  They are intentionally
label-blind and derived from public contract/profile/scaffold observations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .data_availability import DataAvailabilityReport
from .profiler import DataProfile
from .protocol import ModelingProtocolPlan
from .quality import QualityReport
from .submission import SubmissionValidationResult
from .task_router import TaskRoute
from .verified_evidence import VerifiedModelingEvidence


def write_pre_execution_stage_artifacts(
    artifact_dir: str | Path,
    *,
    contract: Any,
    profile: DataProfile,
    availability: DataAvailabilityReport,
    route: TaskRoute,
    protocol_plan: ModelingProtocolPlan,
    verified_evidence: VerifiedModelingEvidence | None,
) -> dict[str, str]:
    target = Path(artifact_dir) / "stage_artifacts"
    target.mkdir(parents=True, exist_ok=True)
    evidence = verified_evidence.to_dict() if verified_evidence is not None else {}
    paths: dict[str, str] = {}
    paths["data_audit"] = _write_json(
        target / "data_audit.json",
        {
            "task": contract.name,
            "route_family": route.family,
            "data_availability": availability.to_dict(),
            "train_rows_sampled": profile.train_rows_sampled,
            "test_rows_sampled": profile.test_rows_sampled,
            "warnings": profile.warnings,
            "schema_columns_sample": profile.to_dict().get("columns", [])[:80],
        },
    )
    paths["split_plan"] = _write_json(
        target / "split_plan.json",
        {
            "protocol_split_constraint": protocol_plan.split_constraint,
            "protocol_target_type": protocol_plan.target_type,
            "contract_group_columns": getattr(contract, "group_columns", []),
            "scaffold_split_policy": evidence.get("split_policy"),
            "scaffold_group_column": (evidence.get("summary") or {}).get("group_column"),
            "scaffold_group_count": (evidence.get("summary") or {}).get("group_count"),
            "scaffold_trust": evidence.get("validation_metric_trust"),
            "leakage_warnings": evidence.get("warnings", []),
        },
    )
    paths["feature_plan"] = _write_json(
        target / "feature_plan.json",
        {
            "data_structure": protocol_plan.data_structure,
            "features_used_by_scaffold": evidence.get("features_used", []),
            "route_facets": route.facets,
            "feature_recommendations": (evidence.get("summary") or {}).get("recommendations", []),
        },
    )
    leaderboard_path = (evidence.get("artifact_paths") or {}).get("candidate_leaderboard")
    paths["modeling_leaderboard"] = _write_json(
        target / "modeling_leaderboard.json",
        {
            "source_candidate_leaderboard": leaderboard_path,
            "best_candidate_name": evidence.get("best_candidate_name"),
            "best_candidate_metric_value": evidence.get("best_candidate_metric_value"),
            "baseline_metric_value": evidence.get("baseline_metric_value"),
            "score_gap_vs_dummy": (evidence.get("summary") or {}).get("score_gap_vs_dummy"),
        },
    )
    return paths


def write_finalization_stage_artifact(
    artifact_dir: str | Path,
    *,
    result_path: str,
    validation: SubmissionValidationResult,
    quality: QualityReport,
    quality_confidence: str,
) -> str:
    target = Path(artifact_dir) / "stage_artifacts"
    target.mkdir(parents=True, exist_ok=True)
    return _write_json(
        target / "finalization_check.json",
        {
            "result_path": result_path,
            "validation_passed": validation.passed,
            "quality_passed": quality.passed,
            "quality_confidence": quality_confidence,
            "validation_errors": validation.errors,
            "quality_errors": quality.errors,
            "quality_checks": quality.checks,
        },
    )


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(path)
