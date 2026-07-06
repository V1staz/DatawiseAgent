"""Submission manifest helpers for DataModeling quality gating."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


MANIFEST_FILENAME = "submission_manifest.json"
FALLBACK_FILENAME = "fallback_submission.csv"
FINAL_FILENAME = "final_submission.csv"


@dataclass(slots=True)
class ManifestReadResult:
    path: str
    exists: bool
    valid_json: bool
    payload: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def stage(self) -> str:
        return str(self.payload.get("stage") or "") if self.payload else ""


def manifest_path_for_submission(submission_path: str | Path) -> Path:
    return Path(submission_path).parent / MANIFEST_FILENAME


def fallback_path_for_submission(submission_path: str | Path) -> Path:
    return Path(submission_path).parent / FALLBACK_FILENAME


def read_submission_manifest(submission_path: str | Path) -> ManifestReadResult:
    path = manifest_path_for_submission(submission_path)
    if not path.exists():
        return ManifestReadResult(path=str(path), exists=False, valid_json=False, errors=["manifest_missing"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return ManifestReadResult(path=str(path), exists=True, valid_json=False, errors=[f"manifest_invalid_json:{exc.msg}"])
    if not isinstance(payload, dict):
        return ManifestReadResult(path=str(path), exists=True, valid_json=False, errors=["manifest_not_object"])
    result = ManifestReadResult(path=str(path), exists=True, valid_json=True, payload=payload)
    _validate_manifest_fields(result)
    return result


def metric_improved(payload: dict[str, Any], default_direction: str = "maximize") -> bool | None:
    """Return True/False when manifest has comparable metric values, else None."""

    value = _as_float(payload.get("validation_metric_value"))
    baseline = _as_float(payload.get("baseline_metric_value"))
    if value is None or baseline is None:
        return None
    direction = str(payload.get("metric_direction") or default_direction or "maximize").lower()
    tolerance = abs(baseline) * 1e-9 + 1e-12
    if direction == "minimize":
        return value < baseline - tolerance
    return value > baseline + tolerance


def write_manifest_snapshot(submission_path: str | Path, target_path: str | Path) -> ManifestReadResult:
    result = read_submission_manifest(submission_path)
    Path(target_path).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _validate_manifest_fields(result: ManifestReadResult) -> None:
    payload = result.payload
    required = ("stage", "modeling_method", "selected_reason", "prediction_summary")
    for field in required:
        if field not in payload or payload.get(field) in (None, "", []):
            result.errors.append(f"manifest_missing_field:{field}")
    stage = str(payload.get("stage") or "")
    if stage and stage not in {"improved", "fallback_recovery", "failed", "candidate"}:
        result.warnings.append(f"manifest_unknown_stage:{stage}")
    if stage == "improved":
        for field in ("validation_metric_name", "metric_direction", "validation_metric_value", "baseline_metric_value", "trained_rows"):
            if field not in payload or payload.get(field) in (None, ""):
                result.warnings.append(f"manifest_missing_validation_field:{field}")
    trained_rows = _as_float(payload.get("trained_rows"))
    if trained_rows is not None and trained_rows <= 0 and stage == "improved":
        result.errors.append("manifest_nonpositive_trained_rows")


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
