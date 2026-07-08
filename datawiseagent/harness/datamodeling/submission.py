"""Submission validation and offline baseline helpers for DataModeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .contracts import DataModelingContract


@dataclass(slots=True)
class SubmissionValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_submission(
    submission_path: str | Path,
    contract: DataModelingContract,
    *,
    strict: bool = True,
    reject_sample_target_copy: bool = False,
    write_path: str | Path | None = None,
) -> SubmissionValidationResult:
    """Validate a DataModeling submission against public, label-blind constraints.

    Parameters
    ----------
    strict:
        Treat hard metric-domain violations, such as probabilities outside
        ``[0, 1]``, as errors.  This should remain enabled for official agent
        runs.
    reject_sample_target_copy:
        Reject submissions whose target columns are exactly copied from the
        sample submission.  This is intentionally opt-in because the offline
        deterministic baseline may produce constant sanity-check submissions;
        the LLM-in-the-loop harness enables it to avoid accepting "no modeling"
        outputs as successful model runs.
    """
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []
    sample_copied_targets: list[str] = []
    path = Path(submission_path)
    sample = submission_template(contract)
    if not path.exists():
        result = SubmissionValidationResult(False, errors=["submission_file_missing"], summary={"path": str(path)})
        _maybe_write(result, write_path)
        return result
    try:
        sub = pd.read_csv(path)
    except Exception as exc:
        result = SubmissionValidationResult(False, errors=[f"submission_csv_read_error:{type(exc).__name__}"], summary={"path": str(path)})
        _maybe_write(result, write_path)
        return result

    if list(sub.columns) != contract.submission.columns:
        errors.append("columns_mismatch:" + json.dumps({"expected": contract.submission.columns, "actual": list(sub.columns)}, ensure_ascii=False))
    else:
        checks.append("columns_exact")

    if len(sub) != len(sample):
        errors.append(f"row_count_mismatch:expected={len(sample)} actual={len(sub)}")
    else:
        checks.append("row_count_matches_sample")

    for col in contract.id_columns:
        if col not in sub.columns:
            errors.append(f"id_column_missing:{col}")
            continue
        if col in sample.columns and len(sub) == len(sample):
            if not sub[col].astype(str).equals(sample[col].astype(str)):
                errors.append(f"id_values_or_order_mismatch:{col}")
            else:
                checks.append(f"id_values_match:{col}")
        if sub[col].duplicated().any():
            errors.append(f"duplicate_id_values:{col}")

    text_required = contract.metric.prediction_kind == "text" or "span_extraction" in contract.task_kinds
    for col in contract.submission.target_columns:
        if col not in sub.columns:
            errors.append(f"target_column_missing:{col}")
            continue
        if sub[col].isna().any():
            errors.append(f"target_contains_missing:{col}")
        if text_required:
            numeric_ratio = pd.to_numeric(sub[col], errors="coerce").notna().mean() if len(sub) else 0.0
            if numeric_ratio > 0.95:
                errors.append(f"text_target_looks_numeric:{col}")
            else:
                checks.append(f"text_target_dtype_ok:{col}")
            if sub[col].astype(str).str.len().eq(0).any():
                errors.append(f"text_target_empty_string:{col}")
        elif _accepts_categorical_class_target(contract, sample, col):
            as_text = sub[col].astype(str)
            if as_text.str.len().eq(0).any():
                errors.append(f"class_target_empty_string:{col}")
            else:
                checks.append(f"class_target_dtype_ok:{col}")
        else:
            numeric = pd.to_numeric(sub[col], errors="coerce")
            if numeric.isna().any():
                errors.append(f"numeric_target_not_parseable:{col}")
            else:
                checks.append(f"numeric_target_parseable:{col}")
                if contract.metric.requires_non_negative and (numeric < 0).any():
                    errors.append(f"negative_prediction_for_nonnegative_metric:{col}")
                if contract.metric.bounded_range is not None:
                    lo, hi = contract.metric.bounded_range
                    if ((numeric < lo) | (numeric > hi)).any():
                        message = f"prediction_outside_metric_range:{col}:{lo}:{hi}"
                        if strict:
                            errors.append(message)
                        else:
                            warnings.append(message)
        if col in sample.columns and sub[col].astype(str).equals(sample[col].astype(str)):
            warnings.append(f"target_identical_to_sample_submission:{col}")
            sample_copied_targets.append(col)

    if reject_sample_target_copy and sample_copied_targets:
        errors.append("target_identical_to_sample_submission:" + ",".join(sample_copied_targets))

    result = SubmissionValidationResult(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        checks=checks,
        summary={"path": str(path), "rows": len(sub), "columns": list(sub.columns), "strict": strict},
    )
    _maybe_write(result, write_path)
    return result


def build_constant_baseline(contract: DataModelingContract, output_path: str | Path) -> str:
    result = submission_template(contract)
    text_required = contract.metric.prediction_kind == "text" or "span_extraction" in contract.task_kinds
    for col in contract.submission.target_columns:
        if col not in result:
            continue
        if text_required:
            result[col] = result[col].astype(str).where(result[col].astype(str).str.len() > 0, "placeholder")
        elif _accepts_categorical_class_target(contract, result, col):
            as_text = result[col].astype(str)
            mode = as_text[as_text.str.len() > 0].mode()
            fill = mode.iloc[0] if not mode.empty else "placeholder"
            result[col] = as_text.where(as_text.str.len() > 0, fill)
        else:
            numeric = pd.to_numeric(result[col], errors="coerce")
            fill = 0.5 if contract.metric.probability_output else 0.0
            if numeric.notna().any():
                fill = float(np.nanmedian(numeric))
            result[col] = numeric.fillna(fill)
            if contract.metric.requires_non_negative:
                result[col] = result[col].clip(lower=0)
            if contract.metric.bounded_range is not None:
                lo, hi = contract.metric.bounded_range
                result[col] = result[col].clip(lower=lo, upper=hi)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    return str(target)


def _maybe_write(result: SubmissionValidationResult, path: str | Path | None) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _accepts_categorical_class_target(contract: DataModelingContract, sample: pd.DataFrame, col: str) -> bool:
    if contract.metric.prediction_kind not in {"class", "top_k_class"} or contract.metric.probability_output:
        return False
    if col not in sample.columns:
        return False
    numeric = pd.to_numeric(sample[col], errors="coerce")
    return bool(numeric.notna().mean() < 0.95)


def submission_template(contract: DataModelingContract) -> pd.DataFrame:
    """Return the row template expected by the local resplit test set.

    Several DSBench-resplit tasks keep the original Kaggle sample submission
    file even when `test.csv` has been resplit to fewer rows.  The sample still
    defines column names and target dtypes, but current evaluation expects one
    prediction per local test row.  Prefer test IDs when row counts disagree.
    """

    sample = pd.read_csv(contract.sample_submission_path)
    try:
        test = pd.read_csv(contract.test_path, usecols=[c for c in contract.id_columns if c])
    except Exception:
        return sample
    if len(test) == len(sample):
        return sample
    result = pd.DataFrame()
    for col in contract.id_columns:
        if col in test.columns:
            result[col] = test[col]
        elif col in sample.columns:
            result[col] = sample[col].head(len(test)).reset_index(drop=True)
    for col in contract.submission.target_columns:
        if col in sample.columns and len(sample):
            result[col] = sample[col].iloc[0]
        else:
            result[col] = 0
    return result[contract.submission.columns]
