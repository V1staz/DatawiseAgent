"""Lightweight schema/data profiling for DataModeling harness tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import DataModelingContract


@dataclass(slots=True)
class ColumnProfile:
    name: str
    dtype: str
    missing_fraction: float
    unique_count: int
    unique_fraction: float
    role: str
    sample_values: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DataProfile:
    task_name: str
    train_rows_sampled: int
    test_rows_sampled: int
    train_columns: int
    test_columns: int
    columns: list[ColumnProfile]
    target_summary: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_profile(contract: DataModelingContract, sample_rows: int = 2000) -> DataProfile:
    train = pd.read_csv(contract.train_path, nrows=sample_rows) if Path(contract.train_path).exists() else pd.DataFrame()
    test = pd.read_csv(contract.test_path, nrows=sample_rows) if Path(contract.test_path).exists() else pd.DataFrame()
    columns: list[ColumnProfile] = []
    warnings: list[str] = []
    roles = _roles(contract)
    for col in train.columns:
        series = train[col]
        role = roles.get(col, "feature")
        non_null = series.dropna()
        unique = int(non_null.nunique(dropna=True)) if len(non_null) else 0
        columns.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                missing_fraction=round(float(series.isna().mean()), 6),
                unique_count=unique,
                unique_fraction=round(float(unique / max(1, len(series))), 6),
                role=role,
                sample_values=[str(v)[:80] for v in non_null.head(5).tolist()],
            )
        )
        if role == "feature" and unique <= 1:
            warnings.append(f"constant_or_empty_feature:{col}")
        if role == "feature" and col in contract.id_columns:
            warnings.append(f"id_column_in_features:{col}")

    target_summary: dict[str, Any] = {}
    for col in contract.target_columns:
        if col in train:
            s = train[col]
            target_summary[col] = {
                "dtype": str(s.dtype),
                "missing_fraction": round(float(s.isna().mean()), 6),
                "unique_count": int(s.nunique(dropna=True)),
                "min": _maybe_number(s.min()),
                "max": _maybe_number(s.max()),
                "mean": _maybe_number(s.mean()) if pd.api.types.is_numeric_dtype(s) else None,
            }
    if not target_summary:
        warnings.append("target_columns_not_found_in_train_sample")

    return DataProfile(
        task_name=contract.name,
        train_rows_sampled=len(train),
        test_rows_sampled=len(test),
        train_columns=len(train.columns),
        test_columns=len(test.columns),
        columns=columns,
        target_summary=target_summary,
        warnings=warnings,
    )


def write_profile(profile: DataProfile, path: str | Path) -> None:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _roles(contract: DataModelingContract) -> dict[str, str]:
    result: dict[str, str] = {}
    for col in contract.id_columns:
        result[col] = "id"
    for col in contract.target_columns:
        result[col] = "target"
    for col in contract.text_columns:
        result.setdefault(col, "text_feature")
    for col in contract.datetime_columns:
        result.setdefault(col, "datetime_feature")
    return result


def _maybe_number(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            return value.item()
        return value
    except Exception:
        return None
