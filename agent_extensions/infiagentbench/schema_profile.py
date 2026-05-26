from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import write_json


_NORMALIZE_RE = re.compile(r"[^0-9a-zA-Z_]+")


def normalize_column_name(name: str) -> str:
    normalized = _NORMALIZE_RE.sub("_", str(name).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "column"


def _json_number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _examples(series: pd.Series, limit: int) -> list[Any]:
    values = []
    for value in series.dropna().drop_duplicates().head(limit).tolist():
        if hasattr(value, "item"):
            value = value.item()
        values.append(value)
    return values


def _datetime_success_rate(series: pd.Series, sample_size: int) -> float:
    sample = series.dropna().astype(str).head(sample_size)
    if sample.empty:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean())


def _is_probable_datetime_column(name: str, series: pd.Series, sample_size: int) -> tuple[bool, float]:
    lowered = name.lower()
    name_hint = any(token in lowered for token in ["date", "time", "year", "month", "day"])
    success_rate = _datetime_success_rate(series, sample_size)
    if pd.api.types.is_datetime64_any_dtype(series):
        return True, 1.0
    if name_hint and success_rate >= 0.6:
        return True, success_rate
    if not pd.api.types.is_numeric_dtype(series) and success_rate >= 0.9:
        return True, success_rate
    return False, success_rate


def _normalized_mapping(columns: list[str]) -> dict[str, str]:
    used: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for column in columns:
        base = normalize_column_name(column)
        count = used.get(base, 0) + 1
        used[base] = count
        mapping[column] = base if count == 1 else f"{base}_{count}"
    return mapping


def profile_csv(
    csv_path: str | Path,
    *,
    max_category_examples: int = 5,
    categorical_unique_threshold: int = 50,
    datetime_sample_size: int = 1000,
) -> dict[str, Any]:
    path = Path(csv_path)
    df = pd.read_csv(path, low_memory=False)
    row_count, column_count = df.shape
    columns = [str(column) for column in df.columns]

    missing_count = {column: int(df[column].isna().sum()) for column in df.columns}
    missing_ratio = {
        str(column): (round(count / row_count, 6) if row_count else 0.0)
        for column, count in missing_count.items()
    }

    numeric_summary: dict[str, dict[str, Any]] = {}
    for column in df.select_dtypes(include="number").columns:
        series = df[column]
        numeric_summary[str(column)] = {
            "min": _json_number(series.min()),
            "max": _json_number(series.max()),
            "mean": _json_number(series.mean()),
            "std": _json_number(series.std(ddof=1)),
        }

    unique_counts = {str(column): int(df[column].nunique(dropna=True)) for column in df.columns}
    categorical_summary: dict[str, dict[str, Any]] = {}
    possible_categorical_columns: list[str] = []
    for column in df.columns:
        unique_count = unique_counts[str(column)]
        unique_ratio = unique_count / row_count if row_count else 0.0
        is_categorical = (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_bool_dtype(df[column])
            or isinstance(df[column].dtype, pd.CategoricalDtype)
            or unique_count <= categorical_unique_threshold
            or unique_ratio <= 0.05
        )
        if is_categorical:
            column_name = str(column)
            possible_categorical_columns.append(column_name)
            categorical_summary[column_name] = {
                "unique_count": unique_count,
                "examples": _examples(df[column], max_category_examples),
            }

    datetime_candidates: list[dict[str, Any]] = []
    for column in df.columns:
        is_candidate, success_rate = _is_probable_datetime_column(
            str(column), df[column], datetime_sample_size
        )
        if is_candidate:
            datetime_candidates.append(
                {"column": str(column), "parse_success_rate": round(success_rate, 4)}
            )

    id_candidates: list[dict[str, Any]] = []
    for column in df.columns:
        column_name = str(column)
        unique_count = unique_counts[column_name]
        missing = missing_count[column]
        unique_ratio = unique_count / row_count if row_count else 0.0
        name_hint = bool(re.search(r"(^id$|_id$|id$|uuid|identifier|passengerid)", column_name, re.I))
        if missing == 0 and (name_hint or unique_ratio >= 0.98):
            reasons = []
            if name_hint:
                reasons.append("name_hint")
            if unique_ratio >= 0.98:
                reasons.append("near_unique")
            id_candidates.append(
                {
                    "column": column_name,
                    "unique_ratio": round(unique_ratio, 4),
                    "reasons": reasons,
                }
            )

    key_column_candidates = [
        column
        for column in columns
        if re.search(r"(target|label|class|survived|score|price|fare|sales|value)", column, re.I)
    ]

    return {
        "file": str(path),
        "file_name": path.name,
        "shape": [row_count, column_count],
        "columns": columns,
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "missing_count": {str(column): int(count) for column, count in missing_count.items()},
        "missing_ratio": missing_ratio,
        "numeric_summary": numeric_summary,
        "unique_counts": unique_counts,
        "categorical_summary": categorical_summary,
        "possible_datetime_columns": datetime_candidates,
        "possible_categorical_columns": possible_categorical_columns,
        "possible_id_columns": id_candidates,
        "possible_target_or_key_columns": key_column_candidates,
        "normalized_column_name_mapping": _normalized_mapping(columns),
    }


def write_schema_profile(csv_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    profile = profile_csv(csv_path)
    write_json(profile, output_path)
    return profile


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate a compact InfiAgentBench schema profile.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    parser.add_argument(
        "--output",
        default="schema_profile.json",
        help="Output JSON path. Defaults to ./schema_profile.json.",
    )
    args = parser.parse_args()
    write_schema_profile(args.csv, args.output)


if __name__ == "__main__":
    _main()
