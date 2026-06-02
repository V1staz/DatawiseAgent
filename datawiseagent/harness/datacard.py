"""Data-card generation for uploaded tabular files."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import math

from .schemas import DataCard, DataCardColumn

_MISSING = {"", "na", "n/a", "nan", "null", "none", "missing", "?"}
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%b %d %Y",
    "%B %d %Y",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _MISSING


def _to_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    text = str(value).strip().replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _date_like(value: Any) -> bool:
    if _is_missing(value):
        return False
    text = str(value).strip()
    if not any(ch in text for ch in "-/ :"):
        return False
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            continue
    try:
        datetime.fromisoformat(text)
        return True
    except ValueError:
        return False


def _infer_dtype(values: list[Any]) -> tuple[str, bool, float | None, float | None]:
    non_missing = [v for v in values if not _is_missing(v)]
    if not non_missing:
        return "empty", False, None, None

    numeric_values = [_to_float(v) for v in non_missing]
    numeric_ratio = sum(v is not None for v in numeric_values) / len(non_missing)
    date_ratio = sum(_date_like(v) for v in non_missing[:200]) / min(len(non_missing), 200)

    if numeric_ratio >= 0.95:
        nums = [v for v in numeric_values if v is not None]
        all_int = all(float(v).is_integer() for v in nums)
        return ("integer" if all_int else "float"), False, min(nums), max(nums)
    if date_ratio >= 0.8:
        return "datetime", True, None, None
    return "string", False, None, None


def _role_hints(name: str, dtype: str, row_count: int, unique_count: int | None) -> list[str]:
    lowered = name.lower()
    hints: list[str] = []
    if dtype in {"integer", "float"}:
        hints.append("numeric")
    if dtype == "datetime" or any(token in lowered for token in ["date", "time", "year"]):
        hints.append("date_candidate")
    if any(token in lowered for token in ["id", "uuid", "key"]):
        hints.append("id_candidate")
    if unique_count is not None and row_count > 0:
        unique_rate = unique_count / row_count
        if unique_count <= 50 or unique_rate <= 0.2:
            hints.append("category_candidate")
        if unique_rate >= 0.95 and dtype in {"integer", "string"}:
            hints.append("id_candidate")
    return sorted(set(hints))


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    warnings: list[str] = []
    sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
        warnings.append("csv_sniffer_failed_default_excel")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))
    return fieldnames, rows, warnings


class DataCardBuilder:
    """Build concise, serializable schema summaries for prompt and trace use."""

    def build(self, file_path: str | Path) -> DataCard:
        path = Path(file_path)
        warnings: list[str] = []
        if not path.exists():
            return DataCard(
                path=str(path),
                row_count=0,
                column_count=0,
                warnings=["file_not_found"],
            )

        try:
            fieldnames, rows, read_warnings = _read_csv_rows(path)
            warnings.extend(read_warnings)
        except Exception as exc:
            return DataCard(
                path=str(path),
                row_count=0,
                column_count=0,
                warnings=[f"read_failed:{type(exc).__name__}:{exc}"],
            )

        values_by_column: dict[str, list[Any]] = defaultdict(list)
        for row in rows:
            for col in fieldnames:
                values_by_column[col].append(row.get(col))

        columns: list[DataCardColumn] = []
        row_count = len(rows)
        for col in fieldnames:
            values = values_by_column[col]
            missing_count = sum(_is_missing(v) for v in values)
            dtype, is_date, numeric_min, numeric_max = _infer_dtype(values)
            seen: list[Any] = []
            unique_values: set[str] = set()
            for value in values:
                if _is_missing(value):
                    continue
                key = str(value)
                unique_values.add(key)
                if len(seen) < 5 and key not in {str(v) for v in seen}:
                    seen.append(value)
            unique_count = len(unique_values)
            hints = _role_hints(col, dtype, row_count, unique_count)
            columns.append(
                DataCardColumn(
                    name=col,
                    dtype=dtype,
                    missing_count=missing_count,
                    missing_rate=round(missing_count / row_count, 6)
                    if row_count
                    else 0.0,
                    unique_count=unique_count,
                    sample_values=seen,
                    numeric_min=numeric_min,
                    numeric_max=numeric_max,
                    date_like=is_date,
                    role_hints=hints,
                )
            )

        numeric_columns = [c.name for c in columns if "numeric" in c.role_hints]
        date_columns = [c.name for c in columns if "date_candidate" in c.role_hints]
        category_columns = [c.name for c in columns if "category_candidate" in c.role_hints]
        id_columns = [c.name for c in columns if "id_candidate" in c.role_hints]

        return DataCard(
            path=str(path),
            row_count=row_count,
            column_count=len(fieldnames),
            columns=columns,
            date_columns=date_columns,
            category_columns=category_columns,
            numeric_columns=numeric_columns,
            id_columns=id_columns,
            warnings=warnings,
        )

    def to_prompt(self, data_card: DataCard, max_columns: int = 80) -> str:
        lines = [
            "# Data Card",
            f"file: {Path(data_card.path).name}",
            f"shape: {data_card.row_count} rows x {data_card.column_count} columns",
        ]
        if data_card.warnings:
            lines.append("warnings: " + ", ".join(data_card.warnings))
        lines.append("columns:")
        for col in data_card.columns[:max_columns]:
            pieces = [
                f"- {col.name}",
                f"dtype={col.dtype}",
                f"missing={col.missing_count} ({col.missing_rate:.2%})",
            ]
            if col.unique_count is not None:
                pieces.append(f"unique={col.unique_count}")
            if col.numeric_min is not None and col.numeric_max is not None:
                pieces.append(f"range=[{col.numeric_min:g}, {col.numeric_max:g}]")
            if col.role_hints:
                pieces.append("hints=" + "/".join(col.role_hints))
            if col.sample_values:
                pieces.append("sample=" + repr(col.sample_values[:3]))
            lines.append("; ".join(pieces))
        if len(data_card.columns) > max_columns:
            lines.append(f"... {len(data_card.columns) - max_columns} more columns omitted")
        return "\n".join(lines)
