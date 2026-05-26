from __future__ import annotations

from collections import Counter
from typing import Any


ERROR_TYPES = [
    "format_error",
    "column_selection_error",
    "filtering_error",
    "aggregation_error",
    "preprocessing_order_error",
    "feature_definition_error",
    "statistical_method_error",
    "significance_judgment_error",
    "outlier_protocol_error",
    "ml_protocol_error",
    "reformat_error",
    "silent_failure",
    "unknown",
]


def error_types_from_report(report: dict[str, Any]) -> list[str]:
    types: list[str] = []
    for check in report.get("checks", []):
        if check.get("status") == "fail":
            error_type = check.get("error_type") or "unknown"
            if error_type not in ERROR_TYPES:
                error_type = "unknown"
            if error_type not in types:
                types.append(error_type)
    if report.get("must_retry") and not types:
        types.append("unknown")
    return types


def count_error_types(reports: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for report in reports:
        counter.update(error_types_from_report(report))
    return {error_type: int(counter.get(error_type, 0)) for error_type in ERROR_TYPES}

