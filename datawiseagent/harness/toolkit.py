"""Lightweight CSV/dataframe helper functions for harness-guided notebooks.

This module intentionally avoids pandas so it can be uploaded into benchmark
sessions as `input/harness_tools.py` and used even in minimal environments. The
agent may still choose pandas; these helpers provide deterministic cross-checks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean as _mean, median as _median
import csv
import math

MISSING = {"", "na", "n/a", "nan", "null", "none", "missing", "?"}


def read_csv(path):
    path = Path(path)
    sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f, dialect=dialect))


def is_missing(value):
    return value is None or str(value).strip().lower() in MISSING


def to_float(value):
    if is_missing(value):
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def numeric_column(rows, column, drop_missing=True):
    values = []
    missing = 0
    for row in rows:
        value = to_float(row.get(column))
        if value is None:
            missing += 1
            if not drop_missing:
                values.append(value)
        else:
            values.append(value)
    return values, {"column": column, "sample_size": len(values), "missing_count": missing}


def summary_stats(rows, column, ddof=1):
    values, meta = numeric_column(rows, column, drop_missing=True)
    if not values:
        raise ValueError(f"No numeric values in column {column!r}")
    variance = None
    std = None
    if len(values) > ddof:
        mu = _mean(values)
        variance = sum((x - mu) ** 2 for x in values) / (len(values) - ddof)
        std = math.sqrt(variance)
    result = {
        "mean": _mean(values),
        "median": _median(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
        "std": std,
        "variance": variance,
        "ddof": ddof,
    }
    result.update(meta)
    return result


def grouped_mean(rows, group_column, value_column):
    groups = defaultdict(list)
    missing = 0
    for row in rows:
        value = to_float(row.get(value_column))
        if value is None:
            missing += 1
            continue
        groups[row.get(group_column)].append(value)
    return {
        group: {"mean": _mean(values), "sample_size": len(values)}
        for group, values in groups.items()
        if values
    }, {"missing_count": missing, "group_count": len(groups)}


def pearson_correlation(rows, x_column, y_column):
    pairs = []
    missing = 0
    for row in rows:
        x = to_float(row.get(x_column))
        y = to_float(row.get(y_column))
        if x is None or y is None:
            missing += 1
            continue
        pairs.append((x, y))
    n = len(pairs)
    if n < 2:
        raise ValueError("Need at least two complete pairs for Pearson correlation")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = _mean(xs)
    my = _mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        raise ValueError("Zero variance column for Pearson correlation")
    return {"r": numerator / (denom_x * denom_y), "sample_size": n, "missing_pair_count": missing}


def _quantile(sorted_values, q):
    if not sorted_values:
        raise ValueError("empty values")
    pos = (len(sorted_values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[int(pos)]
    return sorted_values[low] * (high - pos) + sorted_values[high] * (pos - low)


def iqr_outliers(rows, column, multiplier=1.5):
    values, meta = numeric_column(rows, column, drop_missing=True)
    ordered = sorted(values)
    q1 = _quantile(ordered, 0.25)
    q3 = _quantile(ordered, 0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    outliers = []
    for idx, row in enumerate(rows):
        value = to_float(row.get(column))
        if value is not None and (value < lower or value > upper):
            outliers.append({"index": idx, "value": value})
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": lower,
        "upper": upper,
        "outlier_count": len(outliers),
        "outliers": outliers,
        **meta,
    }


def value_counts(rows, column):
    counts = Counter(row.get(column) for row in rows if not is_missing(row.get(column)))
    return dict(counts)
