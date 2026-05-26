from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json
from .schema_profile import normalize_column_name


TASK_TYPES = [
    "summary_statistics",
    "distribution_analysis",
    "correlation_analysis",
    "outlier_detection",
    "comprehensive_preprocessing",
    "feature_engineering",
    "machine_learning",
    "final_formatting",
]

_WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_CONCEPT_TASK_MAP = {
    "summary statistics": "summary_statistics",
    "distribution analysis": "distribution_analysis",
    "correlation analysis": "correlation_analysis",
    "outlier detection": "outlier_detection",
    "comprehensive data preprocessing": "comprehensive_preprocessing",
    "data preprocessing": "comprehensive_preprocessing",
    "feature engineering": "feature_engineering",
    "machine learning": "machine_learning",
}

_STAT_KEYWORDS = {
    "mean": ["mean", "average"],
    "median": ["median"],
    "std": ["standard deviation", "std", "sd"],
    "count": ["count", "number of", "how many"],
    "sum": ["sum", "total"],
    "min": ["minimum", "min", "lowest"],
    "max": ["maximum", "max", "highest", "largest"],
    "correlation": ["correlation", "coefficient", "relationship"],
    "p_value": ["p-value", "p value", "significance"],
    "normality": ["normality", "normal distribution", "normally distributed", "shapiro"],
    "skewness": ["skewness", "skew"],
    "accuracy": ["accuracy"],
    "mse": ["mean squared error", "mse"],
    "rmse": ["root mean squared error", "rmse"],
    "r2": ["r2", "r-squared", "r squared"],
}


def _has_keyword(text: str, keyword: str) -> bool:
    if re.fullmatch(r"[A-Za-z0-9_ -]+", keyword):
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(keyword).replace(r"\ ", r"\s+") + r"(?![A-Za-z0-9_])"
        return bool(re.search(pattern, text, re.I))
    return keyword.lower() in text.lower()


def _remove_forbidden_spans(text: str) -> str:
    return re.sub(r"(?:do not|don't|without|no further)\s+[^.\n]+", " ", text, flags=re.I)


def _join_question_parts(question_record: dict[str, Any]) -> str:
    return "\n".join(
        str(question_record.get(key, ""))
        for key in ["question", "constraints", "format"]
        if question_record.get(key)
    )


def _quoted_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(r"`([^`]+)`|'([^']+)'|\"([^\"]+)\"", text):
        value = next(group for group in match.groups() if group is not None)
        if value not in terms:
            terms.append(value)
    return terms


def extract_answer_names(format_text: str) -> list[str]:
    names = re.findall(r"@([A-Za-z_][A-Za-z0-9_]*)\[", format_text or "")
    return list(dict.fromkeys(names))


def _schema_columns(schema_profile: dict[str, Any] | None) -> list[str]:
    if not schema_profile:
        return []
    return [str(column) for column in schema_profile.get("columns", [])]


def _match_columns(text: str, schema_profile: dict[str, Any] | None) -> list[str]:
    columns = _schema_columns(schema_profile)
    if not columns:
        return []
    found: list[str] = []
    normalized_lookup = {normalize_column_name(column): column for column in columns}
    lowered = text.lower()

    for term in _quoted_terms(text):
        normalized = normalize_column_name(term)
        if normalized in normalized_lookup and normalized_lookup[normalized] not in found:
            found.append(normalized_lookup[normalized])

    for column in columns:
        escaped = re.escape(column)
        if re.search(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", text, re.I):
            if column not in found:
                found.append(column)
            continue
        normalized = normalize_column_name(column)
        variants = {normalized, normalized.replace("_", " ")}
        if any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(v)}(?![A-Za-z0-9_])", lowered) for v in variants):
            if column not in found:
                found.append(column)
    return found


def _parse_decimal_places(text: str) -> int | None:
    patterns = [
        r"round(?:ing)?(?:\s+off)?(?:\s+\w+){0,6}?\s+to\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+decimal",
        r"rounded\s+to\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+decimal",
        r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+decimal\s+places?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = match.group(1).lower()
            return int(value) if value.isdigit() else _WORD_NUMBERS[value]
    return None


def _parse_output_format(format_text: str, full_text: str) -> dict[str, Any]:
    lowered = (format_text or "").lower()
    if "dictionary" in lowered or "dict" in lowered:
        container_type = "dict"
    elif "list" in lowered or "array" in lowered:
        container_type = "list"
    elif "string" in lowered or "name" in lowered or "date" in lowered:
        container_type = "string"
    elif "integer" in lowered or "float" in lowered or "number" in lowered:
        container_type = "numeric"
    else:
        container_type = None

    quote_string = None
    if re.search(r"@\w+\[\"", format_text or "") or "quotation mark" in lowered:
        quote_string = True
    elif container_type == "string":
        quote_string = False

    return {
        "raw_format": format_text or "",
        "answer_pattern": "@answer_name[value]",
        "answer_names": extract_answer_names(format_text or ""),
        "decimal_places": _parse_decimal_places(full_text),
        "quote_string": quote_string,
        "container_type": container_type,
    }


def _parse_task_types(question_record: dict[str, Any], text: str) -> list[str]:
    lowered = text.lower()
    actionable_lowered = _remove_forbidden_spans(text).lower()
    task_types: list[str] = []
    for concept in question_record.get("concepts", []) or []:
        task_type = _CONCEPT_TASK_MAP.get(str(concept).strip().lower())
        if task_type and task_type not in task_types:
            task_types.append(task_type)

    keyword_rules = [
        ("correlation_analysis", ["correlation", "pearson", "spearman", "coefficient"]),
        ("outlier_detection", ["outlier", "iqr", "z-score", "z score", "interquartile"]),
        ("comprehensive_preprocessing", ["missing", "impute", "drop", "fillna", "preprocess", "normalize", "standardize", "encode", "one-hot"]),
        ("feature_engineering", ["new feature", "new column", "create a feature", "generate a feature", "ratio", "indicator"]),
        ("machine_learning", ["train_test_split", "train/test", "training set", "testing set", "sklearn", "model", "classifier", "regression", "accuracy", "mse", "rmse"]),
        ("distribution_analysis", ["distribution", "normality", "shapiro", "skewness", "histogram"]),
        ("summary_statistics", ["mean", "median", "standard deviation", "average", "count", "total", "minimum", "maximum"]),
    ]
    for task_type, keywords in keyword_rules:
        if any(_has_keyword(actionable_lowered, keyword) for keyword in keywords) and task_type not in task_types:
            task_types.append(task_type)

    if "final_formatting" not in task_types:
        task_types.append("final_formatting")
    return [task_type for task_type in TASK_TYPES if task_type in task_types]


def _parse_target_statistics(text: str) -> list[str]:
    lowered = text.lower()
    stats = []
    for statistic, keywords in _STAT_KEYWORDS.items():
        if any(_has_keyword(lowered, keyword) for keyword in keywords):
            stats.append(statistic)
    return stats


def _parse_required_methods(text: str) -> list[str]:
    lowered = text.lower()
    rules = [
        ("pearson", ["pearson"]),
        ("spearman", ["spearman"]),
        ("shapiro_wilk", ["shapiro", "shapiro-wilk"]),
        ("iqr", ["iqr", "interquartile range"]),
        ("z_score", ["z-score", "z score"]),
        ("linear_regression", ["linear regression"]),
        ("logistic_regression", ["logistic regression"]),
        ("random_forest", ["random forest"]),
        ("train_test_split", ["train_test_split", "training set", "testing set", "train/test"]),
        ("one_hot_encoding", ["one-hot", "one hot"]),
        ("label_encoding", ["label encoding"]),
    ]
    return [method for method, keywords in rules if any(keyword in lowered for keyword in keywords)]


def _parse_forbidden_actions(text: str) -> list[str]:
    forbidden: list[str] = []
    patterns = [
        r"do not [^.:\n]+",
        r"don't [^.:\n]+",
        r"without [^.:\n]+",
        r"no further [^.:\n]+",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            value = match.group(0).strip()
            if value not in forbidden:
                forbidden.append(value)
    return forbidden


def _parse_filter_conditions(text: str) -> list[str]:
    conditions: list[str] = []
    patterns = [
        r"(?:where|with|for|among)\s+[^.\n;]+(?:=|>|<|greater than|less than|older than|younger than|equal to)[^.\n;]*",
        r"passengers?\s+(?:who|with|older|younger)[^.\n;]*",
        r"rows?\s+(?:where|with)[^.\n;]*",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            value = match.group(0).strip(" .")
            if value and value not in conditions:
                conditions.append(value)
    return conditions


def _parse_groupby_keys(text: str, required_columns: list[str]) -> list[str]:
    lowered = text.lower()
    groupby: list[str] = []
    for column in required_columns:
        column_lower = column.lower()
        patterns = [
            rf"for each\s+[^.\n]*{re.escape(column_lower)}",
            rf"group(?:ed)? by\s+[^.\n]*{re.escape(column_lower)}",
            rf"per\s+[^.\n]*{re.escape(column_lower)}",
        ]
        if any(re.search(pattern, lowered) for pattern in patterns):
            groupby.append(column)
    return groupby


def _parse_preprocessing_steps(text: str) -> list[dict[str, Any]]:
    lowered = _remove_forbidden_spans(text).lower()
    rules = [
        ("drop_missing", ["drop missing", "remove missing", "drop rows", "delete rows"]),
        ("impute_missing", ["impute", "fill missing", "fillna", "replace missing"]),
        ("encode_categorical", ["encode", "one-hot", "one hot", "label encoding"]),
        ("normalize", ["normalize", "min-max", "min max"]),
        ("standardize", ["standardize", "standard scaling", "z-score scaling"]),
        ("log_transform", ["log transform", "logarithm", "natural log", "log10"]),
        ("remove_outliers", ["remove outlier", "drop outlier", "exclude outlier"]),
    ]
    steps: list[dict[str, Any]] = []
    for operation, keywords in rules:
        indexes = [lowered.find(keyword) for keyword in keywords if lowered.find(keyword) >= 0]
        if indexes:
            steps.append({"operation": operation, "text_position": min(indexes)})
    return sorted(steps, key=lambda item: item["text_position"])


def _parse_feature_definitions(text: str) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for match in re.finditer(
        r"(?:new\s+(?:feature|column)\s+(?:called|named)?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?.{0,120}?)(?:\.|\n)",
        text,
        re.I | re.S,
    ):
        snippet = re.sub(r"\s+", " ", match.group(0)).strip(" .")
        name = match.group(1)
        quoted = _quoted_terms(snippet)
        source_columns = [term for term in quoted if term != name]
        formula = None
        if re.search(r"\bsum|summing|add", snippet, re.I) and len(source_columns) >= 2:
            formula = f"{name} = " + " + ".join(source_columns[:2])
        elif re.search(r"\bratio|divide|dividing|per\b", snippet, re.I) and len(source_columns) >= 2:
            formula = f"{name} = {source_columns[0]} / {source_columns[1]}"
        definition = {
            "feature_name": name,
            "source_columns": source_columns,
            "formula": formula,
            "instruction": snippet,
        }
        if not any(
            existing.get("feature_name") == definition["feature_name"]
            and existing.get("formula") == definition["formula"]
            for existing in definitions
        ):
            definitions.append(definition)
    return definitions


def _parse_ml_protocol(text: str, required_columns: list[str]) -> dict[str, Any]:
    lowered = text.lower()
    protocol: dict[str, Any] = {}
    if "machine_learning" not in _parse_task_types({"concepts": []}, text):
        return protocol

    random_state_match = re.search(r"random_state(?:\s+parameter)?\s*(?:is\s+set\s+to|=|:)?\s*(\d+)", text, re.I)
    if random_state_match:
        protocol["random_state"] = int(random_state_match.group(1))

    split_match = re.search(r"training set\s*\((\d+)%\).*?testing set\s*\((\d+)%\)", text, re.I | re.S)
    if split_match:
        train_pct = int(split_match.group(1))
        test_pct = int(split_match.group(2))
        protocol["train_size"] = train_pct / 100
        protocol["test_size"] = test_pct / 100
    test_size_match = re.search(r"test_size\s*=\s*(0?\.\d+|\d+%)", text, re.I)
    if test_size_match:
        value = test_size_match.group(1)
        protocol["test_size"] = float(value.strip("%")) / 100 if value.endswith("%") else float(value)

    features_match = re.search(r"features?\s+(.{0,240}?)(?:\.| before| split| train|$)", text, re.I | re.S)
    if features_match:
        quoted = _quoted_terms(features_match.group(1))
        protocol["feature_names"] = [term for term in quoted if term in required_columns]
    elif required_columns:
        protocol["feature_names"] = required_columns

    target_match = re.search(r"predict\s+(?:whether\s+)?(?:a\s+|the\s+)?[^.\n]*?['\"]([A-Za-z_][A-Za-z0-9_ ]+)['\"]", text, re.I)
    if target_match:
        protocol["target"] = target_match.group(1)

    for model_name in [
        "linear regression",
        "logistic regression",
        "random forest",
        "decision tree",
        "svm",
        "knn",
    ]:
        if model_name in lowered:
            protocol["model"] = model_name
            break

    for metric in ["accuracy", "mean squared error", "mse", "rmse", "r2", "r-squared", "f1"]:
        if metric in lowered:
            protocol["metric"] = metric
            break

    return protocol


def _verification_checks(task_types: list[str]) -> list[str]:
    checks = ["required_columns_exist", "final_answer_format"]
    if "comprehensive_preprocessing" in task_types:
        checks.extend(["preprocessing_order", "missing_values_shape_recorded"])
    if "correlation_analysis" in task_types:
        checks.extend(["correlation_abs_r_if_strongest", "correlation_sign_retained", "p_value_alpha_if_required"])
    if "outlier_detection" in task_types:
        checks.append("outlier_method_matches_contract")
    if "feature_engineering" in task_types:
        checks.extend(["feature_formula_echoed", "feature_first_five_rows_checked"])
    if "machine_learning" in task_types:
        checks.extend(["ml_random_state", "ml_split", "ml_feature_names", "ml_metric"])
    return list(dict.fromkeys(checks))


def parse_question_contract(
    question_record: dict[str, Any],
    schema_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = _join_question_parts(question_record)
    format_text = str(question_record.get("format", ""))
    task_types = _parse_task_types(question_record, text)
    required_columns = _match_columns(text, schema_profile)
    feature_definitions = _parse_feature_definitions(text)
    target_statistics = _parse_target_statistics(text)
    if (
        "feature_engineering" in task_types
        and "summary_statistics" not in task_types
        and "sum" in target_statistics
        and not any("sum" in name.lower() or "total" in name.lower() for name in extract_answer_names(format_text))
    ):
        target_statistics = [statistic for statistic in target_statistics if statistic != "sum"]

    new_features = list(dict.fromkeys(definition["feature_name"] for definition in feature_definitions))
    for definition in feature_definitions:
        for source_column in definition.get("source_columns", []):
            if source_column in _schema_columns(schema_profile) and source_column not in required_columns:
                required_columns.append(source_column)

    contract = {
        "task_id": question_record.get("id"),
        "question": question_record.get("question", ""),
        "constraints": question_record.get("constraints", ""),
        "concepts": question_record.get("concepts", []),
        "answer_names": extract_answer_names(format_text),
        "required_columns": required_columns,
        "new_columns": new_features,
        "filter_conditions": _parse_filter_conditions(text),
        "groupby_keys": _parse_groupby_keys(text, required_columns),
        "target_statistics": target_statistics,
        "required_methods": _parse_required_methods(text),
        "preprocessing_steps": _parse_preprocessing_steps(text),
        "feature_definitions": feature_definitions,
        "ml_protocol": _parse_ml_protocol(text, required_columns),
        "rounding": {"decimal_places": _parse_decimal_places(text)},
        "decimal_places": _parse_decimal_places(text),
        "output_format": _parse_output_format(format_text, text),
        "format_requirements": _parse_output_format(format_text, text),
        "forbidden_actions": _parse_forbidden_actions(text),
        "task_types": task_types,
        "verification_checks": _verification_checks(task_types),
    }
    return contract


def write_question_contract(
    question_record: dict[str, Any],
    schema_profile: dict[str, Any] | None,
    output_path: str | Path,
) -> dict[str, Any]:
    contract = parse_question_contract(question_record, schema_profile)
    write_json(contract, output_path)
    return contract


def _main() -> None:
    parser = argparse.ArgumentParser(description="Parse an InfiAgentBench question contract.")
    parser.add_argument("--question-json", required=True, help="Path to a JSON question record.")
    parser.add_argument("--schema-profile", help="Optional schema_profile.json path.")
    parser.add_argument("--output", default="question_contract.json")
    args = parser.parse_args()
    question = read_json(args.question_json)
    schema = read_json(args.schema_profile) if args.schema_profile else None
    write_question_contract(question, schema, args.output)


if __name__ == "__main__":
    _main()
