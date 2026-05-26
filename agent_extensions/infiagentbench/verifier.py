from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from .final_block import extract_final_block, final_block_to_response, validate_final_block
from .io_utils import read_json, write_json


def _check(name: str, status: str, message: str, *, error_type: str | None = None, suggestion: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"name": name, "status": status, "message": message}
    if error_type:
        item["error_type"] = error_type
    if suggestion:
        item["suggestion"] = suggestion
    return item


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _schema_columns(schema_profile: dict[str, Any] | None) -> set[str]:
    return set(str(column) for column in (schema_profile or {}).get("columns", []))


def _status_summary(checks: list[dict[str, Any]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    if any(check["status"] == "warn" for check in checks):
        return "warn"
    return "pass"


def _verify_columns(schema_profile: dict[str, Any] | None, contract: dict[str, Any]) -> dict[str, Any]:
    columns = _schema_columns(schema_profile)
    new_columns = set(contract.get("new_columns", []))
    required = [column for column in contract.get("required_columns", []) if column not in new_columns]
    missing = [column for column in required if column not in columns]
    if missing:
        return _check(
            "required_columns_exist",
            "fail",
            f"Missing required columns: {missing}",
            error_type="column_selection_error",
            suggestion="Re-read schema_profile and remap required columns before execution.",
        )
    return _check("required_columns_exist", "pass", "All required columns exist in schema_profile.")


def _verify_filter_sample_count(execution_log: str, contract: dict[str, Any]) -> dict[str, Any]:
    if not contract.get("filter_conditions"):
        return _check("filter_sample_count_nonzero", "pass", "No explicit filter condition in contract.")
    zero_patterns = [
        r"shape\s*[:=]?\s*\(0\s*,",
        r"0\s+rows",
        r"empty\s+dataframe",
        r"len\([^)]*\)\s*[:=]\s*0",
        r"sample\s+size\s*[:=]\s*0",
    ]
    if any(re.search(pattern, execution_log, re.I) for pattern in zero_patterns):
        return _check(
            "filter_sample_count_nonzero",
            "fail",
            "Filtered sample appears to be zero.",
            error_type="filtering_error",
            suggestion="Check filter predicates and column value types before computing the answer.",
        )
    if not _contains_any(execution_log, ["shape", "len(", "sample", "count"]):
        return _check(
            "filter_sample_count_nonzero",
            "warn",
            "Filter conditions exist but filtered sample count is not recorded.",
            error_type="filtering_error",
            suggestion="Record row count after filtering.",
        )
    return _check("filter_sample_count_nonzero", "pass", "Filtered sample count is recorded and not obviously zero.")


def _verify_preprocessing(execution_log: str, contract: dict[str, Any]) -> list[dict[str, Any]]:
    steps = contract.get("preprocessing_steps", [])
    if not steps:
        return [_check("preprocessing_order", "pass", "No ordered preprocessing steps in contract.")]

    lowered = execution_log.lower()
    positions: list[int] = []
    missing_ops: list[str] = []
    keyword_map = {
        "drop_missing": ["dropna", "drop missing", "remove missing", "drop rows"],
        "impute_missing": ["fillna", "impute", "fill missing", "replace missing"],
        "encode_categorical": ["get_dummies", "one-hot", "one hot", "labelencoder", "encode"],
        "normalize": ["normalize", "minmax", "min-max"],
        "standardize": ["standardscaler", "standardize", "z-score scaling"],
        "log_transform": ["np.log", "log transform", "log10", "log1p"],
        "remove_outliers": ["remove outlier", "drop outlier", "exclude outlier"],
    }
    for step in steps:
        operation = step.get("operation")
        indexes = [lowered.find(keyword) for keyword in keyword_map.get(operation, [operation]) if lowered.find(keyword) >= 0]
        if indexes:
            positions.append(min(indexes))
        else:
            missing_ops.append(operation)

    checks: list[dict[str, Any]] = []
    if missing_ops:
        checks.append(
            _check(
                "preprocessing_order",
                "fail",
                f"Preprocessing steps not found in execution log: {missing_ops}",
                error_type="preprocessing_order_error",
                suggestion="Execute and log preprocessing steps exactly in contract order.",
            )
        )
    elif positions != sorted(positions):
        checks.append(
            _check(
                "preprocessing_order",
                "fail",
                "Preprocessing operations appear out of contract order.",
                error_type="preprocessing_order_error",
                suggestion="Re-run preprocessing in the ordered contract sequence.",
            )
        )
    else:
        checks.append(_check("preprocessing_order", "pass", "Preprocessing operations appear in contract order."))

    if _contains_any(execution_log, ["shape"]) and _contains_any(execution_log, ["missing", "isna", "null"]):
        checks.append(_check("missing_values_shape_recorded", "pass", "Shape and missing-value checks are recorded."))
    else:
        checks.append(
            _check(
                "missing_values_shape_recorded",
                "warn",
                "Preprocessing is requested but shape/missing-value checkpoints are incomplete.",
                error_type="preprocessing_order_error",
                suggestion="Record shape and missing counts before and after each preprocessing step.",
            )
        )
    return checks


def _verify_correlation(execution_log: str, contract: dict[str, Any]) -> list[dict[str, Any]]:
    if "correlation_analysis" not in contract.get("task_types", []):
        return []
    lowered_question = " ".join(
        str(contract.get(key, "")) for key in ["question", "constraints"]
    ).lower()
    lowered_log = execution_log.lower()
    checks: list[dict[str, Any]] = []
    if "strongest" in lowered_question or "highest correlation" in lowered_question:
        if "abs(" in lowered_log or ".abs(" in lowered_log or "absolute" in lowered_log:
            checks.append(_check("correlation_abs_r_if_strongest", "pass", "Strongest correlation appears to compare abs(r)."))
        else:
            checks.append(
                _check(
                    "correlation_abs_r_if_strongest",
                    "fail",
                    "Question asks for strongest correlation but execution log does not show abs(r) comparison.",
                    error_type="statistical_method_error",
                    suggestion="Compare absolute correlation magnitude and then return the signed coefficient.",
                )
            )
    else:
        checks.append(_check("correlation_abs_r_if_strongest", "pass", "Strongest-correlation protocol not required."))

    if "p_value" in contract.get("target_statistics", []) or "significance" in lowered_question:
        has_p_value = _contains_any(execution_log, ["p-value", "p_value", "pvalue"])
        has_alpha = _contains_any(execution_log, ["alpha", "0.05", "significance level"])
        if has_p_value and has_alpha:
            checks.append(_check("p_value_alpha_if_required", "pass", "p-value and alpha decision are recorded."))
        else:
            checks.append(
                _check(
                    "p_value_alpha_if_required",
                    "fail",
                    "Significance is requested but p-value/alpha decision is incomplete.",
                    error_type="significance_judgment_error",
                    suggestion="Compute p-value and compare it with the stated alpha threshold.",
                )
            )
    return checks


def _verify_outlier(execution_log: str, contract: dict[str, Any]) -> list[dict[str, Any]]:
    if "outlier_detection" not in contract.get("task_types", []):
        return []
    required_methods = set(contract.get("required_methods", []))
    lowered = execution_log.lower()
    checks: list[dict[str, Any]] = []
    if "iqr" in required_methods and not _contains_any(lowered, ["iqr", "interquartile", "quantile(0.25", "quantile(0.75"]):
        checks.append(
            _check(
                "outlier_method_matches_contract",
                "fail",
                "IQR method is required but not visible in execution log.",
                error_type="outlier_protocol_error",
                suggestion="Use Q1/Q3/IQR bounds exactly as requested.",
            )
        )
    elif "z_score" in required_methods and not _contains_any(lowered, ["z-score", "z_score", "zscore"]):
        checks.append(
            _check(
                "outlier_method_matches_contract",
                "fail",
                "Z-score method is required but not visible in execution log.",
                error_type="outlier_protocol_error",
                suggestion="Use the requested z-score threshold exactly.",
            )
        )
    else:
        checks.append(_check("outlier_method_matches_contract", "pass", "Outlier method is compatible with contract or unspecified."))

    if _contains_any(lowered, ["...", "truncated"]) and _contains_any(lowered, ["outlier"]):
        checks.append(
            _check(
                "outlier_list_not_truncated",
                "warn",
                "Outlier list may be truncated.",
                error_type="outlier_protocol_error",
                suggestion="Write complete outlier list or save it as structured JSON.",
            )
        )
    return checks


def _verify_feature_engineering(execution_log: str, contract: dict[str, Any]) -> list[dict[str, Any]]:
    if "feature_engineering" not in contract.get("task_types", []):
        return []
    lowered = execution_log.lower()
    checks: list[dict[str, Any]] = []
    definitions = contract.get("feature_definitions", [])
    missing_formula = []
    for definition in definitions:
        feature_name = str(definition.get("feature_name", ""))
        formula = str(definition.get("formula") or definition.get("instruction") or "")
        if feature_name and feature_name.lower() not in lowered:
            missing_formula.append(feature_name)
            continue
        if formula and not all(part.lower() in lowered for part in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)[:3]):
            missing_formula.append(feature_name or formula)
    if missing_formula:
        checks.append(
            _check(
                "feature_formula_echoed",
                "fail",
                f"Feature formula/name not echoed clearly: {missing_formula}",
                error_type="feature_definition_error",
                suggestion="Echo the formula before computing and show source columns.",
            )
        )
    else:
        checks.append(_check("feature_formula_echoed", "pass", "Feature formula/name is echoed."))

    if _contains_any(lowered, ["head()", ".head(", "first 5", "first five", "前 5"]):
        checks.append(_check("feature_first_five_rows_checked", "pass", "First-five-row feature check is present."))
    else:
        checks.append(
            _check(
                "feature_first_five_rows_checked",
                "warn",
                "Feature engineering did not show a first-five-row verification.",
                error_type="feature_definition_error",
                suggestion="Print the first 5 rows of source and derived columns.",
            )
        )
    return checks


def _verify_ml_protocol(execution_log: str, contract: dict[str, Any]) -> list[dict[str, Any]]:
    if "machine_learning" not in contract.get("task_types", []):
        return []
    protocol = contract.get("ml_protocol", {})
    lowered = execution_log.lower()
    checks: list[dict[str, Any]] = []

    expected_random_state = protocol.get("random_state")
    if expected_random_state is not None:
        if re.search(rf"random_state\s*=\s*{expected_random_state}\b", execution_log) or f"random_state {expected_random_state}" in lowered:
            checks.append(_check("ml_random_state", "pass", "Expected random_state is visible."))
        else:
            checks.append(
                _check(
                    "ml_random_state",
                    "fail",
                    f"Expected random_state={expected_random_state} is not visible.",
                    error_type="ml_protocol_error",
                    suggestion="Set train_test_split random_state exactly as specified.",
                )
            )

    if protocol.get("test_size") is not None:
        expected = protocol["test_size"]
        candidates = [f"test_size={expected}", f"test_size = {expected}", f"{int(expected * 100)}%"]
        if any(candidate in execution_log for candidate in candidates):
            checks.append(_check("ml_split", "pass", "Expected split ratio is visible."))
        else:
            checks.append(
                _check(
                    "ml_split",
                    "warn",
                    "Expected train/test split ratio is not clearly logged.",
                    error_type="ml_protocol_error",
                    suggestion="Log train/test split parameters and resulting shapes.",
                )
            )

    feature_names = protocol.get("feature_names") or []
    missing_features = [feature for feature in feature_names if feature.lower() not in lowered]
    if missing_features:
        checks.append(
            _check(
                "ml_feature_names",
                "fail",
                f"ML feature names not all visible: {missing_features}",
                error_type="ml_protocol_error",
                suggestion="Print exact feature_names used to build X.",
            )
        )
    elif feature_names:
        checks.append(_check("ml_feature_names", "pass", "ML feature names are visible."))

    metric = protocol.get("metric")
    if metric and metric.lower() not in lowered:
        checks.append(
            _check(
                "ml_metric",
                "fail",
                f"Requested metric not visible: {metric}",
                error_type="ml_protocol_error",
                suggestion="Compute and log only the requested metric on the test split.",
            )
        )
    elif metric:
        checks.append(_check("ml_metric", "pass", "Requested metric is visible."))

    if _contains_any(lowered, ["gridsearch", "randomizedsearch", "hyperparameter tuning", "optuna"]):
        checks.append(
            _check(
                "no_unrequested_tuning",
                "warn",
                "Execution log suggests unrequested hyperparameter tuning.",
                error_type="ml_protocol_error",
                suggestion="Remove tuning unless the question explicitly asks for it.",
            )
        )
    return checks


def _verify_final_format(
    execution_log: str,
    final_block: dict[str, Any] | None,
    contract: dict[str, Any],
    require_final_block: bool,
) -> list[dict[str, Any]]:
    if final_block is None:
        final_block = extract_final_block(execution_log)
    if final_block is None and not require_final_block:
        matches = re.findall(r"@(\w+)\[(.*?)\]", execution_log or "", flags=re.S)
        if not matches:
            return [
                _check(
                    "final_answer_format",
                    "fail",
                    "No @answer[value] response found.",
                    error_type="reformat_error",
                    suggestion="Emit all required @answer_name[value] entries or enable final block formatting.",
                )
            ]
        pseudo_block = {
            "answers": [
                {"name": name, "value": value, "format_checked": True}
                for name, value in matches
            ]
        }
        validation = validate_final_block(pseudo_block, contract)
        if not validation["passed"]:
            return [
                _check(
                    "final_answer_format",
                    "fail",
                    f"@answer[value] validation failed: {validation['errors']}",
                    error_type="format_error",
                    suggestion="Fix answer names, placeholders, containers, quotes, or decimal places.",
                )
            ]
        return [_check("final_answer_format", "pass", "Raw @answer[value] output matches the contract.")]
    validation = validate_final_block(final_block, contract)
    if not validation["passed"]:
        return [
            _check(
                "final_answer_format",
                "fail",
                f"Final block validation failed: {validation['errors']}",
                error_type="format_error",
                suggestion="Recompute or rewrite the final block with checked answer names and values.",
            )
        ]
    response, response_validation = final_block_to_response(final_block, contract)
    if response == "ERROR_NEED_RETRY" or not response_validation["passed"]:
        return [
            _check(
                "final_answer_format",
                "fail",
                f"Final response conversion failed: {response_validation['errors']}",
                error_type="reformat_error",
                suggestion="Regenerate final block before reformatting.",
            )
        ]
    return [_check("final_answer_format", "pass", "Final block can be converted to @answer[value].")]


def verify_execution(
    *,
    schema_profile: dict[str, Any] | None,
    question_contract: dict[str, Any],
    skill_plan: dict[str, Any] | None = None,
    execution_log: str = "",
    final_block: dict[str, Any] | None = None,
    require_final_block: bool = True,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(_verify_columns(schema_profile, question_contract))
    checks.append(_verify_filter_sample_count(execution_log, question_contract))
    checks.extend(_verify_preprocessing(execution_log, question_contract))
    checks.extend(_verify_correlation(execution_log, question_contract))
    checks.extend(_verify_outlier(execution_log, question_contract))
    checks.extend(_verify_feature_engineering(execution_log, question_contract))
    checks.extend(_verify_ml_protocol(execution_log, question_contract))
    checks.extend(_verify_final_format(execution_log, final_block, question_contract, require_final_block))

    warnings = [check["message"] for check in checks if check["status"] == "warn"]
    errors = [check["message"] for check in checks if check["status"] == "fail"]
    retry_suggestions = [check["suggestion"] for check in checks if check.get("suggestion") and check["status"] == "fail"]
    status = _status_summary(checks)
    return {
        "task_id": question_contract.get("task_id"),
        "status": status,
        "passed": status != "fail",
        "must_retry": status == "fail",
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "retry_suggestion": " ".join(retry_suggestions),
        "skill_plan_used": bool(skill_plan),
    }


def write_verifier_report(
    *,
    schema_profile: dict[str, Any] | None,
    question_contract: dict[str, Any],
    skill_plan: dict[str, Any] | None,
    execution_log: str,
    final_block: dict[str, Any] | None,
    output_path: str | Path,
) -> dict[str, Any]:
    report = verify_execution(
        schema_profile=schema_profile,
        question_contract=question_contract,
        skill_plan=skill_plan,
        execution_log=execution_log,
        final_block=final_block,
        require_final_block=final_block is not None,
    )
    write_json(report, output_path)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Verify an InfiAgentBench execution log and final block.")
    parser.add_argument("--schema-profile", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--skill-plan")
    parser.add_argument("--execution-log", required=True, help="Text file containing the notebook/agent response.")
    parser.add_argument("--final-block")
    parser.add_argument("--output", default="verifier_report.json")
    args = parser.parse_args()

    schema = read_json(args.schema_profile)
    contract = read_json(args.contract)
    skill_plan = read_json(args.skill_plan) if args.skill_plan else None
    execution_log = Path(args.execution_log).read_text(encoding="utf-8")
    final_block = read_json(args.final_block) if args.final_block else extract_final_block(execution_log)
    write_verifier_report(
        schema_profile=schema,
        question_contract=contract,
        skill_plan=skill_plan,
        execution_log=execution_log,
        final_block=final_block,
        output_path=args.output,
    )


if __name__ == "__main__":
    _main()
