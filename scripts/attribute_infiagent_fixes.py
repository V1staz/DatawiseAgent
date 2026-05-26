from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_extensions.infiagentbench.io_utils import read_json, read_jsonl, write_json


MODULE_BY_ERROR_TYPE = {
    "format_error": "final_formatter",
    "column_selection_error": "schema/contract",
    "filtering_error": "contract/verifier",
    "aggregation_error": "contract/skill",
    "preprocessing_order_error": "contract/skill/verifier",
    "feature_definition_error": "contract/skill/verifier",
    "statistical_method_error": "contract/skill/verifier",
    "significance_judgment_error": "contract/skill/verifier",
    "outlier_protocol_error": "contract/skill/verifier",
    "ml_protocol_error": "contract/skill/verifier",
    "reformat_error": "final_formatter",
    "silent_failure": "verifier/final_formatter",
    "unknown": "manual_analysis",
}


def _load_cases(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    raise ValueError("small error set must be a JSON list or an object with a cases field")


def _eval_by_id(eval_path: str | None) -> dict[int, dict[str, Any]]:
    if not eval_path:
        return {}
    report = read_json(eval_path)
    return {int(row["id"]): row for row in report.get("results", [])}


def _responses_by_id(path: str | None) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    return {int(row["id"]): row for row in read_jsonl(path)}


def _questions_by_id(path: str | None) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    return {int(row["id"]): row for row in read_jsonl(path)}


def _answer_text(eval_row: dict[str, Any] | None, response_row: dict[str, Any] | None) -> str:
    if eval_row and eval_row.get("predicted_answers"):
        return json.dumps(eval_row["predicted_answers"], ensure_ascii=False, sort_keys=True)
    if response_row and response_row.get("response") is not None:
        return str(response_row["response"])
    return ""


def _gold_text(eval_row: dict[str, Any] | None) -> str:
    if eval_row and eval_row.get("label_answers"):
        return json.dumps(eval_row["label_answers"], ensure_ascii=False, sort_keys=True)
    return ""


def _correct(eval_row: dict[str, Any] | None) -> bool | None:
    return bool(eval_row.get("is_correct")) if eval_row else None


def _first_present(*values: dict[str, Any] | None) -> dict[str, Any] | None:
    for value in values:
        if value:
            return value
    return None


def _is_verifier_blocked(response_row: dict[str, Any] | None) -> bool:
    if not response_row:
        return False
    verifier = response_row.get("verifier_report") or {}
    return response_row.get("response") == "ERROR_NEED_RETRY" or bool(verifier.get("must_retry"))


def _comparison_status(
    baseline_correct: bool | None,
    full_correct: bool | None,
    full_response: dict[str, Any] | None,
) -> str:
    if _is_verifier_blocked(full_response):
        return "verifier_blocked"
    if baseline_correct is None or full_correct is None:
        return "not_run"
    if not baseline_correct and full_correct:
        return "fixed"
    if baseline_correct and not full_correct:
        return "regressed"
    if baseline_correct and full_correct:
        return "both_correct"
    return "unchanged_wrong"


def _markdown_cell(value: Any, max_chars: int = 180) -> str:
    text = str(value if value is not None else "")
    text = text.replace("\n", "<br>").replace("|", "\\|")
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-task fix attribution table for InfiAgentBench small error set.")
    parser.add_argument("--small-error-set", required=True)
    parser.add_argument("--questions-file", default="evaluation/InfiAgentBench/data/da-dev-questions.jsonl")
    parser.add_argument("--baseline-eval", help="Backward-compatible alias for --baseline-raw-eval.")
    parser.add_argument("--baseline-raw-eval")
    parser.add_argument("--baseline-reformat-eval")
    parser.add_argument("--full-eval")
    parser.add_argument("--baseline-predictions", help="Backward-compatible alias for --baseline-raw-predictions.")
    parser.add_argument("--baseline-raw-predictions")
    parser.add_argument("--baseline-reformat-predictions")
    parser.add_argument("--full-predictions")
    parser.add_argument("--output", default="runs/infiagent_improve/small_error_fix_attribution.json")
    args = parser.parse_args()

    cases = _load_cases(args.small_error_set)
    questions = _questions_by_id(args.questions_file)
    baseline_raw_eval = _eval_by_id(args.baseline_raw_eval or args.baseline_eval)
    baseline_reformat_eval = _eval_by_id(args.baseline_reformat_eval)
    full_eval = _eval_by_id(args.full_eval)
    baseline_raw_predictions = _responses_by_id(args.baseline_raw_predictions or args.baseline_predictions)
    baseline_reformat_predictions = _responses_by_id(args.baseline_reformat_predictions)
    full_predictions = _responses_by_id(args.full_predictions)

    rows = []
    for case in cases:
        task_id = int(case["task_id"])
        error_type = case.get("original_error_type", "unknown")
        question = questions.get(task_id, {})
        baseline_raw_row = baseline_raw_eval.get(task_id)
        baseline_reformat_row = baseline_reformat_eval.get(task_id)
        full_row = full_eval.get(task_id)
        baseline_raw_correct = _correct(baseline_raw_row)
        baseline_reformat_correct = _correct(baseline_reformat_row)
        baseline_correct = baseline_reformat_correct if baseline_reformat_correct is not None else baseline_raw_correct
        full_correct = _correct(full_row)
        module = MODULE_BY_ERROR_TYPE.get(error_type, "manual_analysis")
        fix_status = _comparison_status(baseline_correct, full_correct, full_predictions.get(task_id))
        if fix_status == "not_run":
            repair_reason = ""
            still_failed_reason = "No comparable live baseline/full evaluation was provided."
        elif fix_status == "fixed":
            repair_reason = f"Likely addressed by {module}."
            still_failed_reason = ""
        elif fix_status == "both_correct":
            repair_reason = "Baseline already matched the label on this run."
            still_failed_reason = ""
        elif fix_status == "regressed":
            repair_reason = ""
            still_failed_reason = f"Full system introduced or exposed a failure in {module}."
        elif fix_status == "verifier_blocked":
            repair_reason = ""
            still_failed_reason = "Verifier or final block validation blocked the final response."
        else:
            repair_reason = ""
            still_failed_reason = f"Original failure type remains in scope for {module}; inspect verifier_report and final_block."
        rows.append(
            {
                "task_id": task_id,
                "difficulty": question.get("level", ""),
                "concept": ",".join(question.get("concepts", []) or []),
                "original_error_type": error_type,
                "known_error_type": error_type,
                "baseline_raw_answer": _answer_text(baseline_raw_row, baseline_raw_predictions.get(task_id)),
                "baseline_reformat_answer": _answer_text(
                    baseline_reformat_row, baseline_reformat_predictions.get(task_id)
                ),
                "full_final_block_answer": _answer_text(full_row, full_predictions.get(task_id)),
                "baseline_answer": _answer_text(
                    baseline_reformat_row or baseline_raw_row,
                    baseline_reformat_predictions.get(task_id) or baseline_raw_predictions.get(task_id),
                ),
                "full_answer": _answer_text(full_row, full_predictions.get(task_id)),
                "gold_answer": _gold_text(_first_present(full_row, baseline_reformat_row, baseline_raw_row)),
                "baseline_raw_correct": baseline_raw_correct,
                "baseline_reformat_correct": baseline_reformat_correct,
                "baseline_correct": baseline_correct,
                "full_correct": full_correct,
                "status": fix_status,
                "fix_status": fix_status,
                "suspected_fix_module": module,
                "repair_reason": repair_reason,
                "failure_reason": still_failed_reason,
                "still_failed_reason": still_failed_reason,
                "module": module,
                "note": case.get("note", ""),
            }
        )

    columns = [
        "task_id",
        "known_error_type",
        "baseline_raw_answer",
        "baseline_reformat_answer",
        "full_final_block_answer",
        "gold_answer",
        "baseline_correct",
        "full_correct",
        "status",
        "failure_reason",
    ]
    markdown = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        markdown.append("| " + " | ".join(_markdown_cell(row.get(column, "")) for column in columns) + " |")

    output = {
        "small_error_set": args.small_error_set,
        "baseline_raw_eval": args.baseline_raw_eval or args.baseline_eval,
        "baseline_reformat_eval": args.baseline_reformat_eval,
        "full_eval": args.full_eval,
        "rows": rows,
        "fair_comparison": {
            "baseline_raw_correct": sum(row["baseline_raw_correct"] is True for row in rows),
            "baseline_reformat_correct": sum(row["baseline_reformat_correct"] is True for row in rows),
            "full_final_block_correct": sum(row["full_correct"] is True for row in rows),
            "fixed": sum(row["status"] == "fixed" for row in rows),
            "regressed": sum(row["status"] == "regressed" for row in rows),
            "unchanged_wrong": sum(row["status"] == "unchanged_wrong" for row in rows),
            "both_correct": sum(row["status"] == "both_correct" for row in rows),
            "not_run": sum(row["status"] == "not_run" for row in rows),
        },
        "markdown_table": "\n".join(markdown),
    }
    write_json(output, args.output)
    print(output["markdown_table"])
    print(args.output)


if __name__ == "__main__":
    main()
