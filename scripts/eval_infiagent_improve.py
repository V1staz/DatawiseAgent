from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_extensions.infiagentbench.error_taxonomy import ERROR_TYPES
from agent_extensions.infiagentbench.io_utils import read_jsonl, write_json


def extract_format(input_string: str) -> tuple[list[str], list[str]]:
    matches = re.findall(r"@(\w+)\[(.*?)\]", input_string or "", flags=re.S)
    return [match[0] for match in matches], [match[1] for match in matches]


def is_equal(response: Any, label: Any) -> bool:
    if response == label:
        return True
    try:
        return abs(float(response) - float(label)) < 1e-6
    except (TypeError, ValueError):
        return False


def _levels(path: str) -> dict[int, str]:
    return {int(row["id"]): row.get("level", "") for row in read_jsonl(path)}


def _concepts(path: str) -> dict[int, list[str]]:
    return {int(row["id"]): row.get("concepts", []) for row in read_jsonl(path)}


def _response_map(path: str) -> dict[int, dict[str, Any]]:
    return {int(row["id"]): row for row in read_jsonl(path) if "id" in row}


def _format_errors(expected: dict[str, str], predicted: dict[str, str], response_text: str) -> tuple[int, int]:
    if not predicted or "ERROR_NEED_RETRY" in (response_text or ""):
        return 1, 1
    missing = [name for name in expected if name not in predicted]
    extra = [name for name in predicted if name not in expected]
    return (1 if missing or extra else 0), 0


def _derive_task_status(
    response: dict[str, Any] | None,
    expected: dict[str, str],
    predicted: dict[str, str],
    response_text: str,
    is_correct: bool,
) -> str:
    if response is None:
        return "execution_error"
    existing_status = response.get("task_status")
    if existing_status in {"model_api_error", "execution_error", "verifier_blocked"}:
        return str(existing_status)
    if existing_status == "reformat_missing" or "ERROR_NEED_RETRY" in (response_text or ""):
        return "reformat_missing"
    missing = [name for name in expected if name not in predicted]
    extra = [name for name in predicted if name not in expected]
    if not predicted or missing or extra:
        return "format_error" if predicted else "reformat_missing"
    if not is_correct:
        return "evaluation_wrong"
    return "success"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    labels = read_jsonl(args.labels_file)
    responses = _response_map(args.responses_file)
    levels = _levels(args.questions_file)
    concepts = _concepts(args.questions_file)

    results = []
    error_counts: Counter[str] = Counter({error_type: 0 for error_type in ERROR_TYPES})
    format_error_count = 0
    reformat_missing_count = 0

    for label in labels:
        task_id = int(label["id"])
        expected = {name: value for name, value in label["common_answers"]}
        response = responses.get(task_id)
        if response is None and not args.count_missing:
            continue
        response_text = str((response or {}).get("response", ""))
        names, values = extract_format(response_text)
        predicted = dict(zip(names, values))
        correctness = {name: is_equal(predicted.get(name), expected[name]) for name in expected}
        is_correct = bool(correctness) and all(correctness.values())
        if response is None:
            error_counts["silent_failure"] += 1
        fmt_error, missing_error = _format_errors(expected, predicted, response_text)
        format_error_count += fmt_error
        reformat_missing_count += missing_error

        for error_type in (response or {}).get("error_types", []) or []:
            if error_type in ERROR_TYPES:
                error_counts[error_type] += 1
            else:
                error_counts["unknown"] += 1
        verifier_report = (response or {}).get("verifier_report") or {}
        for check in verifier_report.get("checks", []):
            if check.get("status") == "fail":
                error_counts[check.get("error_type") or "unknown"] += 1

        results.append(
            {
                "id": task_id,
                "level": levels.get(task_id),
                "concepts": concepts.get(task_id, []),
                "label_answers": expected,
                "predicted_answers": predicted,
                "correctness": correctness,
                "is_correct": is_correct,
                "task_status": _derive_task_status(response, expected, predicted, response_text, is_correct),
            }
        )

    total_questions = len(results)
    correct_questions = sum(result["is_correct"] for result in results)
    total_sub = sum(len(result["correctness"]) for result in results)
    correct_sub = sum(sum(result["correctness"].values()) for result in results)
    proportional = sum(
        (sum(result["correctness"].values()) / len(result["correctness"])) if result["correctness"] else 0
        for result in results
    )
    hard = [result for result in results if result.get("level") == "hard"]
    hard_abq = sum(result["is_correct"] for result in hard) / len(hard) if hard else 0.0

    concept_scores: dict[str, dict[str, float | int]] = {}
    concept_totals: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        for concept in result.get("concepts", []):
            concept_totals[concept].append(result["is_correct"])
    for concept, values in concept_totals.items():
        concept_scores[concept] = {"total": len(values), "ABQ": round(sum(values) / len(values), 4)}

    task_status_counts = Counter(str(result.get("task_status", "unknown")) for result in results)
    runtime_values = [float(row.get("runtime_seconds", 0.0)) for row in responses.values()]
    token_values = [int(row.get("token_estimate", 0)) for row in responses.values()]
    log_lengths = [int(row.get("log_length", len(str(row.get("raw_response") or row.get("response") or "")))) for row in responses.values()]

    metrics = {
        "ABQ": round(correct_questions / total_questions, 4) if total_questions else 0.0,
        "PASQ": round(proportional / total_questions, 4) if total_questions else 0.0,
        "UASQ": round(correct_sub / total_sub, 4) if total_sub else 0.0,
        "hard_ABQ": round(hard_abq, 4),
        "format_error_count": format_error_count,
        "column_mismatch_count": error_counts["column_selection_error"],
        "preprocessing_order_error_count": error_counts["preprocessing_order_error"],
        "statistical_judgment_error_count": error_counts["statistical_method_error"] + error_counts["significance_judgment_error"],
        "reformat_missing_count": reformat_missing_count,
        "average_token_estimate": round(sum(token_values) / len(token_values), 2) if token_values else 0.0,
        "average_log_length": round(sum(log_lengths) / len(log_lengths), 2) if log_lengths else 0.0,
        "average_runtime_seconds": round(sum(runtime_values) / len(runtime_values), 4) if runtime_values else 0.0,
    }
    return {
        "responses_file": args.responses_file,
        "metrics": metrics,
        "error_type_counts": {error_type: int(error_counts[error_type]) for error_type in ERROR_TYPES},
        "task_status_counts": dict(task_status_counts),
        "concept_scores": concept_scores,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate InfiAgentBench improved predictions with error taxonomy.")
    parser.add_argument("--questions-file", default="evaluation/InfiAgentBench/data/da-dev-questions.jsonl")
    parser.add_argument("--labels-file", default="evaluation/InfiAgentBench/data/da-dev-labels.jsonl")
    parser.add_argument("--responses-file", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--count-missing",
        action="store_true",
        help="Count questions without responses as failures. By default only responded tasks are evaluated, matching eval_closed_form.py.",
    )
    args = parser.parse_args()
    report = evaluate(args)
    output = args.output or str(Path(args.responses_file).with_suffix("")) + "_improve_eval.json"
    write_json(report, output)
    print(json.dumps({"output": output, "metrics": report["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
