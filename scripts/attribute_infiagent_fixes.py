from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_extensions.infiagentbench.io_utils import read_json, write_json


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


def _correctness_by_id(eval_path: str | None) -> dict[int, bool]:
    if not eval_path:
        return {}
    report = read_json(eval_path)
    return {int(row["id"]): bool(row.get("is_correct")) for row in report.get("results", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-task fix attribution table for InfiAgentBench small error set.")
    parser.add_argument("--small-error-set", required=True)
    parser.add_argument("--baseline-eval")
    parser.add_argument("--full-eval")
    parser.add_argument("--output", default="runs/infiagent_improve/small_error_fix_attribution.json")
    args = parser.parse_args()

    cases = _load_cases(args.small_error_set)
    baseline = _correctness_by_id(args.baseline_eval)
    full = _correctness_by_id(args.full_eval)

    rows = []
    for case in cases:
        task_id = int(case["task_id"])
        error_type = case.get("original_error_type", "unknown")
        baseline_correct = baseline.get(task_id)
        full_correct = full.get(task_id)
        module = MODULE_BY_ERROR_TYPE.get(error_type, "manual_analysis")
        if baseline_correct is None or full_correct is None:
            fix_status = "not_run"
            repair_reason = ""
            still_failed_reason = "No comparable live baseline/full evaluation was provided."
        elif not baseline_correct and full_correct:
            fix_status = "fixed"
            repair_reason = f"Likely addressed by {module}."
            still_failed_reason = ""
        elif baseline_correct and full_correct:
            fix_status = "already_correct"
            repair_reason = "Baseline already matched the label on this run."
            still_failed_reason = ""
        elif baseline_correct and not full_correct:
            fix_status = "regressed"
            repair_reason = ""
            still_failed_reason = f"Full system introduced or exposed a failure in {module}."
        else:
            fix_status = "still_failed"
            repair_reason = ""
            still_failed_reason = f"Original failure type remains in scope for {module}; inspect verifier_report and final_block."
        rows.append(
            {
                "task_id": task_id,
                "original_error_type": error_type,
                "baseline_correct": baseline_correct,
                "full_correct": full_correct,
                "fix_status": fix_status,
                "repair_reason": repair_reason,
                "still_failed_reason": still_failed_reason,
                "module": module,
                "note": case.get("note", ""),
            }
        )

    columns = [
        "task_id",
        "original_error_type",
        "baseline_correct",
        "full_correct",
        "fix_status",
        "module",
        "repair_reason",
        "still_failed_reason",
    ]
    markdown = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        markdown.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")

    output = {
        "small_error_set": args.small_error_set,
        "baseline_eval": args.baseline_eval,
        "full_eval": args.full_eval,
        "rows": rows,
        "markdown_table": "\n".join(markdown),
    }
    write_json(output, args.output)
    print(output["markdown_table"])
    print(args.output)


if __name__ == "__main__":
    main()

