from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_extensions.infiagentbench.io_utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize InfiAgentBench error cases from an improved eval report.")
    parser.add_argument("--eval-report", required=True)
    parser.add_argument("--output", default="runs/infiagent_improve/error_analysis.json")
    parser.add_argument("--max-cases-per-type", type=int, default=5)
    args = parser.parse_args()

    report = read_json(args.eval_report)
    cases_by_type = defaultdict(list)
    for result in report.get("results", []):
        if result.get("is_correct"):
            continue
        expected = set(result.get("label_answers", {}))
        predicted = set(result.get("predicted_answers", {}))
        if not predicted:
            error_type = "reformat_error"
        elif expected != predicted:
            error_type = "format_error"
        else:
            error_type = "unknown"
        if len(cases_by_type[error_type]) < args.max_cases_per_type:
            cases_by_type[error_type].append(
                {
                    "id": result.get("id"),
                    "level": result.get("level"),
                    "concepts": result.get("concepts", []),
                    "label_answers": result.get("label_answers", {}),
                    "predicted_answers": result.get("predicted_answers", {}),
                    "correctness": result.get("correctness", {}),
                }
            )

    output = {
        "source_eval_report": args.eval_report,
        "metrics": report.get("metrics", {}),
        "error_type_counts": report.get("error_type_counts", {}),
        "typical_cases": dict(cases_by_type),
    }
    write_json(output, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

