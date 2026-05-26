from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_extensions.infiagentbench.io_utils import read_json, write_json


METRIC_COLUMNS = [
    "ABQ",
    "PASQ",
    "UASQ",
    "hard_ABQ",
    "format_error_count",
    "column_mismatch_count",
    "preprocessing_order_error_count",
    "statistical_judgment_error_count",
    "reformat_missing_count",
    "average_token_estimate",
    "average_runtime_seconds",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare InfiAgentBench ablation evaluation JSON files.")
    parser.add_argument("--eval-files", nargs="+", required=True)
    parser.add_argument("--output", default="runs/infiagent_improve/ablations/ablation_compare.json")
    args = parser.parse_args()

    rows = []
    for path in args.eval_files:
        report = read_json(path)
        metrics = report.get("metrics", {})
        label = Path(path).stem.replace("_improve_eval", "")
        row = {"name": label}
        row.update({column: metrics.get(column) for column in METRIC_COLUMNS})
        rows.append(row)

    markdown_lines = [
        "| " + " | ".join(["name"] + METRIC_COLUMNS) + " |",
        "| " + " | ".join(["---"] * (len(METRIC_COLUMNS) + 1)) + " |",
    ]
    for row in rows:
        markdown_lines.append("| " + " | ".join(str(row.get(column, "")) for column in ["name"] + METRIC_COLUMNS) + " |")

    output = {"rows": rows, "markdown_table": "\n".join(markdown_lines)}
    write_json(output, args.output)
    print(output["markdown_table"])


if __name__ == "__main__":
    main()

