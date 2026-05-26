from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_extensions.infiagentbench.final_block import extract_final_block, final_block_to_response
from agent_extensions.infiagentbench.io_utils import read_jsonl, write_jsonl
from agent_extensions.infiagentbench.question_contract import parse_question_contract


def _question_map(path: str) -> dict[int, dict[str, Any]]:
    return {int(row["id"]): row for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reformat InfiAgentBench responses only from FINAL_BLOCK JSON.")
    parser.add_argument("--questions-file", default="evaluation/InfiAgentBench/data/da-dev-questions.jsonl")
    parser.add_argument("--responses-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    questions = _question_map(args.questions_file)
    outputs = []
    for response in read_jsonl(args.responses_file):
        task_id = int(response["id"])
        question = questions.get(task_id, {})
        contract = parse_question_contract(question, None) if question else None
        block = response.get("final_block") or extract_final_block(response.get("raw_response") or response.get("response") or "")
        formatted, validation = final_block_to_response(block, contract)
        row = dict(response)
        row["response"] = formatted
        row["final_block"] = block
        row["final_block_validation"] = validation
        row["is_solved"] = validation["passed"]
        outputs.append(row)
    write_jsonl(outputs, args.output_file)


if __name__ == "__main__":
    main()

