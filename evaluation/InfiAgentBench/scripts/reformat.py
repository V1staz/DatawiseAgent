"""
Script: reformat.py
-------------------
Working directory: `./evaluation` (relative to the root directory of the project).

Usage:
    ```bash
    python ./InfiAgentBench/scripts/reformat.py \
        --model <MODEL_NAME> \
        --responses_file_path ./data/<FILE_PATH> \
        --output_file_path ./InfiAgentBench/reformat/gpt_4o_mini/<FILE_PATH>
    ```

Example:
    ```bash
    python ./InfiAgentBench/scripts/reformat.py \
        --model "gpt-4o-mini" \
        --responses_file_path ./data/results_datawise-test.jsonl \
        --output_file_path ./experimental_results/InfiAgent-Bench/reformat/results_reformat_datawise-test.jsonl
    ```
"""

import logging
import time
import json
import argparse
import traceback
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import read_jsonl, write_jsonl
from datawiseagent.harness.finalizer import (
    extract_final_answer_block,
    render_reformat_response,
)


def define_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url_file", type=str, default="./InfiAgentBench/scripts/url.txt"
    )
    parser.add_argument(
        "--api_key_file", type=str, default="./InfiAgentBench/scripts/api_key.txt"
    )
    parser.add_argument(
        "--questions_file_path",
        type=str,
        default="./InfiAgentBench/data/da-dev-questions.jsonl",
    )
    parser.add_argument(
        "--responses_file_path",
        type=str,
        default="./data/results_datawise_gpt-4o-mini-2024-07-18.jsonl",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default="./experimental_results/InfiAgent-Bench/reformat/results_reformat_datawise-test.jsonl",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_resp", type=int, default=2048)
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum LLM fallback retry attempts per row.",
    )
    parser.add_argument(
        "--retry_sleep",
        type=float,
        default=10.0,
        help="Seconds to sleep between LLM fallback retry attempts.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=120.0,
        help="OpenAI-compatible client timeout in seconds for LLM fallback.",
    )
    parser.add_argument(
        "--final_block_only",
        action="store_true",
        help="Only reformat from machine-readable FinalAnswerBlock; do not call an LLM fallback.",
    )

    args = parser.parse_args()
    return args


def call(messages, args):
    from openai import OpenAI

    max_retries = max(1, int(args.max_retries))
    data = {
        "max_tokens": args.max_resp,
        "model": args.model,
        "temperature": 0,
        "messages": messages,
    }
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            client = OpenAI(
                api_key=args.api_key,
                base_url=args.url,
                timeout=args.request_timeout,
            )
            logging.info("Calling reformat model: %s", args.model)
            result = client.chat.completions.create(**data)

            return result.choices[0].message.content
        except Exception as e:
            last_error = e
            logging.error("Reformat call failed on attempt %s/%s", attempt, max_retries)
            logging.error(data)
            logging.error(traceback.format_exc())
            if attempt < max_retries:
                time.sleep(max(0.0, float(args.retry_sleep)))
    raise RuntimeError(f"LLM reformat failed after {max_retries} attempts: {last_error}")


demons = r"""\Format{{
@shapiro_wilk_statistic[test_statistic]
@shapiro_wilk_p_value[p_value]
where "test_statistic" is a number between 0 and 1 representing the Shapiro-Wilk test statistic. Rounding off the answer to two decimal places.
where "p_value" is a number between 0 and 1 representing the p-value from the Shapiro-Wilk test. Rounding off the answer to four decimal places.
}}
\Answer{{
@shapiro_wilk_statistic[0.56]
@shapiro_wilk_p_value[0.0002]   
}}

\Format{{
@total_votes_outliers_num[outlier_num]
where "outlier_num" is an integer representing the number of values considered outliers in the 'total_votes' column.
}}
\Answer{{
@total_votes_outliers[10]   
}}
"""

reformat_template = """You should strictly follow the output requirements in the Format part. Here're some examples: 
{demons}. 
Your answer should contain all the \"@answer_name[answer]\" in the order mentioned, each \"answer\" should be in the range of value as required. 
The format requirements of this question is:
{format}. Please give your answer:"""

if __name__ == "__main__":
    args = define_arguments()
    args.url = None
    args.api_key = None
    if not args.final_block_only:
        args.url = open(args.url_file).read().strip()
        args.api_key = open(args.api_key_file).read().strip()

    questions = read_jsonl(args.questions_file_path)
    question_by_id = {question["id"]: question for question in questions}

    responses = []
    with open(args.responses_file_path, encoding="utf-8", mode="r") as f:
        for line in f:
            item = json.loads(line)
            responses.append(item)

    reformatted_responses_ids: set[int | str] = set()
    output_file_path = Path(args.output_file_path)
    if not os.path.exists(output_file_path):
        try:
            # create an empty file
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            output_file_path.touch()

            print(f"Created empty file at: {output_file_path}")
        except Exception as e:
            print(f"Failed to create file: {e}")
    else:
        print(f"File already exists: {output_file_path}")

    with open(args.output_file_path, encoding="utf-8", mode="r") as f:
        for line in f:
            item = json.loads(line)
            if "id" in item:
                reformatted_responses_ids.add(item["id"])

    for res_id, response in enumerate(responses):
        if response["id"] in reformatted_responses_ids:
            continue

        question = question_by_id.get(response["id"], {})
        question_description = question.get("question", "")
        format = question.get("format", response.get("format", ""))

        try:
            final_block = extract_final_answer_block(response)
            if final_block is not None:
                reformatted_response = render_reformat_response(final_block, format)
                response["reformat_response"] = reformatted_response
                response["final_block_missing"] = False
                print(f"{res_id}: {reformatted_response}")
            elif args.final_block_only:
                response["reformat_response"] = ""
                response["final_block_missing"] = True
                print(f"{res_id}: FinalAnswerBlock missing; skipped LLM fallback")
            else:
                messages = [{"role": "user", "content": question_description}]
                messages.append({"role": "assistant", "content": response.get("response", "")})
                messages.append(
                    {
                        "role": "user",
                        "content": reformat_template.format(demons=demons, format=format),
                    }
                )
                reformatted_response = call(messages, args)
                response["reformat_response"] = reformatted_response
                response["final_block_missing"] = True
                print(f"{res_id}: {reformatted_response}")

            write_jsonl([response], args.output_file_path, mode="a+")
        except Exception as e:
            traceback.print_exc()
            response["reformat_response"] = ""
            response["final_block_missing"] = True
            response["reformat_error"] = f"{type(e).__name__}: {e}"
            write_jsonl([response], args.output_file_path, mode="a+")
