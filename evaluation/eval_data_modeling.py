"""
Run DataModeling Evaluation
===========================

This script evaluates models on the DataModeling benchmark by creating a user, 
reading dataset samples, uploading related files, and generating responses 
with chat sessions. The model is expected to perform data modeling, training 
and prediction, and output the final submission file in CSV format.

Usage
-----
First, ensure the server is running (default: http://localhost:8000).

Basic usage with defaults:
    python eval_data_modeling.py

Specify custom parameters:
    python eval_data_modeling.py \
        --user_name "DataModeling-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop" \
        --result_path "./results/DataModeling/gpt-4o-mini/"

Arguments
---------
--user_name : str
    Name of the user as a readable identifier 
    (default: DataModeling-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop)

--result_path : str
    Directory where results will be stored 
    (default: ./results/DataModeling/gpt-4o-mini/)
"""

import argparse
import glob
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

from chat_test_asyncio import create_session, create_user, chat, upload_file, stop_session

def main(
    user_id: uuid.UUID,
    user_name: Optional[str] = None,
    result_path: str | Path = Path(f"./results/DataModeling/gpt-4o-mini/"),
    model: str = "gpt-4o-mini",
    start_index: int = 0,
    end_index: int | None = None,
    limit: int | None = None,
    skip_ids: set[int] | None = None,
):

    Path(result_path).mkdir(parents=True, exist_ok=True)
    results_path = Path(result_path) / "results.jsonl"

    log_path = Path(result_path) / "log.txt"

    if not os.path.exists(results_path):
        Path(results_path).touch()

    if not os.path.exists(log_path):
        Path(log_path).touch()

    id2results = {}
    with open(results_path, encoding="utf-8", mode="r") as f:
        for line in f:
            item = json.loads(line)
            id2results[item["id"]] = item

    samples = []
    with open("./DataModeling/data.jsonl", encoding="utf-8", mode="r") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"samples count: {len(samples)}")

    print(f"id2results count: {len(id2results)}")

    instruction = "I have a data modeling task. You must give me the predicted results as a CSV file as detailed in the following content. You should try your best to predict the answer. I provide you with three files. One is training data, one is test data. There is also a sample file for submission."

    completed_this_run = 0
    if skip_ids is None:
        skip_ids = {3}

    for i, item in enumerate(samples):

        if i < start_index:
            continue

        if end_index is not None and i >= end_index:
            continue

        if limit is not None and completed_this_run >= limit:
            break

        if i in id2results:
            continue

        if i in skip_ids:
            continue

        # if i == 14 or i == 17 or i == 22 or i == 33 or i == 48:
        #     print(item)
        #     continue

        name = item["name"]
        with open(f"./DataModeling/data/task/{name}.txt", "r") as f:
            description = f.read()
        text = f"All three data files can be found in the folder `./input/`. After the data modeling, please give me the prediction results for the test file. You must run the code for the predicted results and save the answer as a csv file following the format of the given sample file for submission. The final submission file should be saved in the path `./working/submission.csv`."

        text = f"""All three data files can be found in the folder `./input/`. **You should use sklearn or pytorch to complete the task.** Any training scripts, models, and experiment log should be saved in `./input/`. After data modeling, provide the prediction results for the test file in the format specified by the sample submission file. Save the final submission to `./input/final_submission.csv`.
        """

        all_context = instruction + "\n" + description + "\n" + text

        session_id: uuid.UUID | None = None
        try:
            session_id = uuid.UUID(
                create_session(
                    user_id=user_id, session_name=name, tool_mode="datamodeling"
                )
            )
            print(f"create session:{session_id} for case {name} with id {name}")

            data_split_path = f"./DataModeling/data/data_resplit/{name}"
            for root, dirs, files in os.walk(data_split_path):
                for file in files:
                    if file.endswith(".csv"):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, data_split_path)

                        upload_file(
                            str(user_id),
                            str(session_id),
                            file_path=full_path,
                            filename_to_save=relative_path,
                        )

            start = time.time()

            chat_response = None
            time_consumed = None

            try:
                chat_response = chat(
                    str(user_id),
                    str(session_id),
                    query=all_context,
                    work_mode="jupyter+script",
                )
            except httpx.TimeoutException:
                print("chat() 调用超时，自动设置 chat_response")
                chat_response = {
                    "user_content": all_context,
                    "response_content": "TIMEOUT=3600",
                }
                time_consumed = 3600

            end = time.time()

            if not time_consumed:
                time_consumed = end - start

            user_content = chat_response["user_content"]
            response_content = chat_response["response_content"]

            input_dir = f"../log/workspace/users/{user_id}/{session_id}/input/"
            work_dir = f"../log/workspace/users/{user_id}/{session_id}/working/"

            submission_files = []
            submission_files.extend(
                glob.glob(os.path.join(input_dir, "*submission*.csv"))
            )
            submission_files.extend(
                glob.glob(os.path.join(work_dir, "*submission*.csv"))
            )

            if submission_files:
                submission_file_path = f"../log/workspace/users/{user_id}/{session_id}/input/final_submission.csv"

                if not os.path.exists(submission_file_path):
                    submission_files.sort(
                        key=lambda x: os.path.getmtime(x), reverse=True
                    )
                    submission_file_path = submission_files[0]

            else:
                submission_file_path = f"../log/workspace/users/{user_id}/{session_id}/input/final_submission.csv"

            iteration_result = {
                "id": i,
                "name": name,
                "input_text": all_context,
                "response": response_content,
                "session_id": str(session_id),
                "user_id": str(user_id),
                "time": time_consumed,
                "cost": None,
                "model": model,
                "input": None,
                "output": None,
                "result_path": submission_file_path,
            }
            print(iteration_result)
            print(f"submission_file_path:{submission_file_path}")

            with open(results_path, encoding="utf-8", mode="a+") as f:
                f.write(json.dumps(iteration_result, ensure_ascii=False) + "\n")
                f.flush()
            completed_this_run += 1

        except Exception as e:
            import traceback

            with open(log_path, encoding="utf-8", mode="a+") as f:
                f.write(traceback.format_exc())
        finally:
            if session_id is not None:
                try:
                    stop_session(str(user_id), str(session_id))
                except Exception:
                    import traceback

                    with open(log_path, encoding="utf-8", mode="a+") as f:
                        f.write(traceback.format_exc())

        # create session

        # upload files

        start = time.time()
        cost = 0
        error = ""

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run DSBench evaluation.")

    parser.add_argument(
        "--user_name",
        type=str,
        default="DSBench-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop",
        help="Name of the user (default: DSBench-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop)",
    )

    parser.add_argument(
        "--result_path",
        type=str,
        default="./results/DSBench/gpt-4o-mini/",
        help="Path to store results (default: ./results/DSBench/gpt-4o-mini/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name to record in results.jsonl.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Inclusive task index to start from.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive task index to stop at.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new tasks to run this invocation.",
    )
    parser.add_argument(
        "--skip_ids",
        type=str,
        default="3",
        help="Comma-separated task indices to skip. Default keeps legacy skip of task 3.",
    )

    args = parser.parse_args()

    user_id = create_user(username=args.user_name)
    print(f"Created user_id: {user_id}")

    import uuid

    # user_name = "DataModeling-qwen2.5-temperature=0-args=(7,6,8)-for-loop"
    # user_id = uuid.UUID("383bf999-5537-4267-a65a-a84012b32101")

    main(
        user_id=user_id,
        user_name=args.user_name,
        result_path=args.result_path,
        model=args.model,
        start_index=args.start,
        end_index=args.end,
        limit=args.limit,
        skip_ids={int(x) for x in args.skip_ids.split(",") if x.strip()},
        # "./results/DataModeling/gpt-4o-test/",
        # result_path="./results/DataModeling/qwen25-72B/",
        # model="Qwen/Qwen2.5-72B-Instruct",
    )
