"""
One script to evaluate Datawise Agent on the InfiAgentBench.

Usage:
    First of all, start the server at http://localhost:8000

    Example:
        python ./eval_infiagent_bench.py --note "temperature-0_2-args-7-6-8"

Arguments:
    --note      Readable identifier for the run (e.g., "temperature-0_2-args-7-6-8")
"""

import json
import os
import uuid
import argparse
import traceback
from pathlib import Path
from chat_test_asyncio import create_session, create_user, chat, upload_file, stop_session
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_dicts_from_file(file_name):
    """
    Read a file with each line containing a JSON string representing a dictionary,
    and return a list of dictionaries.

    :param file_name: Name of the file to read from.
    :return: List of dictionaries.
    """
    dict_list = []
    with open(file_name, "r") as file:
        for line in file:
            # Convert the JSON string back to a dictionary.
            dictionary = json.loads(line.rstrip("\n"))
            dict_list.append(dictionary)
    return dict_list


def read_questions(file_path):
    print(file_path)
    with open(file_path) as f:
        questions = json.load(f)

    return questions


def extract_data_from_folder(folder_path):

    print(f"folder_path {folder_path}")
    extracted_data = {}
    # Traverse the files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".questions"):  # You can filter files based on their type
            file_path = os.path.join(folder_path, file_name)
            file_data = read_questions(file_path)
            file_name_without_extension = os.path.splitext(file_name)[0]
            extracted_data[file_name_without_extension] = file_data

    return extracted_data


def _get_script_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", help="note for running", required=False, type=str)
    parser.add_argument(
        "--level",
        help="optional difficulty filter: easy / medium / hard",
        required=False,
        type=str,
        choices=["easy", "medium", "hard"],
    )
    parser.add_argument(
        "--num_workers",
        help="number of workers for concurrent mode",
        required=False,
        type=int,
        default=1,
    )
    parser.add_argument(
        "--concurrent",
        help="run questions concurrently",
        action="store_true",
    )
    return parser.parse_args()


def process_question(user_id: uuid.UUID, q, table_path):
    """
    处理单个问题并返回结果。

    :param user_id: 用户ID
    :param q: 问题字典
    :param table_path: 表格文件所在路径
    :return: 处理结果字典
    """
    q_id = q["id"]
    input_text = q["question"]
    concepts = q["concepts"]
    file_name = q["file_name"]
    constraints = q["constraints"]
    format_ = q["format"]

    phy_file_path = os.path.join(table_path, file_name)
    prompt = f"Question: {input_text}\n{constraints}\n"

    session_id = None
    try:
        session_id = uuid.UUID(create_session(user_id=str(user_id), session_name=q_id))
        print(f"Created session: {session_id} for question {q_id}")

        append_prompt = f"{file_name} has been uploaded."
        input_prompt = prompt + "\n" + append_prompt

        # 上传文件
        upload_file(str(user_id), str(session_id), file_path=phy_file_path)

        # 对话请求
        chat_response = chat(str(user_id), str(session_id), query=input_prompt)

        response_content = chat_response["response_content"]

        iteration_result = {
            "id": q_id,
            "input_text": prompt,
            "concepts": concepts,
            "file_path": phy_file_path,
            "response": response_content,
            "format": format_,
            "user_id": str(user_id),
            "session_id": str(session_id),
        }

        return iteration_result

    except Exception as e:
        print(f"Error processing question {q_id}: {e}")
        traceback.print_exc()
        return None  # 返回None以表示失败
    finally:
        if session_id is not None:
            try:
                stop_session(str(user_id), str(session_id))
            except Exception as stop_exc:
                print(f"Failed to stop session {session_id} for question {q_id}: {stop_exc}")


def _filter_questions_by_level(questions, level: str | None):
    if level is None:
        return questions
    return [q for q in questions if q.get("level") == level]


def _load_run_inputs(args: argparse.Namespace):
    questions = _filter_questions_by_level(
        read_dicts_from_file("./InfiAgentBench/data/da-dev-questions.jsonl"),
        getattr(args, "level", None),
    )
    table_path = "./InfiAgentBench/data/da-dev-tables"
    results_path = "./experimental_results/InfiAgent-Bench/results_{}.jsonl".format(
        getattr(args, "note", None)
    )
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).touch(exist_ok=True)
    existing_ids = {
        item["id"]
        for item in read_dicts_from_file(results_path)
        if "id" in item
    }
    pending_questions = [q for q in questions if q["id"] not in existing_ids]
    return pending_questions, table_path, results_path


def concurrent_main(user_id: uuid.UUID, args: argparse.Namespace, num_workers: int = 4):
    """
    并发处理问题并返回结果列表。

    :param user_id: 用户ID
    :param pending_questions: 待处理的问题列表
    :param table_path: 表格文件所在路径
    :param num_workers: 工作线程数
    :return: 结果生成器
    """
    import threading

    # 创建一个线程锁，用于安全地写入文件
    write_lock = threading.Lock()

    pending_questions, table_path, results_path = _load_run_inputs(args)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_q = {
            executor.submit(process_question, user_id, q, table_path): q
            for q in pending_questions
        }

        # 打开结果文件一次，按追加模式写入
        with open(results_path, "a+", encoding="utf-8") as f:
            for future in as_completed(future_to_q):
                q = future_to_q[future]
                try:
                    result = future.result()
                    if result:
                        # 安全地写入结果文件
                        with write_lock:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()  # 确保结果及时写入文件
                        print(f"Written result for question {result['id']}")
                    else:
                        print(f"Question {q['id']} processing returned no result.")
                except Exception as exc:
                    print(f"Question {q['id']} generated an exception: {exc}")


def main(user_id: uuid.UUID, args: argparse.Namespace):
    pending_questions, table_path, results_path = _load_run_inputs(args)

    for q in pending_questions:

        input_text = q["question"]
        concepts = q["concepts"]
        file_path = q["file_name"]
        constraints = q["constraints"]
        format = q["format"]

        # if q["level"] == "easy":
        #    continue

        phy_file_path = os.path.join(table_path, file_path)

        print(f"input_text: {input_text}")
        print(f"concepts: {concepts}")
        print(f"phy file_path: {phy_file_path}")

        prompt = f"Question: {input_text}\n{constraints}\n"

        session_id = None
        try:
            session_id = uuid.UUID(
                create_session(user_id=user_id, session_name=q["id"])
            )

            print(f"create session:{session_id} for question {q['id']}")

            append_prompt = f"{q['file_name']} has been uploaded."
            input_prompt = prompt + "\n" + append_prompt

            # upload files
            upload_file(str(user_id), str(session_id), file_path=phy_file_path)

            # query
            chat_response = chat(str(user_id), str(session_id), query=input_prompt)

            user_content = chat_response["user_content"]
            response_content = chat_response["response_content"]

            iteration_result = {
                "id": q["id"],
                "input_text": prompt,
                "concepts": concepts,
                "file_path": phy_file_path,
                "response": response_content,
                "format": format,
                "user_id": str(user_id),
                "session_id": str(session_id),
            }
            with open(results_path, encoding="utf-8", mode="a+") as f:
                f.write(json.dumps(iteration_result, ensure_ascii=False) + "\n")
                f.flush()

        except Exception as e:
            traceback.print_exc()
            print(f"Question {q['id']} failed: {type(e).__name__}: {e}")
        finally:
            if session_id is not None:
                try:
                    stop_session(str(user_id), str(session_id))
                except Exception as stop_exc:
                    print(
                        f"Failed to stop session {session_id} for question {q['id']}: {stop_exc}"
                    )


if __name__ == "__main__":
    # --note "datawise-test"
    args = _get_script_params()
    user_name = args.note or "InfiAgentBench-datawise-test"
    if args.level:
        user_name = f"{user_name}-{args.level}"
    user_id = create_user(username=user_name)
    if args.concurrent:
        concurrent_main(user_id, args, num_workers=args.num_workers)
    else:
        main(user_id, args)
