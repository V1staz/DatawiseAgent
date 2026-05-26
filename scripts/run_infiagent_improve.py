from __future__ import annotations

import argparse
import json
import traceback
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_extensions.infiagentbench.error_taxonomy import error_types_from_report
from agent_extensions.infiagentbench.final_block import extract_final_block, final_block_to_response
from agent_extensions.infiagentbench.io_utils import read_jsonl, write_json, write_jsonl
from agent_extensions.infiagentbench.prompt_builder import build_improved_prompt, build_original_prompt
from agent_extensions.infiagentbench.question_contract import parse_question_contract
from agent_extensions.infiagentbench.schema_profile import profile_csv
from agent_extensions.infiagentbench.skill_router import route_skills
from agent_extensions.infiagentbench.verifier import verify_execution


ABLATIONS = {
    "baseline": {"schema": False, "contract": False, "router": False, "verifier": False, "final": False},
    "schema": {"schema": True, "contract": False, "router": False, "verifier": False, "final": False},
    "contract": {"schema": True, "contract": True, "router": False, "verifier": False, "final": False},
    "skill": {"schema": True, "contract": True, "router": True, "verifier": False, "final": False},
    "router": {"schema": True, "contract": True, "router": True, "verifier": False, "final": False},
    "verifier": {"schema": True, "contract": True, "router": True, "verifier": True, "final": False},
    "final_formatter": {"schema": True, "contract": True, "router": True, "verifier": False, "final": True},
    "full": {"schema": True, "contract": True, "router": True, "verifier": True, "final": True},
}


def _task_dir(output_dir: Path, note: str, ablation: str, task_id: int) -> Path:
    return output_dir / note / ablation / "tasks" / str(task_id)


def _load_existing_responses(path: str | None) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    responses = {}
    for row in read_jsonl(path):
        if "id" in row:
            responses[int(row["id"])] = row
    return responses


def _load_existing_response_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _load_task_ids(path: str | None) -> set[int] | None:
    if not path:
        return None
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return set()
    if raw.startswith("["):
        data = json.loads(raw)
        return {int(item["task_id"] if isinstance(item, dict) else item) for item in data}
    ids: set[int] = set()
    for token in raw.replace(",", "\n").splitlines():
        token = token.strip()
        if token:
            ids.add(int(token))
    return ids


def _select_questions(
    questions: list[dict[str, Any]],
    task_id: int | None,
    task_ids: set[int] | None,
    limit: int | None,
    start_from_task_id: int | None = None,
) -> list[dict[str, Any]]:
    selected = questions
    if task_id is not None:
        selected = [question for question in selected if int(question["id"]) == task_id]
    if task_ids is not None:
        selected = [question for question in selected if int(question["id"]) in task_ids]
    if start_from_task_id is not None:
        selected = [question for question in selected if int(question["id"]) >= start_from_task_id]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _prepare_artifacts(
    question: dict[str, Any],
    csv_path: Path,
    task_dir: Path,
    enabled: dict[str, bool],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    task_dir.mkdir(parents=True, exist_ok=True)
    schema = profile_csv(csv_path) if enabled["schema"] or enabled["contract"] or enabled["router"] or enabled["verifier"] or enabled["final"] else None
    if schema is not None:
        write_json(schema, task_dir / "schema_profile.json")

    contract = parse_question_contract(question, schema) if enabled["contract"] or enabled["router"] or enabled["verifier"] or enabled["final"] else None
    if contract is not None:
        write_json(contract, task_dir / "question_contract.json")

    skill_plan = route_skills(contract) if contract is not None and (enabled["router"] or enabled["verifier"]) else None
    if skill_plan is not None:
        write_json(skill_plan, task_dir / "skill_plan.json")
    return schema, contract, skill_plan


def _build_prompt(
    question: dict[str, Any],
    schema: dict[str, Any] | None,
    contract: dict[str, Any] | None,
    skill_plan: dict[str, Any] | None,
    enabled: dict[str, bool],
) -> str:
    if not any(enabled.values()):
        return build_original_prompt(question)
    return build_improved_prompt(
        question,
        schema_profile=schema,
        question_contract=contract,
        skill_plan=skill_plan,
        include_schema=enabled["schema"],
        include_contract=enabled["contract"],
        include_skill_plan=enabled["router"],
        require_final_block=enabled["final"],
    )


def _call_datawise(user_id: uuid.UUID, question: dict[str, Any], csv_path: Path, prompt: str) -> str:
    evaluation_path = REPO_ROOT / "evaluation"
    if str(evaluation_path) not in sys.path:
        sys.path.insert(0, str(evaluation_path))
    try:
        from chat_test_asyncio import chat, create_session, upload_file
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import evaluation/chat_test_asyncio.py. Use --dry-run or --existing-responses-file without a live DatawiseAgent server."
        ) from exc

    session_id = uuid.UUID(create_session(user_id=str(user_id), session_name=str(question["id"])))
    upload_file(str(user_id), str(session_id), file_path=str(csv_path))
    chat_response = chat(str(user_id), str(session_id), query=prompt)
    return chat_response["response_content"]


def _process_final_and_verify(
    *,
    raw_response: str,
    schema: dict[str, Any] | None,
    contract: dict[str, Any] | None,
    skill_plan: dict[str, Any] | None,
    enabled: dict[str, bool],
    task_dir: Path,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    final_block = extract_final_block(raw_response) if enabled["final"] or enabled["verifier"] else None
    final_validation = None
    response_for_eval = raw_response

    if enabled["final"]:
        response_for_eval, final_validation = final_block_to_response(final_block, contract)
        write_json(
            {"final_block": final_block, "formatted_response": response_for_eval, "validation": final_validation},
            task_dir / "final_block.json",
        )

    verifier_report = None
    if enabled["verifier"]:
        verifier_report = verify_execution(
            schema_profile=schema,
            question_contract=contract or {},
            skill_plan=skill_plan,
            execution_log=raw_response,
            final_block=final_block,
            require_final_block=enabled["final"],
        )
        write_json(verifier_report, task_dir / "verifier_report.json")
        if verifier_report.get("must_retry"):
            response_for_eval = "ERROR_NEED_RETRY"
    return response_for_eval, final_validation, verifier_report


def _classify_exception(exc: BaseException) -> str:
    text = "".join(traceback.format_exception_only(type(exc), exc)).lower()
    model_api_markers = [
        "arrearage",
        "access denied",
        "api key",
        "apikey",
        "quota",
        "rate limit",
        "badrequesterror",
        "openai",
        "chat failed",
        "create_user failed",
        "connection refused",
        "connecterror",
        "timeout",
    ]
    return "model_api_error" if any(marker in text for marker in model_api_markers) else "execution_error"


def _task_status(
    *,
    response_for_eval: str,
    final_validation: dict[str, Any] | None,
    verifier_report: dict[str, Any] | None,
    enabled: dict[str, bool],
) -> str:
    if verifier_report and verifier_report.get("must_retry"):
        return "verifier_blocked"
    if enabled["final"] and final_validation and not final_validation.get("passed"):
        errors = final_validation.get("errors") or []
        if any("missing" in str(error) or "final_block_missing" in str(error) for error in errors):
            return "reformat_missing"
        return "format_error"
    if response_for_eval == "ERROR_NEED_RETRY":
        return "reformat_missing"
    return "success"


def _failure_row(
    *,
    question: dict[str, Any],
    csv_path: Path,
    task_dir: Path,
    ablation: str,
    prompt: str,
    start: float,
    exc: BaseException,
) -> dict[str, Any]:
    task_id = int(question["id"])
    error_type = _classify_exception(exc)
    traceback_text = traceback.format_exc()
    write_json(
        {
            "task_id": task_id,
            "error_type": error_type,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback_text,
        },
        task_dir / "error.json",
    )
    runtime = time.perf_counter() - start
    return {
        "id": task_id,
        "input_text": f"Question: {question.get('question', '')}\n{question.get('constraints', '')}\n",
        "concepts": question.get("concepts", []),
        "level": question.get("level"),
        "file_path": str(csv_path),
        "format": question.get("format", ""),
        "response": "ERROR_NEED_RETRY",
        "raw_response": "",
        "ablation": ablation,
        "artifact_dir": str(task_dir),
        "runtime_seconds": round(runtime, 4),
        "log_length": 0,
        "prompt_length": len(prompt),
        "token_estimate": int(len(prompt) / 4),
        "final_block_validation": None,
        "verifier_report": None,
        "error_type": error_type,
        "error_types": [error_type],
        "task_status": error_type,
        "error_message": str(exc),
        "traceback": traceback_text,
    }


def run(args: argparse.Namespace) -> Path:
    enabled = ABLATIONS[args.ablation]
    questions = read_jsonl(args.questions_file)
    task_ids = _load_task_ids(args.task_ids_file)
    selected_questions = _select_questions(
        questions,
        args.task_id,
        task_ids,
        args.limit,
        start_from_task_id=args.start_from_task_id,
    )
    output_dir = Path(args.output_dir)
    run_dir = output_dir / args.note / args.ablation
    predictions_path = run_dir / "predictions" / f"results_{args.note}_{args.ablation}.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    existing_responses = _load_existing_responses(args.existing_responses_file)
    existing_prediction_rows = _load_existing_response_rows(predictions_path) if (args.resume or args.skip_existing) else []
    existing_prediction_ids = {int(row["id"]) for row in existing_prediction_rows if "id" in row}
    rows: list[dict[str, Any]] = [] if args.write_incremental else list(existing_prediction_rows)
    stopped_on_error = False
    processed_count = 0

    user_id = None
    create_user_error: BaseException | None = None
    if not args.dry_run and not existing_responses:
        evaluation_path = REPO_ROOT / "evaluation"
        if str(evaluation_path) not in sys.path:
            sys.path.insert(0, str(evaluation_path))
        from chat_test_asyncio import create_user

        try:
            user_id = create_user(username=f"InfiAgentBench-improve-{args.note}-{args.ablation}")
        except Exception as exc:
            create_user_error = exc

    for question in selected_questions:
        task_id = int(question["id"])
        if task_id in existing_prediction_ids and (args.resume or args.skip_existing):
            continue
        if args.max_tasks is not None and processed_count >= args.max_tasks:
            break

        csv_path = Path(args.tables_dir) / question["file_name"]
        task_dir = _task_dir(output_dir, args.note, args.ablation, task_id)
        start = time.perf_counter()
        prompt = ""

        try:
            schema, contract, skill_plan = _prepare_artifacts(question, csv_path, task_dir, enabled)
            prompt = _build_prompt(question, schema, contract, skill_plan, enabled)
            write_json({"prompt": prompt}, task_dir / "agent_prompt.json")

            if args.dry_run:
                raw_response = ""
                response_for_eval, final_validation, verifier_report = _process_final_and_verify(
                    raw_response=raw_response,
                    schema=schema,
                    contract=contract,
                    skill_plan=skill_plan,
                    enabled=enabled,
                    task_dir=task_dir,
                )
                if not (enabled["final"] or enabled["verifier"]):
                    response_for_eval = "DRY_RUN_NO_RESPONSE"
            elif task_id in existing_responses:
                source_row = existing_responses[task_id]
                raw_response = str(source_row.get("raw_response") or source_row.get("response") or "")
                response_for_eval, final_validation, verifier_report = _process_final_and_verify(
                    raw_response=raw_response,
                    schema=schema,
                    contract=contract,
                    skill_plan=skill_plan,
                    enabled=enabled,
                    task_dir=task_dir,
                )
            else:
                if create_user_error is not None:
                    raise RuntimeError("create_user failed before task execution") from create_user_error
                raw_response = _call_datawise(user_id, question, csv_path, prompt)  # type: ignore[arg-type]
                response_for_eval, final_validation, verifier_report = _process_final_and_verify(
                    raw_response=raw_response,
                    schema=schema,
                    contract=contract,
                    skill_plan=skill_plan,
                    enabled=enabled,
                    task_dir=task_dir,
                )

            runtime = time.perf_counter() - start
            task_status = _task_status(
                response_for_eval=response_for_eval,
                final_validation=final_validation,
                verifier_report=verifier_report,
                enabled=enabled,
            )
            row = {
                "id": task_id,
                "input_text": f"Question: {question.get('question', '')}\n{question.get('constraints', '')}\n",
                "concepts": question.get("concepts", []),
                "level": question.get("level"),
                "file_path": str(csv_path),
                "format": question.get("format", ""),
                "response": response_for_eval,
                "raw_response": raw_response,
                "ablation": args.ablation,
                "artifact_dir": str(task_dir),
                "runtime_seconds": round(runtime, 4),
                "log_length": len(raw_response),
                "prompt_length": len(prompt),
                "token_estimate": int((len(prompt) + len(raw_response)) / 4),
                "final_block_validation": final_validation,
                "verifier_report": verifier_report,
                "error_types": error_types_from_report(verifier_report) if verifier_report else [],
                "task_status": task_status,
            }
        except Exception as exc:
            row = _failure_row(
                question=question,
                csv_path=csv_path,
                task_dir=task_dir,
                ablation=args.ablation,
                prompt=prompt,
                start=start,
                exc=exc,
            )
            stopped_on_error = bool(args.stop_on_error)

        rows.append(row)
        processed_count += 1
        if args.write_incremental:
            write_jsonl([row], predictions_path, mode="a")
        if stopped_on_error:
            break

    if not args.write_incremental:
        write_jsonl(rows, predictions_path, mode="w")

    write_json(
        {
            "note": args.note,
            "ablation": args.ablation,
            "enabled_modules": enabled,
            "questions_file": args.questions_file,
            "tables_dir": args.tables_dir,
            "predictions_path": str(predictions_path),
            "num_questions": len(rows),
            "existing_prediction_count": len(existing_prediction_rows),
            "total_prediction_rows_after_run": (
                len(existing_prediction_rows) + processed_count if args.write_incremental else len(rows)
            ),
            "processed_this_run": processed_count,
            "resume": args.resume,
            "skip_existing": args.skip_existing,
            "start_from_task_id": args.start_from_task_id,
            "max_tasks": args.max_tasks,
            "stopped_on_error": stopped_on_error,
        },
        run_dir / "run_metadata.json",
    )
    return predictions_path


def define_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pluggable InfiAgentBench improvement pipeline.")
    parser.add_argument("--questions-file", default="evaluation/InfiAgentBench/data/da-dev-questions.jsonl")
    parser.add_argument("--tables-dir", default="evaluation/InfiAgentBench/data/da-dev-tables")
    parser.add_argument("--output-dir", default="runs/infiagent_improve")
    parser.add_argument("--note", default="dev")
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="full")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--task-ids-file", help="JSON list or text file containing task ids to run.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tasks", type=int, help="Maximum number of new, non-skipped tasks to run in this invocation.")
    parser.add_argument("--start-from-task-id", type=int, help="Only run tasks with id greater than or equal to this id.")
    parser.add_argument("--dry-run", action="store_true", help="Generate artifacts and prompts without calling DatawiseAgent.")
    parser.add_argument(
        "--existing-responses-file",
        help="Existing JSONL responses to post-process and verify instead of calling DatawiseAgent.",
    )
    parser.add_argument("--write-incremental", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing predictions JSONL and skip completed task ids.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip task ids already present in the target predictions JSONL.")
    error_group = parser.add_mutually_exclusive_group()
    error_group.add_argument("--stop-on-error", dest="stop_on_error", action="store_true", help="Write the failed task row, then stop.")
    error_group.add_argument(
        "--continue-on-error",
        dest="stop_on_error",
        action="store_false",
        help="Write failed task rows and continue with later tasks.",
    )
    parser.set_defaults(stop_on_error=False)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(define_args())
    print(output)
