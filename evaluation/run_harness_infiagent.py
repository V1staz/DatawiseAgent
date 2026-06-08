"""Harness-enabled InfiAgentBench runner.

This script keeps the existing FastAPI/DatawiseAgent execution path intact and
adds an outer, feature-flagged harness layer for data cards, contracts, skill
protocols, final-block validation, memory, and hard-question branch search.

Example:
    python evaluation/run_harness_infiagent.py \
      --note qwen35b_a3b_full_harness \
      --level hard \
      --model-config configs/setting_model_qwen35b_a3b.yaml \
      --harness-mode full \
      --search-policy hard-only \
      --num-workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - yaml is available in the target env.
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from datawiseagent.harness import HarnessConfig, HarnessContext, HarnessController
from datawiseagent.harness.controller import build_runtime
from datawiseagent.harness.schemas import to_plain


def get_chat_client():
    # Import lazily so --help and static harness tooling work in lightweight
    # environments that have not installed httpx/openai runtime dependencies.
    from chat_test_asyncio import chat, create_session, create_user, stop_session, upload_file

    return chat, create_session, create_user, stop_session, upload_file

QUESTION_PATH = EVALUATION_ROOT / "InfiAgentBench" / "data" / "da-dev-questions.jsonl"
TABLE_ROOT = EVALUATION_ROOT / "InfiAgentBench" / "data" / "da-dev-tables"
DEFAULT_OUTPUT_ROOT = EVALUATION_ROOT / "experimental_results" / "InfiAgent-Bench"
TOOLKIT_PATH = PROJECT_ROOT / "datawiseagent" / "harness" / "toolkit.py"

ABLATION_CHOICES = [
    "baseline",
    "harness_light",
    "harness_full",
    "no_datacard",
    "no_contract",
    "no_skills",
    "no_oracle_or_verifier",
    "no_finalizer",
    "no_memory",
    "no_search",
]

MODEL_API_ERROR_MARKERS = (
    "api key",
    "apikey",
    "authentication",
    "unauthorized",
    "invalid token",
    "无效的令牌",
    "insufficient",
    "arrearage",
    "quota",
    "rate limit",
    "status=401",
    "status=429",
    "model_api_error",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, item: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InfiAgentBench with harness controls.")
    parser.add_argument("--note", type=str, default="harness")
    parser.add_argument("--level", type=str, choices=["easy", "medium", "hard", "full"], default="full")
    parser.add_argument("--ablation", type=str, choices=ABLATION_CHOICES, default=None)
    parser.add_argument("--model-config", "--model_config", dest="model_config", type=str, default=None)
    parser.add_argument("--model-name", "--model_name", dest="model_name", type=str, default=None)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=None)
    parser.add_argument(
        "--harness-mode",
        "--harness_mode",
        dest="harness_mode",
        type=str,
        choices=["baseline", "datacard", "contract", "skills", "verify", "memory", "search", "full"],
        default="full",
    )
    parser.add_argument(
        "--search-policy",
        "--search_policy",
        dest="search_policy",
        type=str,
        choices=["off", "hard-only", "all", "validator-fail"],
        default="hard-only",
    )
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test question limit.")
    parser.add_argument("--max-tasks", "--max_tasks", dest="max_tasks", type=int, default=None)
    parser.add_argument("--start-from-task-id", "--start_from_task_id", dest="start_from_task_id", type=int, default=None)
    parser.add_argument("--skip-existing", "--skip_existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", help="Alias for the default skip-existing behavior.")
    parser.add_argument("--continue-on-error", "--continue_on_error", dest="continue_on_error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", "--stop_on_error", dest="stop_on_error", action="store_true", default=False)
    parser.add_argument("--question-id", "--question_id", dest="question_id", type=str, default=None)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--artifact-dir", "--artifact_dir", dest="artifact_dir", type=str, default=None)
    parser.add_argument("--memory-path", "--memory_path", dest="memory_path", type=str, default=None)
    parser.add_argument("--rulebook-path", "--rulebook_path", dest="rulebook_path", type=str, default=None)
    parser.add_argument(
        "--request-timeout",
        "--request_timeout",
        dest="request_timeout",
        type=float,
        default=900.0,
        help="Per agent chat request timeout in seconds.",
    )
    parser.add_argument("--disable-semantic-oracles", action="store_true", help="Disable contract-derived semantic oracle planning/scoring.")
    parser.add_argument("--disable-memory-retrieval", action="store_true", help="Keep writing memory but do not inject retrieved memories into prompts.")
    parser.add_argument("--disable-memory-recording", action="store_true", help="Keep retrieval available but do not append new memory records.")
    parser.add_argument(
        "--disable-capability",
        dest="disabled_capabilities",
        action="append",
        default=[],
        choices=["datacard", "contract", "skills", "verify", "memory", "search", "finalizer"],
        help="Disable one harness capability for ablation. Can be repeated.",
    )
    parser.add_argument("--tool-mode", "--tool_mode", dest="tool_mode", type=str, default="default")
    parser.add_argument("--work-mode", "--work_mode", dest="work_mode", type=str, default="jupyter")
    args = parser.parse_args()
    apply_ablation(args)
    return args


def apply_ablation(args: argparse.Namespace) -> None:
    if not args.ablation:
        return
    disabled = set(args.disabled_capabilities or [])
    if args.ablation == "baseline":
        args.harness_mode = "baseline"
        args.search_policy = "off"
    elif args.ablation == "harness_light":
        args.harness_mode = "skills"
        args.search_policy = "off"
        args.disable_semantic_oracles = True
        args.disable_memory_retrieval = True
        args.disable_memory_recording = True
    else:
        args.harness_mode = "full"
    if args.ablation == "no_datacard":
        disabled.add("datacard")
    elif args.ablation == "no_contract":
        disabled.add("contract")
    elif args.ablation == "no_skills":
        disabled.add("skills")
    elif args.ablation == "no_oracle_or_verifier":
        disabled.add("verify")
        args.disable_semantic_oracles = True
    elif args.ablation == "no_finalizer":
        disabled.add("finalizer")
    elif args.ablation == "no_memory":
        disabled.add("memory")
        args.disable_memory_retrieval = True
        args.disable_memory_recording = True
    elif args.ablation == "no_search":
        disabled.add("search")
        args.search_policy = "off"
    args.disabled_capabilities = sorted(disabled)


def load_model_config(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    if yaml is None:
        raise RuntimeError("PyYAML is required to load --model-config")
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return payload.get("llm", payload)


def apply_llm_overrides(llm_config: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any] | None:
    if llm_config is None and not any([args.model_name, args.temperature is not None, args.max_tokens is not None]):
        return None
    merged = dict(llm_config or {})
    merged.setdefault("llm_type", "openai-chat")
    if args.model_name:
        merged["model"] = args.model_name
    if args.temperature is not None:
        merged["temperature"] = args.temperature
    if args.max_tokens is not None:
        merged["max_tokens"] = args.max_tokens
    return merged


def infer_provider(args: argparse.Namespace) -> str:
    if args.provider:
        return args.provider
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    url_file = EVALUATION_ROOT / "InfiAgentBench" / "scripts" / "url.txt"
    if not base_url and url_file.exists():
        base_url = url_file.read_text(encoding="utf-8").strip()
    lowered = base_url.lower()
    if "dashscope" in lowered or "aliyun" in lowered:
        return "dashscope"
    if "openai" in lowered:
        return "openai"
    if base_url:
        return "openai_compatible"
    return "unknown"


def experiment_metadata(args: argparse.Namespace, llm_config: dict[str, Any] | None) -> dict[str, Any]:
    llm_config = llm_config or {}
    return {
        "provider": infer_provider(args),
        "model_name": llm_config.get("model"),
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "temperature": llm_config.get("temperature"),
        "max_tokens": llm_config.get("max_tokens"),
        "ablation": args.ablation or args.harness_mode,
        "harness_mode": args.harness_mode,
        "search_policy": args.search_policy,
        "disabled_capabilities": list(args.disabled_capabilities or []),
        "memory_retrieval": not args.disable_memory_retrieval,
        "memory_recording": not args.disable_memory_recording,
        "semantic_oracles": not args.disable_semantic_oracles,
        "force_finalizer": args.ablation == "harness_light",
        "request_timeout": args.request_timeout,
    }


def filter_questions(
    questions: list[dict[str, Any]],
    level: str,
    question_id: str | None,
    limit: int | None,
    max_tasks: int | None = None,
    start_from_task_id: int | None = None,
) -> list[dict[str, Any]]:
    if level != "full":
        questions = [q for q in questions if q.get("level") == level]
    if question_id is not None:
        wanted = {part.strip() for part in question_id.split(",") if part.strip()}
        questions = [q for q in questions if str(q.get("id")) in wanted]
    if start_from_task_id is not None:
        questions = [q for q in questions if int(q.get("id", -1)) >= start_from_task_id]
    if limit is not None:
        questions = questions[:limit]
    if max_tasks is not None:
        questions = questions[:max_tasks]
    return questions


def classify_runtime_error(runtime: dict[str, Any] | None) -> str | None:
    runtime = runtime or {}
    errors = runtime.get("errors")
    if isinstance(errors, list):
        text = "\n".join(str(item) for item in errors if item)
    else:
        text = str(runtime.get("error") or "")
    if not text:
        return None
    lowered = text.lower()
    if any(marker in lowered for marker in MODEL_API_ERROR_MARKERS):
        return "model_api_error"
    return "execution_error"


def derive_task_status(result: dict[str, Any]) -> str:
    runtime_status = classify_runtime_error(result.get("runtime"))
    if runtime_status:
        return runtime_status
    validator_report = result.get("validator_report") or {}
    if validator_report and not validator_report.get("passed", True):
        return "verifier_blocked"
    if result.get("reformat_response") == "" and result.get("harness_mode") != "baseline":
        return "reformat_missing"
    return "success"


def original_prompt(question: dict[str, Any]) -> str:
    return (
        f"Question: {question.get('question', '')}\n"
        f"{question.get('constraints', '')}\n\n"
        f"{question.get('file_name', '')} has been uploaded."
    )


def run_agent_once(
    user_id: uuid.UUID,
    question: dict[str, Any],
    prompt: str,
    table_root: Path,
    llm_config: dict[str, Any] | None,
    tool_mode: str,
    work_mode: str,
    branch_name: str,
    upload_harness_tools: bool = False,
    request_timeout: float | None = 900.0,
) -> dict[str, Any]:
    start = time.time()
    session_id: uuid.UUID | None = None
    stop_session_func = None
    try:
        chat_func, create_session_func, _create_user_func, stop_session_func, upload_file_func = get_chat_client()
        session_label = f"q{question.get('id')}-{branch_name}"
        session_id = uuid.UUID(
            create_session_func(
                user_id=str(user_id),
                session_name=session_label,
                tool_mode=tool_mode,
                reset_llm_config=llm_config,
            )
        )
        file_path = table_root / str(question.get("file_name"))
        upload_file_func(str(user_id), str(session_id), file_path=str(file_path))
        if upload_harness_tools and TOOLKIT_PATH.exists():
            upload_file_func(
                str(user_id),
                str(session_id),
                file_path=str(TOOLKIT_PATH),
                filename_to_save="harness_tools.py",
            )
        response = chat_func(
            str(user_id),
            str(session_id),
            query=prompt,
            work_mode=work_mode,
            request_timeout=request_timeout,
        )
        return {
            "response": response.get("response_content", ""),
            "user_content": response.get("user_content", ""),
            "session_id": str(session_id),
            "runtime": build_runtime(start, session_id=str(session_id)),
        }
    except Exception as exc:
        return {
            "response": "",
            "user_content": "",
            "session_id": str(session_id) if session_id else None,
            "runtime": build_runtime(start, error=f"{type(exc).__name__}: {exc}", session_id=str(session_id) if session_id else None),
        }
    finally:
        if session_id is not None and stop_session_func is not None:
            try:
                stop_session_func(str(user_id), str(session_id))
            except Exception as stop_exc:
                print(f"Failed to stop session {session_id}: {stop_exc}")


def run_harness_branch(
    controller: HarnessController,
    context: HarnessContext,
    user_id: uuid.UUID,
    question: dict[str, Any],
    branch: dict[str, Any],
    args: argparse.Namespace,
    llm_config: dict[str, Any] | None,
) -> dict[str, Any]:
    agent_result = run_agent_once(
        user_id,
        question,
        branch["prompt"],
        TABLE_ROOT,
        llm_config,
        args.tool_mode,
        args.work_mode,
        branch["name"],
        upload_harness_tools=True,
        request_timeout=args.request_timeout,
    )
    return controller.postprocess(
        context,
        agent_result["response"],
        branch_index=branch["index"],
        branch_name=branch["name"],
        runtime=agent_result.get("runtime"),
    )


def process_question(
    user_id: uuid.UUID,
    question: dict[str, Any],
    args: argparse.Namespace,
    llm_config: dict[str, Any] | None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_root = args.artifact_dir or str(Path(args.output_dir) / "harness_artifacts" / args.note)
    memory_path = args.memory_path or str(Path(args.output_dir) / "harness_memory" / f"{args.note}.jsonl")
    controller = HarnessController(
        HarnessConfig(
            mode=args.harness_mode,
            search_policy=args.search_policy,
            artifact_root=artifact_root,
            memory_path=memory_path,
            rulebook_path=args.rulebook_path,
            semantic_oracles=not args.disable_semantic_oracles,
            memory_retrieval=not args.disable_memory_retrieval,
            memory_recording=not args.disable_memory_recording,
            disabled_capabilities=tuple(args.disabled_capabilities or ()),
            force_finalizer=args.ablation == "harness_light",
        )
    )

    if args.harness_mode == "baseline":
        context = None
        prompt = original_prompt(question)
        primary = run_agent_once(
            user_id,
            question,
            prompt,
            TABLE_ROOT,
            llm_config,
            args.tool_mode,
            args.work_mode,
            "baseline",
            request_timeout=args.request_timeout,
        )
        result = {
            "id": question.get("id"),
            "input_text": prompt,
            "concepts": question.get("concepts", []),
            "level": question.get("level"),
            "file_path": str(TABLE_ROOT / str(question.get("file_name"))),
            "response": primary["response"],
            "format": question.get("format"),
            "user_id": str(user_id),
            "session_id": primary.get("session_id"),
            "final_block": {},
            "contract": {},
            "data_card_path": None,
            "trace_path": None,
            "validator_report_path": None,
            "search_summary": {},
            "memory_hits": [],
            "rule_ids": [],
            "oracle_checks": [],
            "recovery_events": [],
            "runtime": primary.get("runtime", {}),
            "harness_mode": args.harness_mode,
            "ablation": args.ablation or args.harness_mode,
            "experiment_metadata": run_metadata or {},
        }
        result["task_status"] = derive_task_status(result)
        return result

    context = controller.prepare(question, TABLE_ROOT)
    branch_specs = controller.branch_prompts(context) if controller.should_search_before_primary(context) else [
        {"index": 0, "name": "deterministic_skill_branch", "prompt": context.prompt}
    ]

    branch_results: list[dict[str, Any]] = []
    for branch in branch_specs:
        branch_results.append(
            run_harness_branch(controller, context, user_id, question, branch, args, llm_config)
        )

    if (
        len(branch_results) == 1
        and args.search_policy == "validator-fail"
        and controller.should_search_after_primary(branch_results[0]["validator_report"])
    ):
        for branch in controller.branch_prompts(context)[1:]:
            branch_results.append(
                run_harness_branch(controller, context, user_id, question, branch, args, llm_config)
            )

    selected, search_summary = controller.select_branch(branch_results)
    runtime = merge_runtime([item.get("runtime", {}) for item in branch_results])
    artifacts = context.artifacts
    result = {
        "id": question.get("id"),
        "input_text": context.prompt,
        "concepts": question.get("concepts", []),
        "level": question.get("level"),
        "file_path": context.table_path,
        "response": selected.get("response", ""),
        "reformat_response": selected.get("reformat_response", ""),
        "format": question.get("format"),
        "user_id": str(user_id),
        "session_id": selected.get("runtime", {}).get("session_id"),
        "final_block": selected.get("final_block", {}),
        "contract": to_plain(context.contract) if context.contract else {},
        "data_card_path": artifacts.data_card_path if artifacts else None,
        "trace_path": artifacts.trace_path if artifacts else None,
        "validator_report_path": selected.get("validator_report_path"),
        "semantic_report_path": selected.get("semantic_report_path"),
        "validator_report": selected.get("validator_report"),
        "semantic_report": selected.get("semantic_report"),
        "search_summary": search_summary,
        "memory_hits": context.memory_hits,
        "rule_ids": [rule.get("id") for rule in context.rules],
        "rules": context.rules,
        "oracle_checks": context.oracle_checks,
        "recovery_events": selected.get("recovery_events", []),
        "runtime": runtime,
        "branch_results": summarize_branch_results(branch_results),
        "harness_mode": args.harness_mode,
        "ablation": args.ablation or args.harness_mode,
        "experiment_metadata": run_metadata or {},
    }
    result["task_status"] = derive_task_status(result)
    return result


def error_result(
    question: dict[str, Any],
    args: argparse.Namespace,
    exc: BaseException,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = build_runtime(time.time(), error=f"{type(exc).__name__}: {exc}")
    result = {
        "id": question.get("id"),
        "input_text": original_prompt(question),
        "concepts": question.get("concepts", []),
        "level": question.get("level"),
        "file_path": str(TABLE_ROOT / str(question.get("file_name"))),
        "response": "",
        "reformat_response": "",
        "format": question.get("format"),
        "final_block": {},
        "contract": {},
        "data_card_path": None,
        "trace_path": None,
        "validator_report_path": None,
        "search_summary": {},
        "memory_hits": [],
        "rule_ids": [],
        "oracle_checks": [],
        "recovery_events": [],
        "runtime": runtime,
        "harness_mode": args.harness_mode,
        "ablation": args.ablation or args.harness_mode,
        "experiment_metadata": run_metadata or {},
    }
    result["task_status"] = derive_task_status(result)
    return result


def summarize_branch_results(branch_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for item in branch_results:
        summaries.append(
            {
                "branch_index": item.get("branch_index"),
                "branch_name": item.get("branch_name"),
                "branch_score": item.get("branch_score"),
                "validator_report": item.get("validator_report"),
                "validator_report_path": item.get("validator_report_path"),
                "semantic_report_path": item.get("semantic_report_path"),
                "semantic_report": item.get("semantic_report"),
                "final_block_path": item.get("final_block_path"),
                "runtime": item.get("runtime", {}),
            }
        )
    return summaries


def merge_runtime(runtimes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tokens": sum(int(rt.get("tokens", 0) or 0) for rt in runtimes),
        "llm_calls": sum(int(rt.get("llm_calls", 0) or 0) for rt in runtimes),
        "seconds": round(sum(float(rt.get("seconds", 0) or 0) for rt in runtimes), 3),
        "errors": [rt.get("error") for rt in runtimes if rt.get("error")],
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.note if args.level == "full" else f"{args.note}_{args.level}"
    results_path = output_dir / f"results_{suffix}.jsonl"

    questions = filter_questions(
        read_jsonl(QUESTION_PATH),
        args.level,
        args.question_id,
        args.limit,
        max_tasks=args.max_tasks,
        start_from_task_id=args.start_from_task_id,
    )
    existing_ids: set[Any] = set()
    if results_path.exists():
        for row in read_jsonl(results_path):
            if "id" in row:
                existing_ids.add(row["id"])
    else:
        results_path.touch()

    pending = [q for q in questions if not args.skip_existing or q.get("id") not in existing_ids]
    llm_config = apply_llm_overrides(load_model_config(args.model_config), args)
    run_metadata = experiment_metadata(args, llm_config)
    metadata_path = output_dir / f"run_metadata_{suffix}.json"
    metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    user_name = f"{args.note}-{args.harness_mode}"
    if args.level != "full":
        user_name += f"-{args.level}"
    _chat_func, _create_session_func, create_user_func, _stop_session_func, _upload_file_func = get_chat_client()
    user_id = uuid.UUID(create_user_func(username=user_name))

    print(
        f"Harness run: mode={args.harness_mode}, ablation={args.ablation or args.harness_mode}, "
        f"search={args.search_policy}, model={run_metadata.get('model_name')}, "
        f"questions={len(pending)}/{len(questions)}, output={results_path}"
    )
    write_lock = threading.Lock()

    if args.num_workers <= 1:
        for question in pending:
            print(f"Processing question {question.get('id')} ({question.get('level')})")
            try:
                result = process_question(user_id, question, args, llm_config, run_metadata=run_metadata)
            except Exception as exc:
                if args.stop_on_error:
                    raise
                result = error_result(question, args, exc, run_metadata=run_metadata)
            with write_lock:
                append_jsonl(results_path, result)
            print(f"Written result for question {question.get('id')}")
            if args.stop_on_error and result.get("task_status") != "success":
                raise RuntimeError(f"Stopping after {result.get('task_status')} on question {question.get('id')}")
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_question, user_id, question, args, llm_config, run_metadata): question
                for question in pending
            }
            for future in as_completed(futures):
                question = futures[future]
                try:
                    result = future.result()
                    with write_lock:
                        append_jsonl(results_path, result)
                    print(f"Written result for question {question.get('id')}")
                    if args.stop_on_error and result.get("task_status") != "success":
                        raise RuntimeError(f"Stopping after {result.get('task_status')} on question {question.get('id')}")
                except Exception as exc:
                    if args.stop_on_error:
                        raise
                    result = error_result(question, args, exc, run_metadata=run_metadata)
                    with write_lock:
                        append_jsonl(results_path, result)
                    print(f"Question {question.get('id')} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
