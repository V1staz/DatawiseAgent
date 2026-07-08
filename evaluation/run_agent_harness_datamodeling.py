"""Run DataModeling through the LLM-in-the-loop agent harness.

Unlike evaluation/run_harness_datamodeling.py, this script does not generate
predictions with deterministic Python recipes.  It wraps the original
FastAPI/DatawiseAgent path, injects label-blind contract/profile guidance, and
validates/repairs the LLM-created submission.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
for path in (PROJECT_ROOT, EVALUATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datawiseagent.harness.datamodeling import (  # noqa: E402
    DataModelingAgentHarnessConfig,
    DataModelingAgentHarnessRunner,
    FunctionChatClient,
    load_tasks,
    summarize_agent_harness_results,
)
from datawiseagent_chat_client import chat, create_session, create_user, stop_session, upload_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DataModeling with the LLM-in-the-loop agent harness.")
    parser.add_argument("--benchmark-root", default=str(PROJECT_ROOT / "evaluation" / "DataModeling"))
    parser.add_argument("--workspace-root", default=str(PROJECT_ROOT / "log" / "workspace" / "users"))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tasks", nargs="*", default=None, help="Task ids or names. Omit for all tasks.")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-ids", default="", help="Comma-separated task ids to skip. Default is empty for true full runs.")
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument(
        "--chat-timeout-seconds",
        type=int,
        default=1200,
        help=(
            "Per chat request HTTP timeout. On timeout the harness validates any "
            "final_submission.csv already produced and can issue bounded repairs."
        ),
    )
    parser.add_argument(
        "--early-stop-min-seconds",
        type=float,
        default=60.0,
        help="Before this many seconds, do not stop a chat even if a valid submission appears.",
    )
    parser.add_argument(
        "--early-stop-poll-seconds",
        type=float,
        default=10.0,
        help="Poll interval for detecting a valid agent-created final submission while chat is still running.",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stop on strict-valid final_submission.csv.",
    )
    parser.add_argument(
        "--no-quality-gate",
        action="store_true",
        help="Disable the label-blind manifest/diversity quality gate. Intended only for regression debugging.",
    )
    parser.add_argument(
        "--same-session-repairs",
        action="store_true",
        help=(
            "Reuse the same DatawiseAgent session for repair prompts. Default is fresh repair sessions "
            "to avoid notebook history exceeding small-provider context limits."
        ),
    )
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=1)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-sessions", action="store_true", help="Do not stop sessions after each task; useful only for debugging.")
    return parser.parse_args()


def resolve_tasks(args: argparse.Namespace) -> list[str | int] | None:
    if args.tasks:
        tasks: list[str | int] = [int(t) if str(t).isdigit() else t for t in args.tasks]
    else:
        tasks = list(range(len(load_tasks(args.benchmark_root))))
    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end if args.end is not None else len(tasks)
        tasks = tasks[start:end]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    skip_ids = {int(x) for x in args.skip_ids.split(",") if x.strip()}
    return [task for task in tasks if not ((isinstance(task, int) or str(task).isdigit()) and int(task) in skip_ids)]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or str(PROJECT_ROOT / "evaluation" / "experimental_results" / "Datamodeling-DSBench" / args.model_name)
    client = FunctionChatClient(create_user, create_session, upload_file, chat, stop_session)
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=args.benchmark_root,
            output_dir=output_dir,
            workspace_root=args.workspace_root,
            model_name=args.model_name,
            max_repair_attempts=args.max_repair_attempts,
            chat_timeout_seconds=args.chat_timeout_seconds,
            early_stop_on_valid_submission=not args.no_early_stop,
            early_stop_min_seconds=args.early_stop_min_seconds,
            early_stop_poll_seconds=args.early_stop_poll_seconds,
            enforce_quality_gate=not args.no_quality_gate,
            fresh_session_per_repair=not args.same_session_repairs,
            score=args.score,
            stop_session=not args.keep_sessions,
            num_workers=args.num_workers,
        ),
        client,
    )
    records = runner.run_tasks(resolve_tasks(args), resume=not args.no_resume)
    summary = summarize_agent_harness_results(output_dir)
    print(json.dumps({"ran": len(records), "output_dir": output_dir, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
