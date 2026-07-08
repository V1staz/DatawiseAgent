"""Run the offline deterministic DataModeling baseline/oracle.

This script can sanity-check contracts, validators, and recipe baselines.  It is
not the official LLM-capability evaluation path; use
``evaluation/run_agent_harness_datamodeling.py`` for model runs.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datawiseagent.harness.datamodeling import load_tasks  # noqa: E402
from datawiseagent.harness.datamodeling.runner import (  # noqa: E402
    OfflineDataModelingBaselineConfig,
    OfflineDataModelingBaselineRunner,
    summarize_offline_baseline_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline deterministic DataModeling baseline/oracle.")
    parser.add_argument("--benchmark-root", default="evaluation/DataModeling")
    parser.add_argument("--model-name", default="datamodeling-harness")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tasks", nargs="*", default=None, help="Task ids or names. Omit for all 74 tasks.")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget-seconds", type=int, default=1800)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--candidate-allowlist", default=None, help="Comma-separated candidate names, e.g. dummy,logistic,lightgbm")
    parser.add_argument("--score", action="store_true", help="Run official per-task eval scripts after generating submissions.")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def resolve_tasks(args: argparse.Namespace) -> list[str | int] | None:
    if args.tasks:
        return [int(t) if str(t).isdigit() else t for t in args.tasks]
    tasks = list(range(len(load_tasks(args.benchmark_root))))
    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end if args.end is not None else len(tasks)
        tasks = tasks[start:end]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or f"evaluation/experimental_results/Datamodeling-DSBench/{args.model_name}"
    allowlist = tuple(x.strip() for x in args.candidate_allowlist.split(",") if x.strip()) if args.candidate_allowlist else None
    config = OfflineDataModelingBaselineConfig(
        benchmark_root=args.benchmark_root,
        output_dir=output_dir,
        model_name=args.model_name,
        budget_seconds=args.budget_seconds,
        max_candidates=args.max_candidates,
        n_splits=args.n_splits,
        random_state=args.random_state,
        candidate_allowlist=allowlist,
        score=args.score,
    )
    runner = OfflineDataModelingBaselineRunner(config)
    records = runner.run_tasks(resolve_tasks(args), resume=not args.no_resume)
    summary = summarize_offline_baseline_results(output_dir)
    print(json.dumps({"ran": len(records), "output_dir": str(output_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
