from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_extensions.infiagentbench.io_utils import read_json, write_json
from scripts.run_infiagent_improve import ABLATIONS


ARTIFACTS = [
    "schema_profile.json",
    "question_contract.json",
    "skill_plan.json",
    "verifier_report.json",
    "final_block.json",
    "agent_prompt.json",
]


def _load_task_ids(path: str) -> list[int]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        data = json.loads(raw)
        return [int(item["task_id"] if isinstance(item, dict) else item) for item in data]
    ids = []
    for token in raw.replace(",", "\n").splitlines():
        token = token.strip()
        if token:
            ids.append(int(token))
    return ids


def _expected_artifacts(ablation: str) -> dict[str, bool]:
    enabled = ABLATIONS[ablation]
    return {
        "schema_profile.json": any(enabled[key] for key in ["schema", "contract", "router", "verifier", "final"]),
        "question_contract.json": any(enabled[key] for key in ["contract", "router", "verifier", "final"]),
        "skill_plan.json": enabled["router"] or enabled["verifier"] or enabled["final"],
        "verifier_report.json": enabled["verifier"],
        "final_block.json": enabled["final"],
        "agent_prompt.json": True,
    }


def _run_ablation(args: argparse.Namespace, ablation: str) -> None:
    cmd = [
        sys.executable,
        "scripts/run_infiagent_improve.py",
        "--dry-run",
        "--task-ids-file",
        args.task_ids_file,
        "--ablation",
        ablation,
        "--note",
        args.note,
        "--output-dir",
        args.output_dir,
        "--questions-file",
        args.questions_file,
        "--tables-dir",
        args.tables_dir,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _summarize(args: argparse.Namespace, ablations: list[str], task_ids: list[int]) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    per_task: list[dict[str, Any]] = []
    for ablation in ablations:
        expected = _expected_artifacts(ablation)
        counts = {
            artifact: {"expected": 0, "present": 0, "missing_expected": 0}
            for artifact in ARTIFACTS
        }
        for task_id in task_ids:
            task_dir = Path(args.output_dir) / args.note / ablation / "tasks" / str(task_id)
            task_entry = {"ablation": ablation, "task_id": task_id, "artifacts": {}}
            for artifact in ARTIFACTS:
                present = (task_dir / artifact).exists()
                is_expected = expected[artifact]
                counts[artifact]["expected"] += int(is_expected)
                counts[artifact]["present"] += int(present)
                counts[artifact]["missing_expected"] += int(is_expected and not present)
                task_entry["artifacts"][artifact] = {
                    "present": present,
                    "expected": is_expected,
                }
            per_task.append(task_entry)
        row = {
            "ablation": ablation,
            "missing_expected_total": sum(value["missing_expected"] for value in counts.values()),
        }
        for artifact in ARTIFACTS:
            row[f"{artifact}:present"] = counts[artifact]["present"]
            row[f"{artifact}:expected"] = counts[artifact]["expected"]
            row[f"{artifact}:missing_expected"] = counts[artifact]["missing_expected"]
        summary_rows.append(row)

    columns = ["ablation", "missing_expected_total"] + [
        f"{artifact}:missing_expected" for artifact in ARTIFACTS
    ]
    markdown = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in summary_rows:
        markdown.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")

    return {
        "note": args.note,
        "task_ids_file": args.task_ids_file,
        "task_ids": task_ids,
        "ablations": ablations,
        "summary": summary_rows,
        "per_task": per_task,
        "markdown_table": "\n".join(markdown),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dry-run ablations and audit generated InfiAgentBench artifacts.")
    parser.add_argument("--task-ids-file", required=True)
    parser.add_argument("--questions-file", default="evaluation/InfiAgentBench/data/da-dev-questions.jsonl")
    parser.add_argument("--tables-dir", default="evaluation/InfiAgentBench/data/da-dev-tables")
    parser.add_argument("--output-dir", default="runs/infiagent_improve")
    parser.add_argument("--note", default="small-error-dry-run")
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["baseline", "schema", "contract", "skill", "verifier", "final_formatter", "full"],
    )
    parser.add_argument("--skip-run", action="store_true", help="Only summarize existing dry-run outputs.")
    parser.add_argument("--output", default="runs/infiagent_improve/small_error_dry_run_audit.json")
    args = parser.parse_args()

    unknown = [ablation for ablation in args.ablations if ablation not in ABLATIONS]
    if unknown:
        raise SystemExit(f"Unknown ablations: {unknown}")

    task_ids = _load_task_ids(args.task_ids_file)
    if not args.skip_run:
        for ablation in args.ablations:
            _run_ablation(args, ablation)

    report = _summarize(args, args.ablations, task_ids)
    write_json(report, args.output)
    print(report["markdown_table"])
    print(args.output)


if __name__ == "__main__":
    main()

