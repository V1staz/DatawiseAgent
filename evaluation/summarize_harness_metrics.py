"""Summarize InfiAgentBench harness result/eval artifacts.

This is a lightweight Phase-0/Phase-1 reporting helper. It does not use labels
unless an eval JSON produced by eval_closed_form.py is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    level_counts: dict[str, int] = {}
    concept_counts: dict[str, int] = {}
    concept_count_counts: dict[str, int] = {}
    task_status_counts: dict[str, int] = {}
    for row in rows:
        level = row.get("level", "unknown")
        level_counts[level] = level_counts.get(level, 0) + 1
        status = row.get("task_status", "unknown")
        task_status_counts[status] = task_status_counts.get(status, 0) + 1
        concepts = row.get("concepts") or []
        concept_count_counts[str(len(concepts))] = concept_count_counts.get(str(len(concepts)), 0) + 1
        for concept in concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1

    branch_counts = [
        (row.get("search_summary") or {}).get("branch_count")
        for row in rows
        if (row.get("search_summary") or {}).get("branch_count") is not None
    ]
    runtime_seconds = [
        (row.get("runtime") or {}).get("seconds")
        for row in rows
        if (row.get("runtime") or {}).get("seconds") is not None
    ]
    artifact_counts = []
    for row in rows:
        artifact_counts.append(
            sum(
                1
                for key in [
                    "data_card_path",
                    "trace_path",
                    "validator_report_path",
                    "semantic_report_path",
                ]
                if row.get(key)
            )
            + len(row.get("branch_results") or [])
        )
    metadata = rows[0].get("experiment_metadata", {}) if rows else {}
    return {
        "row_count": len(rows),
        "level_counts": level_counts,
        "concept_counts": concept_counts,
        "concept_count_counts": concept_count_counts,
        "task_status_counts": task_status_counts,
        "final_block_missing_count": sum(1 for row in rows if row.get("final_block") == {} or row.get("final_block_missing") is True),
        "reformat_missing_count": sum(1 for row in rows if not row.get("reformat_response")),
        "verifier_blocked_count": task_status_counts.get("verifier_blocked", 0),
        "validator_failed_count": sum(1 for row in rows if (row.get("validator_report") or {}).get("passed") is False),
        "semantic_oracle_failed_count": sum(1 for row in rows if (row.get("semantic_report") or {}).get("passed") is False),
        "semantic_oracle_warning_count": sum(len((row.get("semantic_report") or {}).get("warnings") or []) for row in rows),
        "average_rule_count": round(sum(len(row.get("rule_ids") or row.get("rules") or []) for row in rows) / len(rows), 4) if rows else 0,
        "average_oracle_count": round(sum(len(row.get("oracle_checks") or []) for row in rows) / len(rows), 4) if rows else 0,
        "average_memory_hit_count": round(sum(len(row.get("memory_hits") or []) for row in rows) / len(rows), 4) if rows else 0,
        "average_artifact_count": round(sum(artifact_counts) / len(artifact_counts), 4) if artifact_counts else 0,
        "recovery_event_count": sum(len(row.get("recovery_events") or []) for row in rows),
        "average_branch_count": round(sum(branch_counts) / len(branch_counts), 4) if branch_counts else 0,
        "average_seconds": round(sum(runtime_seconds) / len(runtime_seconds), 4) if runtime_seconds else 0,
        "experiment_metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize harness JSONL and eval JSON artifacts.")
    parser.add_argument("--results", required=True, help="Harness or baseline result JSONL path.")
    parser.add_argument("--eval-json", default=None, help="Optional eval_closed_form output JSON.")
    parser.add_argument("--output", default=None, help="Optional summary JSON path.")
    args = parser.parse_args()

    summary = summarize_results(read_jsonl(args.results))
    eval_payload = load_json(args.eval_json)
    if eval_payload:
        summary["eval_main_results"] = eval_payload.get("main_results", {})
        summary["eval_harness_metrics"] = eval_payload.get("harness_metrics", {})
        summary["concept_accuracy_analysis"] = eval_payload.get("concept_accuracy_analysis", {})
        summary["concept_count_accuracy_analysis"] = eval_payload.get("concept_count_accuracy_analysis", {})
        summary["two_or_more_concepts_accuracy"] = eval_payload.get("two_or_more_concepts_accuracy")

    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
