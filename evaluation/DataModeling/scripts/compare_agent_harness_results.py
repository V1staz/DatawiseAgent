"""Compare DataModeling result directories with canonical per-task rows.

This is intentionally separate from ``show_results.py`` because several
harnessed runs may append retry rows to ``results.jsonl``.  The comparison here
deduplicates by task id (preferring validator/score-ok rows), then computes the
same normalized performance used by the original DSBench script.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = PROJECT_ROOT / "evaluation"
DM_ROOT = EVAL_ROOT / "DataModeling"
EXP_ROOT = EVAL_ROOT / "experimental_results" / "Datamodeling-DSBench"


def _load_tasks() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DM_ROOT / "data.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _validator_passed(row: dict[str, Any]) -> bool:
    reports = (row.get("harness") or {}).get("validation_reports") or []
    return bool(reports and reports[-1].get("passed"))


def _quality_passed(row: dict[str, Any]) -> bool:
    reports = (row.get("harness") or {}).get("quality_reports") or []
    return bool(reports and reports[-1].get("passed"))


def _score_ok(row: dict[str, Any]) -> bool:
    return ((row.get("harness") or {}).get("official_score") or {}).get("status") == "ok"


def _canonical_rows(model_dir: Path) -> tuple[list[dict[str, Any]], int]:
    canonical_path = model_dir / "results_canonical.jsonl"
    rows = _read_jsonl(canonical_path if canonical_path.exists() else model_dir / "results.jsonl")
    if canonical_path.exists():
        return rows, len(rows)

    by_id: dict[int, tuple[tuple[int, int, int], dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        task_id = row.get("id")
        if not isinstance(task_id, int):
            continue
        rank = (
            1 if (not row.get("failure") and _validator_passed(row) and _score_ok(row)) else 0,
            1 if (not row.get("failure") and _validator_passed(row)) else 0,
            index,
        )
        if task_id not in by_id or rank > by_id[task_id][0]:
            by_id[task_id] = (rank, row)
    return [value[1] for _, value in sorted(by_id.items())], len(rows)


def _read_metric(path: Path) -> tuple[bool, float | None, str | None]:
    if not path.exists():
        return False, None, None
    text = path.read_text(encoding="utf-8").strip()
    if not text or text.lower() == "nan":
        return True, None, text
    try:
        value = ast.literal_eval(text)
    except Exception:
        value = text
    try:
        return True, float(value), text
    except Exception:
        return True, None, text


def _normalized_score(pre: float | None, baseline: float | None, gt: float | None) -> float:
    if pre is None or baseline is None or gt is None or gt == baseline:
        return 0.0
    return max(0.0, (pre - baseline) / (gt - baseline))


def _model_summary(model: str, tasks: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_dir = EXP_ROOT / model
    rows, raw_rows = _canonical_rows(model_dir)
    by_id = {row.get("id"): row for row in rows if isinstance(row.get("id"), int)}
    details: list[dict[str, Any]] = []
    times = [float(row.get("time") or 0) for row in rows if isinstance(row.get("time"), (int, float))]

    for task_id, task in enumerate(tasks):
        name = task["name"]
        row = by_id.get(task_id)
        metric_exists, pre, metric_text = _read_metric(model_dir / "performances" / name / "result.txt")
        _, baseline, _ = _read_metric(DM_ROOT / "save_performance" / "baseline" / name / "result.txt")
        _, gt, _ = _read_metric(DM_ROOT / "save_performance" / "GT" / name / "result.txt")
        details.append(
            {
                "id": task_id,
                "name": name,
                "has_row": row is not None,
                "failure": bool(row.get("failure")) if row else None,
                "failure_category": ((row.get("harness") or {}).get("failure_category") if row else "missing_row"),
                "validator_passed": _validator_passed(row) if row else False,
                "quality_passed": _quality_passed(row) if row else False,
                "score_ok": _score_ok(row) if row else False,
                "time": float(row.get("time") or 0) if row else None,
                "performance_result_exists": metric_exists,
                "raw_metric_text": metric_text,
                "raw_metric": pre,
                "baseline_metric": baseline,
                "gt_metric": gt,
                "normalized_score": _normalized_score(pre, baseline, gt),
            }
        )

    completed = sum(1 for item in details if item["performance_result_exists"])
    scoreable = sum(1 for item in details if item["raw_metric"] is not None)
    summary = {
        "model": model,
        "raw_rows": raw_rows,
        "canonical_rows": len(rows),
        "completed": completed,
        "completion_rate": completed / len(tasks) if tasks else 0,
        "scoreable": scoreable,
        "scoreable_rate": scoreable / len(tasks) if tasks else 0,
        "validator_passed": sum(1 for item in details if item["validator_passed"]),
        "quality_passed": sum(1 for item in details if item["quality_passed"]),
        "score_ok": sum(1 for item in details if item["score_ok"]),
        "failures": sum(1 for item in details if item["failure"]),
        "avg_time": statistics.mean(times) if times else 0,
        "median_time": statistics.median(times) if times else 0,
        "max_time": max(times) if times else 0,
        "performance": statistics.mean(item["normalized_score"] for item in details) if details else 0,
    }
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DataModeling model result directories.")
    parser.add_argument("--models", nargs="+", required=True, help="Model directory names under experimental_results/Datamodeling-DSBench")
    parser.add_argument("--baseline-model", default=None, help="Optional model name used to compute deltas")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    tasks = _load_tasks()
    summaries: dict[str, dict[str, Any]] = {}
    per_model: dict[str, list[dict[str, Any]]] = {}
    for model in args.models:
        summaries[model], per_model[model] = _model_summary(model, tasks)

    comparisons: dict[str, list[dict[str, Any]]] = {}
    if args.baseline_model and args.baseline_model in per_model:
        baseline_items = {item["id"]: item for item in per_model[args.baseline_model]}
        for model, items in per_model.items():
            if model == args.baseline_model:
                continue
            comparisons[f"{model}_vs_{args.baseline_model}"] = [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "delta": item["normalized_score"] - baseline_items[item["id"]]["normalized_score"],
                    "score": item["normalized_score"],
                    "baseline_score": baseline_items[item["id"]]["normalized_score"],
                    "time": item["time"],
                    "baseline_time": baseline_items[item["id"]]["time"],
                    "failure_category": item["failure_category"],
                    "quality_passed": item["quality_passed"],
                }
                for item in items
            ]

    payload = {"summary": summaries, "tasks": per_model, "comparisons": comparisons}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
