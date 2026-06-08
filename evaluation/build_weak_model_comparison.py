"""Build weak-model InfiAgentBench comparison tables from completed eval artifacts.

This script is evaluation-only: it reads labels after runs finish and never feeds
gold answers back into prompts or memory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import re


DEFAULT_ROOT = Path("evaluation/experimental_results/InfiAgent-Bench/weak_model_comparison")
DEFAULT_QUESTIONS = Path("evaluation/InfiAgentBench/data/da-dev-questions.jsonl")
DEFAULT_LABELS = Path("evaluation/InfiAgentBench/data/da-dev-labels.jsonl")


def read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_format(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    return dict(re.findall(r"@(\w+)\[(.*?)\]", text))


def is_equal(response: Any, label: Any) -> bool:
    if response == label:
        return True
    try:
        return abs(float(str(response)) - float(str(label))) < 1e-6
    except Exception:
        return False


def answer_text(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    return str(row.get("reformat_response") or row.get("response") or "")


def correctness(row: dict[str, Any] | None, gold: dict[str, str]) -> tuple[bool | None, dict[str, str]]:
    if not row:
        return None, {}
    predicted = extract_format(answer_text(row))
    if not predicted:
        return False, predicted
    return all(is_equal(predicted.get(name), value) for name, value in gold.items()), predicted


def status_from_correctness(base: bool | None, harness: bool | None, base_row: dict[str, Any] | None, harness_row: dict[str, Any] | None) -> str:
    if base is None or harness is None:
        return "not_run"
    if (harness_row or {}).get("task_status") == "verifier_blocked":
        return "verifier_blocked"
    if not base and harness:
        return "fixed"
    if base and not harness:
        return "regressed"
    if base and harness:
        return "both_correct"
    return "unchanged_wrong"


def metrics(rows: list[dict[str, Any]], labels: dict[int, dict[str, str]], questions: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_level: dict[str, list[tuple[bool, float, int, int, dict[str, Any]]]] = {}
    for row in rows:
        qid = int(row.get("id"))
        gold = labels.get(qid)
        if not gold:
            continue
        ok, predicted = correctness(row, gold)
        if ok is None:
            continue
        sub_correct = sum(is_equal(predicted.get(name), value) for name, value in gold.items())
        sub_total = len(gold)
        prop = sub_correct / sub_total if sub_total else 0
        level = questions.get(qid, {}).get("level", "unknown")
        by_level.setdefault(level, []).append((ok, prop, sub_correct, sub_total, row))
        by_level.setdefault("full", []).append((ok, prop, sub_correct, sub_total, row))

    output: dict[str, dict[str, Any]] = {}
    for level, items in by_level.items():
        total = len(items)
        runtime_values = [(item[4].get("runtime") or {}).get("seconds") for item in items]
        runtime_values = [float(v) for v in runtime_values if v is not None]
        output[level] = {
            "Q-Acc": round(sum(1 for item in items if item[0]) / total, 4) if total else 0,
            "Proportional Sub-Q": round(sum(item[1] for item in items) / total, 4) if total else 0,
            "Sub-Q": round(sum(item[2] for item in items) / max(sum(item[3] for item in items), 1), 4) if total else 0,
            "runtime": round(sum(runtime_values) / len(runtime_values), 4) if runtime_values else 0,
            "format_errors": sum(1 for item in items if not extract_format(answer_text(item[4]))),
            "reformat_missing": sum(1 for item in items if not item[4].get("reformat_response")),
            "verifier_blocked": sum(1 for item in items if item[4].get("task_status") == "verifier_blocked"),
            "fixed": 0,
            "regressed": 0,
        }
    return output


def load_keyed(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    keyed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "id" in row:
            keyed[int(row["id"])] = row
    return keyed


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_ablation(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use setting=responses.jsonl for --ablation-run")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weak-model comparison CSVs.")
    parser.add_argument("--weak-baseline", required=True)
    parser.add_argument("--weak-harness", required=True)
    parser.add_argument("--strong-reference", default=None)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--weak-model", default="")
    parser.add_argument("--strong-name", default="qwen3.5-35b-a3b")
    parser.add_argument("--output-dir", default=str(DEFAULT_ROOT))
    parser.add_argument("--ablation-run", action="append", type=parse_ablation, default=[])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    questions = load_keyed(read_jsonl(args.questions))
    labels = {
        int(row["id"]): {name: value for name, value in row.get("common_answers", [])}
        for row in read_jsonl(args.labels)
    }
    weak_base = load_keyed(read_jsonl(args.weak_baseline))
    weak_harness = load_keyed(read_jsonl(args.weak_harness))
    strong = load_keyed(read_jsonl(args.strong_reference)) if args.strong_reference else {}

    per_task_rows: list[dict[str, Any]] = []
    for qid in sorted(labels):
        question = questions.get(qid, {})
        gold = labels[qid]
        base_row = weak_base.get(qid)
        harness_row = weak_harness.get(qid)
        strong_row = strong.get(qid)
        base_ok, _ = correctness(base_row, gold)
        harness_ok, _ = correctness(harness_row, gold)
        strong_ok, _ = correctness(strong_row, gold)
        status = status_from_correctness(base_ok, harness_ok, base_row, harness_row)
        per_task_rows.append(
            {
                "task_id": qid,
                "difficulty": question.get("level", ""),
                "concepts": "|".join(question.get("concepts", [])),
                "known_error_type": "",
                "weak_baseline_answer": answer_text(base_row),
                "weak_harness_answer": answer_text(harness_row),
                "strong_baseline_answer": answer_text(strong_row),
                "gold_answer": json.dumps(gold, ensure_ascii=False, sort_keys=True),
                "weak_baseline_correct": base_ok,
                "weak_harness_correct": harness_ok,
                "strong_baseline_correct": strong_ok,
                "status": status,
                "suspected_module": "",
                "failure_reason": "",
            }
        )

    write_csv(
        output_dir / "weak_baseline_vs_harness_per_task.csv",
        per_task_rows,
        [
            "task_id",
            "difficulty",
            "concepts",
            "known_error_type",
            "weak_baseline_answer",
            "weak_harness_answer",
            "strong_baseline_answer",
            "gold_answer",
            "weak_baseline_correct",
            "weak_harness_correct",
            "strong_baseline_correct",
            "status",
            "suspected_module",
            "failure_reason",
        ],
    )

    ablation_inputs = args.ablation_run or [
        ("baseline", Path(args.weak_baseline)),
        ("harness_full", Path(args.weak_harness)),
    ]
    ablation_rows: list[dict[str, Any]] = []
    metric_by_setting: dict[str, dict[str, dict[str, Any]]] = {}
    for setting, path in ablation_inputs:
        rows = read_jsonl(path)
        by_level = metrics(rows, labels, questions)
        metric_by_setting[setting] = by_level
        metadata = rows[0].get("experiment_metadata", {}) if rows else {}
        for difficulty, values in sorted(by_level.items()):
            ablation_rows.append(
                {
                    "model_name": metadata.get("model_name") or args.weak_model,
                    "setting": setting,
                    "difficulty": difficulty,
                    "Q-Acc": values["Q-Acc"],
                    "Proportional Sub-Q": values["Proportional Sub-Q"],
                    "Sub-Q": values["Sub-Q"],
                    "runtime": values["runtime"],
                    "format_errors": values["format_errors"],
                    "reformat_missing": values["reformat_missing"],
                    "verifier_blocked": values["verifier_blocked"],
                    "fixed": sum(1 for row in per_task_rows if row["status"] == "fixed") if setting == "harness_full" else 0,
                    "regressed": sum(1 for row in per_task_rows if row["status"] == "regressed") if setting == "harness_full" else 0,
                }
            )
    write_csv(
        output_dir / "ablation_summary.csv",
        ablation_rows,
        [
            "model_name",
            "setting",
            "difficulty",
            "Q-Acc",
            "Proportional Sub-Q",
            "Sub-Q",
            "runtime",
            "format_errors",
            "reformat_missing",
            "verifier_blocked",
            "fixed",
            "regressed",
        ],
    )

    strong_metrics = metrics(list(strong.values()), labels, questions) if strong else {}
    gap_rows: list[dict[str, Any]] = []
    for difficulty in ["easy", "medium", "hard", "full"]:
        strong_score = (strong_metrics.get(difficulty) or {}).get("Q-Acc")
        base_score = (metric_by_setting.get("baseline", {}).get(difficulty) or {}).get("Q-Acc")
        harness_score = (metric_by_setting.get("harness_full", {}).get(difficulty) or {}).get("Q-Acc")
        if strong_score is None:
            continue
        base_gap = None if base_score is None else round(strong_score - base_score, 4)
        harness_gap = None if harness_score is None else round(strong_score - harness_score, 4)
        gap_rows.append(
            {
                "weak_model": args.weak_model,
                "setting": "harness_full",
                "strong_reference": args.strong_name,
                "difficulty": difficulty,
                "weak_score": harness_score,
                "strong_score": strong_score,
                "gap": harness_gap,
                "gap_reduction_from_harness": (
                    round(base_gap - harness_gap, 4)
                    if base_gap is not None and harness_gap is not None
                    else None
                ),
            }
        )
    write_csv(
        output_dir / "weak_to_strong_gap.csv",
        gap_rows,
        [
            "weak_model",
            "setting",
            "strong_reference",
            "difficulty",
            "weak_score",
            "strong_score",
            "gap",
            "gap_reduction_from_harness",
        ],
    )


if __name__ == "__main__":
    main()
