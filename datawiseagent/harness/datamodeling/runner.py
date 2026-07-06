"""Offline deterministic baseline/oracle for DataModeling.

This module is intentionally separate from the official LLM-in-the-loop agent
harness.  It can train recipe-based baselines for validator/benchmark sanity
checks, but its outputs must not be used as model-capability results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json
import subprocess
import sys
import time

from .contracts import build_contract, load_tasks, render_modeling_plan_prompt
from .profiler import build_profile, write_profile
from .recipes import RecipeConfig, RecipeEngine


@dataclass(slots=True)
class OfflineDataModelingBaselineConfig:
    benchmark_root: str | Path = "evaluation/DataModeling"
    output_dir: str | Path = "evaluation/experimental_results/Datamodeling-DSBench/datamodeling-harness"
    model_name: str = "datamodeling-harness"
    budget_seconds: int = 1800
    max_candidates: int = 6
    n_splits: int = 3
    random_state: int = 42
    candidate_allowlist: tuple[str, ...] | None = None
    score: bool = False


@dataclass(slots=True)
class TaskRunRecord:
    id: int | None
    name: str
    input_text: str
    response: str
    session_id: str | None
    user_id: str | None
    time: float
    cost: None
    model: str
    input: None
    output: None
    result_path: str
    harness: dict[str, Any]
    failure: bool = False

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class OfflineDataModelingBaselineRunner:
    def __init__(self, config: OfflineDataModelingBaselineConfig) -> None:
        self.config = config
        self.benchmark_root = Path(config.benchmark_root)
        self.output_dir = Path(config.output_dir)
        self.artifact_root = self.output_dir / "artifacts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.recipe_engine = RecipeEngine(
            RecipeConfig(
                budget_seconds=config.budget_seconds,
                max_candidates=config.max_candidates,
                n_splits=config.n_splits,
                random_state=config.random_state,
                candidate_allowlist=config.candidate_allowlist,
            )
        )

    def run_tasks(self, tasks: Iterable[str | int] | None = None, *, resume: bool = True) -> list[TaskRunRecord]:
        task_items = list(tasks) if tasks is not None else [i for i, _ in enumerate(load_tasks(self.benchmark_root))]
        completed = self._completed_ids() if resume else set()
        records: list[TaskRunRecord] = []
        for task in task_items:
            task_id = int(task) if str(task).isdigit() else None
            if task_id is not None and task_id in completed:
                continue
            record = self.run_task(task)
            self._append_result(record)
            records.append(record)
        return records

    def run_task(self, task: str | int) -> TaskRunRecord:
        start = time.time()
        contract = build_contract(self.benchmark_root, task)
        artifact_dir = self.artifact_root / contract.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "contract.json").write_text(json.dumps(contract.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        plan_prompt = render_modeling_plan_prompt(contract)
        (artifact_dir / "modeling_plan_prompt.txt").write_text(plan_prompt, encoding="utf-8")
        profile = build_profile(contract)
        write_profile(profile, artifact_dir / "profile.json")

        failure = False
        failure_category = "none"
        response = ""
        try:
            recipe_result = self.recipe_engine.run(contract, artifact_dir)
            response = json.dumps({"selected_candidate": recipe_result.selected_candidate, "validation": recipe_result.validation}, ensure_ascii=False)
        except Exception as exc:
            failure = True
            failure_category = f"recipe_exception:{type(exc).__name__}"
            recipe_result = None
            response = f"HARNESS_FAILURE:{type(exc).__name__}:{str(exc)[:500]}"

        seconds = time.time() - start
        score_result = None
        if recipe_result is not None and self.config.score:
            score_result = self._score_task(contract.name, recipe_result.submission_path)
            if score_result.get("status") != "ok":
                failure = True
                failure_category = "score_failed"

        return TaskRunRecord(
            id=contract.task_id,
            name=contract.name,
            input_text=plan_prompt,
            response=response,
            session_id=None,
            user_id=None,
            time=seconds,
            cost=None,
            model=self.config.model_name,
            input=None,
            output=None,
            result_path=recipe_result.submission_path if recipe_result is not None else "",
            harness={
                "contract_path": str(artifact_dir / "contract.json"),
                "profile_path": str(artifact_dir / "profile.json"),
                "modeling_plan_prompt_path": str(artifact_dir / "modeling_plan_prompt.txt"),
                "recipe_result": recipe_result.to_dict() if recipe_result is not None else {},
                "official_score": score_result,
                "label_blind_inputs": True,
                "answers_or_gt_used_for_training": False,
                "mode": "offline_deterministic_baseline",
                "deterministic_recipe_used": recipe_result is not None,
                "failure_category": failure_category,
            },
            failure=failure,
        )

    def _score_task(self, name: str, prediction_path: str) -> dict[str, Any]:
        eval_script = (self.benchmark_root / "evaluation" / f"{name}_eval.py").resolve()
        answer_file = (self.benchmark_root / "data" / "answers" / name / "test_answer.csv").resolve()
        prediction_file = Path(prediction_path).resolve()
        output_dir = (self.output_dir / "performances").resolve()
        if not eval_script.exists() or not answer_file.exists() or not prediction_file.exists():
            return {
                "status": "missing_inputs",
                "eval_script": str(eval_script),
                "answer_file": str(answer_file),
                "prediction_file": str(prediction_file),
            }
        (output_dir / name).mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(eval_script),
                "--answer_file",
                str(answer_file),
                "--predict_file",
                str(prediction_file),
                "--path",
                str(output_dir),
                "--name",
                name,
            ],
            cwd=str(self.benchmark_root.parent.resolve()),
            capture_output=True,
            text=True,
            check=False,
        )
        result_path = output_dir / name / "result.txt"
        score_text = result_path.read_text(encoding="utf-8").strip() if result_path.exists() else None
        score_value = None
        if score_text is not None:
            try:
                score_value = float(score_text)
            except ValueError:
                pass
        return {
            "status": "ok" if completed.returncode == 0 and result_path.exists() else "failed",
            "returncode": completed.returncode,
            "score": score_value,
            "score_text": score_text,
            "result_path": str(result_path),
            "stderr": completed.stderr[-1000:],
            "stdout": completed.stdout[-1000:],
        }

    def _append_result(self, record: TaskRunRecord) -> None:
        with (self.output_dir / "results.jsonl").open("a", encoding="utf-8") as f:
            f.write(record.to_json_line() + "\n")

    def _completed_ids(self) -> set[int]:
        path = self.output_dir / "results.jsonl"
        if not path.exists():
            return set()
        completed: set[int] = set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("failure"):
                    continue
                if self.config.score:
                    score = (item.get("harness") or {}).get("official_score") or {}
                    if score.get("status") != "ok":
                        continue
                if "id" in item and item["id"] is not None:
                    completed.add(int(item["id"]))
        return completed


def summarize_offline_baseline_results(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    results_path = output_dir / "results.jsonl"
    rows = []
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    validations = []
    for path in (output_dir / "artifacts").glob("*/submission_validation.json"):
        try:
            validations.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    summary = {
        "rows": len(rows),
        "failures": sum(1 for row in rows if row.get("failure")),
        "validated_submissions": len(validations),
        "validator_passed": sum(1 for item in validations if item.get("passed")),
        "average_time": sum(float(row.get("time") or 0) for row in rows) / len(rows) if rows else 0,
        "selected_candidates": {},
    }
    for row in rows:
        selected = (((row.get("harness") or {}).get("recipe_result") or {}).get("selected_candidate") or "unknown")
        summary["selected_candidates"][selected] = summary["selected_candidates"].get(selected, 0) + 1
    (output_dir / "harness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# Backwards-compatible aliases for older local scripts.  Prefer the explicit
# Offline* names in new code so this recipe runner cannot be confused with the
# official LLM-in-the-loop harness.
DataModelingHarnessConfig = OfflineDataModelingBaselineConfig
DataModelingHarnessRunner = OfflineDataModelingBaselineRunner
summarize_results = summarize_offline_baseline_results
