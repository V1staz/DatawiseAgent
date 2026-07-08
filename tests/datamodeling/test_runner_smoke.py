from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datawiseagent.harness.datamodeling.runner import (
    OfflineDataModelingBaselineConfig,
    OfflineDataModelingBaselineRunner,
    summarize_offline_baseline_results,
)
from datawiseagent.harness.datamodeling.contracts import build_contract
from datawiseagent.harness.datamodeling.cv_engine import make_cv_split_plan
from datawiseagent.harness.datamodeling.profiler import build_profile
from datawiseagent.harness.datamodeling.protocol import build_protocol_plan
from datawiseagent.harness.datamodeling.recipes import RecipeConfig, RecipeEngine
from datawiseagent.harness.datamodeling.task_router import route_task


def make_regression_benchmark(tmp_path: Path) -> Path:
    root = tmp_path / "DataModeling"
    name = "toy-regression"
    data_dir = root / "data" / "data_resplit" / name
    task_dir = root / "data" / "task"
    eval_dir = root / "evaluation"
    data_dir.mkdir(parents=True)
    task_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (root / "data.jsonl").write_text(json.dumps({"name": name, "url": "local", "size": "tiny", "year": 2026}) + "\n")
    train = pd.DataFrame({"id": list(range(12)), "x": list(range(12)), "group": ["a", "b"] * 6})
    train["target"] = train["x"] * 2 + 1
    train.to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [12, 13], "x": [12, 13], "group": ["a", "b"]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [12, 13], "target": [0.0, 0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)
    (task_dir / f"{name}.txt").write_text("Evaluation is root mean squared error. Predict target.")
    (eval_dir / f"{name}_eval.py").write_text("from sklearn.metrics import mean_squared_error\n")
    return root


def make_string_class_benchmark(tmp_path: Path) -> Path:
    root = tmp_path / "DataModeling"
    name = "toy-string-class"
    data_dir = root / "data" / "data_resplit" / name
    task_dir = root / "data" / "task"
    eval_dir = root / "evaluation"
    data_dir.mkdir(parents=True)
    task_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (root / "data.jsonl").write_text(json.dumps({"name": name, "url": "local", "size": "tiny", "year": 2026}) + "\n")
    train = pd.DataFrame(
        {
            "id": list(range(12)),
            "x": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 0.3, 1.3, 2.3],
            "segment": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "a", "b", "c"],
            "label": ["low", "low", "low", "mid", "mid", "mid", "high", "high", "high", "low", "mid", "high"],
        }
    )
    train.to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [12, 13, 14], "x": [0.15, 1.15, 2.15], "segment": ["a", "b", "c"]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [12, 13, 14], "label": ["low", "low", "low"]}).to_csv(data_dir / "sample_submission.csv", index=False)
    (task_dir / f"{name}.txt").write_text("Evaluation is classification accuracy for label.")
    (eval_dir / f"{name}_eval.py").write_text("from sklearn.metrics import accuracy_score\nperformance = accuracy_score(answers['label'], predictions['label'])\n")
    return root


def make_target_encoding_benchmark(tmp_path: Path) -> Path:
    root = tmp_path / "DataModeling"
    name = "toy-target-encoding"
    data_dir = root / "data" / "data_resplit" / name
    task_dir = root / "data" / "task"
    eval_dir = root / "evaluation"
    data_dir.mkdir(parents=True)
    task_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (root / "data.jsonl").write_text(json.dumps({"name": name, "url": "local", "size": "tiny", "year": 2026}) + "\n")
    rows = []
    for i in range(36):
        category = f"category_{i % 9}"
        category_signal = (i % 9) * 3.0
        rows.append({"id": i, "x": float(i % 4), "category_code": category, "target": category_signal + (i % 4) * 0.2})
    train = pd.DataFrame(rows)
    train.to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [100, 101, 102],
            "x": [0.0, 1.0, 2.0],
            "category_code": ["category_0", "category_5", "category_new"],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "target": [0.0, 0.0, 0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)
    (task_dir / f"{name}.txt").write_text("Evaluation is root mean squared error. Predict target from numeric and categorical signals.")
    (eval_dir / f"{name}_eval.py").write_text("from sklearn.metrics import mean_squared_error\n")
    return root


def test_runner_writes_result_and_artifacts(tmp_path: Path) -> None:
    root = make_regression_benchmark(tmp_path)
    out = tmp_path / "out"
    runner = OfflineDataModelingBaselineRunner(
        OfflineDataModelingBaselineConfig(
            benchmark_root=root,
            output_dir=out,
            model_name="toy-harness",
            budget_seconds=60,
            max_candidates=2,
            candidate_allowlist=("dummy", "ridge"),
            score=False,
        )
    )

    records = runner.run_tasks([0], resume=False)
    summary = summarize_offline_baseline_results(out)

    assert len(records) == 1
    assert Path(records[0].result_path).exists()
    assert (out / "results.jsonl").exists()
    assert (out / "artifacts" / "toy-regression" / "contract.json").exists()
    assert (out / "artifacts" / "toy-regression" / "submission_validation.json").exists()
    artifact_dir = out / "artifacts" / "toy-regression"
    assert (artifact_dir / "candidate_leaderboard.json").exists()
    assert (artifact_dir / "fold_metrics.json").exists()
    assert (artifact_dir / "feature_pipeline_summary.json").exists()
    assert (artifact_dir / "model_recommendations.json").exists()
    assert summary["rows"] == 1
    assert summary["validator_passed"] == 1


def test_runner_preserves_string_class_submission_labels(tmp_path: Path) -> None:
    root = make_string_class_benchmark(tmp_path)
    out = tmp_path / "string-out"
    runner = OfflineDataModelingBaselineRunner(
        OfflineDataModelingBaselineConfig(
            benchmark_root=root,
            output_dir=out,
            model_name="toy-string-harness",
            budget_seconds=60,
            max_candidates=2,
            candidate_allowlist=("dummy", "logistic"),
            score=False,
        )
    )

    records = runner.run_tasks([0], resume=False)

    submission = pd.read_csv(records[0].result_path)
    assert set(submission["label"]).issubset({"low", "mid", "high"})
    assert not pd.to_numeric(submission["label"], errors="coerce").notna().all()


def test_recipe_scaffold_uses_fold_local_preprocessing_metadata(tmp_path: Path) -> None:
    root = make_regression_benchmark(tmp_path)
    contract = build_contract(root, "toy-regression")
    out = tmp_path / "recipe-artifacts"

    assert contract.group_columns == ["group"]
    route = route_task(contract, build_profile(contract))
    protocol = build_protocol_plan(contract, build_profile(contract), route)
    assert "time_series_or_grouped" in route.facets
    assert protocol.split_constraint == "group_aware"

    result = RecipeEngine(
        RecipeConfig(
            budget_seconds=60,
            max_candidates=2,
            candidate_allowlist=("dummy", "ridge"),
        )
    ).run(contract, out)

    assert result.validation["passed"] is True
    assert Path(result.artifacts["candidate_leaderboard"]).exists()
    assert Path(result.artifacts["fold_metrics"]).exists()
    assert Path(result.artifacts["feature_pipeline_summary"]).exists()
    assert Path(result.artifacts["model_recommendations"]).exists()
    assert Path(result.artifacts["oof_predictions"]).exists()
    feature_summary = json.loads(Path(result.artifacts["feature_pipeline_summary"]).read_text())
    assert feature_summary["preprocessing_scope"] == "fold_local"
    assert feature_summary["group_column"] == "group"
    fold_metrics = json.loads(Path(result.artifacts["fold_metrics"]).read_text())
    assert fold_metrics
    assert {row["split_kind"] for row in fold_metrics} == {"group_kfold"}
    leaderboard = json.loads(Path(result.artifacts["leaderboard"]).read_text())
    assert all(
        item["cv_result"]["metadata"]["preprocessing_scope"] == "fold_local_when_estimator_is_pipeline"
        for item in leaderboard
        if item.get("cv_result")
    )
    assert all(
        item["cv_result"]["metadata"]["group_column"] == "group"
        for item in leaderboard
        if item.get("cv_result")
    )
    assert all(
        item["cv_result"]["metadata"]["group_fold_overlap_count"] == 0
        for item in leaderboard
        if item.get("cv_result")
    )


def test_recipe_adds_fold_safe_target_encoding_candidate(tmp_path: Path) -> None:
    root = make_target_encoding_benchmark(tmp_path)
    contract = build_contract(root, "toy-target-encoding")
    out = tmp_path / "target-encoding-artifacts"

    result = RecipeEngine(
        RecipeConfig(
            budget_seconds=60,
            max_candidates=1,
            candidate_allowlist=("target_encoded_ridge",),
        )
    ).run(contract, out)

    assert result.validation["passed"] is True
    assert result.selected_candidate == "target_encoded_ridge"
    feature_summary = json.loads(Path(result.artifacts["feature_pipeline_summary"]).read_text())
    assert feature_summary["preprocessing_scope"] == "fold_local"
    assert feature_summary["target_encoding_candidate_columns"] == ["category_code"]
    leaderboard = json.loads(Path(result.artifacts["candidate_leaderboard"]).read_text())
    assert leaderboard[0]["uses_target_encoding"] is True
    recommendations = json.loads(Path(result.artifacts["model_recommendations"]).read_text())
    assert any("target encoding" in item for item in recommendations["recommendations"])
    assert any("native categorical" in item for item in recommendations["recommendations"])


def test_recipe_can_run_native_catboost_candidate(tmp_path: Path) -> None:
    pytest.importorskip("catboost")
    root = make_target_encoding_benchmark(tmp_path)
    contract = build_contract(root, "toy-target-encoding")
    out = tmp_path / "catboost-artifacts"

    result = RecipeEngine(
        RecipeConfig(
            budget_seconds=60,
            max_candidates=1,
            candidate_allowlist=("catboost_native",),
        )
    ).run(contract, out)

    assert result.validation["passed"] is True
    assert result.selected_candidate == "catboost_native"
    leaderboard = json.loads(Path(result.artifacts["candidate_leaderboard"]).read_text())
    assert leaderboard[0]["uses_native_categorical"] is True


def test_cv_split_plan_keeps_groups_exclusive(tmp_path: Path) -> None:
    root = make_regression_benchmark(tmp_path)
    contract = build_contract(root, "toy-regression")
    y = np.arange(12)
    groups = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"])

    splits, metadata = make_cv_split_plan(
        pd.DataFrame({"x": np.arange(12)}),
        y,
        contract,
        n_splits=3,
        groups=groups,
        group_column="group",
    )

    assert metadata["split_kind"] == "group_kfold"
    assert metadata["group_column"] == "group"
    assert metadata["group_fold_overlap_count"] == 0
    for train_idx, valid_idx in splits:
        assert set(groups[train_idx]).isdisjoint(set(groups[valid_idx]))


def test_offline_baseline_marks_score_failure_as_failed_row(tmp_path: Path) -> None:
    root = make_regression_benchmark(tmp_path)
    out = tmp_path / "score-failure-out"
    runner = OfflineDataModelingBaselineRunner(
        OfflineDataModelingBaselineConfig(
            benchmark_root=root,
            output_dir=out,
            model_name="toy-string-harness",
            budget_seconds=60,
            max_candidates=2,
            candidate_allowlist=("dummy", "ridge"),
            score=True,
        )
    )

    records = runner.run_tasks([0], resume=False)
    summary = summarize_offline_baseline_results(out)

    assert len(records) == 1
    assert records[0].failure
    assert records[0].harness["failure_category"] == "score_failed"
    assert records[0].harness["official_score"]["status"] != "ok"
    assert summary["failures"] == 1
