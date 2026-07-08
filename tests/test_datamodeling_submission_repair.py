from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from datawiseagent.harness.datamodeling.contracts import DataModelingContract, MetricSpec, SubmissionSpec
from datawiseagent.harness.datamodeling.quality import evaluate_submission_quality
from datawiseagent.harness.datamodeling.submission import validate_submission
from datawiseagent.harness.datamodeling.submission_repair import repair_submission_layers
from datawiseagent.harness.datamodeling.task_router import route_task


def _contract(tmp_path: Path, *, probability: bool = True) -> DataModelingContract:
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.5, 0.5]}).to_csv(data / "sample_submission.csv", index=False)
    pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]}).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": [10, 11, 12], "x": [1, 2, 3], "target": [0, 1, 0]}).to_csv(data / "train.csv", index=False)
    metric = MetricSpec("roc_auc" if probability else "rmse", "maximize" if probability else "minimize", prediction_kind="probability" if probability else "regression", probability_output=probability, bounded_range=(0.0, 1.0) if probability else None)
    return DataModelingContract(
        task_id=0,
        name="toy",
        benchmark_root=str(tmp_path),
        task_description_path=str(tmp_path / "task.txt"),
        data_dir=str(data),
        train_path=str(data / "train.csv"),
        test_path=str(data / "test.csv"),
        sample_submission_path=str(data / "sample_submission.csv"),
        eval_script_path=None,
        metric=metric,
        submission=SubmissionSpec(str(data / "sample_submission.csv"), ["id", "target"], ["id"], ["target"], 3, {"target": "float64"}),
        task_kinds=["tabular_classification" if probability else "tabular_regression"],
        target_columns=["target"],
        id_columns=["id"],
        feature_columns=["x"],
        text_columns=[],
        datetime_columns=[],
        group_columns=[],
        categorical_columns=[],
        numeric_columns=["x"],
        risk_flags=[],
        recipe_hints=[],
        description_excerpt="toy",
    )


def test_repair_aligns_candidate_by_id_and_clips_probability(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    final = tmp_path / "workspace" / "input" / "final_submission.csv"
    final.parent.mkdir(parents=True)
    # Wrong order, wrong prediction column name, and out-of-range values.
    pd.DataFrame({"id": [3, 1, 2], "prediction": [1.4, -0.2, 0.8]}).to_csv(final, index=False)

    report = repair_submission_layers(final, contract, tmp_path / "artifacts", attempt=0)

    assert report.changed
    assert report.stage == "schema_id_row_repaired"
    repaired = pd.read_csv(final)
    assert repaired.to_dict("list") == {"id": [1, 2, 3], "target": [0.0, 0.8, 1.0]}
    assert validate_submission(final, contract, reject_sample_target_copy=True).passed


def test_repair_sample_copy_adopts_public_scaffold(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    final = tmp_path / "workspace" / "input" / "final_submission.csv"
    final.parent.mkdir(parents=True)
    # Structurally valid but target is identical to sample submission.
    pd.read_csv(contract.sample_submission_path).to_csv(final, index=False)
    recipe_dir = tmp_path / "artifacts" / "verified" / "recipe_proxy"
    recipe_dir.mkdir(parents=True)
    scaffold = recipe_dir / "final_submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.7, 0.2]}).to_csv(scaffold, index=False)
    evidence = SimpleNamespace(artifact_paths={"recipe_proxy_dir": str(recipe_dir)})

    report = repair_submission_layers(final, contract, tmp_path / "artifacts", attempt=1, verified_evidence=evidence, prefer_scaffold=True)

    assert report.changed
    assert report.stage == "scaffold_adopted"
    assert pd.read_csv(final)["target"].tolist() == [0.1, 0.7, 0.2]
    assert validate_submission(final, contract, reject_sample_target_copy=True).passed


def test_unalignable_row_count_uses_scaffold_when_preferred(tmp_path: Path) -> None:
    contract = _contract(tmp_path, probability=False)
    final = tmp_path / "workspace" / "input" / "final_submission.csv"
    final.parent.mkdir(parents=True)
    pd.DataFrame({"prediction": [10.0, 20.0]}).to_csv(final, index=False)
    recipe_dir = tmp_path / "artifacts" / "verified" / "recipe_proxy"
    recipe_dir.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "target": [1.5, 2.5, 3.5]}).to_csv(recipe_dir / "final_submission.csv", index=False)
    evidence = SimpleNamespace(artifact_paths={"recipe_proxy_dir": str(recipe_dir)})

    report = repair_submission_layers(final, contract, tmp_path / "artifacts", attempt=2, verified_evidence=evidence, prefer_scaffold=True)

    assert report.changed
    assert report.stage == "scaffold_adopted"
    assert pd.read_csv(final).to_dict("list") == {"id": [1, 2, 3], "target": [1.5, 2.5, 3.5]}
    assert validate_submission(final, contract, reject_sample_target_copy=True).passed

def test_scaffold_repair_writes_quality_manifest_with_protocol_and_stage_evidence(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    final = tmp_path / "workspace" / "input" / "final_submission.csv"
    final.parent.mkdir(parents=True)
    # Hard labels for an AUC-like task are structurally valid but low-information.
    pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]}).to_csv(final, index=False)
    recipe_dir = tmp_path / "artifacts" / "verified" / "recipe_proxy"
    recipe_dir.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "target": [0.12, 0.73, 0.34]}).to_csv(recipe_dir / "final_submission.csv", index=False)
    evidence = {
        "status": "ok",
        "metric_name": "roc_auc",
        "metric_direction": "maximize",
        "baseline_metric_value": 0.5,
        "best_candidate_metric_value": 0.7,
        "best_candidate_name": "logistic_recipe",
        "split_policy": "stratified_public_train_cv",
        "features_used": ["x"],
        "validation_metric_trust": "harness_verified",
        "artifact_paths": {"recipe_proxy_dir": str(recipe_dir)},
    }
    protocol_plan = SimpleNamespace(
        target_type="binary_probability",
        split_constraint="stratified",
        required_manifest_evidence=[
            "split_policy",
            "features_used",
            "leakage_controls",
            "probability_range",
            "calibration_or_ranking_check",
            "prediction_summary",
        ],
    )
    stage_agent_plan = SimpleNamespace(
        required_manifest_key="stage_agent_outputs",
        required_agent_ids=["data_auditor", "split_guardian", "modeling_specialist", "verifier_finalizer"],
    )

    report = repair_submission_layers(
        final,
        contract,
        tmp_path / "artifacts",
        attempt=3,
        verified_evidence=evidence,
        route=route_task(contract, None),
        protocol_plan=protocol_plan,
        stage_agent_plan=stage_agent_plan,
        prefer_scaffold=True,
        reason="quality_scaffold_repair",
    )

    validation = validate_submission(final, contract, reject_sample_target_copy=True)
    quality = evaluate_submission_quality(
        final,
        contract,
        route_task(contract, None),
        validation,
        protocol_plan=protocol_plan,
        stage_agent_plan=stage_agent_plan,
        verified_evidence=evidence,
    )
    manifest = Path(report.manifest_path) if report.manifest_path else None
    assert report.stage == "scaffold_adopted"
    assert report.manifest_stage == "improved"
    assert manifest is not None and manifest.exists()
    assert quality.passed, quality.errors
    assert "manifest_validation_metric_harness_verified" in quality.checks


def test_candidate_audit_records_low_information_rejection_before_scaffold(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    final = tmp_path / "workspace" / "input" / "final_submission.csv"
    final.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "target": [1, 1, 1]}).to_csv(final, index=False)
    recipe_dir = tmp_path / "artifacts" / "verified" / "recipe_proxy"
    recipe_dir.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "target": [0.2, 0.4, 0.8]}).to_csv(recipe_dir / "final_submission.csv", index=False)
    evidence = SimpleNamespace(artifact_paths={"recipe_proxy_dir": str(recipe_dir)})

    report = repair_submission_layers(final, contract, tmp_path / "artifacts", attempt=4, verified_evidence=evidence, allow_scaffold=True)

    assert report.stage == "scaffold_adopted"
    low_info = [ev for ev in report.candidate_evaluations if ev["kind"] == "agent_final"][0]
    assert any("constant_metric_uninformative" in reason or "probability_hard_labels" in reason for reason in low_info["rejection_reasons"])


def test_validation_repair_does_not_scaffold_by_default(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    final = tmp_path / "workspace" / "input" / "final_submission.csv"
    final.parent.mkdir(parents=True)
    pd.read_csv(contract.sample_submission_path).to_csv(final, index=False)
    recipe_dir = tmp_path / "artifacts" / "verified" / "recipe_proxy"
    recipe_dir.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.7, 0.2]}).to_csv(recipe_dir / "final_submission.csv", index=False)
    evidence = SimpleNamespace(artifact_paths={"recipe_proxy_dir": str(recipe_dir)})

    report = repair_submission_layers(final, contract, tmp_path / "artifacts", attempt=5, verified_evidence=evidence)

    assert report.stage == "unrepaired"
    assert pd.read_csv(final)["target"].tolist() == [0.5, 0.5, 0.5]
