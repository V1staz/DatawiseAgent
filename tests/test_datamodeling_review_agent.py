from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from datawiseagent.harness.datamodeling.contracts import DataModelingContract, MetricSpec, SubmissionSpec
from datawiseagent.harness.datamodeling.quality import QualityReport
from datawiseagent.harness.datamodeling.review_agent import parse_review_response, render_review_prompt
from datawiseagent.harness.datamodeling.submission import SubmissionValidationResult
from datawiseagent.harness.datamodeling.task_router import route_task


def _contract(tmp_path: Path) -> DataModelingContract:
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.5, 0.5]}).to_csv(data / "sample_submission.csv", index=False)
    pd.DataFrame({"id": [1, 2], "x": [10, 20]}).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": [3, 4], "x": [1, 2], "target": [0, 1]}).to_csv(data / "train.csv", index=False)
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
        metric=MetricSpec("roc_auc", "maximize", prediction_kind="probability", probability_output=True, bounded_range=(0.0, 1.0)),
        submission=SubmissionSpec(str(data / "sample_submission.csv"), ["id", "target"], ["id"], ["target"], 2, {"target": "float64"}),
        task_kinds=["tabular_classification"],
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


def test_parse_review_response_blocks_on_major_finding() -> None:
    response = {
        "response_content": '''prefix
<review_decision_json>
{
  "passed": false,
  "confidence": "high",
  "summary": "AUC output is hard labels.",
  "checklist": {"prediction_informative": false},
  "findings": [
    {"severity": "major", "category": "submission_quality", "evidence": "target has only 0/1", "repair_instruction": "write continuous predict_proba scores"}
  ]
}
</review_decision_json>
'''
    }

    decision = parse_review_response(response)

    assert not decision.passed
    assert decision.blocks_finalization
    assert decision.findings[0].repair_instruction == "write continuous predict_proba scores"


def test_parse_review_response_parse_failure_is_non_blocking() -> None:
    decision = parse_review_response({"response_content": "not json"})

    assert not decision.passed
    assert not decision.blocks_finalization
    assert decision.confidence == "parse_failed_inconclusive"
    assert decision.parse_error is not None


def test_render_review_prompt_contains_read_only_schema_and_common_risks(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    route = route_task(contract, None)
    prompt = render_review_prompt(
        contract,
        SubmissionValidationResult(True),
        QualityReport(True, route_family=route.family),
        attempt=0,
        submission_path="./input/final_submission.csv",
        route=route,
        protocol_plan=SimpleNamespace(to_dict=lambda: {"target_type": "binary_probability"}),
        stage_agent_plan=SimpleNamespace(to_dict=lambda: {"required_manifest_key": "stage_agent_outputs"}),
    )

    assert "read-only review pass" in prompt
    assert "probability metrics receiving hard labels" in prompt
    assert "review_decision_json" in prompt
    assert "./input/final_submission.csv" in prompt
