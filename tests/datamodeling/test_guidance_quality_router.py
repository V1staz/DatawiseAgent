from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datawiseagent.harness.datamodeling.contracts import build_contract
from datawiseagent.harness.datamodeling.data_availability import inspect_data_availability
from datawiseagent.harness.datamodeling.guidance import load_guidance_pack
from datawiseagent.harness.datamodeling.profiler import build_profile
from datawiseagent.harness.datamodeling.protocol import build_protocol_plan
from datawiseagent.harness.datamodeling.quality import evaluate_submission_quality
from datawiseagent.harness.datamodeling.repair_planner import plan_repair_actions
from datawiseagent.harness.datamodeling.skill_cards import get_skill_cards
from datawiseagent.harness.datamodeling.stage_agents import build_stage_agent_plan
from datawiseagent.harness.datamodeling.submission import validate_submission
from datawiseagent.harness.datamodeling.task_router import route_task
from datawiseagent.harness.datamodeling.verified_evidence import VerifiedModelingEvidence
from tests.datamodeling.test_contracts_and_submission import make_benchmark


def _write_manifest(input_dir: Path, *, stage: str = "improved", value: float = 0.75, baseline: float = 0.5, trained_rows: int = 4) -> None:
    (input_dir / "submission_manifest.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "modeling_method": "toy public-data model",
                "selected_reason": "validation beats fallback",
                "validation_metric_name": "roc_auc_proxy",
                "metric_direction": "maximize",
                "validation_metric_value": value,
                "baseline_metric_value": baseline,
                "trained_rows": trained_rows,
                "split_policy": "stratified holdout",
                "features_used": ["num", "cat"],
                "leakage_controls": ["public train only", "fold-local preprocessing"],
                "probability_range": [0.2, 0.8],
                "calibration_or_ranking_check": "validation ranking beats constant fallback",
                "prediction_summary": {"target": {"nunique": 2, "min": 0.2, "max": 0.8}},
                "stage_agent_outputs": {
                    "data_auditor": {"schema_summary": "checked"},
                    "split_guardian": {"split_policy": "stratified holdout"},
                    "feature_builder": {"features_used": ["num", "cat"]},
                    "modeling_specialist": {"validation_metric_value": value},
                    "verifier_finalizer": {"final_submission": "read back"},
                    "probability_calibrator": {"probability_range": [0.2, 0.8]},
                },
            }
        ),
        encoding="utf-8",
    )


def test_guidance_pack_is_layered_and_route_selective() -> None:
    pack = load_guidance_pack()
    excerpt = pack.render_excerpt(["all", "tabular_binary_classification", "tabular"])

    assert "AGENTS.md" in excerpt
    assert "SKILLS.md" in excerpt
    assert "MCP_TOOLS.md" in excerpt
    assert "dm.lifecycle.final_requires_manifest" in excerpt
    assert "dm.binary.probabilities" in excerpt
    assert len(excerpt) < 3000


def test_router_selects_binary_probability_family(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")

    route = route_task(contract)

    assert route.family == "tabular_binary_classification"
    assert "manifest_present" in route.quality_checks
    assert "hard_labels_for_probability_metric" in route.forbidden_shortcuts
    assert "schema_first_submission" in route.skill_card_ids
    assert "classification_probability_modeling" in route.skill_card_ids
    assert get_skill_cards(route.skill_card_ids)[0].id == "schema_first_submission"


def test_protocol_plan_binary_probability_has_common_and_specific_stages(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    profile = build_profile(contract)
    route = route_task(contract, profile)

    protocol = build_protocol_plan(contract, profile, route)

    assert protocol.target_type == "binary_probability"
    assert protocol.split_constraint == "stratified"
    assert "tabular" in protocol.data_structure
    assert "choose_leakage_aware_split_policy" in protocol.common_stages
    assert "emit_probabilities_not_hard_labels" in protocol.task_specific_stages
    assert "calibration_or_ranking_check" in protocol.required_manifest_evidence
    assert "image_multimodal_modeling" in protocol.deferred_task_families


def test_protocol_plan_time_series_multi_output_prioritizes_time_split(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path, "toy-forecast")
    data_dir = root / "data" / "data_resplit" / "toy-forecast"
    task_dir = root / "data" / "task"
    eval_dir = root / "evaluation"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "store": ["a", "a", "b", "b"],
            "sales": [10.0, 12.0, 20.0, 23.0],
            "visits": [100.0, 110.0, 200.0, 210.0],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [5, 6],
            "date": ["2024-01-05", "2024-01-05"],
            "store": ["a", "b"],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "sales": [0.0, 0.0], "visits": [0.0, 0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)
    (task_dir / "toy-forecast.txt").write_text("Forecast future store sales and visits. Evaluation uses RMSE.")
    (eval_dir / "toy-forecast_eval.py").write_text("from sklearn.metrics import mean_squared_error\nperformance = mean_squared_error(y, p)\n")
    contract = build_contract(root, "toy-forecast")
    profile = build_profile(contract)
    route = route_task(contract, profile)

    protocol = build_protocol_plan(contract, profile, route)

    assert protocol.target_type == "time_series_multi_output_forecasting"
    assert protocol.split_constraint == "time_ordered_or_group_aware"
    assert "datetime" in protocol.data_structure
    assert "multi_output" in protocol.data_structure
    assert "create_lag_rolling_calendar_features_without_future_leakage" in protocol.task_specific_stages
    assert "future_leakage_checks" in protocol.required_manifest_evidence


def test_data_availability_detects_git_lfs_pointer_and_protocol_blocks_modeling(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path, "toy-lfs")
    data_dir = root / "data" / "data_resplit" / "toy-lfs"
    pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123\nsize 123456\n"
    (data_dir / "train.csv").write_text(pointer, encoding="utf-8")
    (data_dir / "test.csv").write_text(pointer, encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.5\n", encoding="utf-8")
    contract = build_contract(root, "toy-lfs")
    profile = build_profile(contract)

    availability = inspect_data_availability(contract)
    route = route_task(contract, profile, availability=availability)
    protocol = build_protocol_plan(contract, profile, route, availability=availability)

    assert not availability.usable_for_modeling
    assert availability.lfs_pointer_files == ["train", "test"]
    assert route.family == "data_unavailable_or_pointer"
    assert "public_data_availability_guard" in route.skill_card_ids
    assert protocol.target_type == "data_unavailable_or_pointer"
    assert "lfs_pointer" in protocol.data_structure
    assert "avoid_text_or_tabular_misrouting_from_pointer_bytes" in protocol.task_specific_stages


def test_stage_agent_plan_adds_common_and_probability_specialist_agents(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    profile = build_profile(contract)
    route = route_task(contract, profile)
    protocol = build_protocol_plan(contract, profile, route)

    stage_plan = build_stage_agent_plan(protocol)

    assert stage_plan.orchestration_mode == "single_session_internal_stage_roles"
    assert stage_plan.required_manifest_key == "stage_agent_outputs"
    assert "data_auditor" in stage_plan.required_agent_ids
    assert "split_guardian" in stage_plan.required_agent_ids
    assert "probability_calibrator" in stage_plan.required_agent_ids


def test_protocol_quality_requires_manifest_evidence_when_protocol_plan_is_supplied(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    profile = build_profile(contract)
    route = route_task(contract, profile)
    protocol = build_protocol_plan(contract, profile, route)
    stage_plan = build_stage_agent_plan(protocol)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(sub, index=False)
    (tmp_path / "submission_manifest.json").write_text(
        json.dumps(
            {
                "stage": "improved",
                "modeling_method": "toy model",
                "selected_reason": "validation beats fallback",
                "validation_metric_name": "roc_auc_proxy",
                "metric_direction": "maximize",
                "validation_metric_value": 0.75,
                "baseline_metric_value": 0.5,
                "trained_rows": 4,
                "prediction_summary": {"target": {"nunique": 2}},
            }
        ),
        encoding="utf-8",
    )
    validation = validate_submission(sub, contract)

    quality = evaluate_submission_quality(sub, contract, route, validation, require_manifest=True, protocol_plan=protocol, stage_agent_plan=stage_plan)

    assert not quality.passed
    assert any(err.startswith("protocol_manifest_missing_required_evidence") for err in quality.errors)
    assert "protocol_probability_manifest_missing:probability_range,calibration_or_ranking_check" in quality.errors
    assert "stage_agent_manifest_missing:stage_agent_outputs" in quality.errors
    actions = plan_repair_actions(contract, validation, quality=quality, route=route, protocol_plan=protocol)
    assert "repair_protocol_manifest_evidence" in {action.id for action in actions}


def test_protocol_quality_passes_with_protocol_and_stage_agent_evidence(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    profile = build_profile(contract)
    route = route_task(contract, profile)
    protocol = build_protocol_plan(contract, profile, route)
    stage_plan = build_stage_agent_plan(protocol)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(sub, index=False)
    _write_manifest(tmp_path)
    validation = validate_submission(sub, contract)

    quality = evaluate_submission_quality(sub, contract, route, validation, require_manifest=True, protocol_plan=protocol, stage_agent_plan=stage_plan)

    assert quality.passed
    assert "quality_gate_passed" in quality.checks
    assert quality.summary["validation_metric_trust"] == "agent_attested"


def test_quality_uses_harness_verified_evidence_when_manifest_matches(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    profile = build_profile(contract)
    route = route_task(contract, profile)
    protocol = build_protocol_plan(contract, profile, route)
    stage_plan = build_stage_agent_plan(protocol)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(sub, index=False)
    _write_manifest(tmp_path, value=0.75, baseline=0.5)
    validation = validate_submission(sub, contract)
    evidence = VerifiedModelingEvidence(
        status="ok",
        metric_name="roc_auc",
        metric_direction="maximize",
        baseline_metric_value=0.5,
        best_candidate_metric_value=0.75,
        best_candidate_name="logistic_regression",
        validation_metric_trust="harness_verified",
    )

    quality = evaluate_submission_quality(
        sub,
        contract,
        route,
        validation,
        require_manifest=True,
        protocol_plan=protocol,
        stage_agent_plan=stage_plan,
        verified_evidence=evidence,
    )

    assert quality.passed
    assert quality.summary["validation_metric_trust"] == "harness_verified"
    assert "verified_modeling_evidence_available" in quality.checks
    assert "manifest_metrics_match_verified_evidence" in quality.checks
    assert "manifest_validation_metric_harness_verified" in quality.checks
    assert quality.verified_evidence["modeling_evidence"]["best_candidate_metric_value"] == 0.75


def test_quality_rejects_manifest_metric_conflict_with_verified_evidence(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    profile = build_profile(contract)
    route = route_task(contract, profile)
    protocol = build_protocol_plan(contract, profile, route)
    stage_plan = build_stage_agent_plan(protocol)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(sub, index=False)
    _write_manifest(tmp_path, value=0.75, baseline=0.5)
    validation = validate_submission(sub, contract)
    evidence = VerifiedModelingEvidence(
        status="ok",
        metric_name="roc_auc",
        metric_direction="maximize",
        baseline_metric_value=0.55,
        best_candidate_metric_value=0.6,
        best_candidate_name="dummy",
        validation_metric_trust="harness_verified",
    )

    quality = evaluate_submission_quality(
        sub,
        contract,
        route,
        validation,
        require_manifest=True,
        protocol_plan=protocol,
        stage_agent_plan=stage_plan,
        verified_evidence=evidence,
    )

    assert not quality.passed
    assert "verified_evidence_metric_conflict:validation_metric_value" in quality.errors
    assert "verified_evidence_metric_conflict:baseline_metric_value" in quality.errors
    actions = plan_repair_actions(contract, validation, quality=quality, route=route, protocol_plan=protocol)
    assert "repair_verified_evidence_conflict" in {action.id for action in actions}


def test_quality_allows_explicit_agent_additional_validation_source(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    profile = build_profile(contract)
    route = route_task(contract, profile)
    protocol = build_protocol_plan(contract, profile, route)
    stage_plan = build_stage_agent_plan(protocol)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(sub, index=False)
    _write_manifest(tmp_path, value=0.75, baseline=0.5)
    manifest_path = tmp_path / "submission_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["validation_source"] = "agent_additional_validation"
    payload["validation_strategy"] = "agent reran public-data holdout"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    validation = validate_submission(sub, contract)
    evidence = VerifiedModelingEvidence(
        status="ok",
        metric_name="roc_auc",
        metric_direction="maximize",
        baseline_metric_value=0.55,
        best_candidate_metric_value=0.6,
        validation_metric_trust="harness_verified",
    )

    quality = evaluate_submission_quality(
        sub,
        contract,
        route,
        validation,
        require_manifest=True,
        protocol_plan=protocol,
        stage_agent_plan=stage_plan,
        verified_evidence=evidence,
    )

    assert quality.passed
    assert quality.summary["validation_metric_trust"] == "agent_attested"
    assert "verified_modeling_evidence_bypassed_by_agent_additional_validation" in quality.warnings


def test_protocol_quality_rejects_manifest_prediction_summary_lies(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    profile = build_profile(contract)
    route = route_task(contract, profile)
    protocol = build_protocol_plan(contract, profile, route)
    stage_plan = build_stage_agent_plan(protocol)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(sub, index=False)
    _write_manifest(tmp_path)
    manifest_path = tmp_path / "submission_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["prediction_summary"]["target"]["nunique"] = 99
    payload["probability_range"] = [0.0, 1.0]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    validation = validate_submission(sub, contract)

    quality = evaluate_submission_quality(sub, contract, route, validation, require_manifest=True, protocol_plan=protocol, stage_agent_plan=stage_plan)

    assert not quality.passed
    assert "manifest_prediction_summary_mismatch:target.nunique" in quality.errors
    assert "manifest_probability_range_mismatch" in quality.errors


def test_router_selects_text_modeling_family(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path, "toy-text")
    data_dir = root / "data" / "data_resplit" / "toy-text"
    task_dir = root / "data" / "task"
    eval_dir = root / "evaluation"
    pd.DataFrame({"id": [1, 2], "comment_text": ["good movie", "bad movie"], "target": [1, 0]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [3], "comment_text": ["good" ]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [3], "target": [0.5]}).to_csv(data_dir / "sample_submission.csv", index=False)
    (task_dir / "toy-text.txt").write_text("Evaluation uses ROC AUC over comment text classification.")
    (eval_dir / "toy-text_eval.py").write_text("from sklearn.metrics import roc_auc_score\nperformance = roc_auc_score(y, p)\n")
    contract = build_contract(root, "toy-text")

    route = route_task(contract)

    assert route.family == "text_modeling"
    assert "text_modeling" in route.required_skills
    assert "text_feature_baseline" in route.skill_card_ids


def test_router_facets_keep_probability_guidance_for_multi_output_probability(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path, "toy-multilabel")
    data_dir = root / "data" / "data_resplit" / "toy-multilabel"
    task_dir = root / "data" / "task"
    eval_dir = root / "evaluation"
    pd.DataFrame({"id": [1, 2, 3, 4], "x": [0, 1, 2, 3], "a": [0, 1, 0, 1], "b": [1, 0, 1, 0]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [5, 6], "x": [1, 2]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "a": [0.5, 0.5], "b": [0.5, 0.5]}).to_csv(data_dir / "sample_submission.csv", index=False)
    (task_dir / "toy-multilabel.txt").write_text("Each target is evaluated with ROC AUC.")
    (eval_dir / "toy-multilabel_eval.py").write_text("from sklearn.metrics import roc_auc_score\nperformance = roc_auc_score(y, p)\n")
    contract = build_contract(root, "toy-multilabel")

    route = route_task(contract, build_profile(contract))

    assert route.family == "multi_output"
    assert "tabular_binary_classification" in route.facets
    assert "classification_probability_modeling" in route.skill_card_ids


def test_quality_gate_requires_manifest_for_valid_csv(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    route = route_task(contract)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(sub, index=False)
    validation = validate_submission(sub, contract)

    quality = evaluate_submission_quality(sub, contract, route, validation, require_manifest=True)

    assert validation.passed
    assert not quality.passed
    assert "manifest_missing" in quality.errors


def test_quality_gate_passes_varied_improved_submission_with_manifest(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    route = route_task(contract)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(sub, index=False)
    _write_manifest(tmp_path)
    validation = validate_submission(sub, contract)

    quality = evaluate_submission_quality(sub, contract, route, validation, require_manifest=True)

    assert quality.passed
    assert "manifest_validation_beats_baseline" in quality.checks
    assert quality.summary["all_target_columns_constant"] is False


def test_quality_gate_rejects_improved_manifest_without_validation_comparison(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    route = route_task(contract)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(sub, index=False)
    (tmp_path / "submission_manifest.json").write_text(
        json.dumps(
            {
                "stage": "improved",
                "modeling_method": "toy model",
                "selected_reason": "looked better",
                "trained_rows": 4,
                "prediction_summary": {"target": {"nunique": 2}},
            }
        ),
        encoding="utf-8",
    )
    validation = validate_submission(sub, contract)

    quality = evaluate_submission_quality(sub, contract, route, validation, require_manifest=True)

    assert not quality.passed
    assert "manifest_validation_comparison_unavailable" in quality.errors


def test_quality_gate_rejects_constant_predictions_without_improvement(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    route = route_task(contract)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [5, 6], "target": [0.5, 0.5]}).to_csv(sub, index=False)
    _write_manifest(tmp_path, value=0.5, baseline=0.5)
    validation = validate_submission(sub, contract, reject_sample_target_copy=False)

    quality = evaluate_submission_quality(sub, contract, route, validation, require_manifest=True)

    assert not quality.passed
    assert "all_target_columns_constant_without_validation_improvement" in quality.errors
    assert "manifest_validation_not_better_than_baseline" in quality.errors


def test_quality_gate_rejects_probability_hard_labels_even_with_improved_manifest(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    route = route_task(contract)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [5, 6], "target": [0, 1]}).to_csv(sub, index=False)
    _write_manifest(tmp_path, value=0.75, baseline=0.5)
    validation = validate_submission(sub, contract)

    quality = evaluate_submission_quality(sub, contract, route, validation, require_manifest=True)

    assert not quality.passed
    assert "probability_metric_predictions_look_like_hard_labels" in quality.errors

    actions = plan_repair_actions(contract, validation, quality=quality, route=route)
    assert actions[0].id == "repair_probability_scores"


def test_quality_gate_rejects_multi_output_constant_targets_with_improved_manifest(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path, "toy-multilabel")
    data_dir = root / "data" / "data_resplit" / "toy-multilabel"
    task_dir = root / "data" / "task"
    eval_dir = root / "evaluation"
    pd.DataFrame({"id": [1, 2, 3, 4], "x": [0.0, 1.0, 2.0, 3.0], "a": [0, 1, 0, 1], "b": [1, 1, 0, 0]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [5, 6], "x": [1.5, 2.5]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "a": [0.5, 0.5], "b": [0.5, 0.5]}).to_csv(data_dir / "sample_submission.csv", index=False)
    (task_dir / "toy-multilabel.txt").write_text("Each target uses ROC AUC.")
    (eval_dir / "toy-multilabel_eval.py").write_text("from sklearn.metrics import roc_auc_score\nperformance = roc_auc_score(y, p)\n")
    contract = build_contract(root, "toy-multilabel")
    route = route_task(contract)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [5, 6], "a": [0.2, 0.8], "b": [0.5, 0.5]}).to_csv(sub, index=False)
    _write_manifest(tmp_path, value=0.75, baseline=0.5)
    validation = validate_submission(sub, contract)

    quality = evaluate_submission_quality(sub, contract, route, validation, require_manifest=True)

    assert route.family == "multi_output"
    assert not quality.passed
    assert "multi_output_target_columns_constant:b" in quality.errors


def test_quality_gate_rejects_multiclass_single_class_even_with_improved_manifest(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path, "toy-labels")
    data_dir = root / "data" / "data_resplit" / "toy-labels"
    task_dir = root / "data" / "task"
    eval_dir = root / "evaluation"
    pd.DataFrame({"id": [1, 2, 3], "x": [0.0, 1.0, 2.0], "label": ["low", "mid", "high"]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [4, 5], "x": [0.5, 1.5]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [4, 5], "label": ["low", "low"]}).to_csv(data_dir / "sample_submission.csv", index=False)
    (task_dir / "toy-labels.txt").write_text("Evaluation uses accuracy_score for label.")
    (eval_dir / "toy-labels_eval.py").write_text("from sklearn.metrics import accuracy_score\nperformance = accuracy_score(y, p)\n")
    contract = build_contract(root, "toy-labels")
    route = route_task(contract)
    sub = tmp_path / "final_submission.csv"
    pd.DataFrame({"id": [4, 5], "label": ["low", "low"]}).to_csv(sub, index=False)
    _write_manifest(tmp_path, value=0.75, baseline=0.5)
    validation = validate_submission(sub, contract, reject_sample_target_copy=False)

    quality = evaluate_submission_quality(sub, contract, route, validation, require_manifest=True)

    assert route.family == "tabular_multiclass_classification"
    assert not quality.passed
    assert "multiclass_predictions_low_class_coverage:label" in quality.errors
