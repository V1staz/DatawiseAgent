from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from datawiseagent.harness.datamodeling import build_contract, submission_template, validate_submission
from datawiseagent.harness.datamodeling.recipes import _prepare_target, write_submission
from datawiseagent.harness.datamodeling.submission import build_constant_baseline


def make_benchmark(tmp_path: Path, name: str = "toy-binary") -> Path:
    root = tmp_path / "DataModeling"
    data_dir = root / "data" / "data_resplit" / name
    task_dir = root / "data" / "task"
    eval_dir = root / "evaluation"
    data_dir.mkdir(parents=True)
    task_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (root / "data.jsonl").write_text(json.dumps({"name": name, "url": "local", "size": "tiny", "year": 2026}) + "\n")
    pd.DataFrame({"id": [1, 2, 3, 4], "num": [0.1, 0.2, 2.0, 2.5], "cat": ["a", "a", "b", "b"], "target": [0, 0, 1, 1]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [5, 6], "num": [0.15, 2.3], "cat": ["a", "b"]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "target": [0.5, 0.5]}).to_csv(data_dir / "sample_submission.csv", index=False)
    (task_dir / f"{name}.txt").write_text("Submissions are evaluated on ROC AUC. Predict probability for target.")
    (eval_dir / f"{name}_eval.py").write_text("from sklearn.metrics import roc_auc_score\n")
    return root


def test_contract_parses_metric_and_submission_schema(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")

    assert contract.metric.name == "roc_auc"
    assert contract.metric.probability_output is True
    assert contract.id_columns == ["id"]
    assert contract.submission.target_columns == ["target"]
    assert "tabular_classification" in contract.task_kinds
    assert "probability_predictions_required" in contract.risk_flags


def test_submission_validator_rejects_text_task_numeric_prediction(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path, "toy-text")
    task_dir = root / "data" / "task"
    data_dir = root / "data" / "data_resplit" / "toy-text"
    pd.DataFrame({"textID": ["a", "b"], "text": ["very good", "so bad"], "sentiment": ["positive", "negative"], "selected_text": ["good", "bad"]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"textID": ["c"], "text": ["quite good"], "sentiment": ["positive"]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"textID": ["c"], "selected_text": [0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)
    (task_dir / "toy-text.txt").write_text("Evaluation uses word-level Jaccard and selected_text must be quoted text.")
    contract = build_contract(root, "toy-text")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"textID": ["c"], "selected_text": [0.0]}).to_csv(bad, index=False)

    result = validate_submission(bad, contract)

    assert not result.passed
    assert any("text_target_looks_numeric" in err for err in result.errors)


def test_submission_validator_accepts_valid_probability_submission(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    sub = tmp_path / "sub.csv"
    pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(sub, index=False)

    result = validate_submission(sub, contract)

    assert result.passed
    assert "columns_exact" in result.checks


def test_submission_validator_rejects_probability_outside_metric_range(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    sub = tmp_path / "sub.csv"
    pd.DataFrame({"id": [5, 6], "target": [-0.2, 1.4]}).to_csv(sub, index=False)

    result = validate_submission(sub, contract)

    assert not result.passed
    assert any("prediction_outside_metric_range" in err for err in result.errors)


def test_submission_template_prefers_resplit_test_ids_when_sample_row_count_differs(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    data_dir = root / "data" / "data_resplit" / "toy-binary"
    pd.DataFrame({"id": [5, 6, 7], "num": [0.15, 2.3, 3.0], "cat": ["a", "b", "b"]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0.5, 0.5]}).to_csv(data_dir / "sample_submission.csv", index=False)
    contract = build_contract(root, "toy-binary")

    template = submission_template(contract)

    assert list(template["id"]) == [5, 6, 7]
    assert list(template.columns) == ["id", "target"]


def test_eval_script_metric_body_wins_over_unused_imports(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    eval_script = root / "evaluation" / "toy-binary_eval.py"
    eval_script.write_text(
        "from sklearn.metrics import mean_squared_log_error\n"
        "def normalized_gini(actual, pred):\n"
        "    return 0.0\n"
        "performance = normalized_gini(answers['target'], predictions['target'])\n"
    )

    contract = build_contract(root, "toy-binary")

    assert contract.metric.name == "normalized_gini"
    assert contract.metric.probability_output is True


def test_submission_validator_accepts_categorical_class_labels(tmp_path: Path) -> None:
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
    sub = tmp_path / "labels.csv"
    pd.DataFrame({"id": [4, 5], "label": ["low", "high"]}).to_csv(sub, index=False)

    result = validate_submission(sub, contract)

    assert result.passed
    assert "class_target_dtype_ok:label" in result.checks


def test_categorical_class_baseline_stays_valid(tmp_path: Path) -> None:
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
    baseline = tmp_path / "baseline.csv"

    build_constant_baseline(contract, baseline)
    result = validate_submission(baseline, contract, reject_sample_target_copy=False)

    assert result.passed
    assert list(pd.read_csv(baseline)["label"]) == ["low", "low"]


def test_multilabel_auc_submission_probabilities_are_not_row_normalized(tmp_path: Path) -> None:
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
    out = tmp_path / "multi.csv"

    write_submission(np.array([[0.8, 0.7], [0.2, 0.9]]), contract, out)

    sub = pd.read_csv(out)
    assert list(sub["a"]) == [0.8, 0.2]
    assert list(sub["b"]) == [0.7, 0.9]


def test_submission_validator_rejects_any_sample_copied_target_when_requested(tmp_path: Path) -> None:
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
    partial_copy = tmp_path / "partial_copy.csv"
    pd.DataFrame({"id": [5, 6], "a": [0.5, 0.5], "b": [0.2, 0.8]}).to_csv(partial_copy, index=False)

    result = validate_submission(partial_copy, contract, reject_sample_target_copy=True)

    assert not result.passed
    assert "target_identical_to_sample_submission:a" in result.errors


def test_rmsle_target_is_not_prelogged_before_cv_metric(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path, "toy-rmsle")
    data_dir = root / "data" / "data_resplit" / "toy-rmsle"
    task_dir = root / "data" / "task"
    eval_dir = root / "evaluation"
    pd.DataFrame({"id": [1, 2, 3], "x": [0.0, 1.0, 2.0], "target": [1.0, 3.0, 7.0]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [4], "x": [3.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [4], "target": [0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)
    (task_dir / "toy-rmsle.txt").write_text("Evaluation is RMSLE.")
    (eval_dir / "toy-rmsle_eval.py").write_text("from sklearn.metrics import mean_squared_log_error\nperformance = mean_squared_log_error(y, p)\n")
    contract = build_contract(root, "toy-rmsle")

    y, transform = _prepare_target(pd.Series([1.0, 3.0, 7.0]), contract)

    assert list(y) == [1.0, 3.0, 7.0]
    assert transform.kind == "identity"
