"""Contract parsing for DSBench/DataModeling Kaggle-style tasks.

The contract layer is intentionally label-blind: it reads only public task
metadata, task descriptions, train/test/sample_submission schemas, and local
evaluation script text. It must not read benchmark answers or GT performance
files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import json
import re
import warnings

import pandas as pd

MetricDirection = Literal["maximize", "minimize"]
TaskKind = Literal[
    "tabular_classification",
    "tabular_regression",
    "text_regression",
    "text_classification",
    "span_extraction",
    "time_series_forecasting",
    "multi_target",
    "cellular_automaton",
    "unknown",
]


@dataclass(slots=True)
class MetricSpec:
    name: str = "unknown"
    direction: MetricDirection = "maximize"
    prediction_kind: str = "auto"
    requires_non_negative: bool = False
    probability_output: bool = False
    bounded_range: tuple[float, float] | None = None


@dataclass(slots=True)
class SubmissionSpec:
    sample_path: str
    columns: list[str]
    id_columns: list[str]
    target_columns: list[str]
    row_count: int
    target_dtypes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DataModelingContract:
    task_id: int | None
    name: str
    benchmark_root: str
    task_description_path: str
    data_dir: str
    train_path: str
    test_path: str
    sample_submission_path: str
    eval_script_path: str | None
    metric: MetricSpec
    submission: SubmissionSpec
    task_kinds: list[TaskKind]
    target_columns: list[str]
    id_columns: list[str]
    feature_columns: list[str]
    text_columns: list[str]
    datetime_columns: list[str]
    group_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    risk_flags: list[str]
    recipe_hints: list[str]
    description_excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_METRIC_PATTERNS: list[tuple[str, MetricSpec, tuple[str, ...]]] = [
    ("jaccard", MetricSpec("jaccard", "maximize", prediction_kind="text"), ("jaccard", "selected_text", "selected text")),
    ("smape", MetricSpec("smape", "minimize", prediction_kind="regression", requires_non_negative=True), ("smape", "symmetric mean absolute percentage")),
    ("rmsle", MetricSpec("rmsle", "minimize", prediction_kind="regression", requires_non_negative=True), ("root mean squared logarithmic", "rmsle", "mean_squared_log_error")),
    ("rmse", MetricSpec("rmse", "minimize", prediction_kind="regression"), ("root mean squared error", "rmse")),
    ("mse", MetricSpec("mse", "minimize", prediction_kind="regression"), ("mean squared error", "mse")),
    ("mae", MetricSpec("mae", "minimize", prediction_kind="regression"), ("mean absolute error", "mae")),
    ("median_absolute_error", MetricSpec("median_absolute_error", "minimize", prediction_kind="regression"), ("median absolute error", "median_absolute_error")),
    ("r2", MetricSpec("r2", "maximize", prediction_kind="regression"), ("r2_score", "r-squared", "r2 score")),
    ("normalized_gini", MetricSpec("normalized_gini", "maximize", prediction_kind="probability", probability_output=True, bounded_range=(0.0, 1.0)), ("normalized gini", "normalized_gini")),
    ("roc_auc", MetricSpec("roc_auc", "maximize", prediction_kind="probability", probability_output=True, bounded_range=(0.0, 1.0)), ("auc roc", "roc auc", "area under the roc", "roc_auc_score")),
    ("log_loss", MetricSpec("log_loss", "minimize", prediction_kind="probability", probability_output=True, bounded_range=(0.0, 1.0)), ("log loss", "log_loss")),
    ("average_precision", MetricSpec("average_precision", "maximize", prediction_kind="probability", probability_output=True, bounded_range=(0.0, 1.0)), ("average_precision", "average precision")),
    ("qwk", MetricSpec("quadratic_weighted_kappa", "maximize", prediction_kind="class"), ("quadratic weighted kappa", "cohen_kappa_score")),
    ("f1", MetricSpec("f1", "maximize", prediction_kind="class"), ("f1_score", " f1 ", "f1-score")),
    ("accuracy", MetricSpec("accuracy", "maximize", prediction_kind="class"), ("accuracy_score", "classification accuracy", "accuracy")),
    ("mpa_at_3", MetricSpec("mpa_at_3", "maximize", prediction_kind="top_k_class"), ("mpa_at_3", "map@3", "top 3")),
    ("pearson", MetricSpec("pearson", "maximize", prediction_kind="regression", bounded_range=(0.0, 1.0)), ("pearson correlation", "pearson", "correlation coefficient")),
]

_ID_NAME_HINTS = ("id", "index", "forecastid", "textid", "machineidentifier", "passengerid")
_TEXT_NAME_HINTS = ("text", "tweet", "comment", "question", "answer", "essay", "anchor", "target", "phrase", "title", "description", "excerpt")
_DATE_NAME_HINTS = ("date", "time", "timestamp", "datetime")
_GROUP_NAME_HINTS = (
    "group",
    "customer",
    "user",
    "store",
    "shop",
    "item",
    "patient",
    "subject",
    "session",
    "series",
    "site",
    "warehouse",
    "agency",
    "sku",
)


def load_tasks(benchmark_root: str | Path) -> list[dict[str, Any]]:
    root = Path(benchmark_root)
    tasks: list[dict[str, Any]] = []
    with (root / "data.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                item.setdefault("id", i)
                tasks.append(item)
    return tasks


def find_task(benchmark_root: str | Path, task: str | int) -> tuple[int, dict[str, Any]]:
    tasks = load_tasks(benchmark_root)
    if isinstance(task, int) or str(task).isdigit():
        idx = int(task)
        return idx, tasks[idx]
    for i, item in enumerate(tasks):
        if item.get("name") == task:
            return i, item
    raise KeyError(f"Unknown DataModeling task: {task}")


def build_contract(benchmark_root: str | Path, task: str | int) -> DataModelingContract:
    root = Path(benchmark_root)
    task_id, item = find_task(root, task)
    name = item["name"]
    data_dir = root / "data" / "data_resplit" / name
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = _find_sample_submission(data_dir)
    description_path = root / "data" / "task" / f"{name}.txt"
    eval_script = root / "evaluation" / f"{name}_eval.py"

    description = _safe_read(description_path)
    eval_text = _safe_read(eval_script)
    sample = pd.read_csv(sample_path, nrows=25)
    train_head = pd.read_csv(train_path, nrows=512) if train_path.exists() else pd.DataFrame()
    test_head = pd.read_csv(test_path, nrows=512) if test_path.exists() else pd.DataFrame()

    id_columns = _infer_id_columns(sample, test_head)
    target_columns = [col for col in sample.columns if col not in id_columns]
    if not target_columns and len(sample.columns) > 1:
        target_columns = list(sample.columns[1:])
    train_target_columns = [col for col in target_columns if col in train_head.columns]
    if not train_target_columns:
        train_target_columns = _infer_train_targets(train_head, sample, description) or target_columns

    excluded = set(id_columns) | set(train_target_columns)
    feature_columns = [col for col in train_head.columns if col not in excluded and col in set(test_head.columns)]
    feature_frame = train_head[feature_columns] if feature_columns else train_head
    text_columns = _infer_text_columns(feature_frame)
    datetime_columns = _infer_datetime_columns(feature_frame)
    group_columns = _infer_group_columns(feature_frame, description, name)
    numeric_columns = [col for col in feature_columns if col in train_head and pd.api.types.is_numeric_dtype(train_head[col])]
    categorical_columns = [col for col in feature_columns if col not in set(text_columns) | set(datetime_columns) | set(numeric_columns)]
    metric = infer_metric(description, eval_text)
    task_kinds = infer_task_kinds(name, description, metric, target_columns, train_head, text_columns, datetime_columns)
    risk_flags = infer_risk_flags(name, description, train_head, feature_columns, target_columns, metric, task_kinds)
    hints = infer_recipe_hints(name, metric, task_kinds, risk_flags, text_columns, datetime_columns)

    return DataModelingContract(
        task_id=task_id,
        name=name,
        benchmark_root=str(root),
        task_description_path=str(description_path),
        data_dir=str(data_dir),
        train_path=str(train_path),
        test_path=str(test_path),
        sample_submission_path=str(sample_path),
        eval_script_path=str(eval_script) if eval_script.exists() else None,
        metric=metric,
        submission=SubmissionSpec(
            sample_path=str(sample_path),
            columns=list(sample.columns),
            id_columns=id_columns,
            target_columns=target_columns,
            row_count=_csv_row_count(sample_path),
            target_dtypes={col: str(sample[col].dtype) for col in target_columns if col in sample},
        ),
        task_kinds=task_kinds,
        target_columns=train_target_columns,
        id_columns=id_columns,
        feature_columns=feature_columns,
        text_columns=text_columns,
        datetime_columns=datetime_columns,
        group_columns=group_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        risk_flags=risk_flags,
        recipe_hints=hints,
        description_excerpt=_excerpt(description),
    )


def infer_metric(description: str, eval_text: str = "") -> MetricSpec:
    lower_desc = _normalize_text(description)
    from_eval = _metric_from_eval_script(eval_text)
    if from_eval is not None:
        return from_eval
    for _, spec, patterns in _METRIC_PATTERNS:
        if any(pattern in lower_desc for pattern in patterns):
            return spec
    return MetricSpec()


def _metric_from_eval_script(eval_text: str) -> MetricSpec | None:
    """Infer the actually scored metric from evaluator code, ignoring imports.

    DSBench evaluator files often import several unused sklearn metrics.  The
    contract must therefore look at the executable body rather than doing a
    plain full-text search that can confuse, e.g., a normalized-gini evaluator
    with an unused mean_squared_log_error import.
    """

    if not eval_text:
        return None
    body = "\n".join(
        line for line in eval_text.splitlines()
        if not line.lstrip().startswith(("import ", "from "))
    )
    lower = _normalize_text(body)
    if "normalized_gini" in lower or "performance = normalized_gini" in lower:
        return MetricSpec("normalized_gini", "maximize", prediction_kind="probability", probability_output=True, bounded_range=(0.0, 1.0))
    if "jaccard(" in lower or "performance = sum(jaccard" in lower:
        return MetricSpec("jaccard", "maximize", prediction_kind="text")
    if "mpa_at_3" in lower:
        return MetricSpec("mpa_at_3", "maximize", prediction_kind="top_k_class")
    if "log_loss" in lower:
        return MetricSpec("log_loss", "minimize", prediction_kind="probability", probability_output=True, bounded_range=(0.0, 1.0))
    if "roc_auc_score" in lower:
        return MetricSpec("roc_auc", "maximize", prediction_kind="probability", probability_output=True, bounded_range=(0.0, 1.0))
    if "cohen_kappa_score" in lower:
        return MetricSpec("quadratic_weighted_kappa", "maximize", prediction_kind="class")
    if "f1_score" in lower:
        return MetricSpec("f1", "maximize", prediction_kind="class")
    if "accuracy_score" in lower or "== answers[args.value]" in lower:
        return MetricSpec("accuracy", "maximize", prediction_kind="class")
    if "pearsonr" in lower:
        return MetricSpec("pearson", "maximize", prediction_kind="regression", bounded_range=(0.0, 1.0))
    if "smape(" in lower or "performance = smape" in lower:
        return MetricSpec("smape", "minimize", prediction_kind="regression", requires_non_negative=True)
    if "r2_score" in lower:
        return MetricSpec("r2", "maximize", prediction_kind="regression")
    if "mean_squared_log_error" in lower:
        return MetricSpec("rmsle", "minimize", prediction_kind="regression", requires_non_negative=True)
    if "sqrt(mean_squared_error" in lower or "np.sqrt(mean_squared_error" in lower:
        return MetricSpec("rmse", "minimize", prediction_kind="regression")
    if "mean_squared_error" in lower:
        return MetricSpec("mse", "minimize", prediction_kind="regression")
    if "mean_absolute_error" in lower:
        return MetricSpec("mae", "minimize", prediction_kind="regression")
    if "median_absolute_error" in lower:
        return MetricSpec("median_absolute_error", "minimize", prediction_kind="regression")
    return None


def infer_task_kinds(name: str, description: str, metric: MetricSpec, target_columns: list[str], train_head: pd.DataFrame, text_columns: list[str], datetime_columns: list[str]) -> list[TaskKind]:
    lower = _normalize_text(f"{name}\n{description}")
    kinds: list[TaskKind] = []
    if "conway" in lower or "game of life" in lower:
        kinds.append("cellular_automaton")
    if metric.name == "jaccard" or "selected_text" in lower:
        kinds.append("span_extraction")
    if any(token in lower for token in ("forecast", "time series", "public/private split is time", "cumulative")):
        kinds.append("time_series_forecasting")
    if len(target_columns) > 1:
        kinds.append("multi_target")
    if text_columns and "span_extraction" not in kinds:
        kinds.append("text_regression" if metric.prediction_kind == "regression" else "text_classification")
    if metric.prediction_kind in {"probability", "class", "top_k_class"}:
        kinds.append("tabular_classification")
    elif metric.prediction_kind == "regression" or not kinds:
        kinds.append("tabular_regression")
    return _dedupe(kinds)


def infer_risk_flags(name: str, description: str, train_head: pd.DataFrame, feature_columns: list[str], target_columns: list[str], metric: MetricSpec, kinds: list[TaskKind]) -> list[str]:
    lower = _normalize_text(f"{name}\n{description}")
    flags: list[str] = []
    if "dont-overfit" in lower or (len(train_head) and len(train_head) <= 500 and len(feature_columns) >= 100):
        flags.append("small_sample_high_dimensional_overfit_risk")
    if metric.requires_non_negative:
        flags.append("non_negative_predictions_required")
    if metric.probability_output:
        flags.append("probability_predictions_required")
    if metric.prediction_kind == "top_k_class":
        flags.append("top_k_class_labels_required")
    if "time_series_forecasting" in kinds:
        flags.append("time_or_group_split_required")
    if "span_extraction" in kinds:
        flags.append("string_submission_required")
    if "multi_target" in kinds:
        flags.append("multi_target_submission_required")
    if "cellular_automaton" in kinds:
        flags.append("nonstandard_optimization_task")
    if target_columns and any(col.lower() in {"target", "score", "sales"} for col in feature_columns):
        flags.append("possible_target_leakage_column")
    return _dedupe(flags)


def infer_recipe_hints(name: str, metric: MetricSpec, kinds: list[TaskKind], risk_flags: list[str], text_columns: list[str], datetime_columns: list[str]) -> list[str]:
    hints: list[str] = []
    lower = name.lower()
    if "span_extraction" in kinds:
        hints.append("span_extraction_string_baseline")
    if "cellular_automaton" in kinds:
        hints.append("game_of_life_heuristic")
    if "time_series_forecasting" in kinds or datetime_columns:
        hints.append("date_features_and_chronological_validation")
    if "demand-forecasting" in lower:
        hints.append("store_item_seasonal_group_average")
    if "covid19" in lower:
        hints.append("cumulative_forecast_last_observation")
    if metric.name == "rmsle":
        hints.append("log1p_target_nonnegative_clip")
    if metric.probability_output:
        hints.append("predict_proba_not_class_label")
    if metric.prediction_kind == "top_k_class":
        hints.append("top_k_class_labels")
    if text_columns:
        hints.append("tfidf_text_features")
    if "small_sample_high_dimensional_overfit_risk" in risk_flags:
        hints.append("regularized_linear_small_sample")
    if not hints:
        hints.append("tabular_gbdt_default")
    return _dedupe(hints)


def render_modeling_plan_prompt(contract: DataModelingContract) -> str:
    payload = {
        "task": contract.name,
        "metric": asdict(contract.metric),
        "task_kinds": contract.task_kinds,
        "submission_columns": contract.submission.columns,
        "id_columns": contract.id_columns,
        "target_columns": contract.target_columns,
        "feature_columns_sample": contract.feature_columns[:40],
        "text_columns": contract.text_columns,
        "datetime_columns": contract.datetime_columns,
        "group_columns": contract.group_columns,
        "risk_flags": contract.risk_flags,
        "recipe_hints": contract.recipe_hints,
        "description_excerpt": contract.description_excerpt,
    }
    return (
        "You are planning a Kaggle-style data modeling run. Return only JSON with keys: "
        "task_type, metric, cv_strategy, candidate_recipes, leakage_risks, submission_checks, fallback_recipe. "
        "Do not use benchmark labels or ground-truth performance.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _find_sample_submission(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("*sample*submission*.csv")) + sorted(data_dir.glob("submission.csv"))
    if candidates:
        return candidates[0]
    csvs = sorted(data_dir.glob("*.csv"))
    for path in csvs:
        if "sample" in path.name.lower() or "submission" in path.name.lower():
            return path
    raise FileNotFoundError(f"No sample submission found under {data_dir}")


def _infer_id_columns(sample: pd.DataFrame, test_head: pd.DataFrame) -> list[str]:
    if sample.empty:
        return []
    first = sample.columns[0]
    lower_first = first.lower()
    if any(hint == lower_first or hint in lower_first for hint in _ID_NAME_HINTS):
        return [first]
    if first in test_head.columns:
        return [first]
    return [first] if len(sample.columns) > 1 else []


def _infer_train_targets(train_head: pd.DataFrame, sample: pd.DataFrame, description: str) -> list[str]:
    common = [col for col in list(sample.columns)[1:] if col in train_head.columns]
    if common:
        return common
    for col in ["target", "Survived", "Transported", "SalePrice", "sales", "score", "selected_text", "HasDetections"]:
        if col in train_head.columns:
            return [col]
    lower = description.lower()
    for col in train_head.columns:
        if re.search(rf"\b{re.escape(col.lower())}\b", lower) and col.lower() not in _ID_NAME_HINTS:
            return [col]
    return []


def _infer_text_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        series = df[col]
        lower = col.lower()
        if any(hint in lower for hint in _TEXT_NAME_HINTS):
            columns.append(col)
            continue
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sample = series.dropna().astype(str).head(128)
            if not sample.empty and (sample.str.len().median() >= 30 or sample.str.contains(r"\s", regex=True).mean() > 0.6):
                columns.append(col)
    return _dedupe(columns)


def _infer_datetime_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        lower = col.lower()
        if any(hint in lower for hint in _DATE_NAME_HINTS):
            columns.append(col)
            continue
        if pd.api.types.is_object_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(64)
            if len(sample) >= 8:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(sample, errors="coerce", utc=False)
                if parsed.notna().mean() >= 0.85:
                    columns.append(col)
    return _dedupe(columns)


def _infer_group_columns(df: pd.DataFrame, description: str, name: str) -> list[str]:
    columns: list[str] = []
    context = f"{name} {description}".lower()
    for col in df.columns:
        lower = col.lower()
        if lower in _ID_NAME_HINTS:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        unique = int(series.nunique(dropna=True))
        n = int(len(series))
        if unique <= 1 or unique >= max(3, int(n * 0.95)):
            continue
        repeated_entity = unique < n
        name_hint = any(hint == lower or hint in lower for hint in _GROUP_NAME_HINTS)
        context_hint = any(hint in context for hint in _GROUP_NAME_HINTS)
        low_cardinality_entity = unique <= max(20, int(n * 0.4))
        if repeated_entity and (name_hint or (context_hint and low_cardinality_entity)):
            columns.append(col)
    return _dedupe(columns)


def _csv_row_count(path: Path) -> int:
    with path.open("rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def _safe_read(path: Path | None) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path and path.exists() else ""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _excerpt(text: str, limit: int = 2200) -> str:
    return re.sub(r"\s+", " ", text.strip())[:limit]


def _dedupe(items: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
