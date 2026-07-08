"""Leakage-aware CV and metric helpers for DataModeling recipes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    roc_auc_score,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

try:  # pragma: no cover - depends on sklearn version
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore[assignment]

from .contracts import DataModelingContract, MetricSpec


@dataclass(slots=True)
class CVResult:
    candidate_name: str
    metric_name: str
    raw_scores: list[float]
    selection_scores: list[float]
    mean_raw_score: float
    mean_selection_score: float
    std_raw_score: float
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_candidate_cv(
    estimator: Any,
    X: Any,
    y: pd.Series | np.ndarray,
    contract: DataModelingContract,
    *,
    candidate_name: str,
    n_splits: int = 3,
    random_state: int = 42,
    groups: pd.Series | np.ndarray | None = None,
    group_column: str | None = None,
) -> CVResult:
    y_arr = np.asarray(y)
    groups_arr = _as_group_array(groups, len(y_arr))
    raw_scores: list[float] = []
    selection_scores: list[float] = []
    warnings: list[str] = []
    splits, split_metadata = make_cv_split_plan(
        X,
        y_arr,
        contract,
        n_splits=n_splits,
        random_state=random_state,
        groups=groups_arr,
        group_column=group_column,
    )
    for fold, (train_idx, valid_idx) in enumerate(splits):
        model = _clone_estimator(estimator)
        X_train, X_valid = _take_rows(X, train_idx), _take_rows(X, valid_idx)
        y_train, y_valid = y_arr[train_idx], y_arr[valid_idx]
        try:
            model.fit(X_train, y_train)
            pred = predictions_for_metric(model, X_valid, contract.metric)
            raw = metric_value(y_valid, pred, contract.metric)
            raw_scores.append(float(raw))
            selection_scores.append(selection_value(raw, contract.metric))
        except Exception as exc:
            warnings.append(f"fold_{fold}_failed:{type(exc).__name__}:{str(exc)[:160]}")
    if not raw_scores:
        raw_scores = [math.inf if contract.metric.direction == "minimize" else 0.0]
        selection_scores = [-math.inf]
    return CVResult(
        candidate_name=candidate_name,
        metric_name=contract.metric.name,
        raw_scores=raw_scores,
        selection_scores=selection_scores,
        mean_raw_score=float(np.mean(raw_scores)),
        mean_selection_score=float(np.mean(selection_scores)),
        std_raw_score=float(np.std(raw_scores)),
        warnings=warnings,
        metadata={
            **split_metadata,
            "folds": len(splits),
            "fold_sizes": [
                {"train": int(len(train_idx)), "valid": int(len(valid_idx))}
                for train_idx, valid_idx in splits
            ],
            "preprocessing_scope": "fold_local_when_estimator_is_pipeline",
        },
    )


def make_cv_splits(X: Any, y: np.ndarray, contract: DataModelingContract, *, n_splits: int = 3, random_state: int = 42) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    splits, _metadata = make_cv_split_plan(X, y, contract, n_splits=n_splits, random_state=random_state)
    yield from splits


def make_cv_split_plan(
    X: Any,
    y: np.ndarray,
    contract: DataModelingContract,
    *,
    n_splits: int = 3,
    random_state: int = 42,
    groups: pd.Series | np.ndarray | None = None,
    group_column: str | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    n = len(y)
    n_splits = max(2, min(n_splits, n))
    groups_arr = _as_group_array(groups, n)
    if n < 4:
        idx = np.arange(n)
        midpoint = max(1, n // 2)
        return [(idx[:midpoint], idx[midpoint:])], {
            "split_kind": "small_holdout",
            "shuffle": False,
            "leakage_guard": "minimal split for tiny public train",
            "group_column": group_column,
        }
    if "time_series_forecasting" in contract.task_kinds and n >= n_splits + 2:
        return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n))), {
            "split_kind": "time_series",
            "shuffle": False,
            "leakage_guard": "ordered TimeSeriesSplit",
            "group_column": group_column,
        }
    if groups_arr is not None:
        group_plan = _make_group_split_plan(y, groups_arr, contract, n_splits=n_splits, random_state=random_state, group_column=group_column)
        if group_plan is not None:
            return group_plan
    if contract.metric.prediction_kind in {"probability", "class", "top_k_class"}:
        values, counts = np.unique(y, return_counts=True)
        if len(values) > 1 and counts.min() >= n_splits:
            return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.arange(n), y)), {
                "split_kind": "stratified_kfold",
                "shuffle": True,
                "leakage_guard": "class balance preserved per fold",
                "group_column": group_column,
            }
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.arange(n))), {
        "split_kind": "kfold",
        "shuffle": True,
        "leakage_guard": "generic random KFold; no group column detected",
        "group_column": group_column,
    }


def cross_val_oof_predictions(
    estimator: Any,
    X: Any,
    y: pd.Series | np.ndarray,
    contract: DataModelingContract,
    *,
    n_splits: int = 3,
    random_state: int = 42,
    groups: pd.Series | np.ndarray | None = None,
    group_column: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return out-of-fold predictions using the same fold-local fit policy."""

    y_arr = np.asarray(y)
    groups_arr = _as_group_array(groups, len(y_arr))
    splits, split_metadata = make_cv_split_plan(
        X,
        y_arr,
        contract,
        n_splits=n_splits,
        random_state=random_state,
        groups=groups_arr,
        group_column=group_column,
    )
    oof: np.ndarray | None = None
    filled = np.zeros(len(y_arr), dtype=bool)
    warnings: list[str] = []
    for fold, (train_idx, valid_idx) in enumerate(splits):
        model = _clone_estimator(estimator)
        X_train, X_valid = _take_rows(X, train_idx), _take_rows(X, valid_idx)
        y_train = y_arr[train_idx]
        try:
            model.fit(X_train, y_train)
            pred = np.asarray(predictions_for_metric(model, X_valid, contract.metric))
            if pred.ndim == 1:
                if oof is None:
                    oof = np.full(len(y_arr), np.nan, dtype=float)
                oof[valid_idx] = pred
            else:
                if oof is None:
                    oof = np.full((len(y_arr), pred.shape[1]), np.nan, dtype=float)
                oof[valid_idx, : pred.shape[1]] = pred
            filled[valid_idx] = True
        except Exception as exc:
            warnings.append(f"oof_fold_{fold}_failed:{type(exc).__name__}:{str(exc)[:160]}")
    if oof is None:
        oof = np.full(len(y_arr), np.nan, dtype=float)
    return oof, {
        **split_metadata,
        "folds": len(splits),
        "filled_rows": int(filled.sum()),
        "warnings": warnings,
        "preprocessing_scope": "fold_local",
    }


def _make_group_split_plan(
    y: np.ndarray,
    groups: np.ndarray,
    contract: DataModelingContract,
    *,
    n_splits: int,
    random_state: int,
    group_column: str | None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]] | None:
    unique_groups = pd.Series(groups).dropna().unique()
    if len(unique_groups) < 2:
        return None
    folds = max(2, min(n_splits, len(unique_groups)))
    idx = np.arange(len(y))
    if folds < 2:
        return None
    if contract.metric.prediction_kind in {"probability", "class", "top_k_class"} and StratifiedGroupKFold is not None:
        values, counts = np.unique(y, return_counts=True)
        if len(values) > 1 and counts.min() >= 2:
            try:
                splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
                splits = list(splitter.split(idx, y, groups))
                return splits, _group_metadata(
                    splits,
                    groups,
                    split_kind="stratified_group_kfold",
                    group_column=group_column,
                    leakage_guard="group-exclusive stratified folds",
                )
            except Exception:
                pass
    try:
        splits = list(GroupKFold(n_splits=folds).split(idx, y, groups))
    except Exception:
        return None
    return splits, _group_metadata(
        splits,
        groups,
        split_kind="group_kfold",
        group_column=group_column,
        leakage_guard="group-exclusive folds",
    )


def _group_metadata(
    splits: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    *,
    split_kind: str,
    group_column: str | None,
    leakage_guard: str,
) -> dict[str, Any]:
    overlaps: list[int] = []
    fold_group_counts: list[dict[str, int]] = []
    for train_idx, valid_idx in splits:
        train_groups = set(pd.Series(groups[train_idx]).dropna().astype(str))
        valid_groups = set(pd.Series(groups[valid_idx]).dropna().astype(str))
        overlaps.append(len(train_groups & valid_groups))
        fold_group_counts.append({"train_groups": len(train_groups), "valid_groups": len(valid_groups)})
    return {
        "split_kind": split_kind,
        "shuffle": split_kind == "stratified_group_kfold",
        "leakage_guard": leakage_guard,
        "group_column": group_column,
        "group_count": int(pd.Series(groups).nunique(dropna=True)),
        "group_fold_overlap_count": int(sum(overlaps)),
        "fold_group_counts": fold_group_counts,
    }


def _as_group_array(groups: pd.Series | np.ndarray | None, expected_len: int) -> np.ndarray | None:
    if groups is None:
        return None
    arr = np.asarray(groups)
    if len(arr) != expected_len:
        return None
    if pd.Series(arr).nunique(dropna=True) < 2:
        return None
    return arr


def predictions_for_metric(model: Any, X: Any, metric: MetricSpec) -> np.ndarray:
    if metric.name == "mpa_at_3" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, list):
            proba = proba[0]
        return np.asarray(proba)
    if metric.probability_output:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if isinstance(proba, list):
                proba = proba[0]
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] > 2:
                return proba
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba.reshape(-1)
        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X)).reshape(-1)
            return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(model.predict(X))


def metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: MetricSpec) -> float:
    name = metric.name
    y_pred = np.asarray(y_pred)
    if name == "roc_auc":
        return _safe_metric(lambda: _roc_auc(y_true, y_pred), fallback=0.5)
    if name == "normalized_gini":
        return _safe_metric(lambda: _normalized_gini(y_true, np.asarray(y_pred).reshape(-1)), fallback=0.0)
    if name == "average_precision":
        return _safe_metric(lambda: average_precision_score(y_true, y_pred), fallback=0.0)
    if name == "log_loss":
        clipped = np.clip(y_pred, 1e-6, 1 - 1e-6)
        return _safe_metric(lambda: log_loss(y_true, clipped), fallback=float("inf"))
    if name == "accuracy":
        return _safe_metric(lambda: accuracy_score(y_true, _as_classes(y_pred)), fallback=0.0)
    if name == "mpa_at_3":
        return _safe_metric(lambda: _mpa_at_3(y_true, y_pred), fallback=0.0)
    if name == "f1":
        return _safe_metric(lambda: f1_score(y_true, _as_classes(y_pred), average=_f1_average(y_true)), fallback=0.0)
    if name == "quadratic_weighted_kappa":
        return _safe_metric(lambda: cohen_kappa_score(y_true, np.rint(y_pred).astype(int), weights="quadratic"), fallback=0.0)
    if name == "mae":
        return _safe_metric(lambda: mean_absolute_error(y_true, y_pred), fallback=float("inf"))
    if name == "median_absolute_error":
        return _safe_metric(lambda: median_absolute_error(y_true, y_pred), fallback=float("inf"))
    if name == "rmse":
        return _safe_metric(lambda: math.sqrt(mean_squared_error(y_true, y_pred)), fallback=float("inf"))
    if name == "mse":
        return _safe_metric(lambda: mean_squared_error(y_true, y_pred), fallback=float("inf"))
    if name == "r2":
        return _safe_metric(lambda: r2_score(y_true, y_pred), fallback=-math.inf)
    if name == "rmsle":
        pred = np.clip(y_pred, 0, None)
        truth = np.clip(np.asarray(y_true, dtype=float), 0, None)
        return _safe_metric(lambda: math.sqrt(mean_squared_error(np.log1p(truth), np.log1p(pred))), fallback=float("inf"))
    if name == "smape":
        truth = np.asarray(y_true, dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        denom = np.abs(truth) + np.abs(pred)
        value = np.where(denom == 0, 0.0, 2.0 * np.abs(pred - truth) / denom)
        return float(np.mean(value))
    if name == "pearson":
        return _safe_metric(lambda: float(pearsonr(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))[0]), fallback=0.0)
    # Unknown regression/classification fallback: select by MAE for numeric targets, accuracy otherwise.
    if np.issubdtype(np.asarray(y_true).dtype, np.number):
        return _safe_metric(lambda: mean_absolute_error(y_true, y_pred), fallback=float("inf"))
    return _safe_metric(lambda: accuracy_score(y_true, _as_classes(y_pred)), fallback=0.0)


def selection_value(raw_value: float, metric: MetricSpec) -> float:
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return -math.inf
    return -float(raw_value) if metric.direction == "minimize" else float(raw_value)


def _as_classes(y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.asarray(y_pred)
    if np.issubdtype(y_pred.dtype, np.floating):
        return (y_pred >= 0.5).astype(int)
    return y_pred


def _roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pred = np.asarray(y_pred)
    if pred.ndim == 2 and pred.shape[1] > 2:
        return float(roc_auc_score(y_true, pred, multi_class="ovr"))
    return float(roc_auc_score(y_true, pred.reshape(-1)))


def _normalized_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(truth) == 0 or float(np.sum(truth)) == 0.0:
        return 0.0
    order = np.lexsort((np.arange(len(truth)), -pred))
    sorted_truth = truth[order]
    gini_sum = sorted_truth.cumsum().sum() / sorted_truth.sum()
    gini_sum -= (len(truth) + 1) / 2.0
    gini_pred = gini_sum / len(truth)

    perfect_order = np.lexsort((np.arange(len(truth)), -truth))
    perfect_truth = truth[perfect_order]
    perfect_sum = perfect_truth.cumsum().sum() / perfect_truth.sum()
    perfect_sum -= (len(truth) + 1) / 2.0
    gini_true = perfect_sum / len(truth)
    return 0.0 if gini_true == 0 else float(gini_pred / gini_true)


def _mpa_at_3(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred)
    if pred.ndim == 1:
        return float(np.mean(_as_classes(pred) == truth))
    top_k = np.argsort(-pred, axis=1)[:, : min(3, pred.shape[1])]
    return float(np.mean([truth[i] in set(top_k[i]) for i in range(len(truth))]))


def _f1_average(y_true: np.ndarray) -> str:
    values = np.unique(y_true)
    if len(values) <= 2:
        return "binary"
    return "macro"


def _safe_metric(fn: Any, fallback: float) -> float:
    try:
        value = fn()
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return fallback
        return float(value)
    except Exception:
        return fallback


def _clone_estimator(estimator: Any) -> Any:
    from sklearn.base import clone

    try:
        return clone(estimator)
    except Exception:
        import copy

        return copy.deepcopy(estimator)


def _take_rows(X: Any, idx: np.ndarray) -> Any:
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.iloc[idx]
    if sparse.issparse(X):
        return X[idx]
    return np.asarray(X)[idx]
