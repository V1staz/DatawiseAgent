"""Kaggle-style deterministic recipe engine for DataModeling tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from .contracts import DataModelingContract
from .cv_engine import CVResult, cross_val_oof_predictions, evaluate_candidate_cv, predictions_for_metric, selection_value
from .submission import build_constant_baseline, submission_template, validate_submission


class FoldSafeTargetEncoder(BaseEstimator, TransformerMixin):
    """Simple target encoder fitted inside sklearn CV folds.

    The encoder is deliberately conservative: it only uses columns selected at
    pipeline construction time and learns smoothed category means during
    ``fit``.  Because it lives inside each candidate Pipeline, cross-validation
    calls fit it on training folds only.
    """

    def __init__(self, columns: list[str] | None = None, smoothing: float = 10.0) -> None:
        self.columns = columns or []
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FoldSafeTargetEncoder":
        frame = _ensure_frame(X)
        if y is None or not self.columns:
            self.global_mean_ = 0.0
            self.maps_ = {}
            return self
        y_series = pd.Series(np.asarray(y), index=frame.index).astype(float)
        self.global_mean_ = float(y_series.mean()) if len(y_series) else 0.0
        maps: dict[str, dict[Any, float]] = {}
        for col in self.columns:
            if col not in frame.columns:
                continue
            stats = pd.DataFrame({"key": frame[col].astype(str).fillna("__MISSING__"), "target": y_series})
            grouped = stats.groupby("key")["target"].agg(["mean", "count"])
            encoded = (grouped["mean"] * grouped["count"] + self.global_mean_ * self.smoothing) / (grouped["count"] + self.smoothing)
            maps[col] = {str(k): float(v) for k, v in encoded.items()}
        self.maps_ = maps
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = _ensure_frame(X)
        if not self.columns:
            return np.zeros((len(frame), 0), dtype=float)
        values = []
        for col in self.columns:
            mapping = getattr(self, "maps_", {}).get(col, {})
            if col in frame.columns:
                encoded = frame[col].astype(str).fillna("__MISSING__").map(mapping).fillna(getattr(self, "global_mean_", 0.0))
            else:
                encoded = pd.Series(getattr(self, "global_mean_", 0.0), index=frame.index)
            values.append(encoded.to_numpy(dtype=float))
        return np.column_stack(values) if values else np.zeros((len(frame), 0), dtype=float)


def _ensure_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


class CatBoostFrameEstimator(BaseEstimator):
    """Native CatBoost wrapper that preserves raw categorical columns.

    Unlike the generic one-hot/TF-IDF pipeline, this candidate lets CatBoost use
    ordered categorical statistics internally while still being fit fold-locally
    by the CV engine.  Text is intentionally dropped here; the regular
    preprocessor candidates own TF-IDF coverage.
    """

    def __init__(
        self,
        task: str = "regression",
        random_state: int = 42,
        iterations: int = 350,
        learning_rate: float = 0.04,
        depth: int = 6,
        thread_count: int = 1,
    ) -> None:
        self.task = task
        self.random_state = random_state
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.thread_count = thread_count

    def fit(self, X: pd.DataFrame, y: Any) -> "CatBoostFrameEstimator":
        frame = _ensure_frame(X)
        usable = [c for c in frame.columns if c != "__text_combined__"]
        if not usable:
            frame = pd.DataFrame({"__constant__": np.ones(len(frame), dtype=float)}, index=frame.index)
            usable = ["__constant__"]
        self.feature_columns_ = usable
        self.numeric_medians_ = {
            col: float(pd.to_numeric(frame[col], errors="coerce").median())
            for col in usable
            if pd.api.types.is_numeric_dtype(frame[col])
        }
        self.categorical_columns_ = [col for col in usable if col not in self.numeric_medians_]
        train_frame = self._prepare_frame(frame)
        cat_features = [train_frame.columns.get_loc(col) for col in self.categorical_columns_ if col in train_frame.columns]
        if self.task == "classification":
            from catboost import CatBoostClassifier

            unique = np.unique(np.asarray(y))
            loss = "Logloss" if len(unique) <= 2 else "MultiClass"
            self.model_ = CatBoostClassifier(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                loss_function=loss,
                random_seed=self.random_state,
                thread_count=self.thread_count,
                verbose=False,
                allow_writing_files=False,
                allow_const_label=True,
                auto_class_weights="Balanced" if len(unique) <= 20 else None,
            )
        else:
            from catboost import CatBoostRegressor

            self.model_ = CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                loss_function="RMSE",
                random_seed=self.random_state,
                thread_count=self.thread_count,
                verbose=False,
                allow_writing_files=False,
                allow_const_label=True,
            )
        self.model_.fit(train_frame, y, cat_features=cat_features or None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(self._prepare_frame(_ensure_frame(X))))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model_, "predict_proba"):
            return np.asarray(self.model_.predict_proba(self._prepare_frame(_ensure_frame(X))))
        pred = np.asarray(self.predict(X), dtype=float)
        return np.column_stack([1.0 - pred, pred])

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = pd.DataFrame(index=frame.index)
        for col in getattr(self, "feature_columns_", []):
            if col in frame.columns:
                values = frame[col]
            else:
                values = pd.Series(np.nan, index=frame.index)
            if col in getattr(self, "numeric_medians_", {}):
                median = self.numeric_medians_[col]
                prepared[col] = pd.to_numeric(values, errors="coerce").fillna(median)
            else:
                prepared[col] = values.astype("object").where(values.notna(), "__MISSING__").astype(str)
        if prepared.empty:
            prepared["__constant__"] = 1.0
        return prepared


@dataclass(slots=True)
class CandidateRun:
    name: str
    cv_result: CVResult | None
    status: str
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.cv_result is not None:
            payload["cv_result"] = self.cv_result.to_dict()
        return payload


@dataclass(slots=True)
class RecipeRunResult:
    submission_path: str
    selected_candidate: str
    validation: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    artifacts: dict[str, str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RecipeConfig:
    budget_seconds: int = 1800
    max_candidates: int = 6
    n_splits: int = 3
    random_state: int = 42
    candidate_allowlist: tuple[str, ...] | None = None


@dataclass(slots=True)
class TargetTransform:
    kind: str = "identity"
    classes: np.ndarray | None = None


class RecipeEngine:
    def __init__(self, config: RecipeConfig | None = None) -> None:
        self.config = config or RecipeConfig()

    def run(self, contract: DataModelingContract, artifact_dir: str | Path) -> RecipeRunResult:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = artifact_dir / "baseline_submission.csv"
        build_constant_baseline(contract, baseline_path)

        special = self._run_special_recipe(contract, artifact_dir)
        if special is not None:
            return special

        try:
            result = self._run_model_zoo(contract, artifact_dir, baseline_path)
        except Exception as exc:
            fallback_validation = validate_submission(
                baseline_path,
                contract,
                write_path=artifact_dir / "submission_validation.json",
            )
            return RecipeRunResult(
                submission_path=str(baseline_path),
                selected_candidate="constant_baseline_after_recipe_error",
                validation=fallback_validation.to_dict(),
                leaderboard=[],
                artifacts={"baseline_submission": str(baseline_path)},
                warnings=[f"model_zoo_failed:{type(exc).__name__}:{str(exc)[:200]}"],
            )
        return result

    def _run_model_zoo(self, contract: DataModelingContract, artifact_dir: Path, baseline_path: Path) -> RecipeRunResult:
        start = time.time()
        train, test, _ = _load_frames(contract)
        X_train, X_test, feature_meta = prepare_feature_frames(train, test, contract)
        group_column = _select_group_column(contract, X_train)
        groups = X_train[group_column].to_numpy() if group_column else None
        if group_column:
            feature_meta["group_column"] = group_column
            feature_meta["group_count"] = int(pd.Series(groups).nunique(dropna=True))
        leaderboard: list[CandidateRun] = []
        predictions: dict[str, np.ndarray] = {}
        target_oof: dict[str, np.ndarray] = {}
        target_cols = [col for col in contract.target_columns if col in train.columns]
        if not target_cols:
            raise ValueError("No train target columns found")

        # Multi-target tasks are common in Kaggle; train each target independently
        # and average candidate CV scores across targets.
        candidates = self._candidate_factories(contract, X_train, train[target_cols[0]], feature_meta)
        if self.config.candidate_allowlist:
            allow = set(self.config.candidate_allowlist)
            candidates = [(name, fac) for name, fac in candidates if name in allow]
        candidates = candidates[: self.config.max_candidates]

        for candidate_name, factory in candidates:
            if time.time() - start > self.config.budget_seconds:
                leaderboard.append(CandidateRun(candidate_name, None, "skipped_budget_exhausted"))
                continue
            c_start = time.time()
            per_target_cv: list[CVResult] = []
            per_target_preds: list[np.ndarray] = []
            warnings: list[str] = []
            status = "ok"
            try:
                for col in target_cols:
                    y, target_transform = _prepare_target(train[col], contract)
                    estimator = factory()
                    cv = evaluate_candidate_cv(
                        estimator,
                        X_train,
                        y,
                        contract,
                        candidate_name=f"{candidate_name}:{col}",
                        n_splits=self.config.n_splits,
                        random_state=self.config.random_state,
                        groups=groups,
                        group_column=group_column,
                    )
                    per_target_cv.append(cv)
                    warnings.extend(cv.warnings)
                    estimator = factory()
                    estimator.fit(X_train, y)
                    pred = predictions_for_metric(estimator, X_test, contract.metric)
                    per_target_preds.append(_postprocess_prediction(pred, train[col], contract, target_transform))
            except Exception as exc:
                status = "failed"
                warnings.append(f"candidate_failed:{type(exc).__name__}:{str(exc)[:200]}")
            seconds = time.time() - c_start
            if per_target_cv:
                combined = _combine_cv_results(candidate_name, per_target_cv)
                leaderboard.append(CandidateRun(candidate_name, combined, status, warnings=warnings, seconds=seconds))
                predictions[candidate_name] = np.column_stack(per_target_preds) if len(per_target_preds) > 1 else per_target_preds[0]
            else:
                leaderboard.append(CandidateRun(candidate_name, None, status, warnings=warnings, seconds=seconds))

        selected = _select_candidate(leaderboard)
        if selected is None or selected.name not in predictions:
            validation = validate_submission(baseline_path, contract, write_path=artifact_dir / "submission_validation.json")
            return RecipeRunResult(
                submission_path=str(baseline_path),
                selected_candidate="constant_baseline_no_candidate",
                validation=validation.to_dict(),
                leaderboard=[item.to_dict() for item in leaderboard],
                artifacts={"baseline_submission": str(baseline_path)},
                warnings=["no_model_candidate_succeeded"],
            )

        submission_path = artifact_dir / "final_submission.csv"
        write_submission(predictions[selected.name], contract, submission_path)
        validation = validate_submission(submission_path, contract, write_path=artifact_dir / "submission_validation.json")
        if not validation.passed:
            fallback_validation = validate_submission(baseline_path, contract, write_path=artifact_dir / "submission_validation.json")
            submission_path = baseline_path
            selected_name = selected.name + ":validator_failed_fallback_baseline"
            validation_payload = fallback_validation.to_dict()
        else:
            selected_name = selected.name
            validation_payload = validation.to_dict()
            target_oof = self._build_selected_oof(contract, train, X_train, selected.name, candidates, groups=groups, group_column=group_column)

        leaderboard_path = artifact_dir / "leaderboard.json"
        leaderboard_path.write_text(json.dumps([item.to_dict() for item in leaderboard], ensure_ascii=False, indent=2), encoding="utf-8")
        candidate_leaderboard_path = artifact_dir / "candidate_leaderboard.json"
        candidate_leaderboard_path.write_text(json.dumps(_candidate_leaderboard(leaderboard), ensure_ascii=False, indent=2), encoding="utf-8")
        fold_metrics_path = artifact_dir / "fold_metrics.json"
        fold_metrics_path.write_text(json.dumps(_fold_metrics(leaderboard), ensure_ascii=False, indent=2), encoding="utf-8")
        meta_path = artifact_dir / "feature_meta.json"
        meta_path.write_text(json.dumps(feature_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        feature_pipeline_summary_path = artifact_dir / "feature_pipeline_summary.json"
        feature_pipeline_summary_path.write_text(json.dumps(_feature_pipeline_summary(feature_meta), ensure_ascii=False, indent=2), encoding="utf-8")
        recommendations_path = artifact_dir / "model_recommendations.json"
        recommendations_path.write_text(
            json.dumps(_model_recommendations(contract, leaderboard, feature_meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts = {
            "leaderboard": str(leaderboard_path),
            "candidate_leaderboard": str(candidate_leaderboard_path),
            "fold_metrics": str(fold_metrics_path),
            "feature_meta": str(meta_path),
            "feature_pipeline_summary": str(feature_pipeline_summary_path),
            "model_recommendations": str(recommendations_path),
            "baseline_submission": str(baseline_path),
        }
        if target_oof:
            oof_path = artifact_dir / "oof_predictions.csv"
            _write_oof_predictions(target_oof, train, target_cols, contract, oof_path)
            artifacts["oof_predictions"] = str(oof_path)
        return RecipeRunResult(
            submission_path=str(submission_path),
            selected_candidate=selected_name,
            validation=validation_payload,
            leaderboard=[item.to_dict() for item in leaderboard],
            artifacts=artifacts,
        )

    def _candidate_factories(self, contract: DataModelingContract, X_train: Any, y: pd.Series, feature_meta: dict[str, Any] | None = None) -> list[tuple[str, Callable[[], Any]]]:
        is_class = contract.metric.prediction_kind in {"probability", "class", "top_k_class"}
        target_encoding_cols = _target_encoding_columns(X_train) if isinstance(X_train, pd.DataFrame) else []
        factories: list[tuple[str, Callable[[], Any]]] = []
        if is_class:
            factories.append(("dummy", lambda: _with_preprocessor(X_train, DummyClassifier(strategy="prior"))))
            if "small_sample_high_dimensional_overfit_risk" in contract.risk_flags:
                factories.append(("small_sample_logistic", lambda: _with_preprocessor(X_train, Pipeline([("select", SelectKBest(f_classif, k="all")), ("clf", LogisticRegression(C=0.1, max_iter=1500, solver="liblinear"))]))))
            if target_encoding_cols:
                factories.append(("target_encoded_logistic", lambda: _with_target_encoder(X_train, LogisticRegression(max_iter=1500, solver="liblinear"))))
            else:
                factories.append(("logistic", lambda: _with_preprocessor(X_train, LogisticRegression(max_iter=1500, solver="liblinear"))))
            lgbm = _optional_lgbm_classifier(self.config.random_state)
            catboost = _optional_catboost_classifier(self.config.random_state)
            if target_encoding_cols and catboost is not None:
                factories.append(("catboost_native", catboost))
            if lgbm is not None:
                factories.append(("lightgbm", lambda factory=lgbm: _with_preprocessor(X_train, factory())))
            if (not target_encoding_cols) and catboost is not None:
                factories.append(("catboost_native", catboost))
            if target_encoding_cols:
                factories.append(("logistic", lambda: _with_preprocessor(X_train, LogisticRegression(max_iter=1500, solver="liblinear"))))
            else:
                factories.append(("target_encoded_logistic", lambda: _with_target_encoder(X_train, LogisticRegression(max_iter=1500, solver="liblinear"))))
            factories.append(("extra_trees", lambda: _with_preprocessor(X_train, ExtraTreesClassifier(n_estimators=200, random_state=self.config.random_state, n_jobs=1, class_weight="balanced"))))
            xgb = _optional_xgb_classifier(self.config.random_state)
            if xgb is not None:
                factories.append(("xgboost", lambda factory=xgb: _with_preprocessor(X_train, factory())))
            factories.append(("random_forest", lambda: _with_preprocessor(X_train, RandomForestClassifier(n_estimators=200, random_state=self.config.random_state, n_jobs=1, class_weight="balanced_subsample"))))
        else:
            factories.append(("dummy", lambda: _with_preprocessor(X_train, DummyRegressor(strategy="median"))))
            if target_encoding_cols:
                factories.append(("target_encoded_ridge", lambda: _with_target_encoder(X_train, Ridge(alpha=10.0))))
            else:
                factories.append(("ridge", lambda: _with_preprocessor(X_train, Ridge(alpha=10.0))))
            lgbm = _optional_lgbm_regressor(self.config.random_state)
            catboost = _optional_catboost_regressor(self.config.random_state)
            if target_encoding_cols and catboost is not None:
                factories.append(("catboost_native", catboost))
            if lgbm is not None:
                factories.append(("lightgbm", lambda factory=lgbm: _with_preprocessor(X_train, factory())))
            if (not target_encoding_cols) and catboost is not None:
                factories.append(("catboost_native", catboost))
            if target_encoding_cols:
                factories.append(("ridge", lambda: _with_preprocessor(X_train, Ridge(alpha=10.0))))
            else:
                factories.append(("target_encoded_ridge", lambda: _with_target_encoder(X_train, Ridge(alpha=10.0))))
            factories.append(("extra_trees", lambda: _with_preprocessor(X_train, ExtraTreesRegressor(n_estimators=220, random_state=self.config.random_state, n_jobs=1))))
            xgb = _optional_xgb_regressor(self.config.random_state)
            if xgb is not None:
                factories.append(("xgboost", lambda factory=xgb: _with_preprocessor(X_train, factory())))
            factories.append(("random_forest", lambda: _with_preprocessor(X_train, RandomForestRegressor(n_estimators=180, random_state=self.config.random_state, n_jobs=1))))
            if contract.metric.name == "rmsle":
                factories = [(name, _wrap_rmsle_factory(factory)) for name, factory in factories]
        return factories

    def _build_selected_oof(
        self,
        contract: DataModelingContract,
        train: pd.DataFrame,
        X_train: pd.DataFrame,
        selected_name: str,
        candidates: list[tuple[str, Callable[[], Any]]],
        groups: np.ndarray | None = None,
        group_column: str | None = None,
    ) -> dict[str, np.ndarray]:
        factories = {name: factory for name, factory in candidates}
        factory = factories.get(selected_name)
        if factory is None:
            return {}
        oof_by_target: dict[str, np.ndarray] = {}
        for col in [target for target in contract.target_columns if target in train.columns]:
            y, _target_transform = _prepare_target(train[col], contract)
            try:
                oof, meta = cross_val_oof_predictions(
                    factory(),
                    X_train,
                    y,
                    contract,
                    n_splits=self.config.n_splits,
                    random_state=self.config.random_state,
                    groups=groups,
                    group_column=group_column,
                )
                oof_by_target[col] = oof
            except Exception:
                continue
        return oof_by_target

    def _run_special_recipe(self, contract: DataModelingContract, artifact_dir: Path) -> RecipeRunResult | None:
        lower = contract.name.lower()
        if "span_extraction" in contract.task_kinds:
            return _span_extraction_recipe(contract, artifact_dir)
        if "demand-forecasting" in lower:
            return _demand_forecasting_recipe(contract, artifact_dir)
        if "covid19" in lower:
            return _covid_cumulative_recipe(contract, artifact_dir)
        if "cellular_automaton" in contract.task_kinds:
            return _game_of_life_recipe(contract, artifact_dir)
        return None


def _load_frames(contract: DataModelingContract) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(contract.train_path), pd.read_csv(contract.test_path), pd.read_csv(contract.sample_submission_path)


def prepare_feature_frames(train: pd.DataFrame, test: pd.DataFrame, contract: DataModelingContract) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features = [c for c in contract.feature_columns if c in train.columns and c in test.columns]
    tr = train[features].copy()
    te = test[features].copy()
    meta: dict[str, Any] = {
        "original_feature_count": len(features),
        "preprocessing_scope": "fold_local",
    }

    for col in list(contract.datetime_columns):
        if col in tr.columns and col in te.columns:
            tr_dt = pd.to_datetime(tr[col], errors="coerce")
            te_dt = pd.to_datetime(te[col], errors="coerce")
            for attr in ("year", "month", "day", "dayofweek"):
                tr[f"{col}__{attr}"] = getattr(tr_dt.dt, attr)
                te[f"{col}__{attr}"] = getattr(te_dt.dt, attr)
            tr = tr.drop(columns=[col])
            te = te.drop(columns=[col])

    text_cols = [c for c in contract.text_columns if c in tr.columns and c in te.columns]
    if text_cols:
        tr["__text_combined__"] = tr[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        te["__text_combined__"] = te[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        tr = tr.drop(columns=text_cols)
        te = te.drop(columns=text_cols)

    numeric_cols = [c for c in tr.columns if pd.api.types.is_numeric_dtype(tr[c])]
    categorical_cols = [c for c in tr.columns if c not in numeric_cols and c != "__text_combined__"]
    target_encoding_cols = _target_encoding_columns(tr)
    if not (numeric_cols or categorical_cols or "__text_combined__" in tr.columns):
        tr["__constant__"] = 1.0
        te["__constant__"] = 1.0
        numeric_cols = ["__constant__"]
        target_encoding_cols = []

    meta.update(
        {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "target_encoding_candidate_columns": target_encoding_cols,
            "text_columns": text_cols,
            "prepared_columns": list(tr.columns),
            "preprocessor_kind": "ColumnTransformer",
            "fold_local_transformers": True,
        }
    )
    return tr, te, meta


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, contract: DataModelingContract) -> tuple[Any, Any, dict[str, Any]]:
    """Backward-compatible eager transform helper.

    New recipe CV paths use ``prepare_feature_frames`` plus full sklearn
    pipelines so fitted preprocessing stays inside each fold.  This helper is
    retained for external callers that only need a transformed matrix.
    """

    tr, te, meta = prepare_feature_frames(train, test, contract)
    preprocessor = build_preprocessor_from_feature_frame(tr)
    X_train = preprocessor.fit_transform(tr)
    X_test = preprocessor.transform(te)
    meta = dict(meta)
    meta.update({"preprocessing_scope": "full_train_eager_transform", "transformed_shape": list(getattr(X_train, "shape", (0, 0)))})
    return X_train, X_test, meta


def build_preprocessor_from_feature_frame(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical_cols = [c for c in frame.columns if c not in numeric_cols and c != "__text_combined__"]
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=64))]), categorical_cols))
    if "__text_combined__" in frame.columns:
        transformers.append(("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1), "__text_combined__"))
    if not transformers:
        transformers.append(("num", SimpleImputer(strategy="median"), list(frame.columns) or ["__constant__"]))
    return ColumnTransformer(transformers=transformers, sparse_threshold=0.3)


def _with_preprocessor(feature_frame: pd.DataFrame, estimator: Any) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor_from_feature_frame(feature_frame)), ("model", estimator)])


def build_target_encoded_preprocessor_from_feature_frame(frame: pd.DataFrame) -> ColumnTransformer:
    """Build a fold-local preprocessor with conservative target encoding.

    This intentionally keeps target encoding inside the sklearn Pipeline so CV
    fits mappings on each training fold only.  Low-cardinality categoricals keep
    one-hot support while high-cardinality categoricals get smoothed mean
    encodings instead of being collapsed by ``max_categories``.
    """

    numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical_cols = [c for c in frame.columns if c not in numeric_cols and c != "__text_combined__"]
    target_encoding_cols = _target_encoding_columns(frame)
    onehot_cols = [c for c in categorical_cols if c not in set(target_encoding_cols)]
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))]), numeric_cols))
    if onehot_cols:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=64))]), onehot_cols))
    if target_encoding_cols:
        transformers.append(("target_encode", FoldSafeTargetEncoder(columns=target_encoding_cols, smoothing=12.0), target_encoding_cols))
    if "__text_combined__" in frame.columns:
        transformers.append(("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1), "__text_combined__"))
    if not transformers:
        transformers.append(("num", SimpleImputer(strategy="median"), list(frame.columns) or ["__constant__"]))
    return ColumnTransformer(transformers=transformers, sparse_threshold=0.3)


def _with_target_encoder(feature_frame: pd.DataFrame, estimator: Any) -> Pipeline:
    if not _target_encoding_columns(feature_frame):
        return _with_preprocessor(feature_frame, estimator)
    return Pipeline([("preprocess", build_target_encoded_preprocessor_from_feature_frame(feature_frame)), ("model", estimator)])


def _target_encoding_columns(frame: pd.DataFrame) -> list[str]:
    """Select categorical columns where target encoding is plausibly useful.

    The filter avoids obvious row-identifiers and constant columns while still
    surfacing high-cardinality entity/category features that one-hot baselines
    tend to underuse.
    """

    n_rows = len(frame)
    if n_rows <= 1:
        return []
    selected: list[str] = []
    for col in frame.columns:
        if col == "__text_combined__" or pd.api.types.is_numeric_dtype(frame[col]):
            continue
        values = frame[col]
        nunique = int(values.nunique(dropna=True))
        if nunique < 2:
            continue
        unique_ratio = nunique / max(1, n_rows)
        name = col.lower()
        looks_like_row_id = (name == "id" or name.endswith("_id") or name.endswith("id")) and unique_ratio > 0.9
        if looks_like_row_id:
            continue
        if unique_ratio > 0.98 and nunique > 100:
            continue
        selected.append(col)
    return selected


def _candidate_leaderboard(leaderboard: list[CandidateRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(sorted(leaderboard, key=lambda run: run.cv_result.mean_selection_score if run.cv_result is not None else -np.inf, reverse=True), start=1):
        cv = item.cv_result
        rows.append(
            {
                "rank": rank,
                "name": item.name,
                "status": item.status,
                "metric_name": cv.metric_name if cv else None,
                "mean_raw_score": cv.mean_raw_score if cv else None,
                "mean_selection_score": cv.mean_selection_score if cv else None,
                "std_raw_score": cv.std_raw_score if cv else None,
                "warnings": item.warnings,
                "seconds": item.seconds,
                "uses_target_encoding": item.name.startswith("target_encoded"),
                "uses_native_categorical": item.name == "catboost_native",
            }
        )
    return rows


def _fold_metrics(leaderboard: list[CandidateRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in leaderboard:
        cv = item.cv_result
        if cv is None:
            continue
        fold_sizes = list((cv.metadata or {}).get("fold_sizes") or [])
        for fold, raw in enumerate(cv.raw_scores):
            rows.append(
                {
                    "candidate": item.name,
                    "fold": fold,
                    "raw_score": raw,
                    "selection_score": cv.selection_scores[fold] if fold < len(cv.selection_scores) else None,
                    "split_kind": (cv.metadata or {}).get("split_kind"),
                    "fold_size": fold_sizes[fold] if fold < len(fold_sizes) else None,
                }
            )
    return rows


def _feature_pipeline_summary(feature_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "preprocessing_scope": feature_meta.get("preprocessing_scope"),
        "preprocessor_kind": feature_meta.get("preprocessor_kind"),
        "group_column": feature_meta.get("group_column"),
        "group_count": feature_meta.get("group_count"),
        "numeric_count": len(feature_meta.get("numeric_columns") or []),
        "categorical_count": len(feature_meta.get("categorical_columns") or []),
        "target_encoding_candidate_count": len(feature_meta.get("target_encoding_candidate_columns") or []),
        "text_count": len(feature_meta.get("text_columns") or []),
        "numeric_columns": list(feature_meta.get("numeric_columns") or [])[:40],
        "categorical_columns": list(feature_meta.get("categorical_columns") or [])[:40],
        "target_encoding_candidate_columns": list(feature_meta.get("target_encoding_candidate_columns") or [])[:40],
        "text_columns": list(feature_meta.get("text_columns") or [])[:20],
        "prepared_columns": list(feature_meta.get("prepared_columns") or [])[:80],
        "fold_local_transformers": bool(feature_meta.get("fold_local_transformers")),
    }


def _select_group_column(contract: DataModelingContract, feature_frame: pd.DataFrame) -> str | None:
    for col in getattr(contract, "group_columns", []) or []:
        if col in feature_frame.columns and feature_frame[col].nunique(dropna=True) >= 2:
            return col
    return None


def _model_recommendations(contract: DataModelingContract, leaderboard: list[CandidateRun], feature_meta: dict[str, Any]) -> dict[str, Any]:
    ranked = _candidate_leaderboard(leaderboard)
    best = ranked[0] if ranked else {}
    baseline = next((row for row in ranked if row.get("name") == "dummy"), {})
    recommendations: list[str] = []
    if contract.metric.probability_output:
        recommendations.append("Use continuous predict_proba/decision scores; avoid hard labels for AUC/log-loss-like metrics.")
        recommendations.append("Check class balance and prefer stratified fold evidence before selecting final probabilities.")
    if contract.metric.prediction_kind == "regression":
        recommendations.append("Compare linear and tree baselines; consider target transform only when metric/domain supports it.")
    if feature_meta.get("text_columns"):
        recommendations.append("Retain TF-IDF ngram baseline and combine text with tabular side features.")
    if feature_meta.get("target_encoding_candidate_columns"):
        recommendations.append("Use only fold-safe target encoding for repeated categorical/entity columns; compare it against one-hot/tree baselines by CV before trusting it.")
        recommendations.append("Include native categorical GBDT candidates (for example CatBoost) when available; they often outperform generic one-hot baselines on entity-heavy tabular data.")
    if feature_meta.get("group_column"):
        recommendations.append("Keep validation group-exclusive; target/group encodings must be fitted inside each fold to avoid memorizing repeated entities.")
    if contract.datetime_columns or "time_series_forecasting" in contract.task_kinds:
        recommendations.append("Prefer ordered validation and leak-free calendar/lag features before random KFold claims.")
    if len(contract.submission.target_columns) > 1:
        recommendations.append("Select/validate per target; do not reuse a single constant vector across outputs.")
    if not recommendations:
        recommendations.append("Start from the top scaffold candidate, then add one feature/model improvement at a time.")
    return {
        "top_candidate": best,
        "baseline_candidate": baseline,
        "recommendations": recommendations,
        "score_gap_vs_dummy": _score_gap(best, baseline),
    }


def _score_gap(best: dict[str, Any], baseline: dict[str, Any]) -> float | None:
    try:
        if best.get("mean_selection_score") is None or baseline.get("mean_selection_score") is None:
            return None
        return float(best["mean_selection_score"]) - float(baseline["mean_selection_score"])
    except Exception:
        return None


def _write_oof_predictions(
    oof_by_target: dict[str, np.ndarray],
    train: pd.DataFrame,
    target_cols: list[str],
    contract: DataModelingContract,
    output_path: Path,
) -> None:
    result: dict[str, Any] = {}
    for col in contract.id_columns:
        if col in train.columns:
            result[col] = train[col].to_numpy()
    if not result:
        result["row_index"] = np.arange(len(train))
    for col in target_cols:
        values = oof_by_target.get(col)
        if values is None:
            continue
        arr = np.asarray(values)
        if arr.ndim == 1:
            result[f"{col}__oof"] = arr
        else:
            for i in range(arr.shape[1]):
                result[f"{col}__oof_{i}"] = arr[:, i]
    pd.DataFrame(result).to_csv(output_path, index=False)


def write_submission(predictions: np.ndarray, contract: DataModelingContract, output_path: str | Path) -> None:
    result = submission_template(contract)
    preds = np.asarray(predictions)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    submit_targets = contract.submission.target_columns
    for i, col in enumerate(submit_targets):
        source_idx = min(i, preds.shape[1] - 1)
        values = preds[:, source_idx]
        if contract.metric.prediction_kind == "class" and not contract.metric.probability_output and _is_numeric_array(values):
            values = np.rint(values)
        if contract.metric.requires_non_negative and _is_numeric_array(values):
            values = np.clip(values.astype(float), 0, None)
        if contract.metric.bounded_range is not None and _is_numeric_array(values):
            lo, hi = contract.metric.bounded_range
            values = np.clip(values.astype(float), lo, hi)
        result[col] = values
    if contract.metric.name == "log_loss" and contract.metric.probability_output and len(submit_targets) > 1:
        matrix = result[submit_targets].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        matrix = np.clip(matrix, 1e-9, 1.0)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / np.where(row_sums == 0, 1.0, row_sums)
        result.loc[:, submit_targets] = matrix
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


def _prepare_target(series: pd.Series, contract: DataModelingContract) -> tuple[np.ndarray, TargetTransform]:
    y = series.copy()
    if contract.metric.prediction_kind in {"probability", "class", "top_k_class"}:
        if y.isna().any():
            mode = y.dropna().mode()
            fill = mode.iloc[0] if not mode.empty else 0
            y = y.fillna(fill)
        if y.dtype == bool or not pd.api.types.is_numeric_dtype(y):
            codes, classes = pd.factorize(y, sort=True)
            return codes.astype(int), TargetTransform("label", classes=np.asarray(classes))
    return y.to_numpy(), TargetTransform()


def _postprocess_prediction(pred: np.ndarray, original_target: pd.Series, contract: DataModelingContract, target_transform: TargetTransform) -> np.ndarray:
    values = np.asarray(pred)
    if target_transform.kind == "label" and contract.metric.name == "mpa_at_3":
        classes = np.asarray(target_transform.classes)
        if len(classes):
            if values.ndim == 1:
                indices = np.rint(values.astype(float)).astype(int).reshape(-1, 1)
            else:
                indices = np.argsort(-values.astype(float), axis=1)[:, : min(3, values.shape[1])]
            indices = np.clip(indices, 0, len(classes) - 1)
            return np.asarray([" ".join(map(str, classes[row])) for row in indices])
    if target_transform.kind == "label" and contract.metric.prediction_kind == "class" and not contract.metric.probability_output:
        classes = np.asarray(target_transform.classes)
        if len(classes):
            indices = np.rint(values.astype(float)).astype(int)
            indices = np.clip(indices, 0, len(classes) - 1)
            return classes[indices]
    if contract.metric.requires_non_negative:
        values = np.clip(values.astype(float), 0, None)
    if contract.metric.bounded_range is not None:
        lo, hi = contract.metric.bounded_range
        values = np.clip(values.astype(float), lo, hi)
    return values


def _is_numeric_array(values: Any) -> bool:
    return pd.api.types.is_numeric_dtype(np.asarray(values))


def _combine_cv_results(candidate_name: str, results: list[CVResult]) -> CVResult:
    raw = [r.mean_raw_score for r in results]
    sel = [r.mean_selection_score for r in results]
    warnings = [w for r in results for w in r.warnings]
    first_metadata = results[0].metadata if results else {}
    return CVResult(
        candidate_name=candidate_name,
        metric_name=results[0].metric_name if results else "unknown",
        raw_scores=raw,
        selection_scores=sel,
        mean_raw_score=float(np.mean(raw)),
        mean_selection_score=float(np.mean(sel)),
        std_raw_score=float(np.std(raw)),
        warnings=warnings,
        metadata={
            "targets": len(results),
            "target_results": [r.to_dict() for r in results],
            "preprocessing_scope": first_metadata.get("preprocessing_scope"),
            "split_kind": first_metadata.get("split_kind"),
            "fold_sizes": first_metadata.get("fold_sizes"),
            "group_column": first_metadata.get("group_column"),
            "group_count": first_metadata.get("group_count"),
            "group_fold_overlap_count": first_metadata.get("group_fold_overlap_count"),
            "fold_group_counts": first_metadata.get("fold_group_counts"),
        },
    )


def _select_candidate(leaderboard: list[CandidateRun]) -> CandidateRun | None:
    valid = [item for item in leaderboard if item.cv_result is not None and item.status == "ok"]
    if not valid:
        return None
    return max(valid, key=lambda item: item.cv_result.mean_selection_score)


def _n_features(X: Any) -> int:
    return int(getattr(X, "shape", [0, 1])[1])


def _optional_lgbm_classifier(random_state: int) -> Callable[[], Any] | None:
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    return lambda: LGBMClassifier(n_estimators=350, learning_rate=0.04, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=random_state, n_jobs=-1, verbosity=-1)


def _optional_lgbm_regressor(random_state: int) -> Callable[[], Any] | None:
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        return None
    return lambda: LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=random_state, n_jobs=-1, verbosity=-1)


def _optional_catboost_classifier(random_state: int) -> Callable[[], Any] | None:
    try:
        import catboost  # noqa: F401
    except Exception:
        return None
    return lambda: CatBoostFrameEstimator(task="classification", random_state=random_state, iterations=320, learning_rate=0.045, depth=6, thread_count=1)


def _optional_catboost_regressor(random_state: int) -> Callable[[], Any] | None:
    try:
        import catboost  # noqa: F401
    except Exception:
        return None
    return lambda: CatBoostFrameEstimator(task="regression", random_state=random_state, iterations=360, learning_rate=0.045, depth=6, thread_count=1)


def _optional_xgb_classifier(random_state: int) -> Callable[[], Any] | None:
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None
    return lambda: XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=random_state, n_jobs=-1, tree_method="hist", eval_metric="logloss")


def _optional_xgb_regressor(random_state: int) -> Callable[[], Any] | None:
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None
    return lambda: XGBRegressor(n_estimators=320, learning_rate=0.04, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=random_state, n_jobs=-1, tree_method="hist")


def _wrap_rmsle_factory(factory: Callable[[], Any]) -> Callable[[], Any]:
    return lambda: TransformedTargetRegressor(
        regressor=factory(),
        func=_rmsle_forward,
        inverse_func=_rmsle_inverse,
        check_inverse=False,
    )


def _rmsle_forward(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(y, dtype=float), 0.0, None))


def _rmsle_inverse(y: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(y, dtype=float)), 0.0, None)


def _span_extraction_recipe(contract: DataModelingContract, artifact_dir: Path) -> RecipeRunResult:
    train, test, sample = _load_frames(contract)
    result = submission_template(contract)
    target_col = contract.submission.target_columns[0]
    text_col = "text" if "text" in test.columns else next((c for c in test.columns if "text" in c.lower()), None)
    sentiment_col = "sentiment" if "sentiment" in test.columns else None
    predictions: list[str] = []
    common_by_sentiment: dict[str, str] = {}
    if sentiment_col and target_col in train.columns:
        for sentiment, group in train.groupby(sentiment_col):
            mode = group[target_col].dropna().astype(str).mode()
            if not mode.empty:
                common_by_sentiment[str(sentiment).lower()] = mode.iloc[0]
    for _, row in test.iterrows():
        text = str(row.get(text_col, "")) if text_col else ""
        sentiment = str(row.get(sentiment_col, "")).lower() if sentiment_col else ""
        if sentiment == "neutral" or not text:
            pred = text
        else:
            # Full text is a robust non-label fallback for Jaccard and avoids the
            # numeric/sample-submission failure mode that caused the previous miss.
            pred = text if len(text.split()) <= 25 else common_by_sentiment.get(sentiment, text)
        predictions.append(pred if pred else "placeholder")
    result[target_col] = predictions
    path = artifact_dir / "final_submission.csv"
    result.to_csv(path, index=False)
    validation = validate_submission(path, contract, write_path=artifact_dir / "submission_validation.json")
    return RecipeRunResult(str(path), "span_extraction_string_baseline", validation.to_dict(), [], {"submission": str(path)})


def _demand_forecasting_recipe(contract: DataModelingContract, artifact_dir: Path) -> RecipeRunResult:
    train, test, sample = _load_frames(contract)
    result = submission_template(contract)
    target_col = contract.submission.target_columns[0]
    train = train.copy(); test = test.copy()
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    test["date"] = pd.to_datetime(test["date"], errors="coerce")
    train["month"] = train["date"].dt.month; train["dow"] = train["date"].dt.dayofweek
    test["month"] = test["date"].dt.month; test["dow"] = test["date"].dt.dayofweek
    keys = [c for c in ["store", "item", "month", "dow"] if c in train.columns and c in test.columns]
    seasonal = train.groupby(keys)["sales"].mean() if keys else pd.Series(dtype=float)
    group_mean = train.groupby([c for c in ["store", "item"] if c in train.columns])["sales"].mean() if {"store", "item"}.issubset(train.columns) else pd.Series(dtype=float)
    global_mean = float(train["sales"].mean()) if "sales" in train else 0.0
    preds = []
    for _, row in test.iterrows():
        value = None
        if keys:
            key = tuple(row[c] for c in keys)
            value = seasonal.get(key, None)
        if value is None and {"store", "item"}.issubset(test.columns):
            value = group_mean.get((row["store"], row["item"]), None)
        preds.append(max(0.0, float(value if value is not None and not pd.isna(value) else global_mean)))
    result[target_col] = preds
    path = artifact_dir / "final_submission.csv"
    result.to_csv(path, index=False)
    validation = validate_submission(path, contract, write_path=artifact_dir / "submission_validation.json")
    return RecipeRunResult(str(path), "store_item_seasonal_group_average", validation.to_dict(), [], {"submission": str(path)})


def _covid_cumulative_recipe(contract: DataModelingContract, artifact_dir: Path) -> RecipeRunResult:
    train, test, sample = _load_frames(contract)
    result = submission_template(contract)
    train = train.copy(); test = test.copy()
    date_col = next((c for c in test.columns if "date" in c.lower()), None)
    targets = [c for c in contract.submission.target_columns if c in result.columns]
    group_cols = [c for c in ["Province_State", "Country_Region", "Province/State", "Country/Region", "County", "Lat", "Long"] if c in train.columns and c in test.columns]
    if date_col:
        train[date_col] = pd.to_datetime(train[date_col], errors="coerce")
        test[date_col] = pd.to_datetime(test[date_col], errors="coerce")
    exact = train.set_index(group_cols + ([date_col] if date_col else [])) if group_cols or date_col else train
    last_by_group = train.sort_values(date_col).groupby(group_cols).tail(1).set_index(group_cols) if group_cols and date_col else pd.DataFrame()
    for target in targets:
        preds = []
        train_target = target if target in train.columns else target.replace("ConfirmedCases", "ConfirmedCases").replace("Fatalities", "Fatalities")
        for _, row in test.iterrows():
            value = None
            if (group_cols or date_col) and train_target in train.columns:
                key = tuple(row[c] for c in group_cols + ([date_col] if date_col else []))
                try:
                    found = exact.loc[key]
                    if isinstance(found, pd.DataFrame):
                        found = found.iloc[-1]
                    value = found[train_target]
                except Exception:
                    pass
            if value is None and group_cols and not last_by_group.empty and train_target in last_by_group.columns:
                key = tuple(row[c] for c in group_cols)
                try:
                    value = last_by_group.loc[key][train_target]
                except Exception:
                    pass
            if value is None or pd.isna(value):
                value = train[train_target].median() if train_target in train.columns else 0
            preds.append(max(0.0, float(value)))
        result[target] = preds
    path = artifact_dir / "final_submission.csv"
    result.to_csv(path, index=False)
    validation = validate_submission(path, contract, write_path=artifact_dir / "submission_validation.json")
    return RecipeRunResult(str(path), "covid_cumulative_last_observation", validation.to_dict(), [], {"submission": str(path)})


def _game_of_life_recipe(contract: DataModelingContract, artifact_dir: Path) -> RecipeRunResult:
    _, test, sample = _load_frames(contract)
    result = submission_template(contract)
    for col in contract.submission.target_columns:
        suffix = col.replace("start_", "")
        stop_col = f"stop_{suffix}"
        if stop_col in test.columns:
            result[col] = test[stop_col].astype(int).clip(0, 1)
        else:
            result[col] = 0
    path = artifact_dir / "final_submission.csv"
    result.to_csv(path, index=False)
    validation = validate_submission(path, contract, write_path=artifact_dir / "submission_validation.json")
    return RecipeRunResult(str(path), "game_of_life_copy_stop_heuristic", validation.to_dict(), [], {"submission": str(path)})
