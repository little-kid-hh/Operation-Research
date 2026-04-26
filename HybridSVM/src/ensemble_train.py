# -*- coding: utf-8 -*-
"""
Project-grade ensemble baseline training and inference.

Implements OOF stacking for tabular binary classification:
  - Base learners: svm, lr, rf, xgb (with sklearn GBDT fallback)
  - Meta learner: linear combiner (logistic regression by default)
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def _as_prob_1(model: Any, X: np.ndarray) -> np.ndarray:
    """Return P(y=1) for a fitted sklearn-like estimator."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.ravel()
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=np.float64).ravel()
        return 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))
    pred = np.asarray(model.predict(X), dtype=np.float64).ravel()
    return np.clip(pred, 0.0, 1.0)


def _split_model_names(names: str | list[str]) -> list[str]:
    if isinstance(names, str):
        out = [x.strip().lower() for x in names.split(",") if x.strip()]
    else:
        out = [str(x).strip().lower() for x in names if str(x).strip()]
    if not out:
        raise ValueError("No base models specified for ensemble")
    valid = {"svm", "lr", "rf", "xgb"}
    bad = [x for x in out if x not in valid]
    if bad:
        raise ValueError(f"Unknown base model(s): {bad}; valid={sorted(valid)}")
    # de-duplicate while preserving order
    seen = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _make_base_model(name: str, random_state: int) -> tuple[Any, str, bool]:
    """
    Build one base learner.

    Returns
    -------
    estimator, resolved_name, xgb_fallback_used
    """
    if name == "svm":
        return (
            SVC(
                kernel="linear",
                C=1.9124740037393821,
                class_weight={1: 1},
                probability=True,
                random_state=random_state,
            ),
            "svm",
            False,
        )
    if name == "lr":
        return (
            LogisticRegression(
                C=4.736286181285866,
                max_iter=2000,
                solver="lbfgs",
                random_state=random_state,
            ),
            "lr",
            False,
        )
    if name == "rf":
        return (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=random_state,
            ),
            "rf",
            False,
        )
    if name == "xgb":
        try:
            from xgboost import XGBClassifier

            return (
                XGBClassifier(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
                "xgb",
                False,
            )
        except Exception:
            # Fallback keeps project runnable without xgboost install.
            return (
                HistGradientBoostingClassifier(
                    learning_rate=0.1,
                    max_depth=6,
                    max_iter=400,
                    random_state=random_state,
                ),
                "gbdt_fallback",
                True,
            )
    raise ValueError(f"Unsupported model name: {name}")


def _make_meta_model(name: str, random_state: int) -> Any:
    n = name.strip().lower()
    if n in {"logreg", "lr", "logistic"}:
        return LogisticRegression(
            C=1.5688869685092588,
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        )
    if n in {"mlp", "nn"}:
        # Small, regularized MLP for low-dimensional meta-features (base model probs).
        return MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            batch_size=64,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported meta model: {name}")


@dataclass
class EnsembleConfig:
    base_models: list[str] = field(default_factory=lambda: ["svm", "lr", "rf", "xgb"])
    meta_model: str = "logreg"
    cv_folds: int = 5
    random_state: int = 42


@dataclass
class EnsemblePipeline:
    scaler: MinMaxScaler
    base_models: dict[str, Any]
    meta_model: Any
    feature_names: list[str] = field(default_factory=list)
    config: EnsembleConfig = field(default_factory=EnsembleConfig)
    resolved_base_models: list[str] = field(default_factory=list)
    xgb_fallback_used: bool = False

    def _transform(self, X_df: pd.DataFrame) -> np.ndarray:
        present = [c for c in self.feature_names if c in X_df.columns]
        missing = set(self.feature_names) - set(present)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        return self.scaler.transform(X_df[present].values)

    def _meta_features(self, X_scaled: np.ndarray) -> np.ndarray:
        cols = []
        for name in self.resolved_base_models:
            model = self.base_models[name]
            cols.append(_as_prob_1(model, X_scaled))
        return np.column_stack(cols)

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        X = self._transform(X_df)
        meta = self._meta_features(X)
        return _as_prob_1(self.meta_model, meta)

    def predict(self, X_df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X_df)
        return (p >= threshold).astype(int)

    def save(self, path: pathlib.Path) -> None:
        info = {
            "feature_names": self.feature_names,
            "base_models_requested": self.config.base_models,
            "base_models_resolved": self.resolved_base_models,
            "meta_model": self.config.meta_model,
            "cv_folds": self.config.cv_folds,
            "random_state": self.config.random_state,
            "xgb_fallback_used": bool(self.xgb_fallback_used),
        }
        path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        joblib.dump(self, path.with_suffix(".joblib"))

    @classmethod
    def load(cls, path: pathlib.Path) -> "EnsemblePipeline":
        p = path.with_suffix(".joblib")
        if not p.exists():
            raise FileNotFoundError(f"Missing ensemble joblib artifact: {p}")
        obj = joblib.load(p)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected EnsemblePipeline in {p}, got {type(obj)}")
        return obj


def fit_ensemble_pipeline(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    *,
    config: EnsembleConfig | None = None,
) -> EnsemblePipeline:
    """
    Fit an ensemble with OOF stacking on the provided train split.
    """
    cfg = config or EnsembleConfig()
    base_requested = _split_model_names(cfg.base_models)
    if int(cfg.cv_folds) < 2:
        raise ValueError("cv_folds must be >= 2")

    y = np.asarray(y_train, dtype=int).ravel()
    feature_names = list(X_train_df.columns)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train_df.values)

    skf = StratifiedKFold(n_splits=int(cfg.cv_folds), shuffle=True, random_state=int(cfg.random_state))
    oof_cols: list[np.ndarray] = []
    fitted_models: dict[str, Any] = {}
    resolved_names: list[str] = []
    xgb_fallback_used = False

    for req_name in base_requested:
        model_for_oof, resolved_name, used_fb = _make_base_model(req_name, int(cfg.random_state))
        xgb_fallback_used = xgb_fallback_used or used_fb
        oof = np.zeros(len(y), dtype=np.float64)
        for tr_idx, va_idx in skf.split(X_scaled, y):
            m, _, _ = _make_base_model(req_name, int(cfg.random_state))
            m.fit(X_scaled[tr_idx], y[tr_idx])
            oof[va_idx] = _as_prob_1(m, X_scaled[va_idx])
        # Fit final model on full train split.
        model_for_oof.fit(X_scaled, y)
        fitted_models[resolved_name] = model_for_oof
        resolved_names.append(resolved_name)
        oof_cols.append(oof)

    meta_X = np.column_stack(oof_cols)
    meta = _make_meta_model(cfg.meta_model, int(cfg.random_state))
    meta.fit(meta_X, y)

    return EnsemblePipeline(
        scaler=scaler,
        base_models=fitted_models,
        meta_model=meta,
        feature_names=feature_names,
        config=EnsembleConfig(
            base_models=base_requested,
            meta_model=cfg.meta_model,
            cv_folds=int(cfg.cv_folds),
            random_state=int(cfg.random_state),
        ),
        resolved_base_models=resolved_names,
        xgb_fallback_used=xgb_fallback_used,
    )


def load_ensemble_pipeline(model_path: pathlib.Path) -> EnsemblePipeline:
    return EnsemblePipeline.load(model_path)

