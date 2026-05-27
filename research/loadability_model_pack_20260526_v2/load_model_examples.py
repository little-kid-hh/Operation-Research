# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


PACK_ROOT = Path(__file__).resolve().parent


@dataclass
class PackedSVMPipeline:
    scaler: MinMaxScaler
    model: Any
    feature_names: list[str]
    drop_cols: list[str]

    def _transform(self, x_df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names if c not in x_df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return self.scaler.transform(x_df[self.feature_names].values)

    def predict(self, x_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._transform(x_df))

    def predict_proba(self, x_df: pd.DataFrame) -> np.ndarray:
        x = self._transform(x_df)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)
        score = np.asarray(self.model.decision_function(x), dtype=float).ravel()
        p1 = 1.0 / (1.0 + np.exp(-np.clip(score, -500, 500)))
        return np.column_stack([1.0 - p1, p1])


def _load_packed_svm(path: Path) -> PackedSVMPipeline:
    joblib_path = path.with_suffix(".joblib")
    if joblib_path.exists():
        loaded = joblib.load(joblib_path)
        if hasattr(loaded, "feature_names") and hasattr(loaded, "scaler") and hasattr(loaded, "model"):
            return PackedSVMPipeline(
                scaler=loaded.scaler,
                model=loaded.model,
                feature_names=list(loaded.feature_names),
                drop_cols=list(getattr(loaded, "drop_cols", [])),
            )

    payload = json.loads(path.read_text(encoding="utf-8"))
    scaler = MinMaxScaler()
    scaler.data_min_ = np.asarray(payload["scaler_min"], dtype=np.float64)
    scaler.data_max_ = np.asarray(payload["scaler_max"], dtype=np.float64)
    scaler.feature_names_in_ = np.asarray(payload["feature_names"], dtype=object)
    scaler.n_features_in_ = len(payload["feature_names"])
    rng = scaler.data_max_ - scaler.data_min_
    rng = np.where(rng == 0.0, 1.0, rng)
    scaler.scale_ = 1.0 / rng
    scaler.min_ = -scaler.data_min_ * scaler.scale_

    coef = np.asarray(payload["coef"], dtype=np.float64)
    intercept = np.asarray(payload["intercept"], dtype=np.float64).ravel()
    model = LinearSVC(C=float(payload["model_C"]), dual=False, random_state=42)
    model.fit(np.zeros((2, coef.shape[1]), dtype=np.float64), np.array([0, 1], dtype=np.int64))
    model.coef_[:] = coef
    model.intercept_[:] = intercept

    return PackedSVMPipeline(
        scaler=scaler,
        model=model,
        feature_names=list(payload["feature_names"]),
        drop_cols=list(payload.get("drop_cols", [])),
    )


def _predict_scaled(model: Any, scaler: Any, x_df: pd.DataFrame) -> np.ndarray:
    x = scaler.transform(x_df.values)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return np.asarray(p).ravel()
    score = np.asarray(model.decision_function(x), dtype=float).ravel()
    return 1.0 / (1.0 + np.exp(-np.clip(score, -500, 500)))


def load_svm_base40() -> PackedSVMPipeline:
    return _load_packed_svm(PACK_ROOT / "models" / "svm_base40" / "linear_svm_20260427_192130.json")


def load_svm_feature_engineered() -> PackedSVMPipeline:
    return _load_packed_svm(PACK_ROOT / "models" / "svm_feature_engineered" / "linear_svm_fe_20260526.json")


def load_lr_base40() -> tuple[Any, Any]:
    model = joblib.load(PACK_ROOT / "models" / "lg_base40" / "lg_model.joblib")
    scaler = joblib.load(PACK_ROOT / "models" / "lg_base40" / "lg_scaler.joblib")
    return model, scaler


def load_rf_base40() -> tuple[Any, Any]:
    model = joblib.load(PACK_ROOT / "models" / "rf_base40" / "rf_model.joblib")
    scaler = joblib.load(PACK_ROOT / "models" / "rf_base40" / "rf_scaler.joblib")
    return model, scaler


def load_xgb_base40() -> tuple[Any, Any]:
    model = joblib.load(PACK_ROOT / "models" / "xgb_base40" / "xgb_model.joblib")
    scaler = joblib.load(PACK_ROOT / "models" / "xgb_base40" / "xgb_scaler.joblib")
    return model, scaler


def predict_lr(x_df: pd.DataFrame) -> np.ndarray:
    model, scaler = load_lr_base40()
    return _predict_scaled(model, scaler, x_df)


def predict_rf(x_df: pd.DataFrame) -> np.ndarray:
    model, scaler = load_rf_base40()
    return _predict_scaled(model, scaler, x_df)


def predict_xgb(x_df: pd.DataFrame) -> np.ndarray:
    model, scaler = load_xgb_base40()
    return _predict_scaled(model, scaler, x_df)
