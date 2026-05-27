# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_model_examples import PackedSVMPipeline

PACK_ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "FunSearch_test" / "training_2orientations.csv"
ITEMS_PATH = ROOT / "FunSearch_test" / "物品信息和dblf信息.csv"
ACTIVE_BANK_PATH = PACK_ROOT / "feature_engineering" / "reference_active_bank.csv"

SVM_TRAIN_PATH = ROOT / "HybridSVM" / "src" / "svm_train.py"
_spec = importlib.util.spec_from_file_location("src.svm_train", SVM_TRAIN_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {SVM_TRAIN_PATH}")
_svm_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _svm_mod
_spec.loader.exec_module(_svm_mod)

DROP_COLS = _svm_mod.DROP_COLS
fit_svm_pipeline = _svm_mod.fit_svm_pipeline
load_raw_data = _svm_mod.load_raw_data

RANDOM_STATE = 42
TEST_SIZE = 0.25
SVM_C = 10.0


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _predict_prob(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return np.asarray(p).ravel()
    s = np.asarray(model.decision_function(X), dtype=float).ravel()
    return 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_packed_svm(pipe, out: Path, stem: str, meta: dict) -> None:
    packed = PackedSVMPipeline(
        scaler=pipe.scaler,
        model=pipe.model,
        feature_names=list(pipe.feature_names),
        drop_cols=list(getattr(pipe, "drop_cols", [])),
    )
    _write_json(
        out / f"{stem}.json",
        {
            "scaler_min": pipe.scaler.data_min_.tolist(),
            "scaler_max": pipe.scaler.data_max_.tolist(),
            "feature_names": list(pipe.feature_names),
            "drop_cols": list(getattr(pipe, "drop_cols", [])),
            "coef": np.asarray(pipe.model.coef_, dtype=float).tolist(),
            "intercept": np.asarray(pipe.model.intercept_, dtype=float).tolist(),
            "model_C": float(pipe.model.C),
            "model_kernel": str(pipe.model.kernel),
            **meta,
        },
    )
    joblib.dump(packed, out / f"{stem}.joblib")


def _fit_base_split():
    raw_df = load_raw_data(DATA_PATH)
    cols_to_drop = [c for c in DROP_COLS if c in raw_df.columns]
    base_df = raw_df.drop(columns=cols_to_drop + ["if_loaded"]).reset_index(drop=True)
    active_df = pd.read_csv(ACTIVE_BANK_PATH).reset_index(drop=True)
    aug_df = pd.concat([base_df, active_df], axis=1)
    y = raw_df["if_loaded"].to_numpy(dtype=int)
    idx = np.arange(len(raw_df))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return {
        "raw_df": raw_df,
        "base_df": base_df,
        "aug_df": aug_df,
        "y": y,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def build_svm_base40(split):
    X_train = split["base_df"].iloc[split["train_idx"]].reset_index(drop=True)
    y_train = split["y"][split["train_idx"]]
    pipe = fit_svm_pipeline(X_train, y_train, C=SVM_C, random_state=RANDOM_STATE)
    pipe.drop_cols = [c for c in DROP_COLS if c in split["raw_df"].columns]
    out = PACK_ROOT / "models" / "svm_base40"
    _ensure(out)
    _save_packed_svm(
        pipe,
        out,
        "linear_svm_20260427_192130",
        {
            "model_type": "linear_svm",
            "feature_set": "base40",
            "svm_c": SVM_C,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
    )


def build_lg_rf_xgb(split):
    X_train = split["base_df"].iloc[split["train_idx"]].reset_index(drop=True)
    y_train = split["y"][split["train_idx"]]
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train.values)

    models = {
        "lg": LogisticRegression(C=4.736286181285866, max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgb": XGBClassifier(
            n_estimators=700,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_lambda=0.5,
            min_child_weight=2,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        out = PACK_ROOT / "models" / f"{name}_base40"
        _ensure(out)
        joblib.dump(model, out / f"{name}_model.joblib")
        joblib.dump(scaler, out / f"{name}_scaler.joblib")
        _write_json(
            out / "model_info.json",
            {
                "model_type": name,
                "feature_set": "base40",
                "feature_names": list(X_train.columns),
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "scaled_with": "MinMaxScaler",
                "params": model.get_params(),
            },
        )


def build_fe_svm(split):
    X_train = split["aug_df"].iloc[split["train_idx"]].reset_index(drop=True)
    y_train = split["y"][split["train_idx"]]
    pipe = fit_svm_pipeline(X_train, y_train, C=SVM_C, random_state=RANDOM_STATE)
    pipe.drop_cols = [c for c in DROP_COLS if c in split["raw_df"].columns]
    out = PACK_ROOT / "models" / "svm_feature_engineered"
    _ensure(out)
    _save_packed_svm(
        pipe,
        out,
        "linear_svm_fe_20260526",
        {
            "model_type": "linear_svm",
            "feature_set": "base40_plus_active_bank",
            "active_bank_path": "feature_engineering/reference_active_bank.csv",
            "svm_c": SVM_C,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
    )


def main():
    split = _fit_base_split()
    build_svm_base40(split)
    build_lg_rf_xgb(split)
    build_fe_svm(split)

    meta = {
        "data_path": str(DATA_PATH),
        "active_bank_path": str(ACTIVE_BANK_PATH),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "models": [
            "svm_base40",
            "lg_base40",
            "rf_base40",
            "xgb_base40",
            "svm_feature_engineered",
        ],
    }
    (PACK_ROOT / "build_pack_manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
