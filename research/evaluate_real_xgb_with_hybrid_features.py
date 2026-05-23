# -*- coding: utf-8 -*-
"""
Evaluate real XGBoost on the fixed split, with and without the 15 HybridSVM
features, and compare a full stack using real XGB.

Outputs:
    research/real_xgb_with_hybrid_features_20260511/
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
HYBRID_ROOT = ROOT / "HybridSVM"
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))

from src.svm_train import DROP_COLS, load_raw_data  # noqa: E402


OUT_ROOT = ROOT / "research" / "real_xgb_with_hybrid_features_20260511"
DATA_PATH = ROOT / "FunSearch_test" / "training_2orientations.csv"
ACTIVE_BANK_PATH = ROOT / "tmp_glm_probe_v4" / "by_model" / "glm-5.1" / "exp_20260510_181255" / "active_feature_bank.csv"

SVM_C = 1.9124740037393821
LR_C = 4.736286181285866
RF_N_EST = 200
RF_MAX_DEPTH = 20
RF_MIN_LEAF = 2
META_C = 1.5688869685092588
RANDOM_STATE = 42

XGB_CFG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def prob_1(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.ravel()
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))
    pred = np.asarray(model.predict(X), dtype=float).ravel()
    return np.clip(pred, 0.0, 1.0)


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> tuple[float, float]:
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(tpr[idx]), float(thresh[idx])


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr1, thresh = tpr_at_fpr(y_true, y_score, 0.01)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "tpr_at_fpr1pct": float(tpr1),
        "threshold_at_fpr1pct": float(thresh),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def make_models() -> dict[str, Any]:
    return {
        "svm": SVC(kernel="linear", C=SVM_C, class_weight={1: 1}, probability=True, random_state=RANDOM_STATE),
        "lr": LogisticRegression(C=LR_C, max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(
            n_estimators=RF_N_EST,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_LEAF,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgb": XGBClassifier(**XGB_CFG),
    }


def fit_eval_single(name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    model = make_models()[name]
    model.fit(X_train, y_train)
    prob = prob_1(model, X_test)
    pred = (prob >= 0.5).astype(int)
    out = metric_bundle(y_test, pred, prob)
    out["model"] = name
    return out


def fit_eval_stack(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    base_names = ["svm", "lr", "rf", "xgb"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_cols = []
    fitted_full = {}
    for name in base_names:
        oof = np.zeros(len(y_train), dtype=float)
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = make_models()[name]
            m.fit(X_train[tr_idx], y_train[tr_idx])
            oof[va_idx] = prob_1(m, X_train[va_idx])
        full = make_models()[name]
        full.fit(X_train, y_train)
        fitted_full[name] = full
        oof_cols.append(oof)
    meta_X_train = np.column_stack(oof_cols)
    meta = LogisticRegression(C=META_C, max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE)
    meta.fit(meta_X_train, y_train)
    meta_X_test = np.column_stack([prob_1(fitted_full[n], X_test) for n in base_names])
    prob = meta.predict_proba(meta_X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    out = metric_bundle(y_test, pred, prob)
    out["base_models"] = base_names
    out["meta_coefficients"] = [float(x) for x in meta.coef_.ravel().tolist()]
    out["meta_intercept"] = float(meta.intercept_.ravel()[0])
    return out


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_df = load_raw_data(DATA_PATH)
    cols_to_drop = [c for c in DROP_COLS if c in raw_df.columns]
    base_df = raw_df.drop(columns=cols_to_drop + ["if_loaded"]).reset_index(drop=True)
    active_df = pd.read_csv(ACTIVE_BANK_PATH).reset_index(drop=True)
    aug_df = pd.concat([base_df, active_df], axis=1)
    y = raw_df["if_loaded"].to_numpy(dtype=int)

    idx = np.arange(len(raw_df))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=RANDOM_STATE)
    y_train = y[train_idx]
    y_test = y[test_idx]

    X_train_base_df = base_df.iloc[train_idx].reset_index(drop=True)
    X_test_base_df = base_df.iloc[test_idx].reset_index(drop=True)
    X_train_aug_df = aug_df.iloc[train_idx].reset_index(drop=True)
    X_test_aug_df = aug_df.iloc[test_idx].reset_index(drop=True)

    scaler_base = MinMaxScaler()
    X_train_base = scaler_base.fit_transform(X_train_base_df.values)
    X_test_base = scaler_base.transform(X_test_base_df.values)

    scaler_aug = MinMaxScaler()
    X_train_aug = scaler_aug.fit_transform(X_train_aug_df.values)
    X_test_aug = scaler_aug.transform(X_test_aug_df.values)

    single_rows = []
    for feature_set, Xtr, Xte in [
        ("base40", X_train_base, X_test_base),
        ("base40_plus_h15", X_train_aug, X_test_aug),
    ]:
        for name in ["svm", "lr", "rf", "xgb"]:
            row = fit_eval_single(name, Xtr, y_train, Xte, y_test)
            row["feature_set"] = feature_set
            single_rows.append(row)

    stack_rows = []
    for feature_set, Xtr, Xte in [
        ("base40", X_train_base, X_test_base),
        ("base40_plus_h15", X_train_aug, X_test_aug),
    ]:
        row = fit_eval_stack(Xtr, y_train, Xte, y_test)
        row["feature_set"] = feature_set
        stack_rows.append(row)

    pd.DataFrame(single_rows).to_csv(OUT_ROOT / "single_model_comparison.csv", index=False, encoding="utf-8")
    pd.DataFrame(stack_rows).to_csv(OUT_ROOT / "stacking_comparison.csv", index=False, encoding="utf-8")
    write_json(
        OUT_ROOT / "summary.json",
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "xgb_cfg": XGB_CFG,
            "split": {"test_size": 0.25, "random_state": RANDOM_STATE},
            "active_feature_count": int(active_df.shape[1]),
            "single_model_comparison": single_rows,
            "stacking_comparison": stack_rows,
        },
    )
    (OUT_ROOT / "README.md").write_text(
        "# Real XGBoost with HybridSVM Features\n\n- `single_model_comparison.csv`\n- `stacking_comparison.csv`\n- `summary.json`\n",
        encoding="utf-8",
    )
    print(f"Artifacts written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
