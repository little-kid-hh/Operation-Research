# -*- coding: utf-8 -*-
"""
Evaluate whether the 15 HybridSVM active-bank features improve tree models and
stacked ensemble models on the fixed local split.

Outputs are written under:
    research/tree_with_15_hybrid_features_20260511/
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

ROOT = pathlib.Path(__file__).resolve().parents[1]
HYBRID_ROOT = ROOT / "HybridSVM"
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))

from src.ensemble_train import EnsembleConfig, fit_ensemble_pipeline  # noqa: E402
from src.feature_search import load_search_data  # noqa: E402
from src.svm_train import DROP_COLS, fit_svm_pipeline, load_raw_data  # noqa: E402


OUT_ROOT = ROOT / "research" / "tree_with_15_hybrid_features_20260511"
GLM_EXP_DIR = ROOT / "tmp_glm_probe_v4" / "by_model" / "glm-5.1" / "exp_20260510_181255"
DATA_PATH = ROOT / "FunSearch_test" / "training_2orientations.csv"
ITEMS_PATH = ROOT / "FunSearch_test" / "物品信息和dblf信息.csv"

ENSEMBLE_CFG = {
    "svm_C": 1.9124740037393821,
    "lr_C": 4.736286181285866,
    "rf_n_estimators": 200,
    "rf_max_depth": 20,
    "rf_min_samples_leaf": 2,
    "meta_C": 1.5688869685092588,
    "hgb_max_iter": 400,
    "hgb_max_depth": 6,
    "hgb_learning_rate": 0.1,
}


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def fit_single_model(name: str, X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> tuple[Any, str]:
    if name == "svm":
        return (
            SVC(
                kernel="linear",
                C=float(ENSEMBLE_CFG["svm_C"]),
                class_weight={1: 1},
                probability=True,
                random_state=random_state,
            ),
            "svm",
        )
    if name == "lr":
        return (
            LogisticRegression(
                C=float(ENSEMBLE_CFG["lr_C"]),
                max_iter=3000,
                solver="lbfgs",
                random_state=random_state,
            ),
            "lr",
        )
    if name == "rf":
        return (
            RandomForestClassifier(
                n_estimators=int(ENSEMBLE_CFG["rf_n_estimators"]),
                max_depth=int(ENSEMBLE_CFG["rf_max_depth"]),
                min_samples_leaf=int(ENSEMBLE_CFG["rf_min_samples_leaf"]),
                n_jobs=-1,
                random_state=random_state,
            ),
            "rf",
        )
    if name == "xgb":
        return (
            HistGradientBoostingClassifier(
                learning_rate=float(ENSEMBLE_CFG["hgb_learning_rate"]),
                max_depth=int(ENSEMBLE_CFG["hgb_max_depth"]),
                max_iter=int(ENSEMBLE_CFG["hgb_max_iter"]),
                random_state=random_state,
            ),
            "gbdt_fallback",
        )
    raise ValueError(f"unknown model: {name}")


def evaluate_single(name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    model, resolved = fit_single_model(name, X_train, y_train)
    model.fit(X_train, y_train)
    prob = prob_1(model, X_test)
    pred = (prob >= 0.5).astype(int)
    out = metric_bundle(y_test, pred, prob)
    out["resolved_name"] = resolved
    return out


def evaluate_svm_c_values(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for c in [10.0, ENSEMBLE_CFG["svm_C"]]:
        pipe = fit_svm_pipeline(X_train_df, y_train, C=float(c), random_state=42)
        decision = pipe.decision_score(X_test_df)
        prob = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(decision, dtype=float), -500, 500)))
        pred = (prob >= 0.5).astype(int)
        m = metric_bundle(y_test, pred, prob)
        m["svm_C"] = float(c)
        rows.append(m)
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data(DATA_PATH)
    cols_to_drop = [c for c in DROP_COLS if c in raw_df.columns]
    base_df = raw_df.drop(columns=cols_to_drop + ["if_loaded"]).reset_index(drop=True)
    y = raw_df["if_loaded"].to_numpy(dtype=int)

    search_data = load_search_data(DATA_PATH, ITEMS_PATH, test_size=0.25, random_state=42)
    active_df = pd.read_csv(GLM_EXP_DIR / "active_feature_bank.csv").reset_index(drop=True)
    aug_df = pd.concat([base_df.reset_index(drop=True), active_df], axis=1)

    idx = np.arange(len(raw_df))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42)
    X_train_base_df = base_df.iloc[train_idx].reset_index(drop=True)
    X_test_base_df = base_df.iloc[test_idx].reset_index(drop=True)
    X_train_aug_df = aug_df.iloc[train_idx].reset_index(drop=True)
    X_test_aug_df = aug_df.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler_base = MinMaxScaler()
    X_train_base = scaler_base.fit_transform(X_train_base_df.values)
    X_test_base = scaler_base.transform(X_test_base_df.values)

    scaler_aug = MinMaxScaler()
    X_train_aug = scaler_aug.fit_transform(X_train_aug_df.values)
    X_test_aug = scaler_aug.transform(X_test_aug_df.values)

    single_rows = []
    for feat_set, Xtr, Xte in [
        ("base40", X_train_base, X_test_base),
        ("base40_plus_h15", X_train_aug, X_test_aug),
    ]:
        for name in ["svm", "lr", "rf", "xgb"]:
            row = evaluate_single(name, Xtr, y_train, Xte, y_test)
            row["feature_set"] = feat_set
            single_rows.append(row)

    stack_rows = []
    for feat_set, Xtr_df, Xte_df in [
        ("base40", X_train_base_df, X_test_base_df),
        ("base40_plus_h15", X_train_aug_df, X_test_aug_df),
    ]:
        pipe = fit_ensemble_pipeline(
            X_train_df=Xtr_df,
            y_train=y_train,
            config=EnsembleConfig(
                base_models=["svm", "lr", "rf", "xgb"],
                meta_model="logreg",
                cv_folds=5,
                random_state=42,
            ),
        )
        prob = pipe.predict_proba(Xte_df)
        pred = (prob >= 0.5).astype(int)
        m = metric_bundle(y_test, pred, prob)
        m["feature_set"] = feat_set
        m["resolved_base_models"] = list(pipe.resolved_base_models)
        m["xgb_fallback_used"] = bool(pipe.xgb_fallback_used)
        meta_coef = []
        if hasattr(pipe.meta_model, "coef_"):
            meta_coef = [float(x) for x in np.asarray(pipe.meta_model.coef_).ravel().tolist()]
        m["meta_coefficients"] = meta_coef
        stack_rows.append(m)

    svm_c_base_rows = evaluate_svm_c_values(X_train_base_df, y_train, X_test_base_df, y_test)
    svm_c_aug_rows = evaluate_svm_c_values(X_train_aug_df, y_train, X_test_aug_df, y_test)

    single_df = pd.DataFrame(single_rows)
    stack_df = pd.DataFrame(stack_rows)
    svm_c_base_df = pd.DataFrame(svm_c_base_rows)
    svm_c_aug_df = pd.DataFrame(svm_c_aug_rows)

    single_df.to_csv(OUT_ROOT / "single_model_comparison.csv", index=False, encoding="utf-8")
    stack_df.to_csv(OUT_ROOT / "stacking_comparison.csv", index=False, encoding="utf-8")
    svm_c_base_df.to_csv(OUT_ROOT / "svm_c_base40_comparison.csv", index=False, encoding="utf-8")
    svm_c_aug_df.to_csv(OUT_ROOT / "svm_c_base40_plus_h15_comparison.csv", index=False, encoding="utf-8")

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": str(DATA_PATH),
        "items_path": str(ITEMS_PATH),
        "glm_active_bank_path": str(GLM_EXP_DIR / "active_feature_bank.csv"),
        "split": {
            "test_size": 0.25,
            "random_state": 42,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "active_feature_count": int(active_df.shape[1]),
        "active_feature_names": list(active_df.columns),
        "single_model_comparison": single_rows,
        "stacking_comparison": stack_rows,
        "svm_c_base40_comparison": svm_c_base_rows,
        "svm_c_base40_plus_h15_comparison": svm_c_aug_rows,
        "notes": {
            "ensemble_svm_C": ENSEMBLE_CFG["svm_C"],
            "hybridsvm_feature_search_svm_C": 10.0,
            "xgb_route": "uses HistGradientBoostingClassifier fallback in this local environment",
        },
    }
    write_json(OUT_ROOT / "summary.json", summary)

    lines = [
        "# Tree Models with HybridSVM 15-Feature Bank",
        "",
        "This experiment tests whether adding the 15 accepted HybridSVM features",
        "improves tree models and the full ensemble on the same fixed split.",
        "",
        "Outputs:",
        "- `single_model_comparison.csv`",
        "- `stacking_comparison.csv`",
        "- `svm_c_base40_comparison.csv`",
        "- `svm_c_base40_plus_h15_comparison.csv`",
        "- `summary.json`",
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Artifacts written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
