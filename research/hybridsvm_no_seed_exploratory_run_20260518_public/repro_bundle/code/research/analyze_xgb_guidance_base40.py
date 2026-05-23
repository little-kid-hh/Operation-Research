# -*- coding: utf-8 -*-
"""
Build a base40-only XGBoost guidance package for the HybridSVM feature-search route.

This file is the formal teacher-analysis entry for the "tree model teaches LLM
feature engineering" workflow. It intentionally does NOT use any later
LLM-derived active-bank features.

Outputs:
    research/xgb_guidance_base40_20260514/
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT.parent
HYBRID_ROOT = ROOT / "HybridSVM"
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))

from src.svm_train import DROP_COLS, fit_svm_pipeline, load_raw_data  # noqa: E402


OUT_DIR = BUNDLE_ROOT / "teacher_analysis" / "xgb_guidance_base40_20260514"
DATA_PATH = BUNDLE_ROOT / "data" / "training_2orientations.csv"
BASE40_SEARCH_SUMMARY = (
    BUNDLE_ROOT
    / "reference"
    / "Ensemble_baseline"
    / "experiments"
    / "xgb_search_20260511_202834_base40"
    / "summary.json"
)

TEST_SIZE = 0.25
RANDOM_STATE = 42
SVM_C = 10.0


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_best_cfg(path: pathlib.Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return dict(obj["best"]["config"])


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr_arr, tpr_arr, thresh = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs(fpr_arr - 0.01)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "tpr_at_fpr1pct": float(tpr_arr[idx]),
        "threshold_at_fpr1pct": float(thresh[idx]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def prob_1(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X)
            if getattr(p, "ndim", 1) == 2 and p.shape[1] >= 2:
                return np.asarray(p[:, 1], dtype=float).ravel()
            return np.asarray(p, dtype=float).ravel()
        except (AttributeError, NotImplementedError):
            pass
    if hasattr(model, "decision_score"):
        s = np.asarray(model.decision_score(X), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))
    pred = np.asarray(model.predict(X), dtype=float).ravel()
    return np.clip(pred, 0.0, 1.0)


def make_xgb(cfg: dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=int(cfg["n_estimators"]),
        max_depth=int(cfg["max_depth"]),
        learning_rate=float(cfg["learning_rate"]),
        subsample=float(cfg["subsample"]),
        colsample_bytree=float(cfg["colsample_bytree"]),
        reg_lambda=float(cfg["reg_lambda"]),
        min_child_weight=float(cfg["min_child_weight"]),
        gamma=float(cfg["gamma"]),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def gain_importance_table(model: XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    rows = []
    for idx, name in enumerate(feature_names):
        key = f"f{idx}"
        rows.append(
            {
                "feature": name,
                "gain": float(gain.get(key, 0.0)),
                "weight": float(weight.get(key, 0.0)),
                "split_used": int(weight.get(key, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["gain", "weight"], ascending=False).reset_index(drop=True)


def recovery_contrast(
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    svm_pred: np.ndarray,
    xgb_pred: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    svm_wrong_xgb_right = (svm_pred != y_test) & (xgb_pred == y_test)
    svm_right_xgb_wrong = (svm_pred == y_test) & (xgb_pred != y_test)

    recovered = X_test_df.loc[svm_wrong_xgb_right].reset_index(drop=True)
    harmed = X_test_df.loc[svm_right_xgb_wrong].reset_index(drop=True)
    whole = X_test_df.reset_index(drop=True)

    meta = {
        "n_test_rows": int(len(y_test)),
        "n_svm_wrong_xgb_right": int(svm_wrong_xgb_right.sum()),
        "n_svm_right_xgb_wrong": int(svm_right_xgb_wrong.sum()),
    }

    rows = []
    numeric_cols = [c for c in X_test_df.columns if pd.api.types.is_numeric_dtype(X_test_df[c])]
    for col in numeric_cols:
        whole_mean = float(whole[col].mean())
        recovered_mean = float(recovered[col].mean()) if len(recovered) else float("nan")
        harmed_mean = float(harmed[col].mean()) if len(harmed) else float("nan")
        rows.append(
            {
                "feature": col,
                "recovered_mean": recovered_mean,
                "whole_mean": whole_mean,
                "harmed_mean": harmed_mean,
                "delta_recovered_vs_whole": recovered_mean - whole_mean if len(recovered) else float("nan"),
                "delta_recovered_vs_harmed": (
                    recovered_mean - harmed_mean if len(recovered) and len(harmed) else float("nan")
                ),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(
            "delta_recovered_vs_whole",
            ascending=False,
            key=lambda s: np.abs(s.fillna(0.0)),
        ).reset_index(drop=True),
        meta,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data(DATA_PATH)
    cols_to_drop = [c for c in DROP_COLS if c in raw_df.columns]
    base_df = raw_df.drop(columns=cols_to_drop + ["if_loaded"]).reset_index(drop=True)
    y = raw_df["if_loaded"].to_numpy(dtype=int)

    idx = np.arange(len(raw_df))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_train = y[train_idx]
    y_test = y[test_idx]

    X_train_base_df = base_df.iloc[train_idx].reset_index(drop=True)
    X_test_base_df = base_df.iloc[test_idx].reset_index(drop=True)

    scaler_base = MinMaxScaler()
    X_train_base = scaler_base.fit_transform(X_train_base_df.values)
    X_test_base = scaler_base.transform(X_test_base_df.values)

    base40_cfg = read_best_cfg(BASE40_SEARCH_SUMMARY)

    xgb_base = make_xgb(base40_cfg)
    xgb_base.fit(X_train_base, y_train)
    xgb_base_prob = prob_1(xgb_base, X_test_base)
    xgb_base_pred = (xgb_base_prob >= 0.5).astype(int)
    xgb_base_metrics = metric_bundle(y_test, xgb_base_pred, xgb_base_prob)

    svm_pipe = fit_svm_pipeline(X_train_base_df, y_train, C=SVM_C, random_state=RANDOM_STATE)
    svm_base_prob = prob_1(svm_pipe, X_test_base_df)
    svm_base_pred = (svm_base_prob >= 0.5).astype(int)
    svm_base_metrics = metric_bundle(y_test, svm_base_pred, svm_base_prob)

    imp_base = gain_importance_table(xgb_base, list(X_train_base_df.columns))
    contrast_df, contrast_meta = recovery_contrast(X_test_base_df, y_test, svm_base_pred, xgb_base_pred)

    imp_base.to_csv(OUT_DIR / "base40_gain_importance.csv", index=False, encoding="utf-8")
    contrast_df.to_csv(OUT_DIR / "xgb_vs_svm_recovery_contrast_base40.csv", index=False, encoding="utf-8")

    summary = {
        "teacher_scope": "base40_only",
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "data_path": str(DATA_PATH),
        "feature_set": "base40",
        "active_bank_used": False,
        "base40_xgb_cfg_source": str(BASE40_SEARCH_SUMMARY),
        "base40_xgb_cfg": base40_cfg,
        "metrics": {
            "svm_base40": svm_base_metrics,
            "xgb_base40": xgb_base_metrics,
        },
        "recovery_meta_base40": contrast_meta,
        "top_base40_features": imp_base.head(20).to_dict(orient="records"),
    }
    write_json(OUT_DIR / "summary.json", summary)

    top_base = imp_base.head(15)["feature"].tolist()
    top_recovery = contrast_df.head(15)["feature"].tolist()
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Base40-Only XGBoost Guidance Package",
                "",
                "This package is the formal teacher-analysis source for the HybridSVM",
                "feature-search route. It intentionally uses only the original base40",
                "aggregate feature set and does not use any later LLM-derived features.",
                "",
                "## Scope",
                "- feature_set: `base40`",
                "- active_bank_used: `false`",
                f"- data: `{DATA_PATH}`",
                f"- xgb_cfg_source: `{BASE40_SEARCH_SUMMARY}`",
                "",
                "## Metrics",
                f"- svm (base40): AUC `{svm_base_metrics['auc']:.4f}`, TPR@1% `{svm_base_metrics['tpr_at_fpr1pct']:.4f}`, ACC `{svm_base_metrics['accuracy']:.4f}`",
                f"- xgb (base40): AUC `{xgb_base_metrics['auc']:.4f}`, TPR@1% `{xgb_base_metrics['tpr_at_fpr1pct']:.4f}`, ACC `{xgb_base_metrics['accuracy']:.4f}`",
                "",
                "## Top base40 XGB features",
                *[f"- `{x}`" for x in top_base],
                "",
                "## High-contrast features on rows recovered by XGB over SVM",
                *[f"- `{x}`" for x in top_recovery],
                "",
                "Files:",
                "- `summary.json`",
                "- `base40_gain_importance.csv`",
                "- `xgb_vs_svm_recovery_contrast_base40.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
