# -*- coding: utf-8 -*-
"""Cross-validate baseline ML models on the OR 2023 3D-BPP package-aware dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


ID_COLS = {
    "instance_name",
    "order_id",
    "package_id",
    "package_l",
    "package_w",
    "package_h",
    "label_2ori",
    "label_6ori",
    "time_2ori_ms",
    "time_6ori_ms",
}


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    if len(np.unique(y_true)) == 2:
        fpr_arr, tpr_arr, thresh = roc_curve(y_true, y_score)
        idx = int(np.argmin(np.abs(fpr_arr - 0.01)))
        auc = float(roc_auc_score(y_true, y_score))
        tpr_at_fpr1pct = float(tpr_arr[idx])
        threshold_at_fpr1pct = float(thresh[idx])
    else:
        auc = float("nan")
        tpr_at_fpr1pct = float("nan")
        threshold_at_fpr1pct = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "tpr_at_fpr1pct": tpr_at_fpr1pct,
        "threshold_at_fpr1pct": threshold_at_fpr1pct,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def make_models(random_state: int, quick: bool) -> dict[str, Any]:
    rf_estimators = 120 if quick else 300
    hgb_iters = 120 if quick else 250
    models: dict[str, Any] = {
        "logreg": make_pipeline(
            MinMaxScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=random_state),
        ),
        "linear_svm": make_pipeline(
            MinMaxScaler(),
            LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=5000, random_state=random_state),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gbdt": HistGradientBoostingClassifier(
            max_iter=hgb_iters,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=random_state,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=250 if quick else 500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
    except Exception as exc:
        print(f"Skipping xgboost: {exc}")
    return models


def score_model(model: Any, x_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    # Pipeline delegates methods, but keep this fallback for unusual estimators.
    return model.predict(x_test)


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["accuracy", "precision", "recall", "auc", "tpr_at_fpr1pct", "fpr", "fnr"]:
        vals = df[col].to_numpy(dtype=float)
        valid = vals[~np.isnan(vals)]
        out[col] = {
            "mean": float(np.mean(valid)) if len(valid) else float("nan"),
            "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            "min": float(np.min(valid)) if len(valid) else float("nan"),
            "max": float(np.max(valid)) if len(valid) else float("nan"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("permin_dataset_processing/processed_features/or2023_bpp_labeled_base40_package.csv"),
    )
    parser.add_argument("--target", choices=["label_2ori", "label_6ori"], default="label_6ori")
    parser.add_argument(
        "--group-level",
        choices=["order", "instance"],
        default="instance",
        help=(
            "Group split level. 'order' keeps all packages for one order together; "
            "'instance' holds out complete XML instances for stricter leakage checks."
        ),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional subset of model names to run, e.g. --models logreg xgboost.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("permin_dataset_processing/experiments"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    df = df[df[args.target].isin([0, 1])].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    x = df[feature_cols].to_numpy(dtype=float)
    y = df[args.target].to_numpy(dtype=int)
    if args.group_level == "instance":
        groups = df["instance_name"].astype(str).to_numpy()
        group_description = "instance_name"
    else:
        groups = (df["instance_name"].astype(str) + "::" + df["order_id"].astype(str)).to_numpy()
        group_description = "instance_name::order_id"

    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    models = make_models(args.random_state, args.quick)
    if args.models:
        missing_models = sorted(set(args.models) - set(models))
        if missing_models:
            raise ValueError(f"Unknown models requested: {', '.join(missing_models)}")
        models = {name: models[name] for name in args.models}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / f"cv_{ts}_{args.target}"
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    for model_name, model in models.items():
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(x, y, groups), start=1):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            overlap = train_groups.intersection(test_groups)
            if overlap:
                raise RuntimeError(f"Group leakage detected in fold {fold_id}: {len(overlap)} overlapping groups")
            model.fit(x[train_idx], y[train_idx])
            score = score_model(model, x[test_idx])
            pred = (score >= 0.5).astype(int) if model_name not in {"linear_svm"} else (score >= 0.0).astype(int)
            metrics = metric_bundle(y[test_idx], pred, score)
            row = {
                "model": model_name,
                "fold_id": int(fold_id),
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "positive_train": int(y[train_idx].sum()),
                "positive_test": int(y[test_idx].sum()),
                **metrics,
            }
            fold_rows.append(row)
            print(
                f"[{model_name} fold {fold_id}] "
                f"ACC={metrics['accuracy']:.5f} AUC={metrics['auc']:.5f} "
                f"TPR@1%={metrics['tpr_at_fpr1pct']:.5f}"
            )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(run_dir / "fold_metrics.csv", index=False)
    summary_rows = []
    summary: dict[str, Any] = {
        "data_path": str(args.data_path),
        "target": args.target,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "cv": {
            "type": "StratifiedGroupKFold",
            "n_splits": int(args.n_splits),
            "group_level": args.group_level,
            "group": group_description,
            "random_state": int(args.random_state),
        },
        "models": {},
    }
    for model_name, model_df in fold_df.groupby("model"):
        agg = summarize(model_df)
        summary["models"][model_name] = agg
        summary_rows.append(
            {
                "model": model_name,
                "accuracy_mean": agg["accuracy"]["mean"],
                "accuracy_std": agg["accuracy"]["std"],
                "auc_mean": agg["auc"]["mean"],
                "auc_std": agg["auc"]["std"],
                "tpr_at_fpr1pct_mean": agg["tpr_at_fpr1pct"]["mean"],
                "tpr_at_fpr1pct_std": agg["tpr_at_fpr1pct"]["std"],
                "precision_mean": agg["precision"]["mean"],
                "recall_mean": agg["recall"]["mean"],
                "fpr_mean": agg["fpr"]["mean"],
                "fnr_mean": agg["fnr"]["mean"],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["auc_mean", "tpr_at_fpr1pct_mean"], ascending=False)
    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved CV results to {run_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
