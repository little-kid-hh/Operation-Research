# -*- coding: utf-8 -*-
"""
Cross-validate the current real XGBoost model on the full 10k-row dataset.

Outputs are written under:
    Ensemble_baseline/experiments/xgb_cv_<timestamp>_<config_name>_<feature_set>/
"""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from search_xgb_hparams import ROOT, load_feature_frame, metric_bundle


DEFAULT_CFG_PATH = ROOT / "research" / "real_xgb_with_hybrid_features_20260511" / "summary.json"
TUNED_CFG_PATHS = {
    "base40": ROOT / "Ensemble_baseline" / "experiments" / "xgb_search_20260511_202834_base40" / "summary.json",
    "base40_plus_h15": ROOT / "Ensemble_baseline" / "experiments" / "xgb_search_20260511_202925_base40_plus_h15" / "summary.json",
}


def load_default_cfg() -> tuple[dict[str, Any], pathlib.Path]:
    payload = json.loads(DEFAULT_CFG_PATH.read_text(encoding="utf-8"))
    cfg = dict(payload["xgb_cfg"])
    return cfg, DEFAULT_CFG_PATH


def load_tuned_cfg(feature_set: str) -> tuple[dict[str, Any], pathlib.Path]:
    path = TUNED_CFG_PATHS[feature_set]
    payload = json.loads(path.read_text(encoding="utf-8"))
    best_cfg = dict(payload["best"]["config"])
    cfg = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
        **best_cfg,
    }
    return cfg, path


def resolve_cfg(config_name: str, feature_set: str) -> tuple[dict[str, Any], pathlib.Path]:
    if config_name == "default":
        return load_default_cfg()
    if config_name == "tuned":
        return load_tuned_cfg(feature_set)
    raise ValueError(f"Unknown config_name: {config_name}")


def make_model(cfg: dict[str, Any], random_state: int) -> XGBClassifier:
    model_cfg = dict(cfg)
    model_cfg["random_state"] = int(random_state)
    model_cfg["n_jobs"] = int(model_cfg.get("n_jobs", -1))
    model_cfg["objective"] = str(model_cfg.get("objective", "binary:logistic"))
    model_cfg["eval_metric"] = str(model_cfg.get("eval_metric", "logloss"))
    return XGBClassifier(**model_cfg)


def summarize_folds(df: pd.DataFrame) -> dict[str, Any]:
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "auc",
        "tpr_at_fpr1pct",
        "threshold_at_fpr1pct",
        "fpr",
        "fnr",
    ]
    out: dict[str, Any] = {}
    for col in metric_cols:
        vals = df[col].to_numpy(dtype=float)
        out[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    out["n_folds_total"] = int(len(df))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate the current real XGBoost model")
    parser.add_argument("--data-path", type=pathlib.Path, default=pathlib.Path("FunSearch_test/training_2orientations.csv"))
    parser.add_argument("--feature-set", choices=["base40", "base40_plus_h15"], default="base40_plus_h15")
    parser.add_argument("--config-name", choices=["default", "tuned"], default="tuned")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-root", type=pathlib.Path, default=pathlib.Path("Ensemble_baseline/experiments"))
    args = parser.parse_args()

    X_df, y = load_feature_frame(args.data_path, args.feature_set)
    cfg, cfg_path = resolve_cfg(args.config_name, args.feature_set)

    cv = RepeatedStratifiedKFold(
        n_splits=int(args.n_splits),
        n_repeats=int(args.n_repeats),
        random_state=int(args.random_state),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root.resolve() if args.out_root.is_absolute() else (ROOT / args.out_root).resolve()
    run_dir = out_root / f"xgb_cv_{ts}_{args.config_name}_{args.feature_set}"
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X_df, y), start=1):
        X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
        X_test_df = X_df.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_df.values)
        X_test = scaler.transform(X_test_df.values)

        model = make_model(cfg, int(args.random_state))
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = metric_bundle(y_test, pred, prob)
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "positive_train": int(np.sum(y_train)),
                "positive_test": int(np.sum(y_test)),
                **metrics,
            }
        )
        print(
            f"[fold {fold_id:02d}] "
            f"ACC={metrics['accuracy']:.5f} "
            f"AUC={metrics['auc']:.5f} "
            f"TPR@1%={metrics['tpr_at_fpr1pct']:.5f}"
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(run_dir / "fold_metrics.csv", index=False, encoding="utf-8")

    summary = {
        "timestamp": ts,
        "data_path": str(args.data_path),
        "feature_set": args.feature_set,
        "config_name": args.config_name,
        "config_source": str(cfg_path),
        "xgb_cfg": cfg,
        "n_rows_total": int(len(X_df)),
        "n_features": int(X_df.shape[1]),
        "target_positive": int(np.sum(y)),
        "target_negative": int(len(y) - np.sum(y)),
        "cv_protocol": {
            "type": "RepeatedStratifiedKFold",
            "n_splits": int(args.n_splits),
            "n_repeats": int(args.n_repeats),
            "random_state": int(args.random_state),
        },
        "aggregate_metrics": summarize_folds(fold_df),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "README.md").write_text(
        "\n".join(
            [
                "# XGBoost Cross Validation",
                "",
                f"- feature_set: `{args.feature_set}`",
                f"- config_name: `{args.config_name}`",
                f"- config_source: `{cfg_path}`",
                f"- rows: `{len(X_df)}`",
                f"- features: `{X_df.shape[1]}`",
                f"- protocol: `RepeatedStratifiedKFold(n_splits={args.n_splits}, n_repeats={args.n_repeats}, random_state={args.random_state})`",
                "",
                "Files:",
                "- `fold_metrics.csv`: per-fold metrics",
                "- `summary.json`: aggregate metrics and exact XGBoost config",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nAggregate:")
    agg = summary["aggregate_metrics"]
    print(
        json.dumps(
            {
                "accuracy_mean": agg["accuracy"]["mean"],
                "accuracy_std": agg["accuracy"]["std"],
                "auc_mean": agg["auc"]["mean"],
                "auc_std": agg["auc"]["std"],
                "tpr_at_fpr1pct_mean": agg["tpr_at_fpr1pct"]["mean"],
                "tpr_at_fpr1pct_std": agg["tpr_at_fpr1pct"]["std"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()
