# -*- coding: utf-8 -*-
"""
Random hyperparameter search for real XGBoost on the fixed local split.

Supports:
1. base40 aggregate features
2. base40 + 15 HybridSVM active-bank features

All trials and the best result are written under:
    Ensemble_baseline/experiments/xgb_search_<timestamp>_<feature_set>/
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
HYBRID_ROOT = ROOT / "HybridSVM"
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))

from src.svm_train import DROP_COLS, load_raw_data  # noqa: E402


DEFAULT_ACTIVE_BANK_PATH = (
    ROOT
    / "HybridSVM"
    / "experiments_feature_search"
    / "by_model"
    / "glm-5.1"
    / "exp_20260513_235323"
    / "active_feature_bank.csv"
)


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


def sample_cfg(rng: random.Random) -> dict[str, Any]:
    return {
        "n_estimators": rng.choice([250, 400, 500, 700, 900]),
        "max_depth": rng.choice([4, 5, 6, 8, 10]),
        "learning_rate": rng.choice([0.02, 0.03, 0.05, 0.08, 0.1]),
        "subsample": rng.choice([0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": rng.choice([0.7, 0.8, 0.9, 1.0]),
        "reg_lambda": rng.choice([0.5, 1.0, 2.0, 4.0, 8.0]),
        "min_child_weight": rng.choice([1, 2, 4, 8]),
        "gamma": rng.choice([0.0, 0.1, 0.3, 0.5, 1.0]),
    }


def make_model(cfg: dict[str, Any], random_state: int) -> XGBClassifier:
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
        random_state=int(random_state),
        n_jobs=-1,
    )


def load_feature_frame(
    data_path: pathlib.Path,
    feature_set: str,
    *,
    active_bank_path: pathlib.Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    raw_df = load_raw_data(data_path)
    cols_to_drop = [c for c in DROP_COLS if c in raw_df.columns]
    base_df = raw_df.drop(columns=cols_to_drop + ["if_loaded"]).reset_index(drop=True)
    y = raw_df["if_loaded"].to_numpy(dtype=int)
    if feature_set == "base40":
        return base_df, y
    if feature_set in {"base40_plus_h15", "base40_plus_active_bank"}:
        bank_path = active_bank_path or DEFAULT_ACTIVE_BANK_PATH
        active_df = pd.read_csv(bank_path).reset_index(drop=True)
        return pd.concat([base_df, active_df], axis=1), y
    raise ValueError(f"Unknown feature_set: {feature_set}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search real XGBoost hyperparameters on fixed split")
    parser.add_argument("--data-path", type=pathlib.Path, default=pathlib.Path("FunSearch_test/training_2orientations.csv"))
    parser.add_argument(
        "--feature-set",
        choices=["base40", "base40_plus_h15", "base40_plus_active_bank"],
        default="base40",
    )
    parser.add_argument("--active-bank-path", type=pathlib.Path, default=DEFAULT_ACTIVE_BANK_PATH)
    parser.add_argument(
        "--feature-set-label",
        type=str,
        default="",
        help="Optional label used in output directory naming, e.g. base40_plus_h21.",
    )
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-root", type=pathlib.Path, default=pathlib.Path("Ensemble_baseline/experiments"))
    args = parser.parse_args()

    rng = random.Random(args.random_state)
    np.random.seed(args.random_state)

    X_df, y = load_feature_frame(
        args.data_path,
        args.feature_set,
        active_bank_path=args.active_bank_path,
    )
    idx = np.arange(len(X_df))
    train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.random_state)
    X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
    X_test_df = X_df.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_test = scaler.transform(X_test_df.values)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root.resolve() if args.out_root.is_absolute() else (ROOT / args.out_root).resolve()
    feature_set_label = args.feature_set_label.strip() or args.feature_set
    run_dir = out_root / f"xgb_search_{ts}_{feature_set_label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best = None
    best_key = (-math.inf, -math.inf, -math.inf)
    rows = []
    all_trials = []

    for i in range(1, int(args.n_trials) + 1):
        cfg = sample_cfg(rng)
        model = make_model(cfg, int(args.random_state))
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = metric_bundle(y_test, pred, prob)
        rec = {
            "trial": i,
            "config": cfg,
            "metrics": metrics,
        }
        all_trials.append(rec)
        rows.append(
            {
                "trial": i,
                **cfg,
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "tpr_at_fpr1pct": metrics["tpr_at_fpr1pct"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )
        key = (metrics["auc"], metrics["tpr_at_fpr1pct"], metrics["accuracy"])
        if key > best_key:
            best_key = key
            best = rec
        print(
            f"[trial {i:02d}] AUC={metrics['auc']:.5f} "
            f"TPR@1%={metrics['tpr_at_fpr1pct']:.5f} "
            f"ACC={metrics['accuracy']:.5f}"
        )

    trials_df = pd.DataFrame(rows).sort_values(["auc", "tpr_at_fpr1pct", "accuracy"], ascending=False)
    trials_df.to_csv(run_dir / "trials.csv", index=False, encoding="utf-8")

    summary = {
        "timestamp": ts,
        "feature_set": args.feature_set,
        "feature_set_label": feature_set_label,
        "data_path": str(args.data_path),
        "active_bank_path": (
            str(args.active_bank_path.resolve())
            if args.feature_set in {"base40_plus_h15", "base40_plus_active_bank"}
            else None
        ),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "n_trials": int(args.n_trials),
        "n_rows_total": int(len(X_df)),
        "n_rows_train": int(len(train_idx)),
        "n_rows_test": int(len(test_idx)),
        "n_features": int(X_df.shape[1]),
        "feature_names": list(X_df.columns),
        "best": best,
        "all_trials": all_trials,
    }
    (run_dir / "README.md").write_text(
        "\n".join(
            [
                "# Real XGBoost Hyperparameter Search",
                "",
                f"- feature_set: `{args.feature_set}`",
                f"- feature_set_label: `{feature_set_label}`",
                f"- data: `{args.data_path}`",
                (
                    f"- active_bank_path: `{args.active_bank_path}`"
                    if args.feature_set in {"base40_plus_h15", "base40_plus_active_bank"}
                    else ""
                ),
                f"- total rows: `{len(X_df)}`",
                f"- train rows: `{len(train_idx)}`",
                f"- test rows: `{len(test_idx)}`",
                f"- feature count: `{X_df.shape[1]}`",
                f"- n_trials: `{args.n_trials}`",
                "",
                "Files:",
                "- `trials.csv`: all trials sorted by AUC / TPR@1% / Accuracy",
                "- `summary.json`: full structured payload including every sampled config",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_path = run_dir / "summary.json"
    write_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nBest trial:")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"\nSaved search result: {write_path}")


if __name__ == "__main__":
    main()
