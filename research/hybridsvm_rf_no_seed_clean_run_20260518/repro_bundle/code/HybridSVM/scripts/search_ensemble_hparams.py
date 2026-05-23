# -*- coding: utf-8 -*-
"""
Random hyperparameter search for Ensemble baseline.

Searches base learners (svm/lr/rf/xgb-or-hgb-fallback) + linear meta combiner.
Evaluates on the fixed holdout split used by run_experiment defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.svm_train import DROP_COLS, load_raw_data


def _prob_1(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))
    p = np.asarray(model.predict(X), dtype=float).ravel()
    return np.clip(p, 0.0, 1.0)


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(fpr - target_fpr))
    return float(tpr[idx])


def _sample_trial(rng: random.Random, xgb_available: bool) -> dict:
    cfg = {
        "svm_C": 10 ** rng.uniform(-0.5, 1.7),         # ~0.32 .. 50
        "lr_C": 10 ** rng.uniform(-1.0, 1.5),          # 0.1 .. 31.6
        "rf_n_estimators": rng.choice([200, 300, 400, 600, 800]),
        "rf_max_depth": rng.choice([None, 8, 12, 16, 20]),
        "rf_min_samples_leaf": rng.choice([1, 2, 3, 4]),
        "meta_C": 10 ** rng.uniform(-1.3, 1.5),        # 0.05 .. 31.6
    }
    if xgb_available:
        cfg.update(
            {
                "xgb_n_estimators": rng.choice([250, 400, 600, 800]),
                "xgb_max_depth": rng.choice([4, 5, 6, 8]),
                "xgb_learning_rate": rng.choice([0.03, 0.05, 0.08, 0.1]),
                "xgb_subsample": rng.choice([0.8, 0.9, 1.0]),
                "xgb_colsample_bytree": rng.choice([0.8, 0.9, 1.0]),
                "xgb_reg_lambda": rng.choice([0.5, 1.0, 2.0, 4.0]),
            }
        )
    else:
        cfg.update(
            {
                "hgb_max_iter": rng.choice([250, 400, 600, 800]),
                "hgb_max_depth": rng.choice([4, 6, 8, 10]),
                "hgb_learning_rate": rng.choice([0.03, 0.05, 0.08, 0.1]),
            }
        )
    return cfg


def _make_models(cfg: dict, random_state: int, xgb_available: bool):
    models = {}
    models["svm"] = SVC(
        kernel="linear",
        C=float(cfg["svm_C"]),
        class_weight={1: 1},
        probability=True,
        random_state=random_state,
    )
    models["lr"] = LogisticRegression(
        C=float(cfg["lr_C"]),
        solver="lbfgs",
        max_iter=3000,
        random_state=random_state,
    )
    models["rf"] = RandomForestClassifier(
        n_estimators=int(cfg["rf_n_estimators"]),
        max_depth=cfg["rf_max_depth"],
        min_samples_leaf=int(cfg["rf_min_samples_leaf"]),
        n_jobs=-1,
        random_state=random_state,
    )

    if xgb_available:
        from xgboost import XGBClassifier

        models["xgb"] = XGBClassifier(
            n_estimators=int(cfg["xgb_n_estimators"]),
            max_depth=int(cfg["xgb_max_depth"]),
            learning_rate=float(cfg["xgb_learning_rate"]),
            subsample=float(cfg["xgb_subsample"]),
            colsample_bytree=float(cfg["xgb_colsample_bytree"]),
            reg_lambda=float(cfg["xgb_reg_lambda"]),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        models["gbdt_fallback"] = HistGradientBoostingClassifier(
            max_iter=int(cfg["hgb_max_iter"]),
            max_depth=int(cfg["hgb_max_depth"]),
            learning_rate=float(cfg["hgb_learning_rate"]),
            random_state=random_state,
        )
    return models


def evaluate_cfg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    *,
    cv_folds: int,
    random_state: int,
    xgb_available: bool,
) -> dict:
    base_models = _make_models(cfg, random_state, xgb_available)
    names = list(base_models.keys())

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_cols = []
    fitted_full = {}
    for name in names:
        oof = np.zeros(len(y_train), dtype=float)
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = _make_models(cfg, random_state, xgb_available)[name]
            m.fit(X_train[tr_idx], y_train[tr_idx])
            oof[va_idx] = _prob_1(m, X_train[va_idx])
        full = _make_models(cfg, random_state, xgb_available)[name]
        full.fit(X_train, y_train)
        fitted_full[name] = full
        oof_cols.append(oof)

    meta_X_train = np.column_stack(oof_cols)
    meta = LogisticRegression(
        C=float(cfg["meta_C"]),
        solver="lbfgs",
        max_iter=3000,
        random_state=random_state,
    )
    meta.fit(meta_X_train, y_train)

    meta_X_test = np.column_stack([_prob_1(fitted_full[n], X_test) for n in names])
    prob = meta.predict_proba(meta_X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, prob))
    acc = float(accuracy_score(y_test, pred))
    tpr1 = _tpr_at_fpr(y_test, prob, 0.01)
    return {
        "auc": auc,
        "accuracy": acc,
        "tpr_at_fpr1pct": float(tpr1),
        "base_models": names,
        "meta_coefficients": [float(x) for x in meta.coef_.ravel()],
        "meta_intercept": float(meta.intercept_.ravel()[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search ensemble hyperparameters")
    parser.add_argument("--data-path", type=pathlib.Path, default=pathlib.Path(r"c:\Operation Research\FunSearch_test\training_2orientations.csv"))
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("experiments_ensemble_eval/hparam_search"))
    args = parser.parse_args()

    rng = random.Random(args.random_state)
    np.random.seed(args.random_state)

    xgb_available = True
    try:
        import xgboost  # noqa: F401
    except Exception:
        xgb_available = False

    df = load_raw_data(args.data_path)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X_full_df = df.drop(columns=cols_to_drop + ["if_loaded"])
    y_full = df["if_loaded"].values.astype(int)

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.random_state
    )
    X_train_df = X_full_df.iloc[train_idx].reset_index(drop=True)
    X_test_df = X_full_df.iloc[test_idx].reset_index(drop=True)
    y_train = y_full[train_idx]
    y_test = y_full[test_idx]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_test = scaler.transform(X_test_df.values)

    results = []
    best = None
    best_key = (-math.inf, -math.inf, -math.inf)  # auc, tpr1, acc
    for i in range(1, int(args.n_trials) + 1):
        cfg = _sample_trial(rng, xgb_available)
        metrics = evaluate_cfg(
            X_train,
            y_train,
            X_test,
            y_test,
            cfg,
            cv_folds=int(args.cv_folds),
            random_state=int(args.random_state),
            xgb_available=xgb_available,
        )
        rec = {
            "trial": i,
            "config": cfg,
            "metrics": metrics,
            "xgb_available": bool(xgb_available),
            "xgb_fallback_used": not bool(xgb_available),
        }
        results.append(rec)
        key = (metrics["auc"], metrics["tpr_at_fpr1pct"], metrics["accuracy"])
        if key > best_key:
            best_key = key
            best = rec
        print(
            f"[trial {i:02d}] AUC={metrics['auc']:.5f} "
            f"TPR@1%={metrics['tpr_at_fpr1pct']:.5f} ACC={metrics['accuracy']:.5f}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = pathlib.Path(__file__).resolve().parents[1] / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ensemble_hparam_search_{ts}.json"
    payload = {
        "timestamp": ts,
        "n_trials": int(args.n_trials),
        "cv_folds": int(args.cv_folds),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "xgb_available": bool(xgb_available),
        "best": best,
        "all_trials": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nBest trial:")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"\nSaved search result: {out_path}")


if __name__ == "__main__":
    main()

