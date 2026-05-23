# -*- coding: utf-8 -*-
"""
Structured ablations for the standalone ensemble route.

This script intentionally lives under `Ensemble_baseline/` rather than
`HybridSVM/`, because the ensemble route is conceptually separate from the
LLM+SVM HybridSVM line.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
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
from src.svm_train import DROP_COLS, load_raw_data  # noqa: E402


BEST_LOCAL_CONFIG = {
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(fpr_arr - target_fpr))
    return float(tpr_arr[idx])


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "tpr_at_fpr1pct": float(_tpr_at_fpr(y_true, y_score, 0.01)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def _as_prob_1(model: Any, X: np.ndarray) -> np.ndarray:
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


def _make_single_model(name: str, cfg: dict[str, Any], random_state: int) -> tuple[Any, str]:
    if name == "svm":
        return (
            SVC(
                kernel="linear",
                C=float(cfg["svm_C"]),
                class_weight={1: 1},
                probability=True,
                random_state=random_state,
            ),
            "svm",
        )
    if name == "lr":
        return (
            LogisticRegression(
                C=float(cfg["lr_C"]),
                max_iter=3000,
                solver="lbfgs",
                random_state=random_state,
            ),
            "lr",
        )
    if name == "rf":
        return (
            RandomForestClassifier(
                n_estimators=int(cfg["rf_n_estimators"]),
                max_depth=int(cfg["rf_max_depth"]),
                min_samples_leaf=int(cfg["rf_min_samples_leaf"]),
                n_jobs=-1,
                random_state=random_state,
            ),
            "rf",
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
            )
        except Exception:
            return (
                HistGradientBoostingClassifier(
                    learning_rate=float(cfg["hgb_learning_rate"]),
                    max_depth=int(cfg["hgb_max_depth"]),
                    max_iter=int(cfg["hgb_max_iter"]),
                    random_state=random_state,
                ),
                "gbdt_fallback",
            )
    raise ValueError(f"Unknown model name: {name}")


def _single_model_spec(name: str, cfg: dict[str, Any], random_state: int) -> dict[str, Any]:
    model, resolved = _make_single_model(name, cfg, random_state)
    params = {}
    if hasattr(model, "get_params"):
        params = model.get_params(deep=False)
    training_rounds = None
    if hasattr(model, "n_estimators"):
        training_rounds = int(getattr(model, "n_estimators"))
    elif hasattr(model, "max_iter"):
        try:
            training_rounds = int(getattr(model, "max_iter"))
        except Exception:
            training_rounds = None
    return {
        "requested_name": name,
        "resolved_name": resolved,
        "class_name": type(model).__name__,
        "params": _json_ready(params),
        "training_rounds_hint": training_rounds,
    }


def _fit_single(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict[str, Any],
    random_state: int,
) -> dict[str, Any]:
    model, resolved = _make_single_model(model_name, cfg, random_state)
    model.fit(X_train, y_train)
    prob = _as_prob_1(model, X_test)
    pred = (prob >= 0.5).astype(int)
    out = _metric_bundle(y_test, pred, prob)
    out["model_name"] = resolved
    params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
    out["model_spec"] = {
        "requested_name": model_name,
        "resolved_name": resolved,
        "class_name": type(model).__name__,
        "params": _json_ready(params),
        "training_rounds_hint": _single_model_spec(model_name, cfg, random_state)["training_rounds_hint"],
    }
    return out


def _flatten_row(kind: str, name: str, meta_model: str | None, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "kind": kind,
        "name": name,
        "meta_model": meta_model or "",
    }
    for k, v in metrics.items():
        if k in {"tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "auc", "tpr_at_fpr1pct", "fpr", "fnr"}:
            row[k] = v
    return row


def _choose_combo_space(base_models: list[str]) -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = []
    for r in range(2, len(base_models) + 1):
        combos.extend(itertools.combinations(base_models, r))
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ensemble ablations on the fixed local split")
    parser.add_argument("--data-path", type=pathlib.Path, default=pathlib.Path("FunSearch_test/training_2orientations.csv"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("Ensemble_baseline/experiments"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--base-models", type=str, default="svm,lr,rf,xgb")
    parser.add_argument("--include-mlp-meta", action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir.resolve() if args.out_dir.is_absolute() else (ROOT / args.out_dir).resolve()
    run_dir = out_dir / f"ablation_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(args.data_path)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X_full_df = df.drop(columns=cols_to_drop + ["if_loaded"]).reset_index(drop=True)
    y_full = df["if_loaded"].to_numpy(dtype=int)

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.random_state)
    X_train_df = X_full_df.iloc[train_idx].reset_index(drop=True)
    X_test_df = X_full_df.iloc[test_idx].reset_index(drop=True)
    y_train = y_full[train_idx]
    y_test = y_full[test_idx]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_test = scaler.transform(X_test_df.values)

    requested_models = [x.strip() for x in args.base_models.split(",") if x.strip()]
    meta_models = ["logreg"] + (["mlp"] if args.include_mlp_meta else [])

    input_summary = {
        "data_path": str(args.data_path),
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(train_idx)),
        "n_rows_test": int(len(test_idx)),
        "n_features": int(X_full_df.shape[1]),
        "feature_names": list(X_full_df.columns),
        "positive_rate_total": float(y_full.mean()),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "drop_cols": list(cols_to_drop),
        "split": {
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
            "cv_folds": int(args.cv_folds),
        },
    }

    model_catalog = {
        name: _single_model_spec(name, BEST_LOCAL_CONFIG, int(args.random_state))
        for name in requested_models
    }

    single_rows: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []
    detailed = {
        "timestamp": ts,
        "input_summary": input_summary,
        "config": {
            "data_path": str(args.data_path),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "cv_folds": args.cv_folds,
            "base_models": requested_models,
            "meta_models": meta_models,
            "best_local_config": BEST_LOCAL_CONFIG,
        },
        "model_catalog": model_catalog,
        "single_models": {},
        "stacking_subsets": {},
    }

    for name in requested_models:
        metrics = _fit_single(
            model_name=name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cfg=BEST_LOCAL_CONFIG,
            random_state=args.random_state,
        )
        resolved_name = metrics.pop("model_name")
        detailed["single_models"][resolved_name] = metrics
        single_rows.append(_flatten_row("single", resolved_name, None, metrics))
        print(
            f"[single:{resolved_name}] "
            f"AUC={metrics['auc']:.5f} "
            f"TPR@1%={metrics['tpr_at_fpr1pct']:.5f} "
            f"ACC={metrics['accuracy']:.5f}"
        )

    for combo in _choose_combo_space(requested_models):
        combo_name = ",".join(combo)
        detailed["stacking_subsets"][combo_name] = {}
        for meta_model in meta_models:
            pipe = fit_ensemble_pipeline(
                X_train_df=X_train_df,
                y_train=y_train,
                config=EnsembleConfig(
                    base_models=list(combo),
                    meta_model=meta_model,
                    cv_folds=int(args.cv_folds),
                    random_state=int(args.random_state),
                ),
            )
            prob = pipe.predict_proba(X_test_df)
            pred = (prob >= 0.5).astype(int)
            metrics = _metric_bundle(y_test, pred, prob)
            metrics["resolved_base_models"] = list(pipe.resolved_base_models)
            metrics["xgb_fallback_used"] = bool(pipe.xgb_fallback_used)
            metrics["pipeline_spec"] = {
                "base_models_requested": list(combo),
                "base_models_resolved": list(pipe.resolved_base_models),
                "meta_model": meta_model,
                "cv_folds": int(args.cv_folds),
                "random_state": int(args.random_state),
                "xgb_fallback_used": bool(pipe.xgb_fallback_used),
                "meta_model_class": type(pipe.meta_model).__name__,
                "meta_model_params": _json_ready(
                    pipe.meta_model.get_params(deep=False) if hasattr(pipe.meta_model, "get_params") else {}
                ),
                "training_rounds_hint": {
                    "n_base_fits_total": int(len(combo) * args.cv_folds + len(combo)),
                    "n_meta_fits_total": 1,
                },
            }
            detailed["stacking_subsets"][combo_name][meta_model] = metrics
            combo_rows.append(_flatten_row("stacking", combo_name, meta_model, metrics))
            print(
                f"[stack:{combo_name}|{meta_model}] "
                f"AUC={metrics['auc']:.5f} "
                f"TPR@1%={metrics['tpr_at_fpr1pct']:.5f} "
                f"ACC={metrics['accuracy']:.5f}"
            )

    single_df = pd.DataFrame(single_rows).sort_values(["auc", "tpr_at_fpr1pct", "accuracy"], ascending=False)
    combo_df = pd.DataFrame(combo_rows).sort_values(["auc", "tpr_at_fpr1pct", "accuracy"], ascending=False)

    single_df.to_csv(run_dir / "single_model_ablation.csv", index=False, encoding="utf-8")
    combo_df.to_csv(run_dir / "stacking_ablation.csv", index=False, encoding="utf-8")

    if len(combo_df):
        detailed["best_stacking"] = combo_df.iloc[0].to_dict()
    if len(single_df):
        detailed["best_single"] = single_df.iloc[0].to_dict()

    detailed["outputs"] = {
        "run_dir": str(run_dir),
        "files": [
            "README.md",
            "summary.json",
            "single_model_ablation.csv",
            "stacking_ablation.csv",
        ],
    }

    (run_dir / "README.md").write_text(
        "\n".join(
            [
                "# Ensemble Ablation",
                "",
                "Generated by `Ensemble_baseline/run_ensemble_ablation.py`.",
                "",
                "## Input",
                f"- data: `{args.data_path}`",
                f"- total rows: `{len(df)}`",
                f"- train rows: `{len(train_idx)}`",
                f"- test rows: `{len(test_idx)}`",
                f"- feature count: `{X_full_df.shape[1]}`",
                f"- base models requested: `{requested_models}`",
                f"- meta models: `{meta_models}`",
                "",
                "Files:",
                "- `single_model_ablation.csv`: each base learner alone on the fixed split",
                "- `stacking_ablation.csv`: stacking subsets and meta-model comparison",
                "- `summary.json`: full structured payload",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps(detailed, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved ablation artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
