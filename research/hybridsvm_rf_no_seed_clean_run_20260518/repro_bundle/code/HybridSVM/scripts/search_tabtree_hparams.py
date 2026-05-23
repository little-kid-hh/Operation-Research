# -*- coding: utf-8 -*-
"""Small diagnostic hyperparameter grid for TabTreeFormer(RF)."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.svm_train import DROP_COLS, load_raw_data
from TabTreeFormer.tabtreeformer import TabTreeFormerConfig, fit_tabtreeformer_pipeline


def tpr_at_fpr(y_true: np.ndarray, prob: np.ndarray, target_fpr: float = 0.01) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(tpr[idx]), float(thresholds[idx])


def metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    tpr1, thr1 = tpr_at_fpr(y_true, prob, 0.01)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, prob)),
        "tpr_at_fpr1pct": tpr1,
        "threshold_at_fpr1pct": thr1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def rf_only(
    X_train_df,
    y_train: np.ndarray,
    X_test_df,
    y_test: np.ndarray,
    cfg: dict,
    random_state: int,
) -> dict:
    model = RandomForestClassifier(
        n_estimators=int(cfg["rf_n_estimators"]),
        max_depth=cfg["rf_max_depth"],
        min_samples_leaf=int(cfg["rf_min_samples_leaf"]),
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train_df.values, y_train)
    prob = model.predict_proba(X_test_df.values)[:, 1]
    return metrics(y_test, prob)


def tabtree(
    X_train_df,
    y_train: np.ndarray,
    X_test_df,
    y_test: np.ndarray,
    cfg: dict,
    random_state: int,
) -> dict:
    model_cfg = TabTreeFormerConfig(
        rf_n_estimators=int(cfg["rf_n_estimators"]),
        rf_max_depth=cfg["rf_max_depth"],
        rf_min_samples_leaf=int(cfg["rf_min_samples_leaf"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        n_layers=int(cfg["n_layers"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
        mlp_hidden=int(cfg["mlp_hidden"]),
        raw_hidden=int(cfg["raw_hidden"]),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        random_state=random_state,
        oof_rf_folds=int(cfg["oof_rf_folds"]),
        correction_scale=float(cfg["correction_scale"]),
        use_raw_features=bool(cfg["use_raw_features"]),
    )
    pipe = fit_tabtreeformer_pipeline(X_train_df, y_train, config=model_cfg)
    prob = pipe.predict_proba(X_test_df)
    out = metrics(y_test, prob)
    out["calibrated_threshold"] = (
        float(pipe.calibrated_threshold) if pipe.calibrated_threshold is not None else None
    )
    out["calibration_info"] = dict(pipe.calibration_info)
    return out


def rf_grid() -> list[dict]:
    out = []
    for n_estimators in [64, 128, 256]:
        for max_depth in [8, 10, 12, None]:
            for min_leaf in [2, 5, 10]:
                out.append(
                    {
                        "rf_n_estimators": n_estimators,
                        "rf_max_depth": max_depth,
                        "rf_min_samples_leaf": min_leaf,
                    }
                )
    return out


def tabtree_grid() -> list[dict]:
    base = {
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 3e-4,
        "oof_rf_folds": 3,
        "nhead": 4,
        "n_layers": 1,
    }
    candidates = [
        # Strong RF-only backbones from the grid; check whether neural residual can
        # improve an already strong tree model.
        dict(rf_n_estimators=128, rf_max_depth=None, rf_min_samples_leaf=2, d_model=48, ff_dim=96, dropout=0.2, mlp_hidden=48, raw_hidden=48, epochs=8, correction_scale=0.7, use_raw_features=True),
        dict(rf_n_estimators=128, rf_max_depth=None, rf_min_samples_leaf=2, d_model=64, ff_dim=128, dropout=0.25, mlp_hidden=64, raw_hidden=64, epochs=8, correction_scale=0.5, use_raw_features=True),
        dict(rf_n_estimators=256, rf_max_depth=None, rf_min_samples_leaf=2, d_model=64, ff_dim=128, dropout=0.25, mlp_hidden=64, raw_hidden=64, epochs=8, correction_scale=0.5, use_raw_features=True),
        dict(rf_n_estimators=256, rf_max_depth=12, rf_min_samples_leaf=2, d_model=64, ff_dim=128, dropout=0.25, mlp_hidden=64, raw_hidden=64, epochs=8, correction_scale=0.5, use_raw_features=True),
        # Same RF backbone, test depth/leaf sensitivity.
        dict(rf_n_estimators=64, rf_max_depth=8, rf_min_samples_leaf=5, d_model=32, ff_dim=64, dropout=0.2, mlp_hidden=32, raw_hidden=32, epochs=6, correction_scale=1.0, use_raw_features=True),
        dict(rf_n_estimators=64, rf_max_depth=10, rf_min_samples_leaf=5, d_model=48, ff_dim=96, dropout=0.2, mlp_hidden=48, raw_hidden=48, epochs=8, correction_scale=1.0, use_raw_features=True),
        dict(rf_n_estimators=64, rf_max_depth=12, rf_min_samples_leaf=5, d_model=48, ff_dim=96, dropout=0.2, mlp_hidden=48, raw_hidden=48, epochs=8, correction_scale=1.0, use_raw_features=True),
        dict(rf_n_estimators=128, rf_max_depth=10, rf_min_samples_leaf=5, d_model=48, ff_dim=96, dropout=0.2, mlp_hidden=48, raw_hidden=48, epochs=8, correction_scale=1.0, use_raw_features=True),
        # Smaller/larger leaves.
        dict(rf_n_estimators=64, rf_max_depth=10, rf_min_samples_leaf=2, d_model=48, ff_dim=96, dropout=0.25, mlp_hidden=48, raw_hidden=48, epochs=8, correction_scale=0.7, use_raw_features=True),
        dict(rf_n_estimators=64, rf_max_depth=10, rf_min_samples_leaf=10, d_model=48, ff_dim=96, dropout=0.15, mlp_hidden=48, raw_hidden=48, epochs=8, correction_scale=1.0, use_raw_features=True),
        # Ablations.
        dict(rf_n_estimators=64, rf_max_depth=10, rf_min_samples_leaf=5, d_model=48, ff_dim=96, dropout=0.2, mlp_hidden=48, raw_hidden=48, epochs=8, correction_scale=1.0, use_raw_features=False),
        dict(rf_n_estimators=64, rf_max_depth=10, rf_min_samples_leaf=5, d_model=48, ff_dim=96, dropout=0.2, mlp_hidden=48, raw_hidden=48, epochs=8, correction_scale=0.0, use_raw_features=True),
    ]
    return [{**base, **c} for c in candidates]


def write_outputs(records: list[dict], out_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"tabtree_hparam_grid_{stamp}.json"
    csv_path = out_dir / f"tabtree_hparam_grid_{stamp}.csv"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "trial",
        "kind",
        "accuracy",
        "auc",
        "tpr_at_fpr1pct",
        "precision",
        "recall",
        "tn",
        "fp",
        "fn",
        "tp",
        "config",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {
                "trial": rec["trial"],
                "kind": rec["kind"],
                "config": json.dumps(rec["config"], ensure_ascii=False, sort_keys=True),
            }
            row.update({k: rec["metrics"].get(k) for k in fieldnames if k in rec["metrics"]})
            writer.writerow(row)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Search TabTreeFormer(RF) hyperparameters")
    parser.add_argument("--data-path", type=pathlib.Path, default=PROJECT_ROOT / "FunSearch_test" / "training_2orientations.csv")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rf-trials", type=int, default=36)
    parser.add_argument("--max-tabtree-trials", type=int, default=8)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("/tmp/hybrid_svm_check/tabtree_grid"))
    args = parser.parse_args()

    np.random.seed(int(args.random_state))
    df = load_raw_data(args.data_path)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X_full_df = df.drop(columns=cols_to_drop + ["if_loaded"])
    y_full = df["if_loaded"].values.astype(int)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )
    X_train_df = X_full_df.iloc[train_idx].reset_index(drop=True)
    X_test_df = X_full_df.iloc[test_idx].reset_index(drop=True)
    y_train = y_full[train_idx]
    y_test = y_full[test_idx]

    records: list[dict] = []
    trial = 0
    for cfg in rf_grid()[: max(0, int(args.max_rf_trials))]:
        trial += 1
        m = rf_only(X_train_df, y_train, X_test_df, y_test, cfg, int(args.random_state))
        rec = {"trial": trial, "kind": "rf_only", "config": cfg, "metrics": m}
        records.append(rec)
        print(
            f"[{trial:03d}] rf_only acc={m['accuracy']:.4f} auc={m['auc']:.4f} "
            f"tpr1={m['tpr_at_fpr1pct']:.4f} cfg={cfg}",
            flush=True,
        )

    for cfg in tabtree_grid()[: max(0, int(args.max_tabtree_trials))]:
        trial += 1
        m = tabtree(X_train_df, y_train, X_test_df, y_test, cfg, int(args.random_state))
        rec = {"trial": trial, "kind": "tabtree", "config": cfg, "metrics": m}
        records.append(rec)
        print(
            f"[{trial:03d}] tabtree acc={m['accuracy']:.4f} auc={m['auc']:.4f} "
            f"tpr1={m['tpr_at_fpr1pct']:.4f} cfg={cfg}",
            flush=True,
        )

    records.sort(
        key=lambda r: (
            r["metrics"]["auc"],
            r["metrics"]["tpr_at_fpr1pct"],
            r["metrics"]["accuracy"],
        ),
        reverse=True,
    )
    json_path, csv_path = write_outputs(records, args.out_dir)

    print("\nTop results:")
    for rec in records[:10]:
        m = rec["metrics"]
        print(
            f"#{rec['trial']:03d} {rec['kind']:<8} "
            f"acc={m['accuracy']:.4f} auc={m['auc']:.4f} "
            f"tpr1={m['tpr_at_fpr1pct']:.4f} cfg={rec['config']}"
        )
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()
