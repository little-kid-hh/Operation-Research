# -*- coding: utf-8 -*-
"""
Data loading, feature engineering, and SVM training pipeline.
Replicates the original SVM_ModelGenerate.py logic with a clean module.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import joblib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


# ─── Column drop list ────────────────────────────────────────────────────────
DROP_COLS = [
    "if_loaded",
    "orderid",
    "发车号",
    "aspect_ratio_var",
    "sku_max_volume",
    "sku_min_volume",
    "total_skuvolume",
    "vehicle_capacity",
    "sku_std_volume",
    "min_asr",
    "std_asr",
    # deviation stats
    "h_dev_l_avg",
    "h_dev_l_min",
    "h_dev_l_max",
    "h_dev_l_std",
    "w_dev_h_avg",
    "w_dev_h_min",
    "w_dev_h_max",
    "w_dev_h_std",
    "w_dev_l_avg",
    "w_dev_l_min",
    "w_dev_l_max",
    "w_dev_l_std",
    # cross-ratio stats
    "lh_to_vehicle_lh_avg",
    "lh_to_vehicle_lh_min",
    "lh_to_vehicle_lh_max",
    "lh_to_vehicle_lh_std",
    "wh_to_vehicle_wh_avg",
    "wh_to_vehicle_wh_min",
    "wh_to_vehicle_wh_max",
    "wh_to_vehicle_wh_std",
    "lh_to_vehicle_lh_total",
    "wh_to_vehicle_wh_total",
]


@dataclass
class SVMPipeline:
    """Trained linear SVM pipeline with scaler and model."""

    scaler: MinMaxScaler
    model: SVC
    feature_names: list[str] = field(default_factory=list)
    drop_cols: list[str] = field(default_factory=list)

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        X = self._transform(X_df)
        return self.model.predict(X)

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        X = self._transform(X_df)
        return self.model.predict_proba(X)

    def decision_score(self, X_df: pd.DataFrame) -> np.ndarray:
        """Raw decision function score (linear = w·x + b)."""
        X = self._transform(X_df)
        return self.model.decision_function(X)

    def _transform(self, X_df: pd.DataFrame) -> np.ndarray:
        present = [c for c in self.feature_names if c in X_df.columns]
        missing = set(self.feature_names) - set(present)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return self.scaler.transform(X_df[present].values)

    def save(self, path: pathlib.Path) -> None:
        data = {
            "scaler_min": self.scaler.data_min_.tolist(),
            "scaler_max": self.scaler.data_max_.tolist(),
            "feature_names": self.feature_names,
            "drop_cols": self.drop_cols,
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
            "model_C": self.model.C,
            "model_kernel": self.model.kernel,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        joblib.dump(self, path.with_suffix(".joblib"))

    @classmethod
    def load(cls, path: pathlib.Path) -> "SVMPipeline":
        joblib_path = path.with_suffix(".joblib")
        if joblib_path.exists():
            loaded = joblib.load(joblib_path)
            if not isinstance(loaded, cls):
                raise TypeError(f"Expected SVMPipeline in {joblib_path}, got {type(loaded)}")
            return loaded

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        scaler = MinMaxScaler()
        scaler.data_min_ = np.array(data["scaler_min"], dtype=np.float64)
        scaler.data_max_ = np.array(data["scaler_max"], dtype=np.float64)
        scaler.feature_names_in_ = np.array(data["feature_names"], dtype=object)
        scaler.n_features_in_ = len(data["feature_names"])
        _rng = scaler.data_max_ - scaler.data_min_
        _rng = np.where(_rng == 0, 1.0, _rng)
        scaler.scale_ = 1.0 / _rng
        scaler.min_ = -scaler.data_min_ * scaler.scale_

        coef = np.asarray(data["coef"], dtype=np.float64)
        intercept = np.asarray(data["intercept"], dtype=np.float64).ravel()
        n_features = coef.shape[1]
        # Legacy JSON: sklearn SVC no longer allows assigning coef_; rebuild as LinearSVC (same w·x+b form).
        model = LinearSVC(
            C=float(data["model_C"]),
            dual=False,
            random_state=42,
        )
        model.fit(np.zeros((2, n_features), dtype=np.float64), np.array([0, 1], dtype=np.int64))
        model.coef_[:] = coef
        model.intercept_[:] = intercept

        return cls(
            scaler=scaler,
            model=model,
            feature_names=data["feature_names"],
            drop_cols=data.get("drop_cols", []),
        )


# ─── Deduplication key ──────────────────────────────────────────────────────
DEDUP_KEY = [
    "sku_counts",
    "sku_average_volume",
    "sku_length_avg",
    "sku_width_avg",
    "sku_height_avg",
    "sku_length_var",
    "sku_width_var",
    "sku_height_var",
    "vehicle_length",
    "vehicle_width",
    "vehicle_height",
    "vehicle_capacity",
    "sku_max_length",
    "sku_min_length",
    "sku_std_length",
    "sku_max_width",
    "sku_min_width",
    "sku_std_width",
]


def load_raw_data(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load and deduplicate raw CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Deduplicate: keep first occurrence per key
    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEY, keep="first")
    df = df.reset_index(drop=True)
    print(f"Loaded {before} rows, dedup → {len(df)} rows")

    return df


def engineer_features(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    """Drop unwanted columns and return feature matrix."""
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Separate target
    if "if_loaded" not in df.columns:
        raise ValueError("if_loaded column missing from data")
    y = df["if_loaded"].values
    df = df.drop(columns=["if_loaded"])

    return df, y


def train_svm(
    data_csv: pathlib.Path,
    test_size: float = 0.25,
    random_state: int = 42,
    C: float = 10.0,
) -> tuple[SVMPipeline, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full training pipeline: load → dedup → engineer → split → scale → SVM.

    Returns
    -------
    pipeline : SVMPipeline
        Trained scaler + SVM model
    X_train, X_test, y_train, y_test : ndarray
    """
    df = load_raw_data(data_csv)

    # Separate GT before dropping
    if "if_loaded" not in df.columns:
        raise ValueError("if_loaded missing")
    y_full = df["if_loaded"].values

    # Engineer features
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X_full = df.drop(columns=cols_to_drop)
    feature_names = list(X_full.columns)

    # Split
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_state
    )

    # Scale
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM (probability=False for speed; we use Platt-style sigmoid separately)
    model = SVC(
        kernel="linear",
        C=C,
        class_weight={1: 1},
        probability=False,  # faster: we use decision_function + sigmoid ourselves
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    pipeline = SVMPipeline(
        scaler=scaler,
        model=model,
        feature_names=feature_names,
        drop_cols=cols_to_drop,
    )

    # Quick sanity print
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")

    return pipeline, X_train_scaled, X_test_scaled, y_train, y_test


def load_svm_pipeline(model_path: pathlib.Path) -> SVMPipeline:
    """Load a saved SVMPipeline from JSON."""
    return SVMPipeline.load(model_path)


# ─── Probability helpers ──────────────────────────────────────────────────────


def decision_to_proba(decision_vals: np.ndarray, y_train: np.ndarray, X_train: np.ndarray) -> tuple:
    """
    Fit Platt scaling: sigmoid on decision_function output.
    Uses sklearn's sigmoid_calibration approach on training set.

    Returns (a, b) such that P(y=1) = sigmoid(a * decision + b).
    """
    from sklearn.calibration import CalibratedClassifierCV

    # Refit with probability=True for calibration wrapper
    clf = SVC(kernel="linear", C=10.0, class_weight={1: 1}, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    # sklearn CalibratedClassifierCV does isotonic/sigmoid internally
    # We just expose the calibrated clf
    return clf


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> tuple:
    """
    Compute TPR (recall) at a specific FPR via ROC curve interpolation.
    """
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(fpr_arr - target_fpr))
    return float(tpr_arr[idx]), float(thresholds[idx])


def print_evaluation(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Print standard evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n── Evaluation ──────────────────────────────")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"ROC AUC   : {roc_auc_score(y_true, y_score):.4f}")
    print(f"CM        : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"FPR       : {fpr:.4f}")

    tpr_at_1pct, thresh = tpr_at_fpr(y_true, y_score, 0.01)
    print(f"TPR@FPR=1%: {tpr_at_1pct:.4f}  (threshold={thresh:.4f})")
    print("─────────────────────────────────────────────\n")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
        "cm": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "fpr": fpr,
        "tpr_at_fpr1pct": tpr_at_1pct,
        "threshold_at_fpr1pct": thresh,
    }


def format_svm_linear_insights_for_prompt(pipeline: SVMPipeline, top_k: int = 30) -> str:
    """
    Markdown for the LLM: linear SVM weights on **MinMax-scaled** features.

    Coefficients apply to scaled inputs x' = (x - min) / (max - min) from training;
    rule patches use raw `features` dict — use this table to see which dimensions
    the SVM trusts most and in which direction (toward class 1 vs 0).
    """
    model = pipeline.model
    coef_arr = getattr(model, "coef_", None)
    if coef_arr is None:
        return "_Linear SVM coefficients not available for this model._\n"

    coef = np.asarray(coef_arr).ravel()
    names = list(pipeline.feature_names)
    if len(coef) != len(names):
        return "_Coefficient vector length does not match feature_names._\n"

    intercept = float(np.asarray(model.intercept_).ravel()[0])
    pairs = sorted(zip(names, coef), key=lambda t: abs(t[1]), reverse=True)
    shown = pairs[: max(1, min(top_k, len(pairs)))]

    lines = [
        "The classifier is **linear** on MinMax-scaled features: "
        "`decision = w·x_scaled + b`, then label 1 if decision ≥ 0.",
        f"- Intercept `b` = **{intercept:.6g}**",
        "- **|w_j|** large → SVM is sensitive to that feature (in scaled space).",
        "- **w_j > 0** → higher scaled value pushes toward **feasible (1)**; **w_j < 0** → toward **not feasible (0)**.",
        "",
        f"| rank | feature | w_j (on scaled x) | |w_j| |",
        "|------|---------|---------------------|------|",
    ]
    for i, (name, w) in enumerate(shown, 1):
        lines.append(f"| {i} | `{name}` | {w:.6g} | {abs(w):.6g} |")

    if len(pairs) > len(shown):
        lines.append("")
        lines.append(f"_({len(pairs) - len(shown)} more features omitted; smallest |w| omitted.)_")

    lines += [
        "",
        "When designing `apply_rule_patch`, raw feature values are **not** scaled like `x_scaled`; "
        "combine this table with the **Feature scales** section to reason about conflicts on hard cases.",
    ]
    return "\n".join(lines)
