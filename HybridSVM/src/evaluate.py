# -*- coding: utf-8 -*-
"""
Evaluation harness: compare SVM vs Hybrid (SVM + evolved rules).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


@dataclass
class EvalResult:
    """Single evaluation result."""

    name: str
    accuracy: float
    precision: float
    recall: float
    auc: float
    tn: int
    fp: int
    fn: int
    tp: int
    fpr: float
    fnr: float
    tpr_at_fpr1pct: float
    hard_case_acc: float  # accuracy on FN + FP cases only
    n_hard_cases: int


def _eval_on(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    hard_mask: np.ndarray | None = None,
) -> EvalResult:
    """Compute all metrics for a given (name, y_true, y_pred, y_score)."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # TPR at FPR=1%
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(fpr_arr - 0.01))
    tpr_at_1pct = float(tpr_arr[idx])

    # Hard case accuracy (FN + FP only)
    if hard_mask is not None:
        hard_acc = accuracy_score(y_true[hard_mask], y_pred[hard_mask])
        n_hard = int(np.sum(hard_mask))
    else:
        hard_acc = 0.0
        n_hard = 0

    return EvalResult(
        name=name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        auc=roc_auc_score(y_true, y_score),
        tn=tn, fp=fp, fn=fn, tp=tp,
        fpr=fpr, fnr=fnr,
        tpr_at_fpr1pct=tpr_at_1pct,
        hard_case_acc=hard_acc,
        n_hard_cases=n_hard,
    )


def evaluate_pipeline(
    y_true: np.ndarray,
    svm_pred: np.ndarray,
    svm_probs: np.ndarray,
    hybrid_pred: np.ndarray,
    hybrid_probs: np.ndarray | None = None,
    fn_mask: np.ndarray | None = None,
    fp_mask: np.ndarray | None = None,
) -> tuple[EvalResult, EvalResult]:
    """
    Compare SVM baseline vs Hybrid (SVM + evolved rules).

    Parameters
    ----------
    y_true       : ground truth (0/1)
    svm_pred     : SVM hard predictions
    svm_probs    : SVM P(y=1)
    hybrid_pred   : Hybrid hard predictions
    hybrid_probs  : Hybrid P(y=1) (optional, falls back to svm_probs)
    fn_mask      : bool array, True for FN case indices
    fp_mask      : bool array, True for FP case indices

    Returns
    -------
    (svm_result, hybrid_result)
    """
    hard_mask = None
    if fn_mask is not None or fp_mask is not None:
        hard_mask = fn_mask | fp_mask if fn_mask is not None else fp_mask

    svm_result = _eval_on("SVM", y_true, svm_pred, svm_probs, hard_mask)

    # Hybrid probs: approximate by flipping svm_probs at overridden indices
    if hybrid_probs is None:
        hybrid_probs = svm_probs.copy()
        diff_mask = svm_pred != hybrid_pred
        # For flipped predictions, push probability in correct direction
        hybrid_probs[diff_mask & (hybrid_pred == 1)] = 0.95
        hybrid_probs[diff_mask & (hybrid_pred == 0)] = 0.05

    hybrid_result = _eval_on("Hybrid", y_true, hybrid_pred, hybrid_probs, hard_mask)

    return svm_result, hybrid_result


def print_comparison(svm_res: EvalResult, hybrid_res: EvalResult) -> None:
    """Print a side-by-side comparison table."""

    def row(label: str, svm_val, hybrid_val: float, fmt=".4f") -> str:
        delta = hybrid_val - svm_val
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        s_str = f"{svm_val:{fmt}}" if isinstance(svm_val, float) else str(svm_val)
        h_str = f"{hybrid_val:{fmt}}" if isinstance(hybrid_val, float) else str(hybrid_val)
        d_str = f"{abs(delta):{fmt}}" if isinstance(delta, float) else str(abs(delta))
        return f"  {label:<30} SVM={s_str}  Hybrid={h_str}  ({arrow}{d_str})"

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║                SVM vs Hybrid (SVM + Evolved Rules) Comparison       ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")

    for label, sv, hv in [
        ("Accuracy",        svm_res.accuracy,       hybrid_res.accuracy),
        ("Precision",       svm_res.precision,      hybrid_res.precision),
        ("Recall (TPR)",    svm_res.recall,        hybrid_res.recall),
        ("ROC AUC",         svm_res.auc,            hybrid_res.auc),
        ("FPR",             svm_res.fpr,            hybrid_res.fpr),
        ("FNR",             svm_res.fnr,            hybrid_res.fnr),
        ("TPR@FPR=1%",     svm_res.tpr_at_fpr1pct, hybrid_res.tpr_at_fpr1pct),
        (
            "Hard-subset acc (FN+FP)",
            svm_res.hard_case_acc,
            hybrid_res.hard_case_acc,
        ),
    ]:
        print(row(label, sv, hv))
        print("╠══════════════════════════════════════════════════════════════════════╣")

    svm_cm = f"{svm_res.tn}/{svm_res.fp}/{svm_res.fn}/{svm_res.tp}"
    hybrid_cm = f"{hybrid_res.tn}/{hybrid_res.fp}/{hybrid_res.fn}/{hybrid_res.tp}"
    print(f"  {'TN / FP / FN / TP':<30}  SVM={svm_cm}  Hybrid={hybrid_cm}")

    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Hard cases (FN+FP): {svm_res.n_hard_cases}")
    print(
        "  Note: Accuracy on this subset uses only rows where SVM erred (FN or FP); "
        "SVM accuracy there is always 0.0 by definition. Hybrid shows how many of those errors are fixed."
    )
    print(f"  Improvement in hard case acc: {hybrid_res.hard_case_acc - svm_res.hard_case_acc:+.4f}")
    print(f"  Improvement in FPR:          {hybrid_res.fpr - svm_res.fpr:+.4f}")
    print(f"  Improvement in FNR:          {hybrid_res.fnr - svm_res.fnr:+.4f}")


def compare_svm_vs_hybrid(
    y_true: np.ndarray,
    svm_pred: np.ndarray,
    svm_probs: np.ndarray,
    hybrid_pred: np.ndarray,
    hybrid_probs: np.ndarray | None = None,
    fn_mask: np.ndarray | None = None,
    fp_mask: np.ndarray | None = None,
) -> tuple[EvalResult, EvalResult]:
    """Main entry point."""
    svm_res, hybrid_res = evaluate_pipeline(
        y_true, svm_pred, svm_probs,
        hybrid_pred, hybrid_probs, fn_mask, fp_mask,
    )
    print_comparison(svm_res, hybrid_res)
    return svm_res, hybrid_res
