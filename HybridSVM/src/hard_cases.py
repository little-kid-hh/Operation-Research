# -*- coding: utf-8 -*-
"""
Hard case mining: extract FP / FN / EasyTP / EasyTN from SVM predictions
and summarize them for feeding to the LLM evolution loop.

Design:
  - "Hard cases" = cases where SVM is uncertain or wrong
  - FN (False Negative) = SVM predicts NO but GT = YES  → SVM under-confident on doable orders
  - FP (False Positive) = SVM predicts YES but GT = NO  → SVM over-confident on undoable orders
  - We mine these and extract their feature signatures
  - The LLM evolution loop will use these as "contrastive examples" to design rule patches
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class Case:
    """Single evaluation case."""

    dispatch_id: int
    svm_prob: float          # P(y=1) from SVM
    svm_pred: int           # 0/1 hard label
    gt: int                 # ground truth
    features: dict          # raw feature dict
    is_fn: bool = False
    is_fp: bool = False
    is_tn: bool = False
    is_tp: bool = False

    @property
    def error_type(self) -> str:
        if self.is_fn:
            return "FN"
        if self.is_fp:
            return "FP"
        if self.is_tn:
            return "TN"
        if self.is_tp:
            return "TP"
        return "OK"


@dataclass
class HardCaseSummary:
    """Aggregated summary of a hard case group."""

    group: str                      # "FN", "FP", "TN", or "TP"
    count: int
    pct_of_group: float             # % this group represents among all GT=group
    avg_prob: float
    std_prob: float
    top_features: list[tuple]       # (feature, mean_val, std_val) sorted by variance
    typical_cases: list[Case]       # k most representative cases


def _compute_top_variance(
    cases: list[Case], feature_names: list[str], k: int = 5
) -> list[tuple]:
    """Compute which features vary most across the case set."""
    if not cases:
        return []

    feat_vals: dict[str, list] = {f: [] for f in feature_names}
    for c in cases:
        for f in feature_names:
            feat_vals[f].append(c.features.get(f, np.nan))

    variances = []
    for f, vals in feat_vals.items():
        v = np.nanvar(vals)
        m = np.nanmean(vals)
        variances.append((f, m, v))

    variances.sort(key=lambda x: x[2], reverse=True)
    return variances[:k]


def mine_hard_cases(
    X_df: pd.DataFrame,
    y_true: np.ndarray,
    svm_probs: np.ndarray,
    threshold: float = 0.5,
    dispatch_ids: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
) -> tuple[list[Case], list[Case], list[Case], list[Case]]:
    """
    Mine error cases from model predictions.

    Parameters
    ----------
    X_df      : raw DataFrame (with dispatch_id column if available)
    y_true    : ground truth labels (0/1)
    svm_probs : P(y=1) from SVM (stored on each Case for analysis)
    threshold : decision boundary when y_pred is None (default 0.5)
    dispatch_ids : optional dispatch_id array for labeling
    y_pred    : optional hard labels (0/1), e.g. Hybrid — if set, used instead of thresholding svm_probs

    Returns
    -------
    fn_cases, fp_cases, easy_tn, easy_tp
    """
    if y_pred is not None:
        svm_pred = np.asarray(y_pred, dtype=int).ravel()
    else:
        svm_pred = (svm_probs >= threshold).astype(int)
    feature_names = list(X_df.columns)

    fn_cases: list[Case] = []
    fp_cases: list[Case] = []
    easy_tn: list[Case] = []
    easy_tp: list[Case] = []

    for i in range(len(y_true)):
        dispatch_id = int(dispatch_ids[i]) if dispatch_ids is not None else i
        features = X_df.iloc[i].to_dict()
        prob = float(svm_probs[i])
        pred = int(svm_pred[i])
        gt = int(y_true[i])

        c = Case(
            dispatch_id=dispatch_id,
            svm_prob=prob,
            svm_pred=pred,
            gt=gt,
            features=features,
            is_fn=(gt == 1 and pred == 0),
            is_fp=(gt == 0 and pred == 1),
            is_tn=(gt == 0 and pred == 0),
            is_tp=(gt == 1 and pred == 1),
        )

        if c.is_fn:
            fn_cases.append(c)
        elif c.is_fp:
            fp_cases.append(c)
        elif c.is_tn:
            easy_tn.append(c)
        elif c.is_tp:
            easy_tp.append(c)

    return fn_cases, fp_cases, easy_tn, easy_tp


def summarize_hard_cases(
    fn_cases: list[Case],
    fp_cases: list[Case],
    all_y_true: np.ndarray,
    feature_names: list[str],
    k: int = 5,
    n_typical: int = 3,
) -> dict:
    """
    Build structured summaries of FN and FP case groups for LLM consumption.

    Output is a dict with keys: 'fn_summary', 'fp_summary', 'metadata'.
    """
    n_gt1 = int(np.sum(all_y_true))
    n_gt0 = int(np.sum(1 - all_y_true))

    def build_summary(group_cases: list[Case], group_name: str) -> HardCaseSummary:
        if not group_cases:
            return HardCaseSummary(
                group=group_name,
                count=0,
                pct_of_group=0.0,
                avg_prob=0.0,
                std_prob=0.0,
                top_features=[],
                typical_cases=[],
            )

        probs = [c.svm_prob for c in group_cases]
        top_feat = _compute_top_variance(group_cases, feature_names, k=k)

        # Pick most extreme cases by prob distance from threshold
        sorted_cases = sorted(
            group_cases,
            key=lambda c: abs(c.svm_prob - 0.5),
            reverse=True,
        )
        typical = sorted_cases[:n_typical]

        total_in_group = n_gt1 if group_name == "FN" else n_gt0
        pct = len(group_cases) / total_in_group * 100 if total_in_group > 0 else 0.0

        return HardCaseSummary(
            group=group_name,
            count=len(group_cases),
            pct_of_group=pct,
            avg_prob=mean(probs),
            std_prob=np.std(probs) if len(probs) > 1 else 0.0,
            top_features=top_feat,
            typical_cases=typical,
        )

    fn_sum = build_summary(fn_cases, "FN")
    fp_sum = build_summary(fp_cases, "FP")

    return {
        "fn_summary": fn_sum,
        "fp_summary": fp_sum,
        "metadata": {
            "total_gt1": n_gt1,
            "total_gt0": n_gt0,
            "fn_rate": len(fn_cases) / n_gt1 if n_gt1 > 0 else 0.0,
            "fp_rate": len(fp_cases) / n_gt0 if n_gt0 > 0 else 0.0,
        },
    }


def _case_feature_vector(case: Case, feature_names: list[str]) -> np.ndarray:
    return np.array([float(case.features.get(f, np.nan)) for f in feature_names], dtype=float)


def _centroid_feature_vector(cases: list[Case], feature_names: list[str]) -> np.ndarray | None:
    if not cases:
        return None
    mat = np.row_stack([_case_feature_vector(c, feature_names) for c in cases])
    return np.nanmean(mat, axis=0)


def _l2_to_centroid(case: Case, centroid: np.ndarray, feature_names: list[str]) -> float:
    v = _case_feature_vector(case, feature_names)
    d = v - centroid
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.linalg.norm(d))


def pick_near_confusable_cases(
    reference_cases: list[Case],
    pool: list[Case],
    feature_names: list[str],
    m: int,
) -> list[Case]:
    """
    From ``pool``, pick up to ``m`` cases closest in L2 (raw feature space) to the
    centroid of ``reference_cases`` (e.g. easy TP near FN cloud, easy TN near FP cloud).
    """
    if m <= 0 or not reference_cases or not pool:
        return []
    centroid = _centroid_feature_vector(reference_cases, feature_names)
    if centroid is None:
        return []
    ranked = sorted(
        pool,
        key=lambda c: _l2_to_centroid(c, centroid, feature_names),
    )
    return ranked[:m]


def summarize_easy_cases(
    easy_tn: list[Case],
    easy_tp: list[Case],
    all_y_true: np.ndarray,
    feature_names: list[str],
    k: int = 5,
    n_typical: int = 3,
) -> tuple[HardCaseSummary, HardCaseSummary]:
    """
    Summarize SVM-correct easy TN / TP for contrastive prompting.

    Typical rows are those **closest to the decision boundary** (smallest |prob - 0.5|),
    i.e. borderline-correct examples — complementary to FN/FP typicals (often far from 0.5).
    """
    n_gt1 = int(np.sum(all_y_true))
    n_gt0 = int(len(all_y_true) - n_gt1)

    def _one(group_cases: list[Case], group_name: str, denom: int) -> HardCaseSummary:
        if not group_cases:
            return HardCaseSummary(
                group=group_name,
                count=0,
                pct_of_group=0.0,
                avg_prob=0.0,
                std_prob=0.0,
                top_features=[],
                typical_cases=[],
            )
        probs = [c.svm_prob for c in group_cases]
        top_feat = _compute_top_variance(group_cases, feature_names, k=k)
        # Borderline-correct: closest to threshold first
        sorted_cases = sorted(
            group_cases,
            key=lambda c: abs(c.svm_prob - 0.5),
        )
        typical = sorted_cases[:n_typical]
        pct = len(group_cases) / denom * 100 if denom > 0 else 0.0
        return HardCaseSummary(
            group=group_name,
            count=len(group_cases),
            pct_of_group=pct,
            avg_prob=mean(probs),
            std_prob=np.std(probs) if len(probs) > 1 else 0.0,
            top_features=top_feat,
            typical_cases=typical,
        )

    tn_sum = _one(easy_tn, "TN", n_gt0)
    tp_sum = _one(easy_tp, "TP", n_gt1)
    return tn_sum, tp_sum


def summarize_for_evolution_prompt(
    fn_cases: list[Case],
    fp_cases: list[Case],
    easy_tn: list[Case],
    easy_tp: list[Case],
    all_y_true: np.ndarray,
    feature_names: list[str],
    *,
    k: int = 5,
    n_typical_hard: int = 3,
    n_easy_typical: int = 3,
    n_near_hard_easy: int = 3,
) -> dict:
    """
    Full structured summary for the evolution LLM: FN/FP + easy TN/TP + optional
    confusable-but-correct rows (easy TP near FN pattern, easy TN near FP pattern).
    """
    out = summarize_hard_cases(
        fn_cases,
        fp_cases,
        all_y_true,
        feature_names,
        k=k,
        n_typical=n_typical_hard,
    )
    tn_s, tp_s = summarize_easy_cases(
        easy_tn,
        easy_tp,
        all_y_true,
        feature_names,
        k=k,
        n_typical=n_easy_typical,
    )
    out["tn_summary"] = tn_s
    out["tp_summary"] = tp_s
    if n_near_hard_easy > 0:
        out["near_tp_like_fn"] = pick_near_confusable_cases(
            fn_cases, easy_tp, feature_names, n_near_hard_easy
        )
        out["near_tn_like_fp"] = pick_near_confusable_cases(
            fp_cases, easy_tn, feature_names, n_near_hard_easy
        )
    else:
        out["near_tp_like_fn"] = []
        out["near_tn_like_fp"] = []
    return out


def _typical_case_line(case: Case) -> str:
    return (
        f"  - dispatch={case.dispatch_id}, prob={case.svm_prob:.4f}, "
        f"sku_counts={case.features.get('sku_counts', 'N/A')}, "
        f"fill_ratio={case.features.get('fill_ratio', 'N/A')}"
    )


def render_hard_case_prompt(
    summary: dict,
    *,
    analysis_title: str = "## 3D-BPP Hard Case Analysis",
    predictor_name: str = "SVM",
    prob_column_caption: str = "SVM probability",
    include_llm_instruction: bool = True,
) -> str:
    """
    Render a hard case summary into an LLM prompt (Markdown).
    Suitable as a 'debug引导词' to feed to the LLM for rule generation.
    """
    lines = [
        analysis_title,
        "",
        f"### 1. FN Cases ({predictor_name} predicts NO, GT = YES — under-confident)",
        f"- Count: {summary['fn_summary'].count} / {summary['metadata']['total_gt1']} = {summary['fn_summary'].pct_of_group:.2f}%",
        f"- Avg {prob_column_caption}: {summary['fn_summary'].avg_prob:.4f}",
        f"- Std probability: {summary['fn_summary'].std_prob:.4f}",
    ]

    if summary["fn_summary"].top_features:
        lines.append("- Most variable features:")
        for feat, mean_val, var_val in summary["fn_summary"].top_features:
            lines.append(f"  - `{feat}`: mean={mean_val:.4f}, var={var_val:.4f}")

    if summary["fn_summary"].typical_cases:
        lines.append("- Typical FN cases:")
        for case in summary["fn_summary"].typical_cases:
            lines.append(_typical_case_line(case))

    lines += [
        "",
        f"### 2. FP Cases ({predictor_name} predicts YES, GT = NO — over-confident)",
        f"- Count: {summary['fp_summary'].count} / {summary['metadata']['total_gt0']} = {summary['fp_summary'].pct_of_group:.2f}%",
        f"- Avg {prob_column_caption}: {summary['fp_summary'].avg_prob:.4f}",
        f"- Std probability: {summary['fp_summary'].std_prob:.4f}",
    ]

    if summary["fp_summary"].top_features:
        lines.append("- Most variable features:")
        for feat, mean_val, var_val in summary["fp_summary"].top_features:
            lines.append(
                f"  - `{feat}`: mean={mean_val:.4f}, var={var_val:.4f}"
            )

    if summary["fp_summary"].typical_cases:
        lines.append("- Typical FP cases:")
        for i, case in enumerate(summary["fp_summary"].typical_cases, 1):
            lines.append(_typical_case_line(case))

    if "tn_summary" in summary and summary["tn_summary"] is not None:
        tn = summary["tn_summary"]
        meta = summary["metadata"]
        lines += [
            "",
            "### 3. Easy TN (SVM predicts NO, GT = NO — correct negatives)",
            f"- Count: {tn.count} / {meta['total_gt0']} = {tn.pct_of_group:.2f}% of GT=NO",
            f"- Avg {prob_column_caption}: {tn.avg_prob:.4f}",
            f"- Std probability: {tn.std_prob:.4f}",
        ]
        if tn.top_features:
            lines.append("- Most variable features:")
            for feat, mean_val, var_val in tn.top_features:
                lines.append(f"  - `{feat}`: mean={mean_val:.4f}, var={var_val:.4f}")
        if tn.typical_cases:
            lines.append(
                "- Borderline-correct TN (closest to prob=0.5 — do not harm these patterns):"
            )
            for case in tn.typical_cases:
                lines.append(_typical_case_line(case))

    if "tp_summary" in summary and summary["tp_summary"] is not None:
        tp = summary["tp_summary"]
        meta = summary["metadata"]
        lines += [
            "",
            "### 4. Easy TP (SVM predicts YES, GT = YES — correct positives)",
            f"- Count: {tp.count} / {meta['total_gt1']} = {tp.pct_of_group:.2f}% of GT=YES",
            f"- Avg {prob_column_caption}: {tp.avg_prob:.4f}",
            f"- Std probability: {tp.std_prob:.4f}",
        ]
        if tp.top_features:
            lines.append("- Most variable features:")
            for feat, mean_val, var_val in tp.top_features:
                lines.append(f"  - `{feat}`: mean={mean_val:.4f}, var={var_val:.4f}")
        if tp.typical_cases:
            lines.append(
                "- Borderline-correct TP (closest to prob=0.5 — do not harm these patterns):"
            )
            for case in tp.typical_cases:
                lines.append(_typical_case_line(case))

    near_tp = summary.get("near_tp_like_fn") or []
    near_tn = summary.get("near_tn_like_fp") or []
    if near_tp:
        lines += [
            "",
            "### 5a. Easy TP rows (feature-similar to FN cloud — SVM still correct)",
            "These are **positive** examples that sit near FN cases in feature space; "
            "your rule must **not** override SVM on rows like these unless necessary.",
        ]
        for case in near_tp:
            lines.append(_typical_case_line(case))
    if near_tn:
        lines += [
            "",
            "### 5b. Easy TN rows (feature-similar to FP cloud — SVM still correct)",
            "These are **negative** examples that sit near FP cases in feature space; "
            "your rule must **not** override SVM on rows like these unless necessary.",
        ]
        for case in near_tn:
            lines.append(_typical_case_line(case))

    if include_llm_instruction:
        inst_num = "6" if "tn_summary" in summary else "3"
        lines += [
            "",
            f"### {inst_num}. Debug Instruction",
            (
                "请分析以上FN/FP错题与Easy TN/TP对照样本，找出SVM线性模型在错题上的失效规律，"
                "设计**窄**的条件：只在确有必要时覆盖SVM。"
                "对 Easy TN/TP 及 §5 中的对照行，除非逻辑上必须触发，否则应保持 `return -1`。"
                "输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，"
                "当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。"
            ),
        ]
    else:
        note_num = "6" if "tn_summary" in summary else "3"
        lines += [
            "",
            f"### {note_num}. Note",
            "",
            "Section titles refer to the **predictor** used for hard labels. "
            "Probability columns still show **SVM P(y=1)** on each row (reference score).",
        ]
    return "\n".join(lines)


# Columns most often referenced in evolved rules (listed first in stats table)
_RULE_FEATURE_PRIORITY = [
    "fill_ratio",
    "sku_counts",
    "spare_capacity",
    "max_asr",
    "sku_average_volume",
    "total_skuvolume",
    "vehicle_capacity",
    "sku_length_var",
    "sku_width_var",
    "sku_height_var",
    "l_to_L_ratio_avg",
    "wl_to_vehicle_wl_max",
]


def format_feature_stats_for_prompt(df: pd.DataFrame, max_columns: int = 48) -> str:
    """
    Build a Markdown table of min / max / median / p25 / p75 for numeric features.

    Use the same DataFrame as SVM features (e.g. test split) so the LLM does not
    guess wrong scales (e.g. fill_ratio is a ratio in [0,1], not m³).
    """
    if df is None or len(df) == 0:
        return "_No feature rows available._\n"

    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        return "_No numeric columns in feature frame._\n"

    cols = list(num_df.columns)
    ordered: list[str] = []
    for c in _RULE_FEATURE_PRIORITY:
        if c in cols and c not in ordered:
            ordered.append(c)
    for c in sorted(cols):
        if c not in ordered:
            ordered.append(c)
    ordered = ordered[:max_columns]

    lines = [
        "| Feature | min | p25 | median | p75 | max |",
        "|---------|-----|-----|--------|-----|-----|",
    ]
    for col in ordered:
        s = num_df[col].dropna()
        if s.empty:
            continue
        lines.append(
            f"| `{col}` | {s.min():.6g} | {s.quantile(0.25):.6g} | {s.median():.6g} | "
            f"{s.quantile(0.75):.6g} | {s.max():.6g} |"
        )

    lines.append("")
    lines.append(
        "**Units:** `fill_ratio` is the fraction of vehicle volume used (typically 0–1), "
        "not cubic meters. Compare thresholds to the table above."
    )
    return "\n".join(lines)
