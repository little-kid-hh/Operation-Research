# -*- coding: utf-8 -*-
"""
LLM-driven feature search utilities for the linear-SVM route.

Goal:
  - keep the existing SVM training/evaluation protocol fixed;
  - let an LLM propose interpretable per-dispatch features;
  - evaluate those features against the same holdout split;
  - persist policy / memory / trial artifacts for iterative search.
"""

from __future__ import annotations

import json
import pathlib
import re
import textwrap
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.svm_train import (
    DROP_COLS,
    fit_svm_pipeline,
    format_svm_linear_insights_for_prompt,
    load_raw_data,
    tpr_at_fpr,
)


DEFAULT_FEATURE_POLICY = textwrap.dedent(
    """\
    # Feature Search Policy

    Objective:
    Improve the linear SVM baseline by adding a small number of interpretable,
    per-dispatch numeric features.

    Hard constraints:
    - Do not change the train/test split.
    - Do not change the model family: keep the stage-1 model as linear SVM.
    - Do not use labels or any target-derived statistics inside feature generation.
    - Do not read files or call external services in candidate feature code.
    - Return one row per dispatch and keep feature names ASCII.

    Preferred feature families:
    - item-level quantiles / tails rather than only mean/std
    - thresholded local bottleneck counts, not only global averages
    - slack / spare-capacity interactions with large-piece pressure
    - repeated-type structure and concentration
    - extreme-piece bottlenecks against vehicle dimensions
    - face-area / edge-pressure style packing stress signals
    - heterogeneity / multimodality / long-tail measures
    - features that are easy to explain to a human reviewer

    Avoid:
    - duplicating obvious existing aggregates unless the new version captures a
      different shape signal
    - opaque embeddings
    - huge feature sets; prefer a compact hypothesis with clear semantics

    Tree-inspired search bias:
    - assume trees are winning partly because they exploit local thresholds and
      interactions on features like `spare_capacity`, `wl_to_vehicle_wl_max`,
      `wl_to_vehicle_wl_total`, `sku_average_volume`, and dimension variances
    - try to convert those nonlinear effects into explicit numeric features that
      a linear SVM can use
    """
)


SEED_POLICY_UPDATE = textwrap.dedent(
    """\
    Keep focusing on compact, interpretable item-distribution features with
    explicit tree-inspired threshold and interaction structure.

    Near-term hypothesis:
    1. repeated-type concentration matters because many identical pieces can be
       easier to tessellate than equally sized but highly diverse piece sets;
    2. upper-tail dimension pressure matters more than means, especially along
       the longest and middle sorted item dimensions;
    3. low-spare-capacity regimes likely interact with large-piece pressure in a
       way the linear baseline cannot express directly;
    4. compact interaction-heavy candidates are preferred over broad feature
       bundles that are harder to ablate.
    """
)


SEED_FEATURE_CODE = textwrap.dedent(
    """\
    import numpy as np
    import pandas as pd

    def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
        base = agg_df[[
            "dispatch_id",
            "vehicle_length",
            "vehicle_width",
            "vehicle_height",
            "vehicle_capacity",
            "spare_capacity",
        ]].copy()

        bin_sorted = np.sort(
            base[["vehicle_length", "vehicle_width", "vehicle_height"]].to_numpy(dtype=float),
            axis=1,
        )
        base["bin_s"] = bin_sorted[:, 0]
        base["bin_m"] = bin_sorted[:, 1]
        base["bin_l"] = bin_sorted[:, 2]
        base["slack_ratio"] = base["spare_capacity"] / np.maximum(base["vehicle_capacity"], 1.0)

        work = items_df.merge(
            base[["dispatch_id", "bin_s", "bin_m", "bin_l", "vehicle_length", "vehicle_width", "slack_ratio"]],
            on="dispatch_id",
            how="left",
        )

        g = work.groupby("dispatch_id", sort=False)

        out = pd.DataFrame({"dispatch_id": list(g.groups.keys())})
        out["dominant_type_share"] = (
            g.apply(
                lambda x: (
                    x.groupby(["dim_s", "dim_m", "dim_l"]).size().max() / max(len(x), 1)
                )
            ).astype(float).values
        )
        out["p90_long_over_bin_long"] = (
            g.apply(lambda x: float(np.quantile(x["dim_l"], 0.90)) / max(float(x["bin_l"].iloc[0]), 1.0)).values
        )
        out["p90_mid_over_bin_mid"] = (
            g.apply(lambda x: float(np.quantile(x["dim_m"], 0.90)) / max(float(x["bin_m"].iloc[0]), 1.0)).values
        )
        out["thin_item_share"] = (
            g.apply(lambda x: float((x["dim_s"] / np.maximum(x["dim_l"], 1.0) < 0.35).mean())).values
        )
        out["max_face_area_load_over_floor"] = (
            g.apply(
                lambda x: float((x["dim_m"] * x["dim_l"]).sum())
                / max(float(x["vehicle_length"].iloc[0] * x["vehicle_width"].iloc[0]), 1.0)
            ).values
        )
        out["tight_bin_large_piece_interaction"] = (
            g.apply(
                lambda x: float(np.quantile(x["dim_l"], 0.90))
                / max(float(x["bin_l"].iloc[0]), 1.0)
                * max(0.0, 0.20 - float(x["slack_ratio"].iloc[0]))
            ).values
        )

        return out
    """
)


SEED_RATIONALE = textwrap.dedent(
    """\
    This seed candidate intentionally uses a small set of explainable
    item-distribution features:

    - `dominant_type_share` captures repetition versus heterogeneity at
      dispatch level.
    - `p90_long_over_bin_long` and `p90_mid_over_bin_mid` emphasize upper-tail
      dimension pressure instead of means.
    - `thin_item_share` captures whether many items have one very small side,
      which can help packing flexibility.
    - `max_face_area_load_over_floor` approximates aggregate face-pressure
      against the vehicle floor footprint.
    - `tight_bin_large_piece_interaction` is an explicit interaction term meant
      to linearize a regime that trees can usually capture more naturally.
    """
)


BANNED_CODE_SNIPPETS = [
    "read_csv",
    "to_csv",
    "open(",
    "pathlib",
    "subprocess",
    "joblib",
    "pickle",
    "if_loaded",
]


ROOT_RESERVED_COLUMNS = {
    "dispatch_id",
    "orderid",
    "if_loaded",
    "发车号",
}


@dataclass
class SearchData:
    raw_df: pd.DataFrame
    agg_input_df: pd.DataFrame
    items_input_df: pd.DataFrame
    base_feature_df: pd.DataFrame
    y: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class EvalArtifact:
    metrics: dict[str, Any]
    pipeline_feature_names: list[str]
    coefficients: list[dict[str, float]]
    new_feature_names: list[str]
    n_missing_dispatches: int
    feature_frame: pd.DataFrame
    pipeline: Any


@dataclass
class TrialRecord:
    iteration: int
    status: str
    trial_dir: str
    feature_names: list[str]
    n_new_features: int
    metrics: dict[str, Any] | None
    delta_vs_baseline: dict[str, float] | None
    objective_key: list[float] | None
    policy_excerpt: str
    rationale_excerpt: str
    error: str | None = None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -500, 500)))


def _metric_sort_key(metrics: dict[str, Any] | None) -> tuple[float, float, float]:
    if not metrics:
        return (-np.inf, -np.inf, -np.inf)
    return (
        float(metrics.get("auc", -np.inf)),
        float(metrics.get("tpr_at_fpr1pct", -np.inf)),
        float(metrics.get("accuracy", -np.inf)),
    )


def _is_success_status(status: str | None) -> bool:
    return status in {"ok", "accepted"}


def _metric_is_better(
    candidate: dict[str, Any],
    current_best: dict[str, Any] | None,
    *,
    auc_margin: float = 5e-4,
    tpr_margin: float = 5e-3,
    accuracy_margin: float = 5e-4,
) -> bool:
    if current_best is None:
        return True

    cand_auc = float(candidate.get("auc", -np.inf))
    best_auc = float(current_best.get("auc", -np.inf))
    if cand_auc > best_auc + auc_margin:
        return True
    if cand_auc < best_auc - auc_margin:
        return False

    cand_tpr = float(candidate.get("tpr_at_fpr1pct", -np.inf))
    best_tpr = float(current_best.get("tpr_at_fpr1pct", -np.inf))
    if cand_tpr > best_tpr + tpr_margin:
        return True
    if cand_tpr < best_tpr - tpr_margin:
        return False

    cand_acc = float(candidate.get("accuracy", -np.inf))
    best_acc = float(current_best.get("accuracy", -np.inf))
    return cand_acc > best_acc + accuracy_margin


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr_at_1pct, thresh = tpr_at_fpr(y_true, y_score, 0.01)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tpr_at_fpr1pct": float(tpr_at_1pct),
        "threshold_at_fpr1pct": float(thresh),
    }


def _flatten_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    keys = [
        "accuracy",
        "precision",
        "recall",
        "auc",
        "fpr",
        "fnr",
        "tpr_at_fpr1pct",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    return {k: metrics.get(k) for k in keys}


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _prepare_items_input(items_path: pathlib.Path, dispatch_ids: set[int]) -> pd.DataFrame:
    raw = pd.read_csv(
        items_path,
        encoding="utf-8",
        usecols=[
            "发车号",
            "SKU长度",
            "SKU宽度",
            "SKU高度",
            "if_fragile",
            "load_parameter",
            "车辆总容积(m3)",
        ],
    )
    raw = raw[raw["发车号"].isin(dispatch_ids)].copy()
    raw = raw.rename(
        columns={
            "发车号": "dispatch_id",
            "SKU长度": "item_length",
            "SKU宽度": "item_width",
            "SKU高度": "item_height",
            "车辆总容积(m3)": "vehicle_capacity",
        }
    )
    dims = np.sort(
        raw[["item_length", "item_width", "item_height"]].to_numpy(dtype=float),
        axis=1,
    )
    raw["dim_s"] = dims[:, 0]
    raw["dim_m"] = dims[:, 1]
    raw["dim_l"] = dims[:, 2]
    raw["item_volume"] = raw["item_length"] * raw["item_width"] * raw["item_height"]
    raw["item_flatness"] = raw["dim_s"] / np.maximum(raw["dim_l"], 1.0)
    raw["dispatch_id"] = raw["dispatch_id"].astype(int)
    return raw.reset_index(drop=True)


def load_search_data(
    data_path: pathlib.Path,
    items_path: pathlib.Path,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> SearchData:
    raw_df = load_raw_data(data_path)
    dispatch_ids = set(raw_df["发车号"].astype(int).tolist())

    agg_input_df = raw_df.drop(columns=["if_loaded"]).copy()
    if "orderid" in agg_input_df.columns:
        agg_input_df = agg_input_df.drop(columns=["orderid"])
    agg_input_df = agg_input_df.rename(columns={"发车号": "dispatch_id"})
    agg_input_df["dispatch_id"] = agg_input_df["dispatch_id"].astype(int)

    items_input_df = _prepare_items_input(items_path, dispatch_ids)

    cols_to_drop = [c for c in DROP_COLS if c in raw_df.columns]
    base_feature_df = raw_df.drop(columns=cols_to_drop + ["if_loaded"]).reset_index(drop=True)
    y = raw_df["if_loaded"].to_numpy(dtype=int)

    indices = np.arange(len(raw_df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
    )
    return SearchData(
        raw_df=raw_df.reset_index(drop=True),
        agg_input_df=agg_input_df.reset_index(drop=True),
        items_input_df=items_input_df,
        base_feature_df=base_feature_df,
        y=y,
        train_idx=np.asarray(train_idx),
        test_idx=np.asarray(test_idx),
    )


def evaluate_svm_features(
    feature_df: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    svm_c: float = 10.0,
    random_state: int = 42,
    new_feature_names: list[str] | None = None,
    n_missing_dispatches: int = 0,
) -> EvalArtifact:
    X_train = feature_df.iloc[train_idx].reset_index(drop=True)
    X_test = feature_df.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    pipeline = fit_svm_pipeline(
        X_train_df=X_train,
        y_train=y_train,
        C=svm_c,
        random_state=random_state,
    )
    decision = pipeline.decision_score(X_test)
    prob = _sigmoid(decision)
    pred = (prob >= 0.5).astype(int)
    metrics = _metric_bundle(y_test, pred, prob)

    coef_map = dict(
        zip(
            pipeline.feature_names,
            np.asarray(pipeline.model.coef_).ravel().astype(float).tolist(),
        )
    )
    coeffs = []
    for name in (new_feature_names or []):
        if name in coef_map:
            coeffs.append(
                {
                    "feature": name,
                    "coef": float(coef_map[name]),
                    "abs_coef": float(abs(coef_map[name])),
                }
            )
    coeffs.sort(key=lambda x: x["abs_coef"], reverse=True)

    return EvalArtifact(
        metrics=metrics,
        pipeline_feature_names=list(pipeline.feature_names),
        coefficients=coeffs,
        new_feature_names=list(new_feature_names or []),
        n_missing_dispatches=int(n_missing_dispatches),
        feature_frame=feature_df,
        pipeline=pipeline,
    )


def build_context_markdown(
    data: SearchData,
    baseline_eval: EvalArtifact,
    *,
    tree_guidance_md: str = "",
    max_feature_names: int = 80,
    include_tree_hints: bool = True,
) -> str:
    base_cols = list(data.base_feature_df.columns)
    feature_preview = ", ".join(f"`{c}`" for c in base_cols[:max_feature_names])
    if len(base_cols) > max_feature_names:
        feature_preview += f", ... ({len(base_cols) - max_feature_names} more)"

    item_counts = data.items_input_df.groupby("dispatch_id").size()
    item_schema = ", ".join(f"`{c}`" for c in data.items_input_df.columns)
    baseline = baseline_eval.metrics
    svm_insights = format_svm_linear_insights_for_prompt(baseline_eval.pipeline, top_k=25)

    lines = [
        "# Context",
        "",
        "Task: improve the current linear-SVM route with a small number of interpretable,",
        "per-dispatch numeric features. The model family and split protocol are fixed.",
        "",
        "## Data Scope",
        f"- dispatch rows after deduplication: `{len(data.raw_df)}`",
        f"- train rows: `{len(data.train_idx)}`",
        f"- test rows: `{len(data.test_idx)}`",
        f"- positive rate: `{data.y.mean():.4f}`",
        f"- matched item rows for these dispatches: `{len(data.items_input_df)}`",
        f"- item-count per dispatch: mean=`{item_counts.mean():.3f}`, median=`{item_counts.median():.1f}`, max=`{item_counts.max()}`",
        "",
        "## Fixed Baseline Metrics",
        f"- Accuracy: `{baseline['accuracy']:.4f}`",
        f"- Precision: `{baseline['precision']:.4f}`",
        f"- Recall: `{baseline['recall']:.4f}`",
        f"- ROC AUC: `{baseline['auc']:.4f}`",
        f"- TPR@FPR=1%: `{baseline['tpr_at_fpr1pct']:.4f}`",
        "",
        "## Existing Aggregate Feature Columns",
        feature_preview,
        "",
        "## Item-Level Table Schema",
        item_schema,
        "",
        "Notes:",
        "- `agg_df` already contains per-dispatch aggregate features and vehicle dimensions.",
        "- `items_df` contains one row per item with both raw dimensions and sorted dimensions:",
        "  `dim_s <= dim_m <= dim_l`.",
        "- Candidate code must return one row per dispatch with ASCII feature names.",
        "",
        "## Current SVM Linear Insights",
        svm_insights,
    ]
    if include_tree_hints:
        lines.extend(
            [
                "",
                "## Tree-Model Contrast Hypotheses",
                "- Purely linear-style combinations did not close the gap to the stronger tree baselines;",
                "  the likely missing piece is explicit threshold / interaction structure rather than more",
                "  smooth averages of the same aggregates.",
                "- In local checks, tree models appear to recover some cases that linear SVM misses,",
                "  especially around `spare_capacity`, `sku_average_volume`, `wl_to_vehicle_wl_max`,",
                "  `wl_to_vehicle_wl_total`, `sku_length_avg`, and item-dimension variance patterns.",
                "- Translate those advantages into explicit features such as: upper-tail dimension pressure,",
                "  low-slack x large-piece interactions, near-limit piece counts/shares, repetition versus",
                "  heterogeneity signals, and footprint or wall-pressure proxies.",
                "- Treat this as a search hint: propose thresholded counts, tail-pressure features,",
                "  and interaction terms that might linearize those nonlinear regimes for the SVM.",
                "",
                "## Search Bias",
                "- prioritize compact, explainable feature sets",
                "- use item-distribution shape, tails, concentration, and bottleneck signals",
                "- explicitly test threshold-style and interaction-style hypotheses suggested by trees",
                "- avoid re-encoding what the existing aggregate means and variances already say",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Search Bias",
                "- prioritize compact, explainable feature sets grounded in packing logic",
                "- use item-distribution shape, tails, concentration, and bottleneck signals",
                "- derive candidate features from domain reasoning over vehicle limits and item geometry",
                "- avoid re-encoding what the existing aggregate means and variances already say",
            ]
        )
    if tree_guidance_md.strip():
        lines.extend(
            [
                "",
                "## Tree-Guidance Snapshot",
                "The following markdown is a frozen guidance snapshot derived from the",
                "current tree-model analysis workflow. It is part of the formal prompt",
                "input for this run and should be treated as a hypothesis source rather",
                "than ground truth.",
                "",
                tree_guidance_md.strip(),
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_generation_prompt(
    *,
    context_md: str,
    policy_md: str,
    memory_md: str,
    best_candidate_md: str,
    iteration: int,
    max_new_features: int,
    current_best_metrics: dict[str, Any],
    acceptance_rule: dict[str, float],
    domain_only: bool = False,
) -> str:
    search_hint_line = (
        "- Favor domain-grounded signals: threshold counts, tail ratios, pressure-slack interactions,\n"
        "  and local awkward-pattern shares."
        if domain_only
        else "- Favor tree-inspired signals: threshold counts, tail ratios, pressure-slack interactions,\n"
        "  and local awkward-pattern shares."
    )
    return textwrap.dedent(
        f"""\
        You are improving a fixed linear-SVM baseline for 3D bin-packing feasibility.

        Produce:
        1. a short policy update for the next iteration;
        2. one Python candidate that adds interpretable per-dispatch features.

        Hard rules:
        - Keep the model family fixed: the candidate only generates features.
        - Use only `agg_df` and `items_df` passed into the function.
        - No file I/O, no API calls, no labels, no target leakage.
        - Return one row per dispatch with `dispatch_id` plus numeric feature columns.
        - Use ASCII feature names only.
        - Prefer at most `{max_new_features}` new features.
        - Use only pandas and numpy.
        - Keep feature computations in pandas objects; do not call `.values` unless you
          immediately wrap the result back into a `pd.Series` or `pd.DataFrame`.
        - Do not call `.rename(...)` on numpy arrays.
        {search_hint_line}
        - You are in an iterative search loop with a cumulative active feature bank.
        - The evaluated model uses: base aggregate features + active feature bank + your new features.
        - Propose only new incremental features to add on top of the active feature bank.
        - Do not repeat, rename, or rewrite any feature already in the active bank.
        - Make a small, explicit local change: usually add 1-3 new feature ideas, not a reset.
        - Propose a candidate only if you believe the cumulative feature set can beat
          the current active bank under the acceptance rule below.

        Current best accepted target to beat:
        - AUC: `{current_best_metrics['auc']:.4f}`
        - TPR@FPR=1%: `{current_best_metrics['tpr_at_fpr1pct']:.4f}`
        - Accuracy: `{current_best_metrics['accuracy']:.4f}`

        Acceptance rule:
        - accept if AUC improves by more than `{acceptance_rule['auc_margin']:.4g}`
        - otherwise require TPR@FPR=1% improvement larger than `{acceptance_rule['tpr_margin']:.4g}`
        - if still tied, require Accuracy improvement larger than `{acceptance_rule['accuracy_margin']:.4g}`

        Candidate function signature:

        ```python
        def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
            ...
        ```

        Response format:

        ## POLICY_UPDATE
        <markdown bullets>

        ## FEATURE_CODE
        ```python
        ...
        ```

        ## RATIONALE
        - bullet 1
        - bullet 2

        Current iteration: `{iteration}`

        {context_md}

        {policy_md}

        {memory_md}

        {best_candidate_md}
        """
    ).strip() + "\n"


def build_repair_prompt(
    *,
    original_prompt: str,
    original_response: str,
    error_text: str,
) -> str:
    return textwrap.dedent(
        f"""\
        You produced a feature-search response that failed validation.

        Fix the candidate and return a new response in the exact same format:

        ## POLICY_UPDATE
        ...

        ## FEATURE_CODE
        ```python
        ...
        ```

        ## RATIONALE
        ...

        Failure summary:
        - error: {error_text}

        Original response:
        ```text
        {original_response}
        ```

        Original prompt:
        ```text
        {original_prompt}
        ```

        Repair rules:
        - Keep the same overall hypothesis if possible.
        - Preserve the exact function signature `build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame`.
        - Return one row per dispatch with `dispatch_id` and only numeric feature columns.
        - Avoid `.values` unless absolutely necessary.
        - Never call `.rename(...)` on numpy arrays.
        - Prefer pandas Series / DataFrame objects all the way through.
        - Make the code syntactically valid and executable.
        """
    ).strip() + "\n"


def ablation_markdown(ablation: dict[str, Any] | None, *, top_k: int = 3) -> str:
    if not ablation:
        return "_No ablation summary available._"

    singles = sorted(
        ablation.get("single_feature", []),
        key=lambda row: (
            float(row.get("delta_auc", float("-inf"))),
            float(row.get("delta_tpr_at_fpr1pct", float("-inf"))),
        ),
        reverse=True,
    )
    loo = sorted(
        ablation.get("leave_one_out", []),
        key=lambda row: (
            float(row.get("delta_auc_vs_baseline", float("-inf"))),
            float(row.get("delta_tpr_at_fpr1pct_vs_baseline", float("-inf"))),
        ),
    )

    lines = []
    if singles:
        lines.append("Strongest single-feature additions:")
        for row in singles[:top_k]:
            lines.append(
                f"- `{row['feature']}`: ΔAUC `{row.get('delta_auc', 0.0):+.4f}`, "
                f"ΔTPR@1% `{row.get('delta_tpr_at_fpr1pct', 0.0):+.4f}`"
            )
    if loo:
        if lines:
            lines.append("")
        lines.append("Most important features by leave-one-out:")
        for row in loo[:top_k]:
            lines.append(
                f"- drop `{row['dropped_feature']}` -> remaining set has "
                f"ΔAUC vs baseline `{row.get('delta_auc_vs_baseline', 0.0):+.4f}`, "
                f"ΔTPR@1% vs baseline `{row.get('delta_tpr_at_fpr1pct_vs_baseline', 0.0):+.4f}`"
            )
    return "\n".join(lines) if lines else "_No ablation summary available._"


def best_candidate_markdown(
    *,
    feature_names: list[str],
    metrics: dict[str, Any],
    code_text: str,
    ablation: dict[str, Any] | None = None,
) -> str:
    return textwrap.dedent(
        f"""\
        ## Current Best Accepted Candidate
        - feature names: {", ".join(f"`{x}`" for x in feature_names)}
        - AUC: `{metrics['auc']:.4f}`
        - TPR@FPR=1%: `{metrics['tpr_at_fpr1pct']:.4f}`
        - Accuracy: `{metrics['accuracy']:.4f}`

        ### Best Candidate Ablation Notes
        {ablation_markdown(ablation)}

        ### Best Candidate Code
        ```python
        {code_text.strip()}
        ```
        """
    ).strip() + "\n"


def active_feature_bank_markdown(
    *,
    feature_names: list[str],
    metrics: dict[str, Any] | None,
    source_trials: list[int] | None = None,
) -> str:
    if not feature_names:
        return textwrap.dedent(
            """\
            ## Current Active Feature Bank
            - no accepted custom features yet
            - propose the first incremental feature block
            """
        ).strip() + "\n"

    lines = [
        "## Current Active Feature Bank",
        f"- source accepted trials: {', '.join(f'`{x}`' for x in (source_trials or [])) or '_unknown_'}",
        f"- active feature count: `{len(feature_names)}`",
        f"- active feature names: {', '.join(f'`{x}`' for x in feature_names)}",
    ]
    if metrics:
        lines.extend(
            [
                f"- active bank AUC: `{metrics['auc']:.4f}`",
                f"- active bank TPR@FPR=1%: `{metrics['tpr_at_fpr1pct']:.4f}`",
                f"- active bank Accuracy: `{metrics['accuracy']:.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "Only propose new incremental features to add on top of this bank.",
            "Do not re-emit, rename, or overwrite any active-bank feature.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def extract_response_sections(response: str) -> tuple[str, str, str]:
    def section(name: str) -> str:
        pat = re.compile(
            rf"##\s*{name}\s*(.*?)(?=\n##\s*[A-Z_]+\s*|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        m = pat.search(response)
        return m.group(1).strip() if m else ""

    policy = section("POLICY_UPDATE")
    rationale = section("RATIONALE")
    feature_section = section("FEATURE_CODE")

    code = ""
    block_pat = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    block = block_pat.search(feature_section or response)
    if block:
        code = block.group(1).strip()
    return policy, code, rationale


def validate_candidate_code(code: str) -> None:
    if not code.strip():
        raise ValueError("Empty candidate code")
    lowered = code.lower()
    for bad in BANNED_CODE_SNIPPETS:
        if bad in lowered:
            raise ValueError(f"Candidate code contains banned snippet: {bad}")
    if "build_candidate_features" not in code:
        raise ValueError("Candidate code must define build_candidate_features")


def materialize_candidate_features(
    code: str,
    *,
    agg_input_df: pd.DataFrame,
    items_input_df: pd.DataFrame,
    base_feature_names: list[str],
) -> tuple[pd.DataFrame, list[str], int]:
    validate_candidate_code(code)
    namespace: dict[str, Any] = {"np": np, "pd": pd}
    exec(code, namespace)
    fn = namespace.get("build_candidate_features")
    if not callable(fn):
        raise ValueError("build_candidate_features is not callable")

    result = fn(agg_input_df.copy(), items_input_df.copy())
    if not isinstance(result, pd.DataFrame):
        raise TypeError("Candidate must return a pandas DataFrame")
    if "dispatch_id" not in result.columns:
        raise ValueError("Candidate feature DataFrame must contain dispatch_id")

    out = result.copy()
    out["dispatch_id"] = pd.to_numeric(out["dispatch_id"], errors="raise").astype(int)
    if out["dispatch_id"].duplicated().any():
        raise ValueError("Candidate feature DataFrame has duplicate dispatch_id rows")

    feature_cols = [c for c in out.columns if c != "dispatch_id"]
    if not feature_cols:
        raise ValueError("Candidate returned no new features")
    collisions = [c for c in feature_cols if c in ROOT_RESERVED_COLUMNS or c in base_feature_names]
    if collisions:
        raise ValueError(f"Candidate feature names collide with reserved/existing columns: {collisions}")
    if any(not c.isascii() for c in feature_cols):
        raise ValueError("Candidate feature names must be ASCII")

    for col in feature_cols:
        if out[col].dtype == bool:
            out[col] = out[col].astype(int)
        out[col] = pd.to_numeric(out[col], errors="coerce")

    aligned = agg_input_df[["dispatch_id"]].merge(out, on="dispatch_id", how="left")
    missing = int(aligned[feature_cols].isna().all(axis=1).sum())
    aligned[feature_cols] = aligned[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return aligned[feature_cols].reset_index(drop=True), feature_cols, missing


def evaluate_candidate(
    *,
    code: str,
    data: SearchData,
    baseline_metrics: dict[str, Any],
    svm_c: float = 10.0,
    random_state: int = 42,
    existing_feature_df: pd.DataFrame | None = None,
    existing_feature_names: list[str] | None = None,
) -> EvalArtifact:
    existing_feature_df = (
        existing_feature_df.reset_index(drop=True)
        if existing_feature_df is not None
        else pd.DataFrame(index=data.base_feature_df.reset_index(drop=True).index)
    )
    existing_feature_names = list(existing_feature_names or [])
    new_feature_df, feature_names, n_missing = materialize_candidate_features(
        code,
        agg_input_df=data.agg_input_df,
        items_input_df=data.items_input_df,
        base_feature_names=list(data.base_feature_df.columns) + existing_feature_names,
    )
    full_feature_df = pd.concat(
        [
            data.base_feature_df.reset_index(drop=True),
            existing_feature_df,
            new_feature_df.reset_index(drop=True),
        ],
        axis=1,
    )
    artifact = evaluate_svm_features(
        feature_df=full_feature_df,
        y=data.y,
        train_idx=data.train_idx,
        test_idx=data.test_idx,
        svm_c=svm_c,
        random_state=random_state,
        new_feature_names=feature_names,
        n_missing_dispatches=n_missing,
    )
    artifact.metrics["delta_accuracy"] = float(artifact.metrics["accuracy"] - baseline_metrics["accuracy"])
    artifact.metrics["delta_recall"] = float(artifact.metrics["recall"] - baseline_metrics["recall"])
    artifact.metrics["delta_auc"] = float(artifact.metrics["auc"] - baseline_metrics["auc"])
    artifact.metrics["delta_tpr_at_fpr1pct"] = float(
        artifact.metrics["tpr_at_fpr1pct"] - baseline_metrics["tpr_at_fpr1pct"]
    )
    return artifact


def run_feature_ablation(
    *,
    data: SearchData,
    candidate_feature_df: pd.DataFrame,
    feature_names: list[str],
    baseline_metrics: dict[str, Any],
    svm_c: float = 10.0,
    random_state: int = 42,
    max_features: int = 12,
    existing_feature_df: pd.DataFrame | None = None,
    existing_feature_names: list[str] | None = None,
) -> dict[str, Any]:
    existing_feature_df = (
        existing_feature_df.reset_index(drop=True)
        if existing_feature_df is not None
        else pd.DataFrame(index=data.base_feature_df.reset_index(drop=True).index)
    )
    existing_feature_names = list(existing_feature_names or [])
    chosen = list(feature_names[:max_features])
    out = {
        "feature_names_evaluated": chosen,
        "single_feature": [],
        "leave_one_out": [],
    }
    if not chosen:
        return out

    for name in chosen:
        single_df = pd.concat(
            [
                data.base_feature_df.reset_index(drop=True),
                existing_feature_df,
                candidate_feature_df[[name]].reset_index(drop=True),
            ],
            axis=1,
        )
        single_eval = evaluate_svm_features(
            feature_df=single_df,
            y=data.y,
            train_idx=data.train_idx,
            test_idx=data.test_idx,
            svm_c=svm_c,
            random_state=random_state,
            new_feature_names=existing_feature_names + [name],
        )
        out["single_feature"].append(
            {
                "feature": name,
                "metrics": _flatten_metrics(single_eval.metrics),
                "delta_auc": float(single_eval.metrics["auc"] - baseline_metrics["auc"]),
                "delta_tpr_at_fpr1pct": float(
                    single_eval.metrics["tpr_at_fpr1pct"] - baseline_metrics["tpr_at_fpr1pct"]
                ),
            }
        )

    if len(chosen) <= 1:
        return out

    for name in chosen:
        keep = [c for c in chosen if c != name]
        loo_df = pd.concat(
            [
                data.base_feature_df.reset_index(drop=True),
                existing_feature_df,
                candidate_feature_df[keep].reset_index(drop=True),
            ],
            axis=1,
        )
        loo_eval = evaluate_svm_features(
            feature_df=loo_df,
            y=data.y,
            train_idx=data.train_idx,
            test_idx=data.test_idx,
            svm_c=svm_c,
            random_state=random_state,
            new_feature_names=existing_feature_names + keep,
        )
        out["leave_one_out"].append(
            {
                "dropped_feature": name,
                "metrics": _flatten_metrics(loo_eval.metrics),
                "delta_auc_vs_baseline": float(loo_eval.metrics["auc"] - baseline_metrics["auc"]),
                "delta_tpr_at_fpr1pct_vs_baseline": float(
                    loo_eval.metrics["tpr_at_fpr1pct"] - baseline_metrics["tpr_at_fpr1pct"]
                ),
            }
        )
    return out


def coefficients_markdown(coefficients: list[dict[str, float]]) -> str:
    if not coefficients:
        return "_No candidate-feature coefficients available._\n"
    lines = [
        "| rank | feature | coef | abs_coef |",
        "|---:|---|---:|---:|",
    ]
    for idx, row in enumerate(coefficients, 1):
        lines.append(
            f"| {idx} | `{row['feature']}` | {row['coef']:.6g} | {row['abs_coef']:.6g} |"
        )
    return "\n".join(lines) + "\n"


def build_memory_markdown(records: list[TrialRecord], baseline_metrics: dict[str, Any]) -> str:
    success = [r for r in records if _is_success_status(r.status) and r.metrics]
    rejected = [r for r in records if r.status == "rejected"]
    failed = [r for r in records if r.status == "failed"]
    success = sorted(success, key=lambda r: _metric_sort_key(r.metrics), reverse=True)

    lines = [
        "# Memory",
        "",
        "Baseline reference:",
        f"- AUC `{baseline_metrics['auc']:.4f}`",
        f"- TPR@FPR=1% `{baseline_metrics['tpr_at_fpr1pct']:.4f}`",
        f"- Accuracy `{baseline_metrics['accuracy']:.4f}`",
        "",
    ]

    if success:
        lines.append("## Accepted Trials")
        for r in success[:3]:
            delta_auc = float(r.metrics["auc"] - baseline_metrics["auc"]) if r.metrics else 0.0
            delta_tpr = (
                float(r.metrics["tpr_at_fpr1pct"] - baseline_metrics["tpr_at_fpr1pct"])
                if r.metrics
                else 0.0
            )
            lines.append(
                f"- iter `{r.iteration}`: AUC `{r.metrics['auc']:.4f}` "
                f"(Δ `{delta_auc:+.4f}`), TPR@1% `{r.metrics['tpr_at_fpr1pct']:.4f}` "
                f"(Δ `{delta_tpr:+.4f}`), features={', '.join(r.feature_names)}"
            )
        lines.append("")

    if rejected:
        lines.append("## Rejected Trials")
        for r in rejected[-3:]:
            delta_auc = float(r.metrics["auc"] - baseline_metrics["auc"]) if r.metrics else 0.0
            delta_tpr = (
                float(r.metrics["tpr_at_fpr1pct"] - baseline_metrics["tpr_at_fpr1pct"])
                if r.metrics
                else 0.0
            )
            lines.append(
                f"- iter `{r.iteration}`: AUC `{r.metrics['auc']:.4f}` "
                f"(Δ `{delta_auc:+.4f}`), TPR@1% `{r.metrics['tpr_at_fpr1pct']:.4f}` "
                f"(Δ `{delta_tpr:+.4f}`), features={', '.join(r.feature_names)}"
            )
        lines.append("")

    if failed:
        lines.append("## Failed Trials")
        for r in failed[-3:]:
            lines.append(f"- iter `{r.iteration}`: {r.error or 'unknown error'}")
        lines.append("")

    if not success and not failed:
        lines.append("_No trial history yet._")
        lines.append("")

    lines.append("## Guidance")
    lines.append("- only promote candidates that beat the current best under the acceptance rule")
    lines.append("- keep trying compact feature sets with explicit physical interpretation")
    lines.append("- prefer features that improve AUC and low-FPR recall without exploding FPR")
    lines.append("- if a feature is weak alone but strong in combination, note that in rationale")
    return "\n".join(lines).strip() + "\n"


def build_summary_payload(
    *,
    baseline_metrics: dict[str, Any],
    records: list[TrialRecord],
    config: dict[str, Any],
) -> dict[str, Any]:
    success = [r for r in records if _is_success_status(r.status) and r.metrics]
    rejected = [r for r in records if r.status == "rejected"]
    failed = [r for r in records if r.status == "failed"]
    success = sorted(success, key=lambda r: _metric_sort_key(r.metrics), reverse=True)
    best = success[0] if success else None
    return {
        "config": config,
        "baseline_metrics": _flatten_metrics(baseline_metrics),
        "n_trials_total": len(records),
        "n_trials_successful": len(success),
        "n_trials_accepted": len(success),
        "n_trials_rejected": len(rejected),
        "n_trials_failed": len(failed),
        "best_trial": asdict(best) if best is not None else None,
        "best_accepted_trial": asdict(best) if best is not None else None,
        "accepted_trials": [asdict(r) for r in success],
        "rejected_trials": [asdict(r) for r in rejected],
        "failed_trials": [asdict(r) for r in failed],
        "trials": [asdict(r) for r in records],
    }


def records_to_csv_frame(records: list[TrialRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {
            "iteration": r.iteration,
            "status": r.status,
            "is_success": _is_success_status(r.status),
            "trial_dir": r.trial_dir,
            "n_new_features": r.n_new_features,
            "feature_names": ",".join(r.feature_names),
            "policy_excerpt": r.policy_excerpt,
            "rationale_excerpt": r.rationale_excerpt,
            "error": r.error or "",
        }
        if r.metrics:
            row.update({f"metric_{k}": v for k, v in _flatten_metrics(r.metrics).items()})
        if r.delta_vs_baseline:
            row.update({f"delta_{k}": v for k, v in r.delta_vs_baseline.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def safe_trial_record(
    *,
    iteration: int,
    trial_dir: pathlib.Path,
    policy_text: str,
    rationale_text: str,
    feature_names: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
    error: Exception | str | None = None,
    status: str | None = None,
) -> TrialRecord:
    delta = None
    objective = None
    if metrics:
        delta = {
            "accuracy": float(metrics.get("delta_accuracy", 0.0)),
            "recall": float(metrics.get("delta_recall", 0.0)),
            "auc": float(metrics.get("delta_auc", 0.0)),
            "tpr_at_fpr1pct": float(metrics.get("delta_tpr_at_fpr1pct", 0.0)),
        }
        objective = [float(x) for x in _metric_sort_key(metrics)]
    err_text = None
    if error is not None:
        if isinstance(error, Exception):
            err_text = f"{type(error).__name__}: {error}"
        else:
            err_text = str(error)
    if status is None:
        status = "ok" if metrics is not None and error is None else "failed"
    return TrialRecord(
        iteration=int(iteration),
        status=status,
        trial_dir=str(trial_dir),
        feature_names=list(feature_names or []),
        n_new_features=len(feature_names or []),
        metrics=_flatten_metrics(metrics),
        delta_vs_baseline=delta,
        objective_key=objective,
        policy_excerpt=(policy_text or "").strip().replace("\n", " ")[:240],
        rationale_excerpt=(rationale_text or "").strip().replace("\n", " ")[:240],
        error=err_text,
    )


def failure_payload(exc: Exception) -> dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
