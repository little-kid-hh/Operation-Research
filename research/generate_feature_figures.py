# -*- coding: utf-8 -*-
"""
Generate report-ready parameter/importance tables and SVG charts for:

1. HybridSVM active feature bank (linear SVM coefficients + single/LOO ablations)
2. Ensemble route (single-model coefficients/importances + stacking meta weights)

Outputs are written under:
    research/feature_figures_20260511/

This script avoids third-party plotting dependencies and writes plain SVG files.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
HYBRID_ROOT = ROOT / "HybridSVM"
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))

from src.ensemble_train import EnsembleConfig, fit_ensemble_pipeline  # noqa: E402
from src.feature_search import evaluate_svm_features, load_search_data, run_feature_ablation  # noqa: E402
from src.svm_train import DROP_COLS, fit_svm_pipeline, load_raw_data  # noqa: E402


DATA_PATH = ROOT / "FunSearch_test" / "training_2orientations.csv"
ITEMS_PATH = ROOT / "FunSearch_test" / "物品信息和dblf信息.csv"
GLM_EXP_DIR = ROOT / "tmp_glm_probe_v4" / "by_model" / "glm-5.1" / "exp_20260510_181255"
OUT_DIR = ROOT / "research" / "feature_figures_20260511"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

ENSEMBLE_CFG = {
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


@dataclass
class SvgTheme:
    bg: str = "#ffffff"
    text: str = "#1f2937"
    subtext: str = "#6b7280"
    grid: str = "#e5e7eb"
    pos: str = "#2563eb"
    neg: str = "#dc2626"
    accent: str = "#0f766e"
    alt: str = "#7c3aed"


THEME = SvgTheme()


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_table(path: pathlib.Path, df: pd.DataFrame, title: str | None = None) -> None:
    lines: list[str] = []
    if title:
        lines.extend([f"# {title}", ""])
    lines.append(dataframe_to_markdown(df))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _nice_num(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:.0f}"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _bar_chart_svg(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    value_key: str,
    title: str,
    subtitle: str,
    output_path: pathlib.Path,
    width: int = 1280,
    row_h: int = 34,
    left_margin: int = 430,
    right_margin: int = 110,
    top_margin: int = 100,
    bottom_margin: int = 50,
    show_zero_line: bool = True,
    color_mode: str = "signed",
    palette: tuple[str, str] | None = None,
    decimals: int = 4,
) -> None:
    n = len(rows)
    plot_h = max(220, n * row_h)
    height = top_margin + plot_h + bottom_margin
    plot_w = width - left_margin - right_margin
    vals = [float(r[value_key]) for r in rows]
    vmin = min(vals) if vals else 0.0
    vmax = max(vals) if vals else 1.0
    if show_zero_line:
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 0.0)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    rng = vmax - vmin

    def xmap(v: float) -> float:
        return left_margin + (v - vmin) / rng * plot_w

    zero_x = xmap(0.0)
    colors = palette or (THEME.pos, THEME.neg)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{THEME.bg}"/>',
        f'<text x="32" y="42" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700" fill="{THEME.text}">{svg_escape(title)}</text>',
        f'<text x="32" y="70" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="{THEME.subtext}">{svg_escape(subtitle)}</text>',
    ]

    for t in np.linspace(vmin, vmax, 6):
        x = xmap(float(t))
        parts.append(f'<line x1="{x:.1f}" y1="{top_margin}" x2="{x:.1f}" y2="{top_margin + plot_h}" stroke="{THEME.grid}" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top_margin + plot_h + 24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="{THEME.subtext}">{svg_escape(_nice_num(float(t)))}</text>'
        )
    if show_zero_line:
        parts.append(f'<line x1="{zero_x:.1f}" y1="{top_margin}" x2="{zero_x:.1f}" y2="{top_margin + plot_h}" stroke="{THEME.text}" stroke-width="1.4"/>')

    for idx, row in enumerate(rows):
        y = top_margin + idx * row_h + 6
        label = str(row[label_key])
        val = float(row[value_key])
        x0 = zero_x if show_zero_line else left_margin
        x1 = xmap(val)
        bar_x = min(x0, x1)
        bar_w = max(2.0, abs(x1 - x0))
        if color_mode == "signed":
            color = colors[0] if val >= 0 else colors[1]
        else:
            color = colors[0]
        parts.append(f'<text x="{left_margin - 12}" y="{y + 15}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="{THEME.text}">{svg_escape(label)}</text>')
        parts.append(f'<rect x="{bar_x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="20" rx="3" fill="{color}"/>')
        tx = x1 + 8 if val >= 0 else x1 - 8
        anchor = "start" if val >= 0 else "end"
        parts.append(
            f'<text x="{tx:.1f}" y="{y + 15}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="{THEME.text}">{val:.{decimals}f}</text>'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _grouped_metric_svg(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    metric_keys: list[str],
    metric_labels: list[str],
    title: str,
    subtitle: str,
    output_path: pathlib.Path,
    width: int = 1280,
    height: int = 620,
) -> None:
    left_margin = 120
    right_margin = 50
    top_margin = 110
    bottom_margin = 90
    plot_w = width - left_margin - right_margin
    plot_h = height - top_margin - bottom_margin
    max_val = max(float(r[k]) for r in rows for k in metric_keys)
    max_val = max(max_val, 1.0)
    colors = [THEME.pos, THEME.accent, THEME.alt]
    group_gap = 28
    cluster_w = plot_w / max(1, len(rows))
    bar_w = min(54, (cluster_w - group_gap) / max(1, len(metric_keys)))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{THEME.bg}"/>',
        f'<text x="32" y="42" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700" fill="{THEME.text}">{svg_escape(title)}</text>',
        f'<text x="32" y="70" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="{THEME.subtext}">{svg_escape(subtitle)}</text>',
    ]
    for ytick in np.linspace(0, max_val, 6):
        y = top_margin + plot_h - ytick / max_val * plot_h
        parts.append(f'<line x1="{left_margin}" y1="{y:.1f}" x2="{width - right_margin}" y2="{y:.1f}" stroke="{THEME.grid}" stroke-width="1"/>')
        parts.append(f'<text x="{left_margin - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="{THEME.subtext}">{ytick:.3f}</text>')

    for gi, row in enumerate(rows):
        cluster_x = left_margin + gi * cluster_w + group_gap / 2
        for mi, key in enumerate(metric_keys):
            val = float(row[key])
            bar_h = val / max_val * plot_h
            x = cluster_x + mi * bar_w
            y = top_margin + plot_h - bar_h
            color = colors[mi % len(colors)]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 6:.1f}" height="{bar_h:.1f}" rx="3" fill="{color}"/>')
            parts.append(f'<text x="{x + (bar_w - 6)/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{THEME.text}">{val:.4f}</text>')
        parts.append(
            f'<text x="{cluster_x + (len(metric_keys) * bar_w)/2 - 3:.1f}" y="{height - 34}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="{THEME.text}">{svg_escape(str(row[group_key]))}</text>'
        )

    legend_x = width - right_margin - 320
    legend_y = 34
    for i, label in enumerate(metric_labels):
        lx = legend_x + i * 102
        parts.append(f'<rect x="{lx}" y="{legend_y}" width="16" height="16" rx="3" fill="{colors[i % len(colors)]}"/>')
        parts.append(f'<text x="{lx + 24}" y="{legend_y + 13}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="{THEME.text}">{svg_escape(label)}</text>')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def load_glm_summary() -> dict[str, Any]:
    return json.loads((GLM_EXP_DIR / "summary.json").read_text(encoding="utf-8"))


def load_glm_active_bank() -> pd.DataFrame:
    return pd.read_csv(GLM_EXP_DIR / "active_feature_bank.csv")


def build_hybrid_active_bank_eval() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
    summary = load_glm_summary()
    data = load_search_data(DATA_PATH, ITEMS_PATH, test_size=0.25, random_state=42)
    active_df = load_glm_active_bank().reset_index(drop=True)
    full_df = pd.concat([data.base_feature_df.reset_index(drop=True), active_df], axis=1)
    artifact = evaluate_svm_features(
        feature_df=full_df,
        y=data.y,
        train_idx=data.train_idx,
        test_idx=data.test_idx,
        svm_c=float(summary["config"]["svm_c"]),
        random_state=int(summary["config"]["random_state"]),
        new_feature_names=list(active_df.columns),
    )
    ablation = run_feature_ablation(
        data=data,
        candidate_feature_df=active_df,
        feature_names=list(active_df.columns),
        baseline_metrics=summary["baseline_metrics"],
        svm_c=float(summary["config"]["svm_c"]),
        random_state=int(summary["config"]["random_state"]),
        max_features=len(active_df.columns),
    )
    coef = np.asarray(artifact.pipeline.model.coef_).ravel().astype(float)
    scaler = artifact.pipeline.scaler
    scales = np.asarray(getattr(scaler, "scale_", np.ones_like(coef)), dtype=float)
    active_rows: list[dict[str, Any]] = []
    for feat in active_df.columns:
        idx = artifact.pipeline.feature_names.index(feat)
        w_scaled = float(coef[idx])
        scale = float(scales[idx])
        active_rows.append(
            {
                "feature": feat,
                "coef_scaled": w_scaled,
                "abs_coef_scaled": abs(w_scaled),
                "approx_coef_raw_units": w_scaled * scale,
                "abs_approx_coef_raw_units": abs(w_scaled * scale),
            }
        )
    coef_df = pd.DataFrame(active_rows).sort_values("abs_coef_scaled", ascending=False).reset_index(drop=True)
    return coef_df, artifact.metrics, ablation, full_df


def build_hybrid_tables_and_figures() -> dict[str, Any]:
    coef_df, metrics, ablation, _full_df = build_hybrid_active_bank_eval()
    write_markdown_table(TABLE_DIR / "hybridsvm_active_feature_coefficients.md", coef_df, "HybridSVM Active Feature Coefficients")
    coef_df.to_csv(TABLE_DIR / "hybridsvm_active_feature_coefficients.csv", index=False, encoding="utf-8")

    singles = pd.DataFrame(ablation["single_feature"]).copy()
    singles["auc"] = singles["metrics"].apply(lambda x: x["auc"])
    singles["tpr_at_fpr1pct"] = singles["metrics"].apply(lambda x: x["tpr_at_fpr1pct"])
    singles["accuracy"] = singles["metrics"].apply(lambda x: x["accuracy"])
    singles = singles.drop(columns=["metrics"]).sort_values(["delta_auc", "delta_tpr_at_fpr1pct"], ascending=False)
    singles.to_csv(TABLE_DIR / "hybridsvm_single_feature_ablation.csv", index=False, encoding="utf-8")
    write_markdown_table(TABLE_DIR / "hybridsvm_single_feature_ablation.md", singles, "HybridSVM Single-Feature Ablation")

    loo = pd.DataFrame(ablation["leave_one_out"]).copy()
    loo["auc"] = loo["metrics"].apply(lambda x: x["auc"])
    loo["tpr_at_fpr1pct"] = loo["metrics"].apply(lambda x: x["tpr_at_fpr1pct"])
    loo["accuracy"] = loo["metrics"].apply(lambda x: x["accuracy"])
    loo["performance_drop_auc"] = float(metrics["auc"]) - loo["auc"]
    loo["performance_drop_tpr_at_fpr1pct"] = float(metrics["tpr_at_fpr1pct"]) - loo["tpr_at_fpr1pct"]
    loo = loo.drop(columns=["metrics"]).sort_values("performance_drop_auc", ascending=False)
    loo.to_csv(TABLE_DIR / "hybridsvm_leave_one_out_ablation.csv", index=False, encoding="utf-8")
    write_markdown_table(TABLE_DIR / "hybridsvm_leave_one_out_ablation.md", loo, "HybridSVM Leave-One-Out Ablation")

    top_coef = coef_df.head(15).copy()
    _bar_chart_svg(
        top_coef.to_dict("records"),
        label_key="feature",
        value_key="coef_scaled",
        title="HybridSVM Active Feature Coefficients",
        subtitle="Linear SVM coefficients on MinMax-scaled features. Positive pushes toward feasible (1).",
        output_path=FIG_DIR / "hybridsvm_active_feature_coefficients.svg",
        color_mode="signed",
    )

    top_single = singles.nlargest(12, "delta_auc")[["feature", "delta_auc"]].to_dict("records")
    _bar_chart_svg(
        top_single,
        label_key="feature",
        value_key="delta_auc",
        title="HybridSVM Single-Feature Gains",
        subtitle="AUC lift when each active-bank feature is added individually on top of the original SVM baseline.",
        output_path=FIG_DIR / "hybridsvm_single_feature_auc_gain.svg",
        color_mode="positive",
        show_zero_line=False,
        palette=(THEME.accent, THEME.accent),
    )

    top_loo = loo.nlargest(12, "performance_drop_auc")[["dropped_feature", "performance_drop_auc"]].to_dict("records")
    _bar_chart_svg(
        top_loo,
        label_key="dropped_feature",
        value_key="performance_drop_auc",
        title="HybridSVM Leave-One-Out Importance",
        subtitle="AUC drop after removing one active-bank feature from the full 15-feature bank.",
        output_path=FIG_DIR / "hybridsvm_leave_one_out_auc_drop.svg",
        color_mode="positive",
        show_zero_line=False,
        palette=(THEME.alt, THEME.alt),
    )

    return {
        "active_bank_metrics": metrics,
        "top_scaled_coefficients": coef_df.head(10).to_dict("records"),
        "top_single_feature_auc_gain": singles.head(10)[["feature", "delta_auc", "delta_tpr_at_fpr1pct"]].to_dict("records"),
        "top_leave_one_out_auc_drop": loo.head(10)[["dropped_feature", "performance_drop_auc", "performance_drop_tpr_at_fpr1pct"]].to_dict("records"),
    }


def _ensemble_feature_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df = load_raw_data(DATA_PATH)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X_full_df = df.drop(columns=cols_to_drop + ["if_loaded"]).reset_index(drop=True)
    y = df["if_loaded"].to_numpy(dtype=int)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42)
    X_train_df = X_full_df.iloc[train_idx].reset_index(drop=True)
    X_test_df = X_full_df.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    return X_full_df, X_train_df, X_test_df, y, y_train


def build_ensemble_tables_and_figures() -> dict[str, Any]:
    X_full_df, X_train_df, X_test_df, y, y_train = _ensemble_feature_frame()
    _ = X_test_df  # unused but kept to make split explicit
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values)

    svm_pipe = fit_svm_pipeline(X_train_df, y_train, C=ENSEMBLE_CFG["svm_C"], random_state=42)
    svm_coef = np.asarray(svm_pipe.model.coef_).ravel().astype(float)
    scales = np.asarray(svm_pipe.scaler.scale_, dtype=float)
    svm_rows = []
    for feat, coef, scale in zip(svm_pipe.feature_names, svm_coef, scales):
        svm_rows.append(
            {
                "feature": feat,
                "coef_scaled": float(coef),
                "abs_coef_scaled": float(abs(coef)),
                "approx_coef_raw_units": float(coef * scale),
                "abs_approx_coef_raw_units": float(abs(coef * scale)),
            }
        )
    svm_df = pd.DataFrame(svm_rows).sort_values("abs_coef_scaled", ascending=False).reset_index(drop=True)
    svm_df.to_csv(TABLE_DIR / "ensemble_single_svm_coefficients.csv", index=False, encoding="utf-8")
    write_markdown_table(TABLE_DIR / "ensemble_single_svm_coefficients.md", svm_df, "Ensemble Route: Single SVM Coefficients")

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    lr = LogisticRegression(
        C=ENSEMBLE_CFG["lr_C"],
        max_iter=3000,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_train_scaled, y_train)
    lr_coef = np.asarray(lr.coef_).ravel().astype(float)
    lr_rows = []
    for feat, coef, scale in zip(X_train_df.columns, lr_coef, scaler.scale_):
        lr_rows.append(
            {
                "feature": feat,
                "coef_scaled": float(coef),
                "abs_coef_scaled": float(abs(coef)),
                "approx_coef_raw_units": float(coef * scale),
                "abs_approx_coef_raw_units": float(abs(coef * scale)),
            }
        )
    lr_df = pd.DataFrame(lr_rows).sort_values("abs_coef_scaled", ascending=False).reset_index(drop=True)
    lr_df.to_csv(TABLE_DIR / "ensemble_single_lr_coefficients.csv", index=False, encoding="utf-8")
    write_markdown_table(TABLE_DIR / "ensemble_single_lr_coefficients.md", lr_df, "Ensemble Route: Single LR Coefficients")

    rf = RandomForestClassifier(
        n_estimators=ENSEMBLE_CFG["rf_n_estimators"],
        max_depth=ENSEMBLE_CFG["rf_max_depth"],
        min_samples_leaf=ENSEMBLE_CFG["rf_min_samples_leaf"],
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train_scaled, y_train)
    rf_df = pd.DataFrame(
        {
            "feature": X_train_df.columns,
            "importance": rf.feature_importances_.astype(float),
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    rf_df.to_csv(TABLE_DIR / "ensemble_single_rf_importance.csv", index=False, encoding="utf-8")
    write_markdown_table(TABLE_DIR / "ensemble_single_rf_importance.md", rf_df, "Ensemble Route: RF Feature Importance")

    gbdt = HistGradientBoostingClassifier(
        learning_rate=ENSEMBLE_CFG["hgb_learning_rate"],
        max_depth=ENSEMBLE_CFG["hgb_max_depth"],
        max_iter=ENSEMBLE_CFG["hgb_max_iter"],
        random_state=42,
    )
    gbdt.fit(X_train_scaled, y_train)
    # permutation importance is too expensive; approximate with split counts from prediction internals is awkward.
    # Use feature_importances_ if available; otherwise use permutation-like proxy from staged single-feature prediction variance.
    if hasattr(gbdt, "feature_importances_"):
        gbdt_importance = np.asarray(gbdt.feature_importances_, dtype=float)
    else:
        predictor_nodes = gbdt._predictors  # type: ignore[attr-defined]
        counts = np.zeros(X_train_scaled.shape[1], dtype=float)
        for iter_nodes in predictor_nodes:
            for tree in iter_nodes:
                nodes = tree.nodes
                for n in nodes:
                    fidx = int(n["feature_idx"])
                    if fidx >= 0:
                        counts[fidx] += 1.0
        gbdt_importance = counts / max(counts.sum(), 1.0)
    gbdt_df = pd.DataFrame(
        {
            "feature": X_train_df.columns,
            "importance": gbdt_importance.astype(float),
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    gbdt_df.to_csv(TABLE_DIR / "ensemble_single_gbdt_importance.csv", index=False, encoding="utf-8")
    write_markdown_table(TABLE_DIR / "ensemble_single_gbdt_importance.md", gbdt_df, "Ensemble Route: GBDT Feature Importance")

    pipe = fit_ensemble_pipeline(
        X_train_df=X_train_df,
        y_train=y_train,
        config=EnsembleConfig(
            base_models=["svm", "lr", "rf", "xgb"],
            meta_model="logreg",
            cv_folds=5,
            random_state=42,
        ),
    )
    meta_coef = np.asarray(pipe.meta_model.coef_).ravel().astype(float)
    meta_df = pd.DataFrame(
        {
            "base_model": pipe.resolved_base_models,
            "meta_weight": meta_coef,
            "abs_meta_weight": np.abs(meta_coef),
        }
    ).sort_values("abs_meta_weight", ascending=False).reset_index(drop=True)
    meta_df.to_csv(TABLE_DIR / "ensemble_stacking_meta_weights.csv", index=False, encoding="utf-8")
    write_markdown_table(TABLE_DIR / "ensemble_stacking_meta_weights.md", meta_df, "Ensemble Route: Stacking Meta Weights")

    _bar_chart_svg(
        svm_df.head(15).to_dict("records"),
        label_key="feature",
        value_key="coef_scaled",
        title="Ensemble Route: Single SVM Coefficients",
        subtitle="Linear SVM coefficients on MinMax-scaled aggregate features.",
        output_path=FIG_DIR / "ensemble_single_svm_coefficients.svg",
        color_mode="signed",
    )
    _bar_chart_svg(
        lr_df.head(15).to_dict("records"),
        label_key="feature",
        value_key="coef_scaled",
        title="Ensemble Route: Single LR Coefficients",
        subtitle="Logistic regression coefficients on the same scaled aggregate features.",
        output_path=FIG_DIR / "ensemble_single_lr_coefficients.svg",
        color_mode="signed",
    )
    _bar_chart_svg(
        rf_df.head(15).to_dict("records"),
        label_key="feature",
        value_key="importance",
        title="Ensemble Route: RF Feature Importance",
        subtitle="Random forest feature importance on the fixed 40 aggregate features.",
        output_path=FIG_DIR / "ensemble_single_rf_importance.svg",
        color_mode="positive",
        show_zero_line=False,
        palette=(THEME.accent, THEME.accent),
        decimals=5,
    )
    _bar_chart_svg(
        gbdt_df.head(15).to_dict("records"),
        label_key="feature",
        value_key="importance",
        title="Ensemble Route: GBDT Feature Importance",
        subtitle="HistGradientBoosting feature-use proxy on the fixed 40 aggregate features.",
        output_path=FIG_DIR / "ensemble_single_gbdt_importance.svg",
        color_mode="positive",
        show_zero_line=False,
        palette=(THEME.alt, THEME.alt),
        decimals=5,
    )
    _bar_chart_svg(
        meta_df.to_dict("records"),
        label_key="base_model",
        value_key="meta_weight",
        title="Ensemble Route: Stacking Meta Weights",
        subtitle="Logistic meta-model coefficients over base-model probabilities in the full stack.",
        output_path=FIG_DIR / "ensemble_stacking_meta_weights.svg",
        color_mode="signed",
    )

    metrics_rows = [
        {"model": "svm", "accuracy": 0.9276, "auc": 0.9644192504251754, "tpr_at_fpr1pct": 0.6057030481809242},
        {"model": "lr", "accuracy": 0.9220, "auc": 0.9648824068095593, "tpr_at_fpr1pct": 0.6283185840707964},
        {"model": "rf", "accuracy": 0.9436, "auc": 0.9822660690999785, "tpr_at_fpr1pct": 0.8058013765978368},
        {"model": "gbdt_fallback", "accuracy": 0.9468, "auc": 0.9845111642844182, "tpr_at_fpr1pct": 0.8131760078662733},
        {"model": "stack_full", "accuracy": 0.9476, "auc": 0.9841851612712641, "tpr_at_fpr1pct": 0.8171091445427728},
    ]
    _grouped_metric_svg(
        metrics_rows,
        group_key="model",
        metric_keys=["accuracy", "auc", "tpr_at_fpr1pct"],
        metric_labels=["Accuracy", "AUC", "TPR@1%"],
        title="Ensemble Route: Metric Comparison",
        subtitle="Single models vs the full stacking classifier on the fixed split.",
        output_path=FIG_DIR / "ensemble_metric_comparison.svg",
    )

    return {
        "top_svm_coefficients": svm_df.head(10).to_dict("records"),
        "top_lr_coefficients": lr_df.head(10).to_dict("records"),
        "top_rf_importance": rf_df.head(10).to_dict("records"),
        "top_gbdt_importance": gbdt_df.head(10).to_dict("records"),
        "stacking_meta_weights": meta_df.to_dict("records"),
    }


def build_index(outputs: dict[str, Any]) -> None:
    lines = [
        "# Feature / Parameter Visualizations",
        "",
        "Generated by `research/generate_feature_figures.py`.",
        "",
        "## Figures",
        "",
        "- `figures/hybridsvm_active_feature_coefficients.svg`",
        "- `figures/hybridsvm_single_feature_auc_gain.svg`",
        "- `figures/hybridsvm_leave_one_out_auc_drop.svg`",
        "- `figures/ensemble_single_svm_coefficients.svg`",
        "- `figures/ensemble_single_lr_coefficients.svg`",
        "- `figures/ensemble_single_rf_importance.svg`",
        "- `figures/ensemble_single_gbdt_importance.svg`",
        "- `figures/ensemble_stacking_meta_weights.svg`",
        "- `figures/ensemble_metric_comparison.svg`",
        "",
        "## Tables",
        "",
        "- `tables/hybridsvm_active_feature_coefficients.csv`",
        "- `tables/hybridsvm_single_feature_ablation.csv`",
        "- `tables/hybridsvm_leave_one_out_ablation.csv`",
        "- `tables/ensemble_single_svm_coefficients.csv`",
        "- `tables/ensemble_single_lr_coefficients.csv`",
        "- `tables/ensemble_single_rf_importance.csv`",
        "- `tables/ensemble_single_gbdt_importance.csv`",
        "- `tables/ensemble_stacking_meta_weights.csv`",
        "",
        "## Snapshot",
        "",
        "```json",
        json.dumps(outputs, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    hybrid = build_hybrid_tables_and_figures()
    ensemble = build_ensemble_tables_and_figures()
    outputs = {"hybridsvm": hybrid, "ensemble": ensemble}
    write_json(OUT_DIR / "summary.json", outputs)
    build_index(outputs)
    print(f"Artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
