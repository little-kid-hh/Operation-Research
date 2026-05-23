# -*- coding: utf-8 -*-
"""
Analyze where SVM mistakes sit relative to the linear decision boundary.

This script consumes the exported CSVs:
  - test_results_with_distances.csv
  - false_negative_samples.csv
  - false_positive_samples.csv

and joins them back to raw dispatch features so we can design a gated
SVM-first Hybrid path: only samples near the decision boundary are sent to
LLM-generated heuristic rules.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
HYBRID_ROOT = ROOT / "HybridSVM"
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))

from src.svm_train import load_raw_data  # noqa: E402


DEFAULT_RESULT_CSV = ROOT / "test_results_with_distances.csv"
DEFAULT_FALSE_NEGATIVE_CSV = ROOT / "false_negative_samples.csv"
DEFAULT_FALSE_POSITIVE_CSV = ROOT / "false_positive_samples.csv"
DEFAULT_DATA_CSV = ROOT / "FunSearch_test" / "training_2orientations.csv"


KEY_FEATURES = [
    "fill_ratio",
    "sku_counts",
    "total_skuvolume",
    "vehicle_capacity",
    "spare_capacity",
    "sku_average_volume",
    "sku_concentration",
    "max_asr",
    "sku_length_avg",
    "sku_width_avg",
    "sku_height_avg",
    "sku_length_var",
    "sku_width_var",
    "sku_height_var",
    "sku_min_length",
    "sku_max_length",
    "sku_min_width",
    "sku_max_width",
    "sku_min_height",
    "sku_max_height",
    "l_to_L_ratio_avg",
    "l_to_L_ratio_max",
    "h_to_H_ratio_avg",
    "h_to_H_ratio_max",
    "w_to_W_ratio_avg",
    "w_to_W_ratio_max",
    "wl_to_vehicle_wl_avg",
    "wl_to_vehicle_wl_max",
    "wl_to_vehicle_wl_total",
]


def _read_results(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "order_id",
        "distance_to_hyperplane",
        "predicted_label",
        "actual_label",
        "prediction_probability",
        "is_false_positive",
        "is_false_negative",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    return df


def _cross_check_error_files(
    results: pd.DataFrame,
    false_negative_csv: pathlib.Path,
    false_positive_csv: pathlib.Path,
) -> None:
    """Verify split FN/FP CSVs agree with flags in the full result CSV."""
    if not false_negative_csv.exists() or not false_positive_csv.exists():
        print("Skipped FN/FP split-file cross-check: one or both files are missing")
        return

    fn_file = _read_results(false_negative_csv)
    fp_file = _read_results(false_positive_csv)
    fn_ids = set(fn_file["order_id"].astype(int))
    fp_ids = set(fp_file["order_id"].astype(int))
    result_fn_ids = set(
        results.loc[results["is_false_negative"].astype(bool), "order_id"].astype(int)
    )
    result_fp_ids = set(
        results.loc[results["is_false_positive"].astype(bool), "order_id"].astype(int)
    )
    fn_ok = fn_ids == result_fn_ids
    fp_ok = fp_ids == result_fp_ids
    print(
        "FN/FP split-file cross-check: "
        f"FN {len(fn_ids)} rows ({'ok' if fn_ok else 'MISMATCH'}), "
        f"FP {len(fp_ids)} rows ({'ok' if fp_ok else 'MISMATCH'})"
    )
    if not fn_ok:
        print(f"  FN id symmetric diff sample: {sorted(fn_ids ^ result_fn_ids)[:10]}")
    if not fp_ok:
        print(f"  FP id symmetric diff sample: {sorted(fp_ids ^ result_fp_ids)[:10]}")


def _load_joined(
    result_csv: pathlib.Path,
    data_csv: pathlib.Path,
    false_negative_csv: pathlib.Path,
    false_positive_csv: pathlib.Path,
) -> pd.DataFrame:
    results = _read_results(result_csv)
    _cross_check_error_files(results, false_negative_csv, false_positive_csv)
    raw = load_raw_data(data_csv)
    raw = raw.copy()
    if "fill_ratio" not in raw.columns:
        raw["fill_ratio"] = raw["total_skuvolume"] / raw["vehicle_capacity"]

    join_candidates = ["orderid", "发车号"]
    joined = None
    best_key = None
    best_missing = len(results) + 1
    for key in join_candidates:
        if key not in raw.columns:
            continue
        tmp = raw.copy()
        tmp["order_id"] = tmp[key].astype(results["order_id"].dtype)
        cur = results.merge(tmp, on="order_id", how="left", validate="one_to_one")
        marker = key if key != "orderid" else "if_loaded"
        missing = int(cur[marker].isna().sum()) if marker in cur.columns else len(cur)
        if missing < best_missing:
            joined = cur
            best_key = key
            best_missing = missing
    if joined is None or best_missing:
        raise ValueError(
            f"Could not join result rows to raw features; best_key={best_key!r}, "
            f"missing={best_missing}"
        )
    print(f"Joined result order_id to raw `{best_key}`")
    joined["abs_distance"] = joined["distance_to_hyperplane"].abs()
    joined["is_error"] = (
        joined["is_false_positive"].astype(bool)
        | joined["is_false_negative"].astype(bool)
    )
    joined["is_correct"] = ~joined["is_error"]
    joined["is_tp"] = (joined["actual_label"] == 1) & (joined["predicted_label"] == 1)
    joined["is_tn"] = (joined["actual_label"] == 0) & (joined["predicted_label"] == 0)
    return joined


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _print_threshold_table(df: pd.DataFrame, thresholds: list[float]) -> None:
    n = len(df)
    n_err = int(df["is_error"].sum())
    n_fn = int(df["is_false_negative"].sum())
    n_fp = int(df["is_false_positive"].sum())
    print("\n## Boundary coverage")
    print(
        "| abs(distance) <= | rows | row % | errors covered | FN covered | FP covered | oracle accuracy |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for t in thresholds:
        m = df["abs_distance"] <= t
        rows = int(m.sum())
        err = int((m & df["is_error"]).sum())
        fn = int((m & df["is_false_negative"].astype(bool)).sum())
        fp = int((m & df["is_false_positive"].astype(bool)).sum())
        oracle_acc = (n - n_err + err) / n
        print(
            f"| {t:g} | {rows} | {_fmt_pct(rows / n)} | "
            f"{err}/{n_err} ({_fmt_pct(err / n_err if n_err else 0)}) | "
            f"{fn}/{n_fn} ({_fmt_pct(fn / n_fn if n_fn else 0)}) | "
            f"{fp}/{n_fp} ({_fmt_pct(fp / n_fp if n_fp else 0)}) | "
            f"{oracle_acc:.4f} |"
        )


def _print_distance_bins(df: pd.DataFrame, bins: list[float]) -> None:
    labels = []
    prev = bins[0]
    for cur in bins[1:]:
        labels.append(f"({prev:g},{cur:g}]")
        prev = cur
    cut = pd.cut(df["abs_distance"], bins=bins, include_lowest=True, labels=labels)
    print("\n## Error rate by abs-distance bin")
    print("| bin | rows | errors | error rate | FN | FP |")
    print("|---|---:|---:|---:|---:|---:|")
    for label in labels:
        m = cut == label
        rows = int(m.sum())
        if rows == 0:
            continue
        err = int((m & df["is_error"]).sum())
        fn = int((m & df["is_false_negative"].astype(bool)).sum())
        fp = int((m & df["is_false_positive"].astype(bool)).sum())
        print(f"| {label} | {rows} | {err} | {_fmt_pct(err / rows)} | {fn} | {fp} |")


def _signed_gate_mask(df: pd.DataFrame, fn_margin: float, fp_margin: float) -> pd.Series:
    d = df["distance_to_hyperplane"]
    return ((d < 0) & (d >= -fn_margin)) | ((d >= 0) & (d <= fp_margin))


def _print_signed_gate_table(df: pd.DataFrame) -> None:
    n = len(df)
    n_err = int(df["is_error"].sum())
    n_fn = int(df["is_false_negative"].sum())
    n_fp = int(df["is_false_positive"].sum())
    grids = [
        (0.5, 0.5),
        (0.75, 0.75),
        (1.0, 1.0),
        (1.0, 1.5),
        (1.0, 2.0),
        (1.5, 1.5),
    ]
    print("\n## Signed two-sided gate candidates")
    print("| FN side margin | FP side margin | rows | row % | errors covered | FN covered | FP covered | oracle accuracy |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fn_margin, fp_margin in grids:
        m = _signed_gate_mask(df, fn_margin, fp_margin)
        rows = int(m.sum())
        err = int((m & df["is_error"]).sum())
        fn = int((m & df["is_false_negative"].astype(bool)).sum())
        fp = int((m & df["is_false_positive"].astype(bool)).sum())
        oracle_acc = (n - n_err + err) / n
        print(
            f"| {fn_margin:g} | {fp_margin:g} | {rows} | {_fmt_pct(rows / n)} | "
            f"{err}/{n_err} ({_fmt_pct(err / n_err if n_err else 0)}) | "
            f"{fn}/{n_fn} ({_fmt_pct(fn / n_fn if n_fn else 0)}) | "
            f"{fp}/{n_fp} ({_fmt_pct(fp / n_fp if n_fp else 0)}) | "
            f"{oracle_acc:.4f} |"
        )


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    b = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (
        len(a) + len(b) - 2
    )
    if var <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(var))


def _top_feature_diffs(
    df: pd.DataFrame,
    left_mask: pd.Series,
    right_mask: pd.Series,
    left_name: str,
    right_name: str,
    top_k: int,
) -> None:
    cols = [c for c in KEY_FEATURES if c in df.columns]
    rows = []
    left = df.loc[left_mask, cols]
    right = df.loc[right_mask, cols]
    for col in cols:
        d = _cohens_d(left[col], right[col])
        rows.append((abs(d), d, col, left[col].median(), right[col].median(), left[col].mean(), right[col].mean()))
    rows.sort(reverse=True)
    print(f"\n## Top feature contrasts: {left_name} vs {right_name}")
    print(f"- n({left_name})={len(left)}, n({right_name})={len(right)}")
    print("| feature | Cohen d | median left | median right | mean left | mean right |")
    print("|---|---:|---:|---:|---:|---:|")
    for _, d, col, l_med, r_med, l_mean, r_mean in rows[:top_k]:
        print(f"| `{col}` | {d:.3f} | {l_med:.6g} | {r_med:.6g} | {l_mean:.6g} | {r_mean:.6g} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SVM boundary errors.")
    parser.add_argument("--results", type=pathlib.Path, default=DEFAULT_RESULT_CSV)
    parser.add_argument("--false-negative", type=pathlib.Path, default=DEFAULT_FALSE_NEGATIVE_CSV)
    parser.add_argument("--false-positive", type=pathlib.Path, default=DEFAULT_FALSE_POSITIVE_CSV)
    parser.add_argument("--data", type=pathlib.Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--near-margin", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    df = _load_joined(
        args.results.resolve(),
        args.data.resolve(),
        args.false_negative.resolve(),
        args.false_positive.resolve(),
    )

    n = len(df)
    err = int(df["is_error"].sum())
    fn = int(df["is_false_negative"].sum())
    fp = int(df["is_false_positive"].sum())
    print(f"# SVM boundary analysis")
    print(f"- rows: {n}")
    print(f"- accuracy: {(n - err) / n:.4f}")
    print(f"- errors: {err} (FN={fn}, FP={fp})")

    for name, mask in [
        ("FN", df["is_false_negative"].astype(bool)),
        ("FP", df["is_false_positive"].astype(bool)),
    ]:
        s = df.loc[mask, "distance_to_hyperplane"]
        qs = s.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        print(f"\n## Signed distance quantiles: {name}")
        print(qs.to_string(float_format=lambda x: f"{x:.6f}"))

    _print_threshold_table(df, thresholds=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    _print_distance_bins(df, bins=[0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf])
    _print_signed_gate_table(df)

    m_near = df["abs_distance"] <= args.near_margin
    fn_near = df["is_false_negative"].astype(bool) & m_near
    fp_near = df["is_false_positive"].astype(bool) & m_near
    tp_near = df["is_tp"] & m_near
    tn_near = df["is_tn"] & m_near
    fp_far = df["is_false_positive"].astype(bool) & (df["abs_distance"] > args.near_margin)
    tn_far = df["is_tn"] & (df["abs_distance"] > args.near_margin)

    _top_feature_diffs(df, fn_near, tp_near, f"FN near |d|<={args.near_margin:g}", "TP near", args.top_k)
    _top_feature_diffs(df, fp_near, tn_near, f"FP near |d|<={args.near_margin:g}", "TN near", args.top_k)
    _top_feature_diffs(df, fp_far, tn_far, f"FP far |d|>{args.near_margin:g}", "TN far", args.top_k)


if __name__ == "__main__":
    main()
