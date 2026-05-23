# -*- coding: utf-8 -*-
"""
Run a manual incremental feature trial on top of an existing active feature bank.

This script is intentionally separate from the LLM search loop so that
hand-written feature experiments do not mix with glm-generated history.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
for p in (ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.feature_search import (  # noqa: E402
    active_feature_bank_markdown,
    coefficients_markdown,
    evaluate_candidate,
    evaluate_svm_features,
    load_search_data,
    run_feature_ablation,
    write_json,
)


DEFAULT_AGG = PROJECT_ROOT / "FunSearch_test" / "training_2orientations.csv"
DEFAULT_ITEMS = PROJECT_ROOT / "FunSearch_test" / "物品信息和dblf信息.csv"
DEFAULT_MANUAL_ROOT = ROOT / "experiments_feature_search" / "manual_incremental"
DEFAULT_BANK_EXP = ROOT / "experiments_feature_search" / "by_model" / "glm-5.1" / "exp_20260510_181255"
DEFAULT_TREE_GUIDANCE = ROOT / "TREE_INSPIRED_FEATURE_HYPOTHESES.md"


MANUAL_CANDIDATES: dict[str, str] = {
    "wall_pressure_v1": """
import numpy as np
import pandas as pd

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    base = agg_df[["dispatch_id", "vehicle_length", "vehicle_width", "vehicle_height", "spare_capacity", "vehicle_capacity"]].copy()
    base["slack_ratio"] = base["spare_capacity"] / np.maximum(base["vehicle_capacity"], 1.0)

    work = items_df.merge(base, on="dispatch_id", how="left")
    g = work.groupby("dispatch_id", sort=False)

    out = pd.DataFrame({"dispatch_id": list(g.groups.keys())})
    out["long_width_competition"] = g.apply(
        lambda x: float(((x["dim_l"] / np.maximum(x["vehicle_length"], 1.0)) * (x["dim_m"] / np.maximum(x["vehicle_width"], 1.0))).sum())
    ).values
    out["tight_wall_share"] = g.apply(
        lambda x: float((((x["dim_m"] / np.maximum(x["vehicle_width"], 1.0)) > 0.78) & ((x["dim_l"] / np.maximum(x["vehicle_length"], 1.0)) > 0.62)).mean())
    ).values
    out["slack_wall_interaction"] = g.apply(
        lambda x: float((((x["dim_m"] / np.maximum(x["vehicle_width"], 1.0)) > 0.78).mean()) * max(0.0, 0.18 - float(x["slack_ratio"].iloc[0])))
    ).values
    return out
""",
    "heterogeneity_v1": """
import numpy as np
import pandas as pd

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    base = agg_df[["dispatch_id", "vehicle_length", "vehicle_width", "vehicle_height"]].copy()
    work = items_df.merge(base, on="dispatch_id", how="left")
    g = work.groupby("dispatch_id", sort=False)

    out = pd.DataFrame({"dispatch_id": list(g.groups.keys())})
    out["volume_cv"] = g.apply(
        lambda x: float(x["item_volume"].std(ddof=0) / max(float(x["item_volume"].mean()), 1.0))
    ).fillna(0.0).values
    out["shape_mix_entropy"] = g.apply(
        lambda x: float(
            -(lambda p: (p * np.log(np.maximum(p, 1e-12))).sum())(
                x.groupby(pd.cut(x["dim_l"] / np.maximum(x["vehicle_length"], 1.0), bins=[-np.inf, 0.45, 0.65, 0.8, np.inf])).size().astype(float)
                / max(float(len(x)), 1.0)
            )
        )
    ).values
    out["p95_volume_over_median"] = g.apply(
        lambda x: float(np.quantile(x["item_volume"], 0.95) / max(float(x["item_volume"].median()), 1.0))
    ).values
    return out
""",
}


def _load_active_bank(bank_exp_dir: pathlib.Path) -> tuple[pd.DataFrame | None, list[str]]:
    bank_csv = bank_exp_dir / "active_feature_bank.csv"
    if not bank_csv.exists():
        return None, []
    df = pd.read_csv(bank_csv)
    return df.reset_index(drop=True), list(df.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a manual incremental feature trial on top of an existing active bank")
    parser.add_argument("--bank-exp-dir", type=pathlib.Path, default=DEFAULT_BANK_EXP)
    parser.add_argument("--experiments-root", type=pathlib.Path, default=DEFAULT_MANUAL_ROOT)
    parser.add_argument("--candidate-name", type=str, choices=sorted(MANUAL_CANDIDATES.keys()), required=True)
    parser.add_argument("--data-path", type=pathlib.Path, default=DEFAULT_AGG)
    parser.add_argument("--items-path", type=pathlib.Path, default=DEFAULT_ITEMS)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--svm-c", type=float, default=10.0)
    parser.add_argument(
        "--tree-guidance-path",
        type=pathlib.Path,
        default=DEFAULT_TREE_GUIDANCE,
        help="Markdown note that documents the tree-inspired feature hypotheses used as manual guidance.",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = (args.experiments_root.resolve() / f"{args.candidate_name}_{ts}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    data = load_search_data(
        args.data_path,
        args.items_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    baseline_eval = evaluate_svm_features(
        feature_df=data.base_feature_df,
        y=data.y,
        train_idx=data.train_idx,
        test_idx=data.test_idx,
        svm_c=args.svm_c,
        random_state=args.random_state,
        new_feature_names=[],
    )
    baseline_metrics = baseline_eval.metrics

    active_bank_df, active_bank_names = _load_active_bank(args.bank_exp_dir.resolve())
    bank_eval = evaluate_svm_features(
        feature_df=pd.concat(
            [data.base_feature_df.reset_index(drop=True), (active_bank_df if active_bank_df is not None else pd.DataFrame(index=data.base_feature_df.index))],
            axis=1,
        ),
        y=data.y,
        train_idx=data.train_idx,
        test_idx=data.test_idx,
        svm_c=args.svm_c,
        random_state=args.random_state,
        new_feature_names=active_bank_names,
    )

    code = MANUAL_CANDIDATES[args.candidate_name].strip() + "\n"
    tree_guidance_path = (
        args.tree_guidance_path.resolve()
        if args.tree_guidance_path.is_absolute()
        else (PROJECT_ROOT / args.tree_guidance_path).resolve()
    )
    if tree_guidance_path.exists():
        (exp_dir / "tree_guidance.md").write_text(
            tree_guidance_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (exp_dir / "feature_candidate.py").write_text(code, encoding="utf-8")
    (exp_dir / "active_feature_bank.md").write_text(
        active_feature_bank_markdown(
            feature_names=active_bank_names,
            metrics=bank_eval.metrics,
            source_trials=[],
        ),
        encoding="utf-8",
    )

    artifact = evaluate_candidate(
        code=code,
        data=data,
        baseline_metrics=baseline_metrics,
        svm_c=args.svm_c,
        random_state=args.random_state,
        existing_feature_df=active_bank_df,
        existing_feature_names=active_bank_names,
    )

    artifact.feature_frame[artifact.new_feature_names].to_csv(exp_dir / "candidate_features.csv", index=False, encoding="utf-8")
    write_json(exp_dir / "metrics.json", artifact.metrics)
    write_json(exp_dir / "candidate_feature_coefficients.json", artifact.coefficients)
    (exp_dir / "candidate_feature_coefficients.md").write_text(
        coefficients_markdown(artifact.coefficients),
        encoding="utf-8",
    )
    ablation = run_feature_ablation(
        data=data,
        candidate_feature_df=artifact.feature_frame[artifact.new_feature_names],
        feature_names=artifact.new_feature_names,
        baseline_metrics=baseline_metrics,
        svm_c=args.svm_c,
        random_state=args.random_state,
        existing_feature_df=active_bank_df,
        existing_feature_names=active_bank_names,
    )
    write_json(exp_dir / "ablation.json", ablation)

    decision = {
        "candidate_name": args.candidate_name,
        "bank_exp_dir": str(args.bank_exp_dir.resolve()),
        "active_feature_count_before": len(active_bank_names),
        "new_feature_names": artifact.new_feature_names,
        "baseline_metrics": baseline_metrics,
        "active_bank_metrics": bank_eval.metrics,
        "candidate_metrics": artifact.metrics,
    }
    write_json(exp_dir / "decision.json", decision)
    write_json(
        exp_dir / "summary.json",
        {
            "mode": "manual_incremental",
            "candidate_name": args.candidate_name,
            "bank_exp_dir": str(args.bank_exp_dir.resolve()),
            "tree_guidance_path": str(tree_guidance_path),
            "active_feature_names_before": active_bank_names,
            "new_feature_names": artifact.new_feature_names,
            "baseline_metrics": baseline_metrics,
            "active_bank_metrics": bank_eval.metrics,
            "candidate_metrics": artifact.metrics,
        },
    )
    print(f"Manual experiment written to: {exp_dir}")
    print(
        f"[{args.candidate_name}] "
        f"AUC={artifact.metrics['auc']:.4f}, "
        f"TPR@1%={artifact.metrics['tpr_at_fpr1pct']:.4f}, "
        f"ACC={artifact.metrics['accuracy']:.4f}, "
        f"new_features={artifact.new_feature_names}"
    )


if __name__ == "__main__":
    main()
