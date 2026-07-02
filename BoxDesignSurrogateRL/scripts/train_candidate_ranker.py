#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from box_design_surrogate.candidate_ranker import (
    CANDIDATE_CATEGORICAL_FEATURES,
    CANDIDATE_NUMERIC_FEATURES,
    write_ranker_metadata,
)


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("top-k values must be positive")
    return values


def git_info(repo_root: Path) -> dict[str, str | bool | None]:
    import subprocess

    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run_git(["status", "--porcelain"])
    return {"commit": commit, "branch": branch, "dirty": bool(status) if status is not None else None}


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_trace_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["trace_source_path"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("at least one trace CSV is required")
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        raise ValueError("candidate trace data is empty")
    required = {"candidate_packaging_factor", "candidate_uncovered_orders", "candidate_unknown_pairs"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"candidate trace is missing required columns: {sorted(missing)}")
    return data


def ensure_feature_columns(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for col in CANDIDATE_NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    for col in CANDIDATE_CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = ""
    return out


def add_training_columns(
    data: pd.DataFrame,
    *,
    uncovered_weight: float,
    unknown_weight: float,
) -> pd.DataFrame:
    out = data.copy()
    if "candidate_objective" not in out.columns:
        out["candidate_objective"] = (
            out["candidate_uncovered_orders"].astype(float) * float(uncovered_weight)
            + out["candidate_unknown_pairs"].astype(float) * float(unknown_weight)
            + out["candidate_packaging_factor"].astype(float)
        )
    else:
        out["candidate_objective"] = (
            out["candidate_uncovered_orders"].astype(float) * float(uncovered_weight)
            + out["candidate_unknown_pairs"].astype(float) * float(unknown_weight)
            + out["candidate_packaging_factor"].astype(float)
        )
    def col(name: str, default: object) -> pd.Series:
        if name in out.columns:
            return out[name]
        return pd.Series([default] * len(out), index=out.index)

    out["step_group_id"] = (
        out["trace_source_path"].astype(str)
        + "::"
        + col("source_run_id", "").astype(str)
        + "::"
        + col("phase", "").astype(str)
        + "::"
        + col("stage", 0).astype(str)
        + "::"
        + col("iteration", 0).astype(str)
    )
    if "candidate_rank" not in out.columns:
        ranks = pd.Series(index=out.index, dtype=float)
        for _, group in out.groupby("step_group_id", sort=False):
            ranked_indices = sorted(
                group.index,
                key=lambda idx: (
                    int(out.loc[idx, "candidate_uncovered_orders"]),
                    int(out.loc[idx, "candidate_unknown_pairs"]),
                    float(out.loc[idx, "candidate_packaging_factor"]),
                    int(col("candidate_index", 0).loc[idx]),
                ),
            )
            for rank, idx in enumerate(ranked_indices, start=1):
                ranks.loc[idx] = rank
        out["candidate_rank"] = ranks.astype(int)
    out["candidate_objective_gap_to_best"] = (
        out["candidate_objective"].astype(float)
        - out.groupby("step_group_id")["candidate_objective"].transform("min").astype(float)
    )
    out["candidate_rank_log_target"] = np.log1p(out["candidate_rank"].astype(float) - 1.0)
    return out


def target_values(data: pd.DataFrame, target_mode: str) -> np.ndarray:
    mode = target_mode.strip().lower()
    if mode == "objective":
        return data["candidate_objective"].astype(float).to_numpy()
    if mode == "objective_gap":
        return data["candidate_objective_gap_to_best"].astype(float).to_numpy()
    if mode == "rank":
        return data["candidate_rank_log_target"].astype(float).to_numpy()
    raise ValueError(f"unknown --target-mode {target_mode!r}")


def split_by_step(
    data: pd.DataFrame,
    *,
    test_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = data["step_group_id"].to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or test_fraction <= 0.0:
        idx = np.arange(len(data))
        return idx, idx
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    train_idx, eval_idx = next(splitter.split(data, groups=groups))
    return np.asarray(train_idx), np.asarray(eval_idx)


def make_pipeline(
    model_name: str,
    random_state: int,
    *,
    hgbt_max_iter: int,
    hgbt_learning_rate: float,
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, CANDIDATE_NUMERIC_FEATURES),
            ("cat", categorical_transformer, CANDIDATE_CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )
    name = model_name.strip().lower()
    if name == "hgbt":
        regressor = HistGradientBoostingRegressor(
            learning_rate=hgbt_learning_rate,
            max_iter=hgbt_max_iter,
            l2_regularization=1e-3,
            random_state=random_state,
        )
    elif name == "rf":
        regressor = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"unknown --model {model_name!r}; expected hgbt or rf")
    return Pipeline(steps=[("preprocess", preprocessor), ("model", regressor)])


def score_tuple(row: pd.Series) -> tuple[int, int, float]:
    return (
        int(row["candidate_uncovered_orders"]),
        int(row["candidate_unknown_pairs"]),
        float(row["candidate_packaging_factor"]),
    )


def selected_row_for_top_k(group: pd.DataFrame, top_k: int) -> pd.Series:
    predicted = group.sort_values(["predicted_objective", "candidate_index"], ascending=[True, True])
    subset = predicted.head(min(top_k, len(predicted)))
    best_idx = sorted(subset.index, key=lambda idx: (score_tuple(subset.loc[idx]), int(subset.loc[idx]["candidate_index"])))[0]
    return subset.loc[best_idx]


def evaluate_predictions(eval_data: pd.DataFrame, top_ks: list[int]) -> dict[str, Any]:
    groups = list(eval_data.groupby("step_group_id", sort=False))
    if not groups:
        return {}

    best_predicted_ranks = []
    accepted_best_predicted_ranks = []
    accepted_groups = 0
    metrics: dict[str, Any] = {
        "eval_groups": len(groups),
        "eval_rows": int(len(eval_data)),
    }
    captures = {k: 0 for k in top_ks}
    accepted_captures = {k: 0 for k in top_ks}
    accepted_preserved = {k: 0 for k in top_ks}
    preserved = {k: 0 for k in top_ks}
    accepted_objective_gaps = {k: [] for k in top_ks}
    accepted_pf_gaps = {k: [] for k in top_ks}
    objective_gaps = {k: [] for k in top_ks}
    pf_gaps = {k: [] for k in top_ks}

    for _, group in groups:
        group = group.copy()
        predicted_order = group.sort_values(["predicted_objective", "candidate_index"], ascending=[True, True])
        predicted_rank_by_idx = {idx: rank for rank, idx in enumerate(predicted_order.index, start=1)}
        exact_best_idx = sorted(group.index, key=lambda idx: (score_tuple(group.loc[idx]), int(group.loc[idx]["candidate_index"])))[0]
        exact_best = group.loc[exact_best_idx]
        exact_best_rank = predicted_rank_by_idx[exact_best_idx]
        best_predicted_ranks.append(exact_best_rank)
        has_accepted = bool(int(exact_best.get("is_accepted", 0)) == 1)
        if has_accepted:
            accepted_groups += 1
            accepted_best_predicted_ranks.append(exact_best_rank)

        exact_best_objective = float(exact_best["candidate_objective"])
        exact_best_pf = float(exact_best["candidate_packaging_factor"])
        for k in top_ks:
            keep = min(k, len(group))
            top_indices = set(predicted_order.head(keep).index)
            if exact_best_idx in top_indices:
                captures[k] += 1
                if has_accepted:
                    accepted_captures[k] += 1
            selected = selected_row_for_top_k(group, k)
            if score_tuple(selected) == score_tuple(exact_best):
                preserved[k] += 1
                if has_accepted:
                    accepted_preserved[k] += 1
            objective_gaps[k].append(float(selected["candidate_objective"]) - exact_best_objective)
            pf_gaps[k].append(float(selected["candidate_packaging_factor"]) - exact_best_pf)
            if has_accepted:
                accepted_objective_gaps[k].append(float(selected["candidate_objective"]) - exact_best_objective)
                accepted_pf_gaps[k].append(float(selected["candidate_packaging_factor"]) - exact_best_pf)

    rank_arr = np.asarray(best_predicted_ranks, dtype=np.float64)
    metrics.update(
        {
            "accepted_eval_groups": int(accepted_groups),
            "mean_exact_best_predicted_rank": float(np.mean(rank_arr)),
            "median_exact_best_predicted_rank": float(np.median(rank_arr)),
            "p90_exact_best_predicted_rank": float(np.percentile(rank_arr, 90)),
            "mean_accepted_exact_best_predicted_rank": (
                float(np.mean(np.asarray(accepted_best_predicted_ranks, dtype=np.float64)))
                if accepted_best_predicted_ranks
                else None
            ),
            "median_accepted_exact_best_predicted_rank": (
                float(np.median(np.asarray(accepted_best_predicted_ranks, dtype=np.float64)))
                if accepted_best_predicted_ranks
                else None
            ),
        }
    )
    for k in top_ks:
        metrics[f"exact_best_capture_at_{k}"] = captures[k] / len(groups)
        metrics[f"accepted_move_capture_at_{k}"] = (
            accepted_captures[k] / accepted_groups if accepted_groups else None
        )
        metrics[f"exact_step_preservation_at_{k}"] = preserved[k] / len(groups)
        metrics[f"accepted_step_preservation_at_{k}"] = (
            accepted_preserved[k] / accepted_groups if accepted_groups else None
        )
        metrics[f"mean_objective_gap_at_{k}"] = float(np.mean(objective_gaps[k]))
        metrics[f"mean_pf_gap_at_{k}"] = float(np.mean(pf_gaps[k]))
        metrics[f"mean_accepted_objective_gap_at_{k}"] = (
            float(np.mean(accepted_objective_gaps[k])) if accepted_objective_gaps[k] else None
        )
        metrics[f"mean_accepted_pf_gap_at_{k}"] = (
            float(np.mean(accepted_pf_gaps[k])) if accepted_pf_gaps[k] else None
        )
        metrics[f"mean_validations_at_{k}"] = float(
            np.mean([min(k, len(group)) for _, group in groups])
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a candidate-level ranker from exact MILP local-search candidate traces."
    )
    parser.add_argument("--trace-csv", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--eval-trace-csv",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Optional held-out trace CSV(s). When supplied, the model trains on "
            "all --trace-csv rows and reports metrics on these held-out rows."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["hgbt", "rf"], default="hgbt")
    parser.add_argument(
        "--hgbt-max-iter",
        type=int,
        default=300,
        help="Number of boosting iterations for --model hgbt.",
    )
    parser.add_argument(
        "--hgbt-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for --model hgbt.",
    )
    parser.add_argument("--top-k", type=parse_int_list, default=parse_int_list("5,10,30"))
    parser.add_argument(
        "--target-mode",
        choices=["objective", "objective_gap", "rank"],
        default="objective",
        help=(
            "Training target. objective reproduces the original absolute score "
            "regression; objective_gap predicts each candidate's exact gap to "
            "the step-best candidate; rank predicts log(1 + exact_rank - 1)."
        ),
    )
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--uncovered-weight", type=float, default=1_000_000.0)
    parser.add_argument("--unknown-weight", type=float, default=1_000.0)
    args = parser.parse_args()

    if not 0.0 <= args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be in [0, 1)")
    if args.hgbt_max_iter <= 0:
        raise ValueError("--hgbt-max-iter must be positive")
    if args.hgbt_learning_rate <= 0.0:
        raise ValueError("--hgbt-learning-rate must be positive")

    data = ensure_feature_columns(
        add_training_columns(
            load_trace_csvs(args.trace_csv),
            uncovered_weight=args.uncovered_weight,
            unknown_weight=args.unknown_weight,
        )
    )
    if args.eval_trace_csv is not None:
        train_data = data.copy()
        eval_data = ensure_feature_columns(
            add_training_columns(
                load_trace_csvs(args.eval_trace_csv),
                uncovered_weight=args.uncovered_weight,
                unknown_weight=args.unknown_weight,
            )
        )
    else:
        train_idx, eval_idx = split_by_step(
            data,
            test_fraction=args.test_fraction,
            random_state=args.random_state,
        )
        train_data = data.iloc[train_idx].copy()
        eval_data = data.iloc[eval_idx].copy()

    pipeline = make_pipeline(
        args.model,
        args.random_state,
        hgbt_max_iter=args.hgbt_max_iter,
        hgbt_learning_rate=args.hgbt_learning_rate,
    )
    pipeline.fit(
        train_data[CANDIDATE_NUMERIC_FEATURES + CANDIDATE_CATEGORICAL_FEATURES],
        target_values(train_data, args.target_mode),
    )

    eval_data["predicted_objective"] = pipeline.predict(
        eval_data[CANDIDATE_NUMERIC_FEATURES + CANDIDATE_CATEGORICAL_FEATURES]
    )
    metrics = evaluate_predictions(eval_data, args.top_k)
    metrics.update(
        {
            "train_rows": int(len(train_data)),
            "eval_rows": int(len(eval_data)),
            "train_groups": int(train_data["step_group_id"].nunique()),
            "eval_groups": int(eval_data["step_group_id"].nunique()),
            "model": args.model,
            "target_mode": args.target_mode,
            "model_params": {
                "hgbt_max_iter": args.hgbt_max_iter if args.model == "hgbt" else None,
                "hgbt_learning_rate": args.hgbt_learning_rate if args.model == "hgbt" else None,
            },
            "top_k": args.top_k,
            "trace_csv": [str(path) for path in args.trace_csv],
            "eval_trace_csv": [str(path) for path in args.eval_trace_csv] if args.eval_trace_csv else None,
        }
    )

    run_id = datetime.now().strftime("candidate_ranker_%Y%m%d_%H%M%S")
    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = out_dir / "candidate_ranker.joblib"
    metadata = {
        "run_id": run_id,
        "model": args.model,
        "model_params": {
            "hgbt_max_iter": args.hgbt_max_iter if args.model == "hgbt" else None,
            "hgbt_learning_rate": args.hgbt_learning_rate if args.model == "hgbt" else None,
        },
        "numeric_features": CANDIDATE_NUMERIC_FEATURES,
        "categorical_features": CANDIDATE_CATEGORICAL_FEATURES,
        "target": "candidate_objective",
        "target_mode": args.target_mode,
        "target_description": {
            "objective": "absolute scalarized exact candidate objective",
            "objective_gap": "within-step candidate_objective minus exact step-best objective",
            "rank": "log1p(candidate_rank - 1), where lower rank is better",
        }[args.target_mode],
        "objective_scalarization": {
            "formula": "uncovered_weight * uncovered_orders + unknown_weight * unknown_pairs + packaging_factor",
            "uncovered_weight": args.uncovered_weight,
            "unknown_weight": args.unknown_weight,
        },
        "metrics": metrics,
        "trace_csv": [str(path) for path in args.trace_csv],
        "eval_trace_csv": [str(path) for path in args.eval_trace_csv] if args.eval_trace_csv else None,
        "git": git_info(REPO_ROOT),
    }
    joblib.dump(
        {
            "pipeline": pipeline,
            "numeric_features": CANDIDATE_NUMERIC_FEATURES,
            "categorical_features": CANDIDATE_CATEGORICAL_FEATURES,
            "metadata": metadata,
        },
        artifact_path,
    )
    write_ranker_metadata(out_dir / "metrics.json", metadata)
    print(json.dumps({"artifact_path": str(artifact_path), **metadata}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
