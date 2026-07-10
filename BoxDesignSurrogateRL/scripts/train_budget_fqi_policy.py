#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from box_design_surrogate.budget_policy import (
    BUDGET_STATE_FEATURES,
    BudgetTransitionDataset,
    budget_state_features,
    counterfactual_budget_preservation,
    fit_budget_fqi,
)
from box_design_surrogate.candidate_ranker import CandidateRanker


GROUP_COLUMNS = ["source_run_id", "phase", "stage", "iteration"]


def parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not budgets or any(value <= 0 for value in budgets) or tuple(sorted(set(budgets))) != budgets:
        raise argparse.ArgumentTypeError("budgets must be unique positive integers in increasing order")
    return budgets


def build_transition_dataset(
    frame: pd.DataFrame,
    ranker: CandidateRanker,
    budgets: tuple[int, ...],
) -> BudgetTransitionDataset:
    records = frame.to_dict("records")
    frame = frame.copy()
    frame["_ranker_score"] = ranker.predict_scores(records)
    groups = list(frame.groupby(GROUP_COLUMNS, sort=False, dropna=False))
    max_iteration_by_run = frame.groupby("source_run_id")["iteration"].max().to_dict()
    states = []
    preserves = []
    episode_ids = []
    keys = []
    for key, group in groups:
        rows = group.to_dict("records")
        scores = group["_ranker_score"].to_numpy(dtype=np.float64)
        source_run_id = str(key[0])
        iteration = int(key[3])
        states.append(
            budget_state_features(
                rows,
                scores,
                iteration=iteration,
                max_iterations=int(max_iteration_by_run[source_run_id]),
            )
        )
        preserves.append(counterfactual_budget_preservation(rows, scores, budgets))
        episode_ids.append(source_run_id)
        keys.append(key)
    next_indices = np.full(len(states), -1, dtype=np.int64)
    for idx in range(len(states) - 1):
        if episode_ids[idx] == episode_ids[idx + 1]:
            next_indices[idx] = idx + 1
    return BudgetTransitionDataset(
        states=np.asarray(states, dtype=np.float64),
        preserves=np.asarray(preserves, dtype=bool),
        next_indices=next_indices,
        episode_ids=np.asarray(episode_ids, dtype=object),
    )


def evaluate_policy(policy, dataset: BudgetTransitionDataset) -> dict[str, Any]:
    q_values = policy.predict_q(dataset.states)
    action_indices = np.argmax(q_values, axis=1)
    budgets = np.asarray(policy.budgets, dtype=np.int64)[action_indices]
    preserves = dataset.preserves[np.arange(len(dataset.states)), action_indices]
    effective_with_full_audit = np.where(preserves, budgets, policy.max_budget)
    return {
        "states": len(dataset.states),
        "preservation_rate": float(np.mean(preserves)),
        "mean_selected_budget": float(np.mean(budgets)),
        "mean_effective_budget_with_full_audit_on_miss": float(np.mean(effective_with_full_audit)),
        "selected_budget_counts": {
            str(budget): int(np.sum(budgets == budget)) for budget in policy.budgets
        },
    }


def evaluate_fixed_budgets(dataset: BudgetTransitionDataset, budgets: tuple[int, ...]) -> dict[str, Any]:
    out = {}
    for action_idx, budget in enumerate(budgets):
        preserves = dataset.preserves[:, action_idx]
        effective = np.where(preserves, budget, max(budgets))
        out[str(budget)] = {
            "preservation_rate": float(np.mean(preserves)),
            "mean_effective_budget_with_full_audit_on_miss": float(np.mean(effective)),
        }
    minimum = np.asarray(
        [next(budget for budget, preserve in zip(budgets, row) if preserve) for row in dataset.preserves],
        dtype=np.int64,
    )
    out["oracle_minimum"] = {
        "preservation_rate": 1.0,
        "mean_effective_budget_with_full_audit_on_miss": float(np.mean(minimum)),
    }
    return out


def subset_dataset(dataset: BudgetTransitionDataset, indices: np.ndarray) -> BudgetTransitionDataset:
    selected = np.asarray(indices, dtype=np.int64)
    states = dataset.states[selected]
    preserves = dataset.preserves[selected]
    episode_ids = dataset.episode_ids[selected]
    next_indices = np.full(len(states), -1, dtype=np.int64)
    for idx in range(len(states) - 1):
        if episode_ids[idx] == episode_ids[idx + 1]:
            next_indices[idx] = idx + 1
    return BudgetTransitionDataset(states, preserves, next_indices, episode_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an offline fitted-Q policy for ranker MILP budgets.")
    parser.add_argument("--candidate-trace", type=Path, action="append", required=True)
    parser.add_argument("--candidate-ranker-path", type=Path, required=True)
    parser.add_argument("--budgets", type=parse_budgets, default=parse_budgets("5,10,20,30,40,50,60"))
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument(
        "--dev-source-run",
        action="append",
        default=None,
        help="Hold out complete source_run_id episodes. Without this, one trace is split temporally for smoke use only.",
    )
    parser.add_argument("--fqi-iterations", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--miss-penalty", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", type=Path, default=Path("BoxDesignSurrogateRL/results/budget_fqi_policy"))
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between zero and one")

    frames = [pd.read_csv(path) for path in args.candidate_trace]
    frame = pd.concat(frames, ignore_index=True)
    ranker = CandidateRanker.load(args.candidate_ranker_path)
    dataset = build_transition_dataset(frame, ranker, args.budgets)
    if args.dev_source_run:
        dev_ids = set(args.dev_source_run)
        known_ids = set(str(value) for value in dataset.episode_ids)
        missing = dev_ids - known_ids
        if missing:
            raise ValueError(f"unknown --dev-source-run values: {sorted(missing)}")
        train_indices = np.flatnonzero([str(value) not in dev_ids for value in dataset.episode_ids])
        dev_indices = np.flatnonzero([str(value) in dev_ids for value in dataset.episode_ids])
        split_mode = "source_run_holdout"
    else:
        split = max(1, min(len(dataset.states) - 1, int(round(len(dataset.states) * args.train_fraction))))
        train_indices = np.arange(0, split, dtype=np.int64)
        dev_indices = np.arange(split, len(dataset.states), dtype=np.int64)
        split_mode = "temporal_smoke"
    if len(train_indices) == 0 or len(dev_indices) == 0:
        raise ValueError("train and dev partitions must both be non-empty")
    train = subset_dataset(dataset, train_indices)
    dev = subset_dataset(dataset, dev_indices)
    policy = fit_budget_fqi(
        train,
        args.budgets,
        iterations=args.fqi_iterations,
        gamma=args.gamma,
        miss_penalty=args.miss_penalty,
        random_state=args.seed,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    policy.save(out_dir / "budget_q_policy.joblib")
    payload = {
        "status": "exploratory",
        "candidate_traces": [str(path) for path in args.candidate_trace],
        "candidate_ranker_path": str(args.candidate_ranker_path),
        "budgets": list(args.budgets),
        "feature_names": list(BUDGET_STATE_FEATURES),
        "train": evaluate_policy(policy, train),
        "dev": evaluate_policy(policy, dev),
        "train_fixed_budget_baselines": evaluate_fixed_budgets(train, args.budgets),
        "dev_fixed_budget_baselines": evaluate_fixed_budgets(dev, args.budgets),
        "split_mode": split_mode,
        "train_source_runs": sorted({str(value) for value in train.episode_ids}),
        "dev_source_runs": sorted({str(value) for value in dev.episode_ids}),
        "parameters": vars(args) | {"out_root": str(args.out_root)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"run_dir": str(out_dir), **payload}, indent=2, default=str))


if __name__ == "__main__":
    main()
