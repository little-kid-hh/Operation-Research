from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np


BUDGET_STATE_FEATURES = (
    "iteration_fraction",
    "current_packaging_factor",
    "step_size",
    "set_volume_cv",
    "ranker_log_spread",
    "gap_rank_2",
    "gap_rank_5",
    "gap_rank_10",
    "gap_rank_20",
    "gap_rank_30",
    "gap_rank_40",
    "gap_rank_50",
    "top5_shrink_fraction",
    "top10_shrink_fraction",
    "top20_shrink_fraction",
    "top1_relative_delta_volume",
)


def _number(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def ranked_candidate_indices(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda idx: (float(scores[idx]), idx))


def budget_state_features(
    rows: list[dict[str, Any]],
    scores: Sequence[float],
    *,
    iteration: int,
    max_iterations: int,
) -> np.ndarray:
    if not rows or len(rows) != len(scores):
        raise ValueError("rows and scores must be non-empty and have equal length")
    ranked = ranked_candidate_indices(scores)
    ordered_scores = np.asarray([float(scores[idx]) for idx in ranked], dtype=np.float64)
    score_min = float(ordered_scores[0])
    score_spread = max(float(ordered_scores[-1] - score_min), 1e-12)

    def normalized_gap(rank: int) -> float:
        idx = min(rank - 1, len(ordered_scores) - 1)
        return float((ordered_scores[idx] - score_min) / score_spread)

    def shrink_fraction(top_k: int) -> float:
        selected = [rows[idx] for idx in ranked[: min(top_k, len(ranked))]]
        return float(np.mean([_number(row, "is_shrink") for row in selected]))

    first = rows[0]
    mean_volume = max(_number(first, "current_set_mean_volume", 1.0), 1e-12)
    top_row = rows[ranked[0]]
    features = np.asarray(
        [
            float(iteration) / max(float(max_iterations), 1.0),
            np.log1p(max(_number(first, "current_packaging_factor"), 0.0)),
            _number(first, "step"),
            _number(first, "current_set_std_volume") / mean_volume,
            np.log1p(abs(score_spread)),
            normalized_gap(2),
            normalized_gap(5),
            normalized_gap(10),
            normalized_gap(20),
            normalized_gap(30),
            normalized_gap(40),
            normalized_gap(50),
            shrink_fraction(5),
            shrink_fraction(10),
            shrink_fraction(20),
            _number(top_row, "relative_delta_volume"),
        ],
        dtype=np.float64,
    )
    return features


def exact_candidate_rank(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        _number(row, "candidate_uncovered_orders", float("inf")),
        _number(row, "candidate_unknown_pairs", float("inf")),
        _number(row, "candidate_objective", float("inf")),
        int(_number(row, "candidate_index", 0)),
    )


def equivalent_exact_quality(left: dict[str, Any], right: dict[str, Any], *, tolerance: float = 1e-12) -> bool:
    return (
        int(_number(left, "candidate_uncovered_orders", -1))
        == int(_number(right, "candidate_uncovered_orders", -2))
        and int(_number(left, "candidate_unknown_pairs", -1))
        == int(_number(right, "candidate_unknown_pairs", -2))
        and abs(_number(left, "candidate_objective", float("inf")) - _number(right, "candidate_objective", float("inf")))
        <= tolerance
    )


def counterfactual_budget_preservation(
    rows: list[dict[str, Any]],
    scores: Sequence[float],
    budgets: Sequence[int],
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    ranked = ranked_candidate_indices(scores)
    exact_best = min(rows, key=exact_candidate_rank)
    outcomes = []
    for budget in budgets:
        if budget <= 0:
            raise ValueError("budgets must be positive")
        selected = min((rows[idx] for idx in ranked[: min(int(budget), len(rows))]), key=exact_candidate_rank)
        outcomes.append(equivalent_exact_quality(selected, exact_best, tolerance=tolerance))
    return np.asarray(outcomes, dtype=bool)


@dataclass
class BudgetTransitionDataset:
    states: np.ndarray
    preserves: np.ndarray
    next_indices: np.ndarray
    episode_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.states.ndim != 2:
            raise ValueError("states must be a 2D matrix")
        if self.preserves.ndim != 2 or len(self.preserves) != len(self.states):
            raise ValueError("preserves must align with states")
        if self.next_indices.shape != (len(self.states),):
            raise ValueError("next_indices must align with states")


@dataclass
class BudgetFQIPolicy:
    budgets: tuple[int, ...]
    feature_names: tuple[str, ...]
    models: list[Any]
    gamma: float
    miss_penalty: float
    max_budget: int

    def predict_q(self, states: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(states, dtype=np.float64))
        return np.column_stack([model.predict(x) for model in self.models])

    def select_budget(self, state: np.ndarray) -> int:
        q_values = self.predict_q(np.asarray(state, dtype=np.float64))[0]
        return int(self.budgets[int(np.argmax(q_values))])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "BudgetFQIPolicy":
        policy = joblib.load(path)
        if not isinstance(policy, cls):
            raise TypeError(f"unexpected budget policy artifact: {type(policy)!r}")
        return policy


def fit_budget_fqi(
    dataset: BudgetTransitionDataset,
    budgets: Sequence[int],
    *,
    iterations: int = 20,
    gamma: float = 0.99,
    miss_penalty: float = 100.0,
    random_state: int = 0,
) -> BudgetFQIPolicy:
    from sklearn.ensemble import HistGradientBoostingRegressor

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    budget_values = tuple(int(value) for value in budgets)
    if len(budget_values) != dataset.preserves.shape[1]:
        raise ValueError("budget count must match preservation columns")
    max_budget = max(budget_values)
    models = [
        HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=3,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=random_state + idx,
        )
        for idx in range(len(budget_values))
    ]
    next_q = np.zeros(len(dataset.states), dtype=np.float64)
    for _iteration in range(iterations):
        fitted = []
        for action_idx, budget in enumerate(budget_values):
            reward = -float(budget) / float(max_budget) - miss_penalty * (~dataset.preserves[:, action_idx])
            continuation = np.zeros(len(dataset.states), dtype=np.float64)
            valid = dataset.preserves[:, action_idx] & (dataset.next_indices >= 0)
            continuation[valid] = gamma * next_q[dataset.next_indices[valid]]
            model = models[action_idx]
            model.fit(dataset.states, reward + continuation)
            fitted.append(model)
        models = fitted
        q_values = np.column_stack([model.predict(dataset.states) for model in models])
        next_q = np.max(q_values, axis=1)
    return BudgetFQIPolicy(
        budgets=budget_values,
        feature_names=BUDGET_STATE_FEATURES,
        models=models,
        gamma=float(gamma),
        miss_penalty=float(miss_penalty),
        max_budget=max_budget,
    )

