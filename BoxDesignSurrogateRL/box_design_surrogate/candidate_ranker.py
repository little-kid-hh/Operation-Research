from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .evaluator import Box
from .kandula_repro import order_requirement
from .milp_oracle import MilpBoxSetScore
from .search import BoxMove


CANDIDATE_NUMERIC_FEATURES = [
    "stage",
    "iteration",
    "step",
    "candidate_index",
    "generated_candidates",
    "current_packaging_factor",
    "current_mean_box_volume",
    "current_mean_order_volume",
    "current_coverage_rate",
    "current_uncovered_orders",
    "current_unknown_pairs",
    "current_orders_with_unknown",
    "current_box_length",
    "current_box_width",
    "current_box_height",
    "current_box_volume",
    "current_box_surface_area",
    "candidate_box_length",
    "candidate_box_width",
    "candidate_box_height",
    "candidate_box_volume",
    "candidate_box_surface_area",
    "delta_length",
    "delta_width",
    "delta_height",
    "delta_volume",
    "delta_surface_area",
    "relative_delta_length",
    "relative_delta_width",
    "relative_delta_height",
    "relative_delta_volume",
    "abs_move_delta",
    "is_shrink",
    "current_set_mean_volume",
    "current_set_min_volume",
    "current_set_max_volume",
    "current_set_std_volume",
    "current_set_total_volume",
    "candidate_set_mean_volume",
    "candidate_set_min_volume",
    "candidate_set_max_volume",
    "candidate_set_std_volume",
    "candidate_set_total_volume",
    "assignment_moved_box_order_count",
    "assignment_moved_box_order_volume",
    "assignment_moved_box_mean_order_volume",
    "assignment_moved_box_order_share",
    "assignment_moved_box_volume_share",
    "assignment_candidate_capture_count",
    "assignment_candidate_capture_volume",
    "assignment_candidate_capture_assigned_box_volume_delta",
    "assignment_candidate_new_capture_count",
    "assignment_candidate_new_capture_volume",
    "assignment_moved_box_at_risk_count",
    "assignment_moved_box_at_risk_volume",
]

CANDIDATE_CATEGORICAL_FEATURES = [
    "move_dimension",
    "move_direction",
]

CANDIDATE_TRACE_COLUMNS = [
    "source_run_id",
    "algorithm",
    "phase",
    *CANDIDATE_NUMERIC_FEATURES,
    *CANDIDATE_CATEGORICAL_FEATURES,
    "move_box_id",
    "move_delta",
    "candidate_packaging_factor",
    "candidate_mean_box_volume",
    "candidate_mean_order_volume",
    "candidate_coverage_rate",
    "candidate_uncovered_orders",
    "candidate_unknown_pairs",
    "candidate_orders_with_unknown",
    "candidate_objective",
    "candidate_pf_delta",
    "candidate_mean_box_volume_delta",
    "candidate_rank",
    "is_exact_best",
    "is_improvement",
    "is_accepted",
]


def positive_class_probability(predictor: Any, x: Any) -> np.ndarray:
    probabilities = np.asarray(predictor.predict_proba(x), dtype=np.float64)
    classes = getattr(predictor, "classes_", None)
    if classes is None and hasattr(predictor, "named_steps"):
        model = getattr(predictor, "named_steps", {}).get("model")
        classes = getattr(model, "classes_", None)
    if classes is not None:
        matches = np.where(np.asarray(classes) == 1)[0]
        if len(matches) == 0:
            return np.zeros(probabilities.shape[0], dtype=np.float64)
        return probabilities[:, int(matches[0])]
    if probabilities.shape[1] <= 1:
        return np.zeros(probabilities.shape[0], dtype=np.float64)
    return probabilities[:, 1]


def score_key(score: MilpBoxSetScore) -> tuple[int, int, float]:
    return score.uncovered_orders, score.unknown_pairs, score.packaging_factor


def score_objective(
    score: MilpBoxSetScore,
    *,
    uncovered_weight: float = 1_000_000.0,
    unknown_weight: float = 1_000.0,
) -> float:
    """Scalarized exact objective used only as a learning target.

    The search itself still compares scores lexicographically. The large
    weights make the regression target follow the same order in ordinary
    all-covered runs while keeping it simple for a tabular learner.
    """

    return (
        float(score.uncovered_orders) * float(uncovered_weight)
        + float(score.unknown_pairs) * float(unknown_weight)
        + float(score.packaging_factor)
    )


def box_surface_area(box: Box) -> float:
    return 2.0 * (box.length * box.width + box.length * box.height + box.width * box.height)


def _box_by_id(boxes: Iterable[Box]) -> dict[int, Box]:
    return {int(box.box_id): box for box in boxes}


def _safe_relative(delta: float, base: float) -> float:
    if abs(base) <= 1e-12:
        return 0.0
    return float(delta) / float(base)


def _set_volume_stats(boxes: list[Box], prefix: str) -> dict[str, float]:
    volumes = np.asarray([box.volume for box in boxes], dtype=np.float64)
    return {
        f"{prefix}_set_mean_volume": float(np.mean(volumes)),
        f"{prefix}_set_min_volume": float(np.min(volumes)),
        f"{prefix}_set_max_volume": float(np.max(volumes)),
        f"{prefix}_set_std_volume": float(np.std(volumes, ddof=0)),
        f"{prefix}_set_total_volume": float(np.sum(volumes)),
    }


def _aggregate_feasible(order: Any, box: Box, eps: float = 1e-9) -> bool:
    req_l, req_m, req_s, total_volume = order_requirement(order)
    return (
        req_l <= box.length + eps
        and req_m <= box.width + eps
        and req_s <= box.height + eps
        and total_volume <= box.volume + eps
    )


def _assignment_feature_stats(
    *,
    orders: list[Any] | None,
    current_boxes: list[Box],
    current_box: Box,
    candidate_box: Box,
    current_score: MilpBoxSetScore,
) -> dict[str, float]:
    zero = {
        "assignment_moved_box_order_count": 0.0,
        "assignment_moved_box_order_volume": 0.0,
        "assignment_moved_box_mean_order_volume": 0.0,
        "assignment_moved_box_order_share": 0.0,
        "assignment_moved_box_volume_share": 0.0,
        "assignment_candidate_capture_count": 0.0,
        "assignment_candidate_capture_volume": 0.0,
        "assignment_candidate_capture_assigned_box_volume_delta": 0.0,
        "assignment_candidate_new_capture_count": 0.0,
        "assignment_candidate_new_capture_volume": 0.0,
        "assignment_moved_box_at_risk_count": 0.0,
        "assignment_moved_box_at_risk_volume": 0.0,
    }
    assignments = tuple(current_score.assignments or ())
    if not orders or not assignments:
        return zero

    current_volume_by_id = {int(box.box_id): float(box.volume) for box in current_boxes}
    total_order_volume = float(sum(float(getattr(order, "total_volume", 0.0)) for order in orders))

    moved_count = 0
    moved_volume = 0.0
    capture_count = 0
    capture_volume = 0.0
    capture_assigned_box_volume_delta = 0.0
    new_capture_count = 0
    new_capture_volume = 0.0
    at_risk_count = 0
    at_risk_volume = 0.0

    for order_idx, order in enumerate(orders):
        if order_idx >= len(assignments):
            break
        order_volume = float(getattr(order, "total_volume", 0.0))
        assigned_box_id = assignments[order_idx]
        if assigned_box_id is None:
            continue
        assigned_box_id = int(assigned_box_id)
        if assigned_box_id == int(current_box.box_id):
            moved_count += 1
            moved_volume += order_volume
            if _aggregate_feasible(order, current_box) and not _aggregate_feasible(order, candidate_box):
                at_risk_count += 1
                at_risk_volume += order_volume
            continue

        assigned_volume = current_volume_by_id.get(assigned_box_id)
        if assigned_volume is None or assigned_volume <= candidate_box.volume + 1e-9:
            continue
        if not _aggregate_feasible(order, candidate_box):
            continue
        capture_count += 1
        capture_volume += order_volume
        capture_assigned_box_volume_delta += assigned_volume - candidate_box.volume
        if not _aggregate_feasible(order, current_box):
            new_capture_count += 1
            new_capture_volume += order_volume

    out = dict(zero)
    out.update(
        {
            "assignment_moved_box_order_count": float(moved_count),
            "assignment_moved_box_order_volume": float(moved_volume),
            "assignment_moved_box_mean_order_volume": float(moved_volume / moved_count) if moved_count else 0.0,
            "assignment_moved_box_order_share": float(moved_count / len(orders)) if orders else 0.0,
            "assignment_moved_box_volume_share": (
                float(moved_volume / total_order_volume) if total_order_volume > 0.0 else 0.0
            ),
            "assignment_candidate_capture_count": float(capture_count),
            "assignment_candidate_capture_volume": float(capture_volume),
            "assignment_candidate_capture_assigned_box_volume_delta": float(capture_assigned_box_volume_delta),
            "assignment_candidate_new_capture_count": float(new_capture_count),
            "assignment_candidate_new_capture_volume": float(new_capture_volume),
            "assignment_moved_box_at_risk_count": float(at_risk_count),
            "assignment_moved_box_at_risk_volume": float(at_risk_volume),
        }
    )
    return out


def candidate_feature_row(
    *,
    current_boxes: list[Box],
    candidate_boxes: list[Box],
    move: BoxMove,
    current_score: MilpBoxSetScore,
    step: float,
    stage: int,
    iteration: int,
    candidate_index: int,
    generated_candidates: int,
    orders: list[Any] | None = None,
) -> dict[str, Any]:
    current_by_id = _box_by_id(current_boxes)
    candidate_by_id = _box_by_id(candidate_boxes)
    current_box = current_by_id[int(move.box_id)]
    candidate_box = candidate_by_id[int(move.box_id)]

    delta_length = float(candidate_box.length - current_box.length)
    delta_width = float(candidate_box.width - current_box.width)
    delta_height = float(candidate_box.height - current_box.height)
    delta_volume = float(candidate_box.volume - current_box.volume)
    current_surface = box_surface_area(current_box)
    candidate_surface = box_surface_area(candidate_box)
    delta_surface = float(candidate_surface - current_surface)

    row: dict[str, Any] = {
        "stage": int(stage),
        "iteration": int(iteration),
        "step": float(step),
        "candidate_index": int(candidate_index),
        "generated_candidates": int(generated_candidates),
        "current_packaging_factor": float(current_score.packaging_factor),
        "current_mean_box_volume": float(current_score.mean_box_volume),
        "current_mean_order_volume": float(current_score.mean_order_volume),
        "current_coverage_rate": float(current_score.coverage_rate),
        "current_uncovered_orders": int(current_score.uncovered_orders),
        "current_unknown_pairs": int(current_score.unknown_pairs),
        "current_orders_with_unknown": int(current_score.orders_with_unknown),
        "current_box_length": float(current_box.length),
        "current_box_width": float(current_box.width),
        "current_box_height": float(current_box.height),
        "current_box_volume": float(current_box.volume),
        "current_box_surface_area": current_surface,
        "candidate_box_length": float(candidate_box.length),
        "candidate_box_width": float(candidate_box.width),
        "candidate_box_height": float(candidate_box.height),
        "candidate_box_volume": float(candidate_box.volume),
        "candidate_box_surface_area": candidate_surface,
        "delta_length": delta_length,
        "delta_width": delta_width,
        "delta_height": delta_height,
        "delta_volume": delta_volume,
        "delta_surface_area": delta_surface,
        "relative_delta_length": _safe_relative(delta_length, current_box.length),
        "relative_delta_width": _safe_relative(delta_width, current_box.width),
        "relative_delta_height": _safe_relative(delta_height, current_box.height),
        "relative_delta_volume": _safe_relative(delta_volume, current_box.volume),
        "abs_move_delta": abs(float(move.delta)),
        "is_shrink": int(float(move.delta) < 0.0),
        "move_dimension": str(move.dimension),
        "move_direction": "shrink" if float(move.delta) < 0.0 else "expand",
    }
    row.update(_set_volume_stats(current_boxes, "current"))
    row.update(_set_volume_stats(candidate_boxes, "candidate"))
    row.update(
        _assignment_feature_stats(
            orders=orders,
            current_boxes=current_boxes,
            current_box=current_box,
            candidate_box=candidate_box,
            current_score=current_score,
        )
    )
    return row


def candidate_trace_rows(
    *,
    source_run_id: str,
    algorithm: str,
    phase: str,
    stage: int,
    iteration: int,
    step: float,
    current_boxes: list[Box],
    current_score: MilpBoxSetScore,
    moves: list[BoxMove],
    candidates: list[list[Box]],
    candidate_scores: list[MilpBoxSetScore],
    orders: list[Any] | None = None,
) -> list[dict[str, Any]]:
    if not (len(moves) == len(candidates) == len(candidate_scores)):
        raise ValueError("moves, candidates, and candidate_scores must have equal length")

    ranked_indices = sorted(range(len(candidate_scores)), key=lambda idx: (score_key(candidate_scores[idx]), idx))
    rank_by_idx = {idx: rank for rank, idx in enumerate(ranked_indices, start=1)}
    best_idx = ranked_indices[0] if ranked_indices else None
    current_key = score_key(current_score)

    rows: list[dict[str, Any]] = []
    for idx, (move, candidate, candidate_score) in enumerate(zip(moves, candidates, candidate_scores)):
        is_improvement = score_key(candidate_score) < current_key
        is_exact_best = idx == best_idx
        row = candidate_feature_row(
            current_boxes=current_boxes,
            candidate_boxes=candidate,
            move=move,
            current_score=current_score,
            step=step,
            stage=stage,
            iteration=iteration,
            candidate_index=idx,
            generated_candidates=len(candidates),
            orders=orders,
        )
        row.update(
            {
                "source_run_id": source_run_id,
                "algorithm": algorithm,
                "phase": phase,
                "move_box_id": int(move.box_id),
                "move_delta": float(move.delta),
                "candidate_packaging_factor": float(candidate_score.packaging_factor),
                "candidate_mean_box_volume": float(candidate_score.mean_box_volume),
                "candidate_mean_order_volume": float(candidate_score.mean_order_volume),
                "candidate_coverage_rate": float(candidate_score.coverage_rate),
                "candidate_uncovered_orders": int(candidate_score.uncovered_orders),
                "candidate_unknown_pairs": int(candidate_score.unknown_pairs),
                "candidate_orders_with_unknown": int(candidate_score.orders_with_unknown),
                "candidate_objective": score_objective(candidate_score),
                "candidate_pf_delta": float(current_score.packaging_factor - candidate_score.packaging_factor),
                "candidate_mean_box_volume_delta": float(
                    current_score.mean_box_volume - candidate_score.mean_box_volume
                ),
                "candidate_rank": int(rank_by_idx[idx]),
                "is_exact_best": int(is_exact_best),
                "is_improvement": int(is_improvement),
                "is_accepted": int(is_exact_best and is_improvement),
            }
        )
        rows.append(row)
    return rows


def make_candidate_feature_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for col in CANDIDATE_NUMERIC_FEATURES:
        if col not in frame.columns:
            frame[col] = 0.0
    for col in CANDIDATE_CATEGORICAL_FEATURES:
        if col not in frame.columns:
            frame[col] = ""
    return frame[CANDIDATE_NUMERIC_FEATURES + CANDIDATE_CATEGORICAL_FEATURES]


@dataclass
class CandidateRanker:
    pipeline: Any
    numeric_features: list[str]
    categorical_features: list[str]
    metadata: dict[str, Any]
    fast_predictor: "FastCandidatePipeline | None" = field(default=None, repr=False)

    @classmethod
    def load(cls, path: Path) -> "CandidateRanker":
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "pipeline" not in payload:
            raise TypeError(f"Expected candidate ranker artifact dict in {path}")
        numeric_features = list(payload.get("numeric_features", CANDIDATE_NUMERIC_FEATURES))
        categorical_features = list(payload.get("categorical_features", CANDIDATE_CATEGORICAL_FEATURES))
        pipeline = payload["pipeline"]
        fast_predictor = FastCandidatePipeline.from_pipeline(
            pipeline,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        if fast_predictor is not None and _skip_fast_candidate_pipeline(fast_predictor.model):
            fast_predictor = None
        return cls(
            pipeline=pipeline,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            metadata=dict(payload.get("metadata", {})),
            fast_predictor=fast_predictor,
        )

    def predict_scores(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=np.float64)
        score_mode = str(self.metadata.get("score_mode", "predict"))
        if self.fast_predictor is not None:
            return self.fast_predictor.predict_scores(rows, score_mode=score_mode)
        return self._predict_scores_slow(rows)

    def _predict_scores_slow(self, rows: list[dict[str, Any]]) -> np.ndarray:
        frame = pd.DataFrame(rows)
        for col in self.numeric_features:
            if col not in frame.columns:
                frame[col] = 0.0
        for col in self.categorical_features:
            if col not in frame.columns:
                frame[col] = ""
        x_df = frame[self.numeric_features + self.categorical_features]
        if str(self.metadata.get("score_mode", "predict")) == "negative_positive_probability":
            return -positive_class_probability(self.pipeline, x_df).ravel()
        return np.asarray(self.pipeline.predict(x_df), dtype=np.float64).ravel()


@dataclass(frozen=True)
class FastCandidatePipeline:
    model: Any
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numeric_fill_values: np.ndarray
    categorical_fill_values: tuple[str, ...]
    categorical_values: tuple[tuple[str, ...], ...]
    categorical_offsets: tuple[int, ...]
    categorical_value_maps: tuple[dict[str, int], ...]

    @classmethod
    def from_pipeline(
        cls,
        pipeline: Any,
        *,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> "FastCandidatePipeline | None":
        try:
            steps = dict(pipeline.steps)
            preprocessor = steps["preprocess"]
            model = steps["model"]
            transformers = {name: (transformer, cols) for name, transformer, cols in preprocessor.transformers_}
            num_transformer, num_cols = transformers["num"]
            cat_transformer, cat_cols = transformers["cat"]
            if list(num_cols) != list(numeric_features) or list(cat_cols) != list(categorical_features):
                return None
            num_steps = dict(num_transformer.steps)
            cat_steps = dict(cat_transformer.steps)
            num_imputer = num_steps["imputer"]
            cat_imputer = cat_steps["imputer"]
            onehot = cat_steps["onehot"]
            return cls(
                model=model,
                numeric_features=tuple(numeric_features),
                categorical_features=tuple(categorical_features),
                numeric_fill_values=np.asarray(num_imputer.statistics_, dtype=np.float64),
                categorical_fill_values=tuple(str(value) for value in cat_imputer.statistics_),
                categorical_values=tuple(tuple(str(value) for value in cats) for cats in onehot.categories_),
                categorical_offsets=tuple(
                    int(value)
                    for value in np.cumsum(
                        [0, *[len(cats) for cats in onehot.categories_[:-1]]]
                    )
                ),
                categorical_value_maps=tuple(
                    {str(value): idx for idx, value in enumerate(cats)}
                    for cats in onehot.categories_
                ),
            )
        except Exception:
            return None

    def predict_scores(self, rows: list[dict[str, Any]], *, score_mode: str = "predict") -> np.ndarray:
        x = self.transform(rows)
        if score_mode == "negative_positive_probability":
            return -positive_class_probability(self.model, x).ravel()
        return np.asarray(self.model.predict(x), dtype=np.float64).ravel()

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return self.predict_scores(rows, score_mode="predict")

    def transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        n_rows = len(rows)
        try:
            numeric = np.asarray(
                [
                    [row[feature] if feature in row else 0.0 for feature in self.numeric_features]
                    for row in rows
                ],
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            numeric = np.empty((n_rows, len(self.numeric_features)), dtype=np.float64)
            for row_idx, row in enumerate(rows):
                for col_idx, feature in enumerate(self.numeric_features):
                    try:
                        value = row[feature] if feature in row else 0.0
                        numeric[row_idx, col_idx] = float(value)
                    except (TypeError, ValueError):
                        numeric[row_idx, col_idx] = np.nan
        if numeric.size:
            missing = ~np.isfinite(numeric)
            if np.any(missing):
                numeric[missing] = np.take(self.numeric_fill_values, np.where(missing)[1])

        categorical_width = sum(len(values) for values in self.categorical_values)
        categorical = np.zeros((n_rows, categorical_width), dtype=np.float64)
        for row_idx, row in enumerate(rows):
            for col_idx, feature in enumerate(self.categorical_features):
                raw_value = row[feature] if feature in row else ""
                if raw_value is None or (isinstance(raw_value, float) and not np.isfinite(raw_value)):
                    value = self.categorical_fill_values[col_idx]
                else:
                    value = str(raw_value)
                encoded_idx = self.categorical_value_maps[col_idx].get(value)
                if encoded_idx is not None:
                    categorical[row_idx, self.categorical_offsets[col_idx] + encoded_idx] = 1.0

        if numeric.size and categorical.size:
            return np.column_stack([numeric, categorical])
        if numeric.size:
            return numeric
        return categorical


def _skip_fast_candidate_pipeline(model: Any) -> bool:
    """Avoid known negative fast-path cases.

    The sklearn HistGradientBoostingRegressor spends nearly all predictor time
    inside its own predict implementation. In real candidate batches, bypassing
    ColumnTransformer saved under 2 ms but made model.predict slower on the
    resulting ndarray, so the end-to-end path was not consistently faster.
    """

    return type(model).__name__ == "HistGradientBoostingRegressor"


def write_ranker_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
