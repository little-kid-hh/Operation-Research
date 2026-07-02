from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .evaluator import Box
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

    @classmethod
    def load(cls, path: Path) -> "CandidateRanker":
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "pipeline" not in payload:
            raise TypeError(f"Expected candidate ranker artifact dict in {path}")
        return cls(
            pipeline=payload["pipeline"],
            numeric_features=list(payload.get("numeric_features", CANDIDATE_NUMERIC_FEATURES)),
            categorical_features=list(payload.get("categorical_features", CANDIDATE_CATEGORICAL_FEATURES)),
            metadata=dict(payload.get("metadata", {})),
        )

    def predict_scores(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=np.float64)
        frame = pd.DataFrame(rows)
        for col in self.numeric_features:
            if col not in frame.columns:
                frame[col] = 0.0
        for col in self.categorical_features:
            if col not in frame.columns:
                frame[col] = ""
        x_df = frame[self.numeric_features + self.categorical_features]
        return np.asarray(self.pipeline.predict(x_df), dtype=np.float64).ravel()


def write_ranker_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
