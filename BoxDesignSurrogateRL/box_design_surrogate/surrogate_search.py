from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .evaluator import BatchSurrogateEvaluator, Box, BoxSetEvaluation, PreparedOrders, SurrogateEvaluator
from .features import OrderSummary
from .search import BoxMove, apply_move, coordinate_moves


@dataclass(frozen=True)
class SurrogateSearchResult:
    boxes: list[Box]
    evaluation: BoxSetEvaluation
    iterations: int
    evaluated_candidates: int
    manifest: dict[str, object] = field(default_factory=dict)


def surrogate_coordinate_descent(
    orders: list[OrderSummary],
    boxes: list[Box],
    evaluator: SurrogateEvaluator,
    *,
    step: float,
    max_iters: int = 10,
) -> SurrogateSearchResult:
    """Greedy local search scored by the learned feasibility surrogate."""

    def rank(result: BoxSetEvaluation) -> tuple[int, float]:
        return result.uncovered_orders, result.total_adjusted_cost

    current = sorted(boxes, key=lambda b: b.box_id)
    current_eval = evaluator.evaluate(orders, current)
    evaluated = 1
    completed_iters = 0

    for _ in range(max_iters):
        best_boxes = current
        best_eval = current_eval
        for move in coordinate_moves(current, step):
            candidate = apply_move(current, move)
            candidate_eval = evaluator.evaluate(orders, candidate)
            evaluated += 1
            if rank(candidate_eval) < rank(best_eval):
                best_boxes = candidate
                best_eval = candidate_eval

        if rank(best_eval) >= rank(current_eval):
            break
        current = best_boxes
        current_eval = best_eval
        completed_iters += 1

    return SurrogateSearchResult(
        boxes=sorted(current, key=lambda b: b.volume),
        evaluation=current_eval,
        iterations=completed_iters,
        evaluated_candidates=evaluated,
    )


def surrogate_repair_search(
    orders: list[OrderSummary],
    boxes: list[Box],
    evaluator: SurrogateEvaluator,
    *,
    step_schedule: tuple[float, ...] | list[float] = (0.5, 0.1, 0.05, 0.01),
    max_iters: int = 50,
    max_repair_orders: int = 64,
    include_coordinate_moves: bool = True,
    max_candidates_per_iter: int | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> SurrogateSearchResult:
    """Deterministic repair-first search scored by surrogate assignment quality."""
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if not step_schedule:
        raise ValueError("step_schedule must be non-empty")

    batch = (
        evaluator
        if isinstance(evaluator, BatchSurrogateEvaluator)
        else BatchSurrogateEvaluator.from_evaluator(evaluator, orders)
    )

    current = sorted(boxes, key=lambda b: b.box_id)
    current_eval = batch.evaluate_prepared(current)
    evaluated = 1
    completed_iters = 0
    history: list[dict[str, object]] = []
    started_at = time.perf_counter()

    for step in step_schedule:
        step_iters = 0
        while completed_iters < max_iters:
            iter_started_at = time.perf_counter()
            moves = repair_moves(
                orders,
                current,
                current_eval,
                step=float(step),
                prepared_orders=batch.prepared_orders,
                max_orders=max_repair_orders,
            )
            if include_coordinate_moves:
                moves.extend(coordinate_moves(current, float(step)))
            moves = _dedupe_moves(moves)
            if max_candidates_per_iter is not None:
                moves = moves[:max_candidates_per_iter]

            best_boxes = current
            best_eval = current_eval
            best_move = None
            candidates = [apply_move(current, move) for move in moves]
            candidate_evals = batch.evaluate_many_box_sets(candidates)
            evaluated += len(candidate_evals)
            for move, candidate, candidate_eval in zip(moves, candidates, candidate_evals):
                if _search_rank(candidate_eval, candidate) < _search_rank(best_eval, best_boxes):
                    best_boxes = candidate
                    best_eval = candidate_eval
                    best_move = move

            progress = {
                "iteration": completed_iters + 1,
                "step": float(step),
                "candidate_moves": len(moves),
                "evaluated_candidates": evaluated,
                "elapsed_seconds": time.perf_counter() - started_at,
                "iteration_seconds": time.perf_counter() - iter_started_at,
                "coverage_rate": best_eval.coverage_rate,
                "uncovered_orders": best_eval.uncovered_orders,
                "low_margin_assignments": best_eval.low_margin_assignments,
                "mean_adjusted_cost": best_eval.total_adjusted_cost / len(orders),
                "accepted": _search_rank(best_eval, best_boxes) < _search_rank(current_eval, current),
                "move": None if best_move is None else best_move.__dict__,
            }
            if progress_callback is not None:
                progress_callback(progress)

            if _search_rank(best_eval, best_boxes) >= _search_rank(current_eval, current):
                history.append(progress)
                break

            current = sorted(best_boxes, key=lambda b: b.box_id)
            current_eval = best_eval
            completed_iters += 1
            step_iters += 1
            progress["iteration"] = completed_iters
            progress["mean_box_volume"] = _mean_box_volume(current)
            history.append(progress)
        if step_iters == 0 and current_eval.uncovered_orders == 0 and current_eval.low_margin_assignments == 0:
            continue

    return SurrogateSearchResult(
        boxes=sorted(current, key=lambda b: b.volume),
        evaluation=current_eval,
        iterations=completed_iters,
        evaluated_candidates=evaluated,
        manifest={
            "algorithm": "surrogate_repair_search",
            "step_schedule": [float(x) for x in step_schedule],
            "max_iters": max_iters,
            "max_repair_orders": max_repair_orders,
            "include_coordinate_moves": include_coordinate_moves,
            "max_candidates_per_iter": max_candidates_per_iter,
            "elapsed_seconds": time.perf_counter() - started_at,
            "history": history,
        },
    )


def surrogate_fast_repair_search(
    orders: list[OrderSummary],
    boxes: list[Box],
    evaluator: SurrogateEvaluator,
    *,
    step_schedule: tuple[float, ...] | list[float] = (0.5, 0.1, 0.05, 0.01),
    max_iters: int = 5,
    max_repair_orders: int = 16,
    max_candidates_per_iter: int = 64,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> SurrogateSearchResult:
    return surrogate_repair_search(
        orders,
        boxes,
        evaluator,
        step_schedule=step_schedule,
        max_iters=max_iters,
        max_repair_orders=max_repair_orders,
        include_coordinate_moves=False,
        max_candidates_per_iter=max_candidates_per_iter,
        progress_callback=progress_callback,
    )


def repair_moves(
    orders: list[OrderSummary],
    boxes: list[Box],
    evaluation: BoxSetEvaluation,
    *,
    step: float,
    prepared_orders: PreparedOrders | None = None,
    max_orders: int = 64,
) -> list[BoxMove]:
    """Generate expansion moves aimed at uncovered and low-margin assignments."""
    if step <= 0:
        raise ValueError("step must be positive")
    if prepared_orders is None:
        prepared_orders = PreparedOrders.from_orders(orders)

    boxes_by_id = {box.box_id: box for box in boxes}
    order_idx_by_key = {key: idx for idx, key in enumerate(prepared_orders.order_keys)}
    targets = sorted(
        evaluation.assignments,
        key=lambda a: (
            0 if not a.covered else 1,
            a.probability if a.covered else a.best_probability,
            a.adjusted_cost,
            a.order_key,
        ),
    )
    moves: list[BoxMove] = []
    used_targets = 0
    for assignment in targets:
        if assignment.covered and not assignment.low_margin:
            continue
        box_id = assignment.box_id if assignment.box_id is not None else assignment.best_box_id
        if box_id is None or box_id not in boxes_by_id:
            continue
        order_idx = order_idx_by_key.get(assignment.order_key)
        if order_idx is None:
            continue
        box = boxes_by_id[box_id]
        reqs = {
            "length": float(prepared_orders.dim_l[order_idx]),
            "width": float(prepared_orders.dim_m[order_idx]),
            "height": float(prepared_orders.dim_s[order_idx]),
        }
        current = {
            "length": box.length,
            "width": box.width,
            "height": box.height,
        }
        deficits = {dim: max(0.0, reqs[dim] - current[dim]) for dim in reqs}
        for dim in ("length", "width", "height"):
            delta = _round_up_to_step(deficits[dim], step) if deficits[dim] > 0 else step
            moves.append(BoxMove(box_id=box_id, dimension=dim, delta=delta))

        volume_deficit = float(prepared_orders.total_volume[order_idx] - box.volume)
        if volume_deficit > 0:
            scale = float(np.cbrt(prepared_orders.total_volume[order_idx] / box.volume))
            for dim in ("length", "width", "height"):
                delta = _round_up_to_step(current[dim] * (scale - 1.0), step)
                if delta > 0:
                    moves.append(BoxMove(box_id=box_id, dimension=dim, delta=delta))

        used_targets += 1
        if used_targets >= max_orders:
            break

    return _coalesce_expansion_moves(moves)


def _search_rank(evaluation: BoxSetEvaluation, boxes: list[Box]) -> tuple[int, int, float, float]:
    return (
        evaluation.uncovered_orders,
        evaluation.low_margin_assignments,
        evaluation.total_adjusted_cost / len(evaluation.assignments),
        _mean_box_volume(boxes),
    )


def _mean_box_volume(boxes: list[Box]) -> float:
    return float(np.mean([box.volume for box in boxes])) if boxes else float("inf")


def _round_up_to_step(value: float, step: float) -> float:
    if value <= 0:
        return 0.0
    return float(math.ceil((value - 1e-12) / step) * step)


def _dedupe_moves(moves: list[BoxMove]) -> list[BoxMove]:
    seen = set()
    out = []
    for move in moves:
        key = (move.box_id, move.dimension, round(move.delta, 12))
        if key in seen:
            continue
        seen.add(key)
        out.append(move)
    return out


def _coalesce_expansion_moves(moves: list[BoxMove]) -> list[BoxMove]:
    best_by_dim: dict[tuple[int, str], BoxMove] = {}
    passthrough: list[BoxMove] = []
    for move in moves:
        if move.delta <= 0:
            passthrough.append(move)
            continue
        key = (move.box_id, move.dimension)
        old = best_by_dim.get(key)
        if old is None or move.delta > old.delta:
            best_by_dim[key] = move
    return _dedupe_moves(passthrough + list(best_by_dim.values()))
