from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from .evaluator import Box
from .features import OrderSummary, read_order_summaries
from .search import apply_move, coordinate_moves


@dataclass(frozen=True)
class SimpleBoxSetScore:
    packaging_factor: float
    mean_box_volume: float
    mean_order_volume: float
    coverage_rate: float
    uncovered_orders: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def order_requirement(order: OrderSummary) -> tuple[float, float, float, float]:
    """Return aggregate sorted-dimension and total-volume requirements."""
    req_l = max(order.dim_l)
    req_m = max(order.dim_m)
    req_s = max(order.dim_s)
    return req_l, req_m, req_s, order.total_volume


def order_vectors(orders: list[OrderSummary]) -> np.ndarray:
    rows = []
    for order in orders:
        req_l, req_m, req_s, total_vol = order_requirement(order)
        rows.append([req_l, req_m, req_s, np.cbrt(total_vol)])
    return np.asarray(rows, dtype=np.float64)


def order_requirement_matrix(orders: list[OrderSummary]) -> np.ndarray:
    """Return columns: req_l, req_m, req_s, total_volume."""
    rows = [order_requirement(order) for order in orders]
    return np.asarray(rows, dtype=np.float64)


def box_matrix(boxes: list[Box]) -> np.ndarray:
    """Return columns: length, width, height, volume."""
    return np.asarray(
        [[box.length, box.width, box.height, box.volume] for box in boxes],
        dtype=np.float64,
    )


def initial_boxes_kmeans(
    orders: list[OrderSummary],
    k: int,
    *,
    random_state: int = 42,
    safety_margin: float = 1.0,
) -> list[Box]:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(orders) < k:
        raise ValueError("number of orders must be >= k")

    x = order_vectors(orders)
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(x)
    boxes: list[Box] = []
    for cluster_id in range(k):
        members = [orders[i] for i in np.flatnonzero(labels == cluster_id)]
        if not members:
            continue
        reqs = [order_requirement(order) for order in members]
        length = max(r[0] for r in reqs) * safety_margin
        width = max(r[1] for r in reqs) * safety_margin
        height = max(r[2] for r in reqs) * safety_margin
        max_order_volume = max(r[3] for r in reqs)
        box_volume = length * width * height
        if box_volume < max_order_volume:
            scale = float(np.cbrt(max_order_volume / box_volume))
            length *= scale
            width *= scale
            height *= scale
        boxes.append(Box(cluster_id, float(length), float(width), float(height)))
    return sorted(boxes, key=lambda b: b.volume)


def simple_feasible(order: OrderSummary, box: Box) -> bool:
    eps = 1e-9
    req_l, req_m, req_s, total_vol = order_requirement(order)
    return (
        req_l <= box.length + eps
        and req_m <= box.width + eps
        and req_s <= box.height + eps
        and total_vol <= box.volume + eps
    )


def evaluate_simple_box_set(
    orders: list[OrderSummary],
    boxes: list[Box],
    *,
    uncovered_penalty_factor: float = 100.0,
) -> SimpleBoxSetScore:
    if not orders:
        raise ValueError("orders must be non-empty")
    if not boxes:
        raise ValueError("boxes must be non-empty")

    req = order_requirement_matrix(orders)
    box = box_matrix(boxes)
    eps = 1e-9
    order_volumes = req[:, 3]
    mean_order_volume = float(np.mean(order_volumes))
    penalty = uncovered_penalty_factor * float(np.max(box[:, 3]))

    feasible = (
        (req[:, None, 0] <= box[None, :, 0] + eps)
        & (req[:, None, 1] <= box[None, :, 1] + eps)
        & (req[:, None, 2] <= box[None, :, 2] + eps)
        & (req[:, None, 3] <= box[None, :, 3] + eps)
    )
    feasible_volumes = np.where(feasible, box[None, :, 3], np.inf)
    assigned = np.min(feasible_volumes, axis=1)
    covered = np.isfinite(assigned)
    uncovered = int(np.size(covered) - np.count_nonzero(covered))
    assigned = np.where(covered, assigned, penalty)
    mean_box_volume = float(np.mean(assigned))
    return SimpleBoxSetScore(
        packaging_factor=mean_box_volume / mean_order_volume if mean_order_volume > 0 else float("inf"),
        mean_box_volume=mean_box_volume,
        mean_order_volume=mean_order_volume,
        coverage_rate=(len(orders) - uncovered) / len(orders),
        uncovered_orders=uncovered,
    )


def greedy_coordinate_descent(
    orders: list[OrderSummary],
    boxes: list[Box],
    *,
    step: float,
    max_iters: int = 50,
) -> tuple[list[Box], SimpleBoxSetScore]:
    def is_better(candidate: SimpleBoxSetScore, incumbent: SimpleBoxSetScore) -> bool:
        return (
            candidate.uncovered_orders < incumbent.uncovered_orders
            or (
                candidate.uncovered_orders == incumbent.uncovered_orders
                and candidate.packaging_factor < incumbent.packaging_factor
            )
        )

    current = sorted(boxes, key=lambda b: b.box_id)
    current_score = evaluate_simple_box_set(orders, current)
    for _ in range(max_iters):
        best_boxes = current
        best_score = current_score
        for move in coordinate_moves(current, step):
            candidate = apply_move(current, move)
            score = evaluate_simple_box_set(orders, candidate)
            if is_better(score, best_score):
                best_boxes = candidate
                best_score = score
        if not is_better(best_score, current_score):
            break
        current = best_boxes
        current_score = best_score
    return sorted(current, key=lambda b: b.volume), current_score


def beam_tree_search(
    orders: list[OrderSummary],
    boxes: list[Box],
    *,
    step: float,
    depth: int = 5,
    beam_width: int = 8,
) -> tuple[list[Box], SimpleBoxSetScore]:
    """Tree-search baseline that keeps the best box sets at each depth."""
    if depth <= 0:
        return sorted(boxes, key=lambda b: b.volume), evaluate_simple_box_set(orders, boxes)
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    def key_for(candidate_boxes: list[Box]) -> tuple[tuple[int, float, float, float], ...]:
        return tuple(
            sorted(
                (
                    box.box_id,
                    round(box.length, 6),
                    round(box.width, 6),
                    round(box.height, 6),
                )
                for box in candidate_boxes
            )
        )

    def rank(score: SimpleBoxSetScore) -> tuple[int, float]:
        return score.uncovered_orders, score.packaging_factor

    start = sorted(boxes, key=lambda b: b.box_id)
    start_score = evaluate_simple_box_set(orders, start)
    beam: list[tuple[tuple[int, float], list[Box], SimpleBoxSetScore]] = [
        (rank(start_score), start, start_score)
    ]
    best_boxes = start
    best_score = start_score

    for _ in range(depth):
        expanded: dict[
            tuple[tuple[int, float, float, float], ...],
            tuple[tuple[int, float], list[Box], SimpleBoxSetScore],
        ] = {}
        for _, candidate_boxes, _ in beam:
            for move in coordinate_moves(candidate_boxes, step):
                child = apply_move(candidate_boxes, move)
                score = evaluate_simple_box_set(orders, child)
                key = key_for(child)
                old = expanded.get(key)
                score_rank = rank(score)
                if old is None or score_rank < old[0]:
                    expanded[key] = (score_rank, child, score)

        if not expanded:
            break
        next_beam = sorted(expanded.values(), key=lambda x: x[0])[:beam_width]
        if next_beam[0][0] < rank(best_score):
            _, best_boxes, best_score = next_beam[0]
        beam = next_beam

    return sorted(best_boxes, key=lambda b: b.volume), best_score


def read_default_orders(limit: int | None = None) -> list[OrderSummary]:
    root = Path(__file__).resolve().parents[1]
    xml_path = root / "assets" / "or2023_bsp_data" / "xml_unique" / "or2023_bsp_unique_orders.xml"
    orders = read_order_summaries(xml_path)
    return orders[:limit] if limit is not None else orders
