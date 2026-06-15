from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .features import FEATURE_COLS, OrderSummary, make_base40_features


class ProbabilityModel(Protocol):
    def predict_proba(self, x_df: pd.DataFrame):
        ...


@dataclass(frozen=True)
class Box:
    box_id: int
    length: float
    width: float
    height: float

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def surface_area(self) -> float:
        return 2.0 * (
            self.length * self.width
            + self.length * self.height
            + self.width * self.height
        )


@dataclass(frozen=True)
class Assignment:
    order_key: str
    box_id: int | None
    probability: float
    base_cost: float
    adjusted_cost: float
    covered: bool
    best_box_id: int | None = None
    best_probability: float = 0.0
    low_margin: bool = False


@dataclass(frozen=True)
class BoxSetEvaluation:
    total_adjusted_cost: float
    total_base_cost: float
    uncovered_orders: int
    coverage_rate: float
    mean_assigned_probability: float
    low_margin_assignments: int
    assignments: tuple[Assignment, ...]


@dataclass
class SurrogateEvaluator:
    model: ProbabilityModel
    tau: float = 0.95
    tau_high: float = 0.99
    lambda_risk: float = 1_000.0
    uncovered_penalty: float = 1_000_000.0
    volume_weight: float = 1.0
    surface_weight: float = 0.0
    low_margin_band: float = 0.03

    def evaluate(self, orders: list[OrderSummary], boxes: list[Box]) -> BoxSetEvaluation:
        if not orders:
            raise ValueError("orders must be non-empty")
        if not boxes:
            raise ValueError("boxes must be non-empty")

        rows = []
        keys: list[tuple[str, int]] = []
        for order in orders:
            order_key = f"{order.instance_name}:{order.order_id}"
            for box in boxes:
                rows.append(make_base40_features(order, box.length, box.width, box.height))
                keys.append((order_key, box.box_id))

        x_df = pd.DataFrame(rows, columns=FEATURE_COLS)
        probs = _positive_probability(self.model.predict_proba(x_df))

        box_by_id = {box.box_id: box for box in boxes}
        by_order: dict[str, list[tuple[int, float]]] = {}
        for (order_key, box_id), prob in zip(keys, probs):
            by_order.setdefault(order_key, []).append((box_id, float(prob)))

        assignments: list[Assignment] = []
        total_base = 0.0
        total_adjusted = 0.0
        low_margin = 0

        for order in orders:
            order_key = f"{order.instance_name}:{order.order_id}"
            candidates = []
            best_box_id = None
            best_probability = 0.0
            for box_id, prob in by_order[order_key]:
                if prob > best_probability:
                    best_box_id = box_id
                    best_probability = prob
                if prob < self.tau:
                    continue
                box = box_by_id[box_id]
                base_cost = self.volume_weight * box.volume + self.surface_weight * box.surface_area
                risk = max(0.0, self.tau_high - prob) ** 2
                adjusted_cost = base_cost + self.lambda_risk * risk
                candidates.append((adjusted_cost, base_cost, box_id, prob))

            if not candidates:
                assignments.append(
                    Assignment(
                        order_key=order_key,
                        box_id=None,
                        probability=0.0,
                        base_cost=self.uncovered_penalty,
                        adjusted_cost=self.uncovered_penalty,
                        covered=False,
                        best_box_id=best_box_id,
                        best_probability=best_probability,
                        low_margin=False,
                    )
                )
                total_base += self.uncovered_penalty
                total_adjusted += self.uncovered_penalty
                continue

            adjusted_cost, base_cost, box_id, prob = min(candidates, key=lambda x: x[0])
            is_low_margin = prob < self.tau + self.low_margin_band
            if is_low_margin:
                low_margin += 1
            assignments.append(
                Assignment(
                    order_key=order_key,
                    box_id=box_id,
                    probability=prob,
                    base_cost=base_cost,
                    adjusted_cost=adjusted_cost,
                    covered=True,
                    best_box_id=best_box_id,
                    best_probability=best_probability,
                    low_margin=is_low_margin,
                )
            )
            total_base += base_cost
            total_adjusted += adjusted_cost

        covered = [a for a in assignments if a.covered]
        return BoxSetEvaluation(
            total_adjusted_cost=total_adjusted,
            total_base_cost=total_base,
            uncovered_orders=len(assignments) - len(covered),
            coverage_rate=len(covered) / len(assignments),
            mean_assigned_probability=float(np.mean([a.probability for a in covered])) if covered else 0.0,
            low_margin_assignments=low_margin,
            assignments=tuple(assignments),
        )


def _positive_probability(raw) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return arr.ravel()


@dataclass(frozen=True)
class PreparedOrders:
    orders: tuple[OrderSummary, ...]
    order_keys: tuple[str, ...]
    frame: pd.DataFrame
    dim_l: np.ndarray
    dim_m: np.ndarray
    dim_s: np.ndarray
    total_volume: np.ndarray

    @classmethod
    def from_orders(cls, orders: list[OrderSummary]) -> "PreparedOrders":
        if not orders:
            raise ValueError("orders must be non-empty")

        rows = []
        order_keys = []
        req_l = []
        req_m = []
        req_s = []
        total_volume = []
        for order in orders:
            dim_l = np.asarray(order.dim_l, dtype=np.float64)
            dim_m = np.asarray(order.dim_m, dtype=np.float64)
            dim_s = np.asarray(order.dim_s, dtype=np.float64)
            volumes = np.asarray(order.volumes, dtype=np.float64)
            footprints = np.asarray(order.footprints, dtype=np.float64)
            total_vol = float(np.sum(volumes))
            rows.append(
                {
                    "sku_counts": float(order.item_count),
                    "sku_average_volume": float(np.mean(volumes)),
                    "sku_length_var": _sample_var_np(dim_l),
                    "sku_width_var": _sample_var_np(dim_m),
                    "sku_height_var": _sample_var_np(dim_s),
                    "sku_length_avg": float(np.mean(dim_l)),
                    "sku_width_avg": float(np.mean(dim_m)),
                    "sku_height_avg": float(np.mean(dim_s)),
                    "max_asr": float(np.max(dim_l / np.where(dim_s != 0.0, dim_s, 1e-9))),
                    "sku_concentration": float(np.max(volumes) / total_vol) if total_vol > 0 else 0.0,
                    "sku_min_length": float(np.min(dim_l)),
                    "sku_max_length": float(np.max(dim_l)),
                    "sku_std_length": _sample_std_np(dim_l),
                    "sku_min_width": float(np.min(dim_m)),
                    "sku_max_width": float(np.max(dim_m)),
                    "sku_std_width": _sample_std_np(dim_m),
                    "sku_min_height": float(np.min(dim_s)),
                    "sku_max_height": float(np.max(dim_s)),
                    "sku_std_height": _sample_std_np(dim_s),
                    "_sum_volume": total_vol,
                    "_sum_footprint": float(np.sum(footprints)),
                    "_min_footprint": float(np.min(footprints)),
                    "_max_footprint": float(np.max(footprints)),
                    "_std_footprint": _sample_std_np(footprints),
                    "_req_l": float(np.max(dim_l)),
                    "_req_m": float(np.max(dim_m)),
                    "_req_s": float(np.max(dim_s)),
                }
            )
            order_keys.append(f"{order.instance_name}:{order.order_id}")
            req_l.append(float(np.max(dim_l)))
            req_m.append(float(np.max(dim_m)))
            req_s.append(float(np.max(dim_s)))
            total_volume.append(total_vol)

        return cls(
            orders=tuple(orders),
            order_keys=tuple(order_keys),
            frame=pd.DataFrame(rows),
            dim_l=np.asarray(req_l, dtype=np.float64),
            dim_m=np.asarray(req_m, dtype=np.float64),
            dim_s=np.asarray(req_s, dtype=np.float64),
            total_volume=np.asarray(total_volume, dtype=np.float64),
        )

    def make_feature_frame(self, boxes: list[Box]) -> pd.DataFrame:
        if not boxes:
            raise ValueError("boxes must be non-empty")

        n_orders = len(self.orders)
        n_boxes = len(boxes)
        base = self.frame
        repeated = pd.DataFrame(
            np.repeat(base.to_numpy(dtype=np.float64), n_boxes, axis=0),
            columns=base.columns,
        )
        box_l = np.tile(np.asarray([box.length for box in boxes], dtype=np.float64), n_orders)
        box_w = np.tile(np.asarray([box.width for box in boxes], dtype=np.float64), n_orders)
        box_h = np.tile(np.asarray([box.height for box in boxes], dtype=np.float64), n_orders)
        box_volume = box_l * box_w * box_h
        box_floor = box_l * box_w

        repeated["vehicle_length"] = box_l
        repeated["vehicle_width"] = box_w
        repeated["vehicle_height"] = box_h
        repeated["spare_capacity"] = box_volume - repeated["_sum_volume"]
        repeated["l_to_L_ratio_avg"] = repeated["sku_length_avg"] / box_l
        repeated["l_to_L_ratio_min"] = repeated["sku_min_length"] / box_l
        repeated["l_to_L_ratio_max"] = repeated["sku_max_length"] / box_l
        repeated["l_to_L_ratio_std"] = repeated["sku_std_length"] / box_l
        repeated["h_to_H_ratio_avg"] = repeated["sku_height_avg"] / box_h
        repeated["h_to_H_ratio_min"] = repeated["sku_min_height"] / box_h
        repeated["h_to_H_ratio_max"] = repeated["sku_max_height"] / box_h
        repeated["h_to_H_ratio_std"] = repeated["sku_std_height"] / box_h
        repeated["w_to_W_ratio_avg"] = repeated["sku_width_avg"] / box_w
        repeated["w_to_W_ratio_min"] = repeated["sku_min_width"] / box_w
        repeated["w_to_W_ratio_max"] = repeated["sku_max_width"] / box_w
        repeated["w_to_W_ratio_std"] = repeated["sku_std_width"] / box_w
        repeated["wl_to_vehicle_wl_avg"] = repeated["_sum_footprint"] / repeated["sku_counts"] / box_floor
        repeated["wl_to_vehicle_wl_min"] = repeated["_min_footprint"] / box_floor
        repeated["wl_to_vehicle_wl_max"] = repeated["_max_footprint"] / box_floor
        repeated["wl_to_vehicle_wl_std"] = repeated["_std_footprint"] / box_floor
        repeated["wl_to_vehicle_wl_total"] = repeated["_sum_footprint"] / box_floor
        return repeated[FEATURE_COLS]


@dataclass
class BatchSurrogateEvaluator(SurrogateEvaluator):
    prepared_orders: PreparedOrders | None = None

    @classmethod
    def from_evaluator(
        cls,
        evaluator: SurrogateEvaluator,
        orders: list[OrderSummary],
    ) -> "BatchSurrogateEvaluator":
        return cls(
            model=evaluator.model,
            tau=evaluator.tau,
            tau_high=evaluator.tau_high,
            lambda_risk=evaluator.lambda_risk,
            uncovered_penalty=evaluator.uncovered_penalty,
            volume_weight=evaluator.volume_weight,
            surface_weight=evaluator.surface_weight,
            low_margin_band=evaluator.low_margin_band,
            prepared_orders=PreparedOrders.from_orders(orders),
        )

    def evaluate(self, orders: list[OrderSummary], boxes: list[Box]) -> BoxSetEvaluation:
        prepared = self.prepared_orders
        if prepared is None or len(prepared.orders) != len(orders):
            prepared = PreparedOrders.from_orders(orders)
            self.prepared_orders = prepared
        return self.evaluate_prepared(boxes)

    def evaluate_prepared(
        self,
        boxes: list[Box],
        *,
        assignment_mode: str = "risk_adjusted",
    ) -> BoxSetEvaluation:
        prepared = self.prepared_orders
        if prepared is None:
            raise ValueError("prepared_orders is required")
        if not boxes:
            raise ValueError("boxes must be non-empty")
        if assignment_mode not in {"risk_adjusted", "min_volume"}:
            raise ValueError(f"unknown assignment_mode: {assignment_mode}")

        sorted_boxes = list(boxes)
        x_df = prepared.make_feature_frame(sorted_boxes)
        probs = _positive_probability(self.model.predict_proba(x_df)).reshape(
            len(prepared.orders),
            len(sorted_boxes),
        )
        return self._evaluation_from_probs(prepared, sorted_boxes, probs, assignment_mode=assignment_mode)

    def _evaluation_from_probs(
        self,
        prepared: PreparedOrders,
        boxes: list[Box],
        probs: np.ndarray,
        *,
        assignment_mode: str = "risk_adjusted",
    ) -> BoxSetEvaluation:
        box_ids = np.asarray([box.box_id for box in boxes])
        box_volumes = np.asarray([box.volume for box in boxes], dtype=np.float64)
        base_costs = np.asarray(
            [self.volume_weight * box.volume + self.surface_weight * box.surface_area for box in boxes],
            dtype=np.float64,
        )
        risks = np.maximum(0.0, self.tau_high - probs) ** 2
        adjusted = base_costs[None, :] + self.lambda_risk * risks
        feasible = probs >= self.tau
        if assignment_mode == "min_volume":
            feasible_adjusted = np.where(feasible, box_volumes[None, :], np.inf)
        elif assignment_mode == "risk_adjusted":
            feasible_adjusted = np.where(feasible, adjusted, np.inf)
        else:
            raise ValueError(f"unknown assignment_mode: {assignment_mode}")
        chosen_idx = np.argmin(feasible_adjusted, axis=1)
        has_choice = np.isfinite(feasible_adjusted[np.arange(len(prepared.orders)), chosen_idx])
        best_idx = np.argmax(probs, axis=1)

        assignments = []
        total_base = 0.0
        total_adjusted = 0.0
        low_margin = 0
        covered_probs = []
        for i, order_key in enumerate(prepared.order_keys):
            best_box_id = int(box_ids[best_idx[i]])
            best_probability = float(probs[i, best_idx[i]])
            if not has_choice[i]:
                assignments.append(
                    Assignment(
                        order_key=order_key,
                        box_id=None,
                        probability=0.0,
                        base_cost=self.uncovered_penalty,
                        adjusted_cost=self.uncovered_penalty,
                        covered=False,
                        best_box_id=best_box_id,
                        best_probability=best_probability,
                        low_margin=False,
                    )
                )
                total_base += self.uncovered_penalty
                total_adjusted += self.uncovered_penalty
                continue

            idx = int(chosen_idx[i])
            probability = float(probs[i, idx])
            if assignment_mode == "min_volume":
                base_cost = float(box_volumes[idx])
                adjusted_cost = base_cost
            else:
                base_cost = float(base_costs[idx])
                adjusted_cost = float(adjusted[i, idx])
            is_low_margin = probability < self.tau + self.low_margin_band
            if is_low_margin:
                low_margin += 1
            covered_probs.append(probability)
            assignments.append(
                Assignment(
                    order_key=order_key,
                    box_id=int(box_ids[idx]),
                    probability=probability,
                    base_cost=base_cost,
                    adjusted_cost=adjusted_cost,
                    covered=True,
                    best_box_id=best_box_id,
                    best_probability=best_probability,
                    low_margin=is_low_margin,
                )
            )
            total_base += base_cost
            total_adjusted += adjusted_cost

        covered_count = len(covered_probs)
        return BoxSetEvaluation(
            total_adjusted_cost=total_adjusted,
            total_base_cost=total_base,
            uncovered_orders=len(prepared.orders) - covered_count,
            coverage_rate=covered_count / len(prepared.orders),
            mean_assigned_probability=float(np.mean(covered_probs)) if covered_probs else 0.0,
            low_margin_assignments=low_margin,
            assignments=tuple(assignments),
        )

    def evaluate_many_box_sets(
        self,
        candidates: list[list[Box]],
        *,
        assignment_mode: str = "risk_adjusted",
        candidate_batch_size: int | None = None,
    ) -> list[BoxSetEvaluation]:
        """Evaluate candidate box sets with shared prepared order state."""
        prepared = self.prepared_orders
        if prepared is None:
            raise ValueError("prepared_orders is required")
        if assignment_mode not in {"risk_adjusted", "min_volume"}:
            raise ValueError(f"unknown assignment_mode: {assignment_mode}")
        if not candidates:
            return []
        if any(not candidate for candidate in candidates):
            raise ValueError("candidate box sets must be non-empty")
        if candidate_batch_size is not None and candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive")

        box_counts = {len(candidate) for candidate in candidates}
        if len(box_counts) != 1:
            return [
                self.evaluate_prepared(candidate, assignment_mode=assignment_mode)
                for candidate in candidates
            ]

        if candidate_batch_size is not None and len(candidates) > candidate_batch_size:
            out: list[BoxSetEvaluation] = []
            for start in range(0, len(candidates), candidate_batch_size):
                out.extend(
                    self.evaluate_many_box_sets(
                        candidates[start : start + candidate_batch_size],
                        assignment_mode=assignment_mode,
                        candidate_batch_size=None,
                    )
                )
            return out

        n_candidates = len(candidates)
        n_orders = len(prepared.orders)
        n_boxes = box_counts.pop()
        frames = [prepared.make_feature_frame(list(candidate)) for candidate in candidates]
        x_df = pd.concat(frames, ignore_index=True)
        raw_probs = _positive_probability(self.model.predict_proba(x_df))
        probs = raw_probs.reshape(n_candidates, n_orders, n_boxes)
        return [
            self._evaluation_from_probs(
                prepared,
                list(candidate),
                probs[idx],
                assignment_mode=assignment_mode,
            )
            for idx, candidate in enumerate(candidates)
        ]


def _sample_var_np(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.var(values, ddof=1))


def _sample_std_np(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))
