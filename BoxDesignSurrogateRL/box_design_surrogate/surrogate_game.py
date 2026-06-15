from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .evaluator import BatchSurrogateEvaluator, Box, BoxSetEvaluation, SurrogateEvaluator
from .features import OrderSummary


@dataclass(frozen=True)
class SurrogateGameStepResult:
    boxes: tuple[Box, ...]
    evaluation: BoxSetEvaluation
    objective: float
    reward: float
    done: bool
    action: int
    terminal_reason: str


@dataclass(frozen=True)
class SurrogateObjectiveBreakdown:
    mean_adjusted_cost: float
    uncovered_penalty: float
    low_margin_penalty: float
    probability_penalty: float
    packaging_factor: float
    mean_assigned_box_volume: float
    mean_order_volume: float

    @property
    def total(self) -> float:
        return float(
            self.mean_adjusted_cost
            + self.uncovered_penalty
            + self.low_margin_penalty
            + self.probability_penalty
        )


class SurrogateBoxSizingGame:
    """Kandula-style box-sizing game scored by ML loadability surrogate.

    This keeps the paper's state/action scaffold:
    - state: K x 3 box dimensions
    - actions: 6K dimension transforms plus one resignation action

    It replaces the paper feasibility oracle with an ML surrogate. In
    ``paper_pf_surrogate`` mode, feasible boxes are chosen by minimum volume
    and scored with paper-style packaging factor. In ``risk_aware_surrogate``
    mode, the older confidence-aware risk objective is retained.
    """

    def __init__(
        self,
        orders: list[OrderSummary],
        initial_boxes: list[Box],
        evaluator: SurrogateEvaluator,
        *,
        step_size: float = 0.5,
        max_steps: int = 100,
        min_dimension: float = 0.01,
        normalize_observation: bool = True,
        uncovered_weight: float = 10_000_000.0,
        low_margin_weight: float = 10_000.0,
        probability_weight: float = 1_000.0,
        objective_mode: Literal["paper_pf_surrogate", "risk_aware_surrogate"] = "paper_pf_surrogate",
        terminate_on_worse_than_initial: bool = False,
        candidate_batch_size: int | None = None,
    ) -> None:
        if not orders:
            raise ValueError("orders must be non-empty")
        if not initial_boxes:
            raise ValueError("initial_boxes must be non-empty")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if objective_mode not in {"paper_pf_surrogate", "risk_aware_surrogate"}:
            raise ValueError(f"unknown objective_mode: {objective_mode}")
        self.orders = list(orders)
        self.initial_boxes = self._normalize_boxes(initial_boxes)
        self.evaluator = (
            evaluator
            if isinstance(evaluator, BatchSurrogateEvaluator)
            else BatchSurrogateEvaluator.from_evaluator(evaluator, orders)
        )
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)
        self.min_dimension = float(min_dimension)
        self.normalize_observation = normalize_observation
        self.uncovered_weight = float(uncovered_weight)
        self.low_margin_weight = float(low_margin_weight)
        self.probability_weight = float(probability_weight)
        self.objective_mode = objective_mode
        self.terminate_on_worse_than_initial = terminate_on_worse_than_initial
        if candidate_batch_size is not None and candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive")
        self.candidate_batch_size = candidate_batch_size
        self.k = len(self.initial_boxes)
        self.resign_action = 6 * self.k
        self.action_count = 6 * self.k + 1
        self.observation_dim = 3 * self.k
        self.scale_dim = max(max(box.length, box.width, box.height) for box in self.initial_boxes)
        self.initial_evaluation = self.evaluate_box_set(self.initial_boxes)
        self.initial_objective = self.objective(self.initial_evaluation)
        self.boxes = list(self.initial_boxes)
        self.evaluation = self.initial_evaluation
        self.current_objective = self.initial_objective
        self.steps = 0

    def reset(self) -> np.ndarray:
        self.boxes = list(self.initial_boxes)
        self.evaluation = self.initial_evaluation
        self.current_objective = self.initial_objective
        self.steps = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        return self.observation_for_boxes(self.boxes)

    def step(self, action: int) -> SurrogateGameStepResult:
        if action < 0 or action >= self.action_count:
            raise ValueError(f"action out of range: {action}")
        if action == self.resign_action:
            return SurrogateGameStepResult(
                boxes=tuple(self.boxes),
                evaluation=self.evaluation,
                objective=self.current_objective,
                reward=0.0,
                done=True,
                action=action,
                terminal_reason="resign",
            )

        prev_objective = self.current_objective
        candidate_boxes = self.apply_action(self.boxes, action)
        candidate_eval = self.evaluate_box_set(candidate_boxes)
        candidate_objective = self.objective(candidate_eval)
        self.boxes = candidate_boxes
        self.evaluation = candidate_eval
        self.current_objective = candidate_objective
        self.steps += 1

        reward = float(prev_objective - candidate_objective)
        if (
            self.terminate_on_worse_than_initial
            and candidate_objective > self.initial_objective + 1e-12
        ):
            return SurrogateGameStepResult(
                boxes=tuple(candidate_boxes),
                evaluation=candidate_eval,
                objective=candidate_objective,
                reward=-1.0,
                done=True,
                action=action,
                terminal_reason="worse_than_initial",
            )
        done = self.steps >= self.max_steps
        return SurrogateGameStepResult(
            boxes=tuple(candidate_boxes),
            evaluation=candidate_eval,
            objective=candidate_objective,
            reward=reward,
            done=done,
            action=action,
            terminal_reason="max_steps" if done else "",
        )

    def objective(self, evaluation: BoxSetEvaluation) -> float:
        return self.objective_breakdown(evaluation).total

    def objective_rank(self, evaluation: BoxSetEvaluation) -> tuple[int, float]:
        breakdown = self.objective_breakdown(evaluation)
        if self.objective_mode == "paper_pf_surrogate":
            return evaluation.uncovered_orders, breakdown.packaging_factor
        return 0, breakdown.total

    def evaluate_box_set(self, boxes: list[Box]) -> BoxSetEvaluation:
        assignment_mode = (
            "min_volume"
            if self.objective_mode == "paper_pf_surrogate"
            else "risk_adjusted"
        )
        return self.evaluator.evaluate_prepared(boxes, assignment_mode=assignment_mode)

    def objective_breakdown(self, evaluation: BoxSetEvaluation) -> SurrogateObjectiveBreakdown:
        n = max(len(evaluation.assignments), 1)
        mean_order_volume = float(np.mean([order.total_volume for order in self.orders]))
        assigned_volumes = [assignment.base_cost for assignment in evaluation.assignments if assignment.covered]
        mean_assigned_box_volume = float(np.mean(assigned_volumes)) if assigned_volumes else float("inf")
        packaging_factor = (
            mean_assigned_box_volume / mean_order_volume
            if mean_order_volume > 0 and np.isfinite(mean_assigned_box_volume)
            else float("inf")
        )
        if self.objective_mode == "paper_pf_surrogate":
            return SurrogateObjectiveBreakdown(
                mean_adjusted_cost=packaging_factor,
                uncovered_penalty=float(self.uncovered_weight * evaluation.uncovered_orders / n),
                low_margin_penalty=0.0,
                probability_penalty=0.0,
                packaging_factor=packaging_factor,
                mean_assigned_box_volume=mean_assigned_box_volume,
                mean_order_volume=mean_order_volume,
            )
        return SurrogateObjectiveBreakdown(
            mean_adjusted_cost=float(evaluation.total_adjusted_cost / n),
            uncovered_penalty=float(self.uncovered_weight * evaluation.uncovered_orders / n),
            low_margin_penalty=float(self.low_margin_weight * evaluation.low_margin_assignments / n),
            probability_penalty=float(
                self.probability_weight * (1.0 - evaluation.mean_assigned_probability)
            ),
            packaging_factor=packaging_factor,
            mean_assigned_box_volume=mean_assigned_box_volume,
            mean_order_volume=mean_order_volume,
        )

    def child_result(self, boxes: list[Box], action: int) -> tuple[list[Box], BoxSetEvaluation, float]:
        child = self.apply_action(boxes, action)
        evaluation = self.evaluate_box_set(child)
        return child, evaluation, self.objective(evaluation)

    def child_results(
        self,
        boxes: list[Box],
        actions: Iterable[int],
    ) -> list[tuple[int, list[Box], BoxSetEvaluation, float]]:
        action_list = list(actions)
        children = [self.apply_action(boxes, action) for action in action_list]
        assignment_mode = (
            "min_volume"
            if self.objective_mode == "paper_pf_surrogate"
            else "risk_adjusted"
        )
        evaluations = self.evaluator.evaluate_many_box_sets(
            children,
            assignment_mode=assignment_mode,
            candidate_batch_size=self.candidate_batch_size,
        )
        return [
            (action, child, evaluation, self.objective(evaluation))
            for action, child, evaluation in zip(action_list, children, evaluations)
        ]

    def apply_action(self, boxes: Iterable[Box], action: int) -> list[Box]:
        if action < 0 or action >= self.resign_action:
            raise ValueError(f"transform action out of range: {action}")
        direction = -1.0 if action < 3 * self.k else 1.0
        flat_idx = action if action < 3 * self.k else action - 3 * self.k
        box_idx = flat_idx // 3
        dim_idx = flat_idx % 3
        out = []
        sorted_boxes = sorted(boxes, key=lambda b: b.box_id)
        for idx, box in enumerate(sorted_boxes):
            dims = [box.length, box.width, box.height]
            if idx == box_idx:
                dims[dim_idx] = max(self.min_dimension, dims[dim_idx] + direction * self.step_size)
                dims = sorted(dims, reverse=True)
            out.append(Box(box.box_id, float(dims[0]), float(dims[1]), float(dims[2])))
        return out

    def observation_for_boxes(self, boxes: list[Box]) -> np.ndarray:
        values = []
        for box in sorted(boxes, key=lambda b: b.box_id):
            values.extend([box.length, box.width, box.height])
        obs = np.asarray(values, dtype=np.float32)
        if self.normalize_observation:
            obs = obs / max(self.scale_dim, 1e-9)
        return obs

    @staticmethod
    def _normalize_boxes(boxes: list[Box]) -> list[Box]:
        out = []
        for box in sorted(boxes, key=lambda b: b.box_id):
            dims = sorted([box.length, box.width, box.height], reverse=True)
            out.append(Box(box.box_id, float(dims[0]), float(dims[1]), float(dims[2])))
        return out


def weighted_objective(values: list[float], beta: float) -> float:
    total = 0.0
    weight = 1.0
    for value in values:
        total += weight * value
        weight *= beta
    return float(total)
