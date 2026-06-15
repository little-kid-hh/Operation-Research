from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .evaluator import BatchSurrogateEvaluator, Box, BoxSetEvaluation, SurrogateEvaluator
from .features import OrderSummary
from .search import BoxMove, apply_move, coordinate_moves
from .surrogate_search import repair_moves


ActionKind = Literal["coordinate", "repair", "noop", "stop"]


@dataclass(frozen=True)
class BoxDesignAction:
    kind: ActionKind
    move: BoxMove | None = None


@dataclass(frozen=True)
class BoxDesignState:
    boxes: tuple[tuple[float, float, float], ...]
    coverage_rate: float
    uncovered_orders: int
    low_margin_assignments: int
    mean_assigned_probability: float
    mean_base_cost: float
    mean_adjusted_cost: float


@dataclass(frozen=True)
class StepResult:
    state: BoxDesignState
    reward: float
    done: bool
    evaluation: BoxSetEvaluation


class SurrogateBoxDesignEnv:
    """Small reproducible RL-style environment for surrogate-guided box design."""

    def __init__(
        self,
        orders: list[OrderSummary],
        initial_boxes: list[Box],
        evaluator: SurrogateEvaluator,
        *,
        step_schedule: tuple[float, ...] | list[float] = (0.5, 0.1, 0.05, 0.01),
        max_steps: int = 100,
        max_repair_orders: int = 64,
        max_actions_per_step: int | None = None,
        seed: int = 0,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not step_schedule:
            raise ValueError("step_schedule must be non-empty")
        if max_repair_orders <= 0:
            raise ValueError("max_repair_orders must be positive")
        if max_actions_per_step is not None and max_actions_per_step <= 0:
            raise ValueError("max_actions_per_step must be positive when set")
        self.orders = list(orders)
        self.initial_boxes = sorted(initial_boxes, key=lambda b: b.box_id)
        self.evaluator = (
            evaluator
            if isinstance(evaluator, BatchSurrogateEvaluator)
            else BatchSurrogateEvaluator.from_evaluator(evaluator, orders)
        )
        self.step_schedule = tuple(float(x) for x in step_schedule)
        self.max_steps = max_steps
        self.max_repair_orders = max_repair_orders
        self.max_actions_per_step = max_actions_per_step
        self.rng = random.Random(seed)
        self.boxes = list(self.initial_boxes)
        self.evaluation = self.evaluator.evaluate_prepared(self.boxes)
        self.steps = 0

    @property
    def fixed_action_count(self) -> int:
        return 2 + len(self.initial_boxes) * 3 * 2

    def reset(self) -> BoxDesignState:
        self.boxes = list(self.initial_boxes)
        self.evaluation = self.evaluator.evaluate_prepared(self.boxes)
        self.steps = 0
        return self.state()

    def state(self) -> BoxDesignState:
        n = len(self.orders)
        return BoxDesignState(
            boxes=tuple((box.length, box.width, box.height) for box in sorted(self.boxes, key=lambda b: b.volume)),
            coverage_rate=self.evaluation.coverage_rate,
            uncovered_orders=self.evaluation.uncovered_orders,
            low_margin_assignments=self.evaluation.low_margin_assignments,
            mean_assigned_probability=self.evaluation.mean_assigned_probability,
            mean_base_cost=self.evaluation.total_base_cost / n,
            mean_adjusted_cost=self.evaluation.total_adjusted_cost / n,
        )

    def observation_vector(self) -> np.ndarray:
        """Return a fixed-size numeric observation for simple trainable policies."""
        n = len(self.orders)
        boxes = sorted(self.boxes, key=lambda b: b.box_id)
        initial = sorted(self.initial_boxes, key=lambda b: b.box_id)
        initial_volumes = np.asarray([box.volume for box in initial], dtype=np.float64)
        scale_dim = max(max(box.length, box.width, box.height) for box in initial)
        scale_volume = max(float(np.max(initial_volumes)), 1.0)
        obs: list[float] = [
            self.steps / max(self.max_steps, 1),
            self.evaluation.coverage_rate,
            self.evaluation.uncovered_orders / max(n, 1),
            self.evaluation.low_margin_assignments / max(n, 1),
            self.evaluation.mean_assigned_probability,
            (self.evaluation.total_base_cost / n) / scale_volume,
            (self.evaluation.total_adjusted_cost / n) / scale_volume,
        ]
        for box in boxes:
            obs.extend(
                [
                    box.length / scale_dim,
                    box.width / scale_dim,
                    box.height / scale_dim,
                    box.volume / scale_volume,
                ]
            )
        return np.asarray(obs, dtype=np.float64)

    def available_actions(self, step: float | None = None) -> list[BoxDesignAction]:
        active_step = self._active_step() if step is None else float(step)
        actions = [BoxDesignAction("stop"), BoxDesignAction("noop")]
        actions.extend(BoxDesignAction("repair", move) for move in repair_moves(
                self.orders,
                self.boxes,
                self.evaluation,
                step=active_step,
                prepared_orders=self.evaluator.prepared_orders,
                max_orders=self.max_repair_orders,
            ))
        actions.extend(BoxDesignAction("coordinate", move) for move in coordinate_moves(self.boxes, active_step))
        if self.max_actions_per_step is not None:
            actions = actions[:2] + actions[2 : 2 + self.max_actions_per_step]
        return actions

    def fixed_action(self, action_index: int, step: float | None = None) -> BoxDesignAction:
        """Decode a stable discrete action index into stop/noop/coordinate moves."""
        if action_index < 0 or action_index >= self.fixed_action_count:
            raise ValueError(f"action_index out of range: {action_index}")
        if action_index == 0:
            return BoxDesignAction("stop")
        if action_index == 1:
            return BoxDesignAction("noop")

        active_step = self._active_step() if step is None else float(step)
        boxes = sorted(self.boxes, key=lambda b: b.box_id)
        idx = action_index - 2
        box_idx = idx // 6
        rem = idx % 6
        dim = ("length", "width", "height")[rem // 2]
        sign = -1.0 if rem % 2 == 0 else 1.0
        return BoxDesignAction("coordinate", BoxMove(boxes[box_idx].box_id, dim, sign * abs(active_step)))

    def step(self, action: BoxDesignAction) -> StepResult:
        if action.kind == "stop":
            return StepResult(self.state(), 0.0, True, self.evaluation)
        if action.kind == "noop":
            self.steps += 1
            done = self.steps >= self.max_steps
            return StepResult(self.state(), -0.01, done, self.evaluation)
        if action.move is None:
            raise ValueError(f"{action.kind} action requires a move")

        prev = self.evaluation
        candidate_boxes = apply_move(self.boxes, action.move)
        candidate_eval = self.evaluator.evaluate_prepared(candidate_boxes)
        reward = self._reward(prev, candidate_eval)
        self.boxes = sorted(candidate_boxes, key=lambda b: b.box_id)
        self.evaluation = candidate_eval
        self.steps += 1
        done = self.steps >= self.max_steps
        return StepResult(self.state(), reward, done, self.evaluation)

    def rule_policy_action(self) -> BoxDesignAction:
        actions = self.available_actions()
        movable = [action for action in actions if action.move is not None]
        if not movable:
            return BoxDesignAction("stop")
        candidates = [apply_move(self.boxes, action.move) for action in movable if action.move is not None]
        candidate_evals = self.evaluator.evaluate_many_box_sets(candidates)
        scored = [(self._rank(candidate_eval), action) for candidate_eval, action in zip(candidate_evals, movable)]
        best_rank, best_action = min(scored, key=lambda x: x[0])
        if best_rank >= self._rank(self.evaluation):
            return BoxDesignAction("stop")
        return best_action

    def epsilon_greedy_action(self, epsilon: float = 0.1) -> BoxDesignAction:
        actions = self.available_actions()
        movable = [action for action in actions if action.move is not None]
        if movable and self.rng.random() < epsilon:
            return self.rng.choice(movable)
        return self.rule_policy_action()

    def random_policy_action(self, stop_probability: float = 0.0) -> BoxDesignAction:
        if stop_probability > 0.0 and self.rng.random() < stop_probability:
            return BoxDesignAction("stop")
        actions = self.available_actions()
        movable = [action for action in actions if action.move is not None]
        if not movable:
            return BoxDesignAction("stop")
        return self.rng.choice(movable)

    @staticmethod
    def action_to_dict(action: BoxDesignAction) -> dict[str, object]:
        row: dict[str, object] = {"kind": action.kind}
        if action.move is None:
            row.update({"box_id": "", "dimension": "", "delta": ""})
        else:
            row.update(
                {
                    "box_id": action.move.box_id,
                    "dimension": action.move.dimension,
                    "delta": action.move.delta,
                }
            )
        return row

    def _active_step(self) -> float:
        idx = min(self.steps, len(self.step_schedule) - 1)
        return self.step_schedule[idx]

    def _reward(self, prev: BoxSetEvaluation, new: BoxSetEvaluation) -> float:
        n = len(self.orders)
        reward = 1000.0 * (prev.uncovered_orders - new.uncovered_orders)
        reward += 100.0 * (prev.low_margin_assignments - new.low_margin_assignments)
        reward += 0.01 * ((prev.total_adjusted_cost - new.total_adjusted_cost) / n)
        if new.coverage_rate < prev.coverage_rate:
            reward -= 1000.0 * (prev.coverage_rate - new.coverage_rate) * n
        return float(reward)

    @staticmethod
    def _rank(evaluation: BoxSetEvaluation) -> tuple[int, int, float, float]:
        assigned_volumes = [a.base_cost for a in evaluation.assignments if a.covered]
        mean_box_volume = float(np.mean(assigned_volumes)) if assigned_volumes else float("inf")
        return (
            evaluation.uncovered_orders,
            evaluation.low_margin_assignments,
            evaluation.total_adjusted_cost / len(evaluation.assignments),
            mean_box_volume,
        )
