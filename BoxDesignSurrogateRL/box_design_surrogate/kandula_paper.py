from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .evaluator import Box
from .features import OrderSummary
from .kandula_repro import SimpleBoxSetScore, evaluate_simple_box_set


@dataclass(frozen=True)
class PaperStepResult:
    boxes: tuple[Box, ...]
    score: SimpleBoxSetScore
    reward: float
    done: bool
    action: int
    terminal_reason: str


class KandulaBoxSizingGame:
    """Paper-aligned box-sizing game from Kandula et al. Section 4.2.

    State: K x 3 box-dimension matrix.
    Actions: 6K transforming actions plus one resignation action.
    Reward: 0 for resignation, -1 for infeasible/worse-than-initial terminal
    states, otherwise PF(s_t) - PF(s_{t+1}).
    """

    def __init__(
        self,
        orders: list[OrderSummary],
        initial_boxes: list[Box],
        *,
        step_size: float = 0.5,
        max_steps: int = 100,
        min_dimension: float = 0.01,
        normalize_observation: bool = True,
    ) -> None:
        if not orders:
            raise ValueError("orders must be non-empty")
        if not initial_boxes:
            raise ValueError("initial_boxes must be non-empty")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.orders = list(orders)
        self.initial_boxes = self._normalize_boxes(initial_boxes)
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)
        self.min_dimension = float(min_dimension)
        self.normalize_observation = normalize_observation
        self.k = len(self.initial_boxes)
        self.resign_action = 6 * self.k
        self.action_count = 6 * self.k + 1
        self.observation_dim = 3 * self.k
        self.scale_dim = max(max(box.length, box.width, box.height) for box in self.initial_boxes)
        self.initial_score = evaluate_simple_box_set(self.orders, self.initial_boxes)
        if self.initial_score.uncovered_orders != 0:
            raise ValueError("Kandula stage-1 initial solution must cover all orders")
        self.boxes = list(self.initial_boxes)
        self.score = self.initial_score
        self.steps = 0

    def reset(self) -> np.ndarray:
        self.boxes = list(self.initial_boxes)
        self.score = self.initial_score
        self.steps = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        values = []
        for box in sorted(self.boxes, key=lambda b: b.box_id):
            values.extend([box.length, box.width, box.height])
        obs = np.asarray(values, dtype=np.float32)
        if self.normalize_observation:
            obs = obs / max(self.scale_dim, 1e-9)
        return obs

    def state_matrix(self) -> np.ndarray:
        return self.observation().reshape(self.k, 3)

    def step(self, action: int) -> PaperStepResult:
        if action < 0 or action >= self.action_count:
            raise ValueError(f"action out of range: {action}")
        if action == self.resign_action:
            return PaperStepResult(
                boxes=tuple(self.boxes),
                score=self.score,
                reward=0.0,
                done=True,
                action=action,
                terminal_reason="resign",
            )

        prev_score = self.score
        candidate_boxes = self.apply_action(self.boxes, action)
        candidate_score = evaluate_simple_box_set(self.orders, candidate_boxes)
        self.boxes = candidate_boxes
        self.score = candidate_score
        self.steps += 1

        if candidate_score.uncovered_orders != 0:
            return PaperStepResult(
                boxes=tuple(candidate_boxes),
                score=candidate_score,
                reward=-1.0,
                done=True,
                action=action,
                terminal_reason="infeasible",
            )
        if candidate_score.packaging_factor > self.initial_score.packaging_factor + 1e-12:
            return PaperStepResult(
                boxes=tuple(candidate_boxes),
                score=candidate_score,
                reward=-1.0,
                done=True,
                action=action,
                terminal_reason="worse_than_initial",
            )

        reward = self.score_delta(prev_score, candidate_score)
        done = self.steps >= self.max_steps
        return PaperStepResult(
            boxes=tuple(candidate_boxes),
            score=candidate_score,
            reward=reward,
            done=done,
            action=action,
            terminal_reason="max_steps" if done else "",
        )

    def score_delta(self, prev: SimpleBoxSetScore, new: SimpleBoxSetScore) -> float:
        return float(prev.packaging_factor - new.packaging_factor)

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

    def child_result(self, boxes: list[Box], action: int) -> tuple[list[Box], SimpleBoxSetScore, bool]:
        child = self.apply_action(boxes, action)
        score = evaluate_simple_box_set(self.orders, child)
        feasible = score.uncovered_orders == 0 and score.packaging_factor <= self.initial_score.packaging_factor + 1e-12
        return child, score, feasible

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


def weighted_packaging_factor(values: list[float], beta: float) -> float:
    total = 0.0
    weight = 1.0
    for value in values:
        total += weight * value
        weight *= beta
    return float(total)
