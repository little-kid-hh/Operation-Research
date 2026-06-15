from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .evaluator import Box


@dataclass(frozen=True)
class BoxMove:
    box_id: int
    dimension: str
    delta: float


def apply_move(boxes: Iterable[Box], move: BoxMove, min_dimension: float = 0.01) -> list[Box]:
    out: list[Box] = []
    for box in boxes:
        if box.box_id != move.box_id:
            out.append(box)
            continue

        length, width, height = box.length, box.width, box.height
        if move.dimension == "length":
            length = max(min_dimension, length + move.delta)
        elif move.dimension == "width":
            width = max(min_dimension, width + move.delta)
        elif move.dimension == "height":
            height = max(min_dimension, height + move.delta)
        else:
            raise ValueError(f"unknown dimension: {move.dimension}")

        out.append(Box(box.box_id, length, width, height))
    return out


def coordinate_moves(boxes: Iterable[Box], step: float) -> list[BoxMove]:
    moves: list[BoxMove] = []
    for box in boxes:
        for dim in ("length", "width", "height"):
            moves.append(BoxMove(box.box_id, dim, -abs(step)))
            moves.append(BoxMove(box.box_id, dim, abs(step)))
    return moves

