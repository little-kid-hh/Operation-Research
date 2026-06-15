"""Surrogate-assisted box-size design utilities."""

from .evaluator import BatchSurrogateEvaluator, Box, BoxSetEvaluation, PreparedOrders, SurrogateEvaluator
from .features import FEATURE_COLS, OrderSummary, read_order_summaries

__all__ = [
    "BatchSurrogateEvaluator",
    "Box",
    "BoxSetEvaluation",
    "FEATURE_COLS",
    "OrderSummary",
    "PreparedOrders",
    "SurrogateEvaluator",
    "read_order_summaries",
]
