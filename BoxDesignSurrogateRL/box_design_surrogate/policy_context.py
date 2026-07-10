from __future__ import annotations

import numpy as np

from .features import OrderSummary


ORDER_CONTEXT_SCHEMA = (
    "mean_max_l",
    "mean_max_m",
    "mean_max_s",
    "p90_max_l",
    "p90_max_m",
    "p90_max_s",
    "mean_total_volume",
    "p90_total_volume",
)


def order_distribution_context(
    orders: list[OrderSummary],
    *,
    scale_dim: float,
    normalize: bool = True,
) -> np.ndarray:
    if not orders:
        raise ValueError("orders must be non-empty")
    max_dims = np.asarray(
        [[max(order.dim_l), max(order.dim_m), max(order.dim_s)] for order in orders],
        dtype=np.float64,
    )
    total_volumes = np.asarray([order.total_volume for order in orders], dtype=np.float64)
    values = np.concatenate(
        [
            np.mean(max_dims, axis=0),
            np.quantile(max_dims, 0.9, axis=0),
            [float(np.mean(total_volumes)), float(np.quantile(total_volumes, 0.9))],
        ]
    )
    if normalize:
        scale = max(float(scale_dim), 1e-9)
        values[:6] /= scale
        values[6:] /= scale**3
    return values.astype(np.float32)

