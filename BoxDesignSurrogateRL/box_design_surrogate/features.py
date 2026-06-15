from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

FEATURE_COLS = [
    "sku_counts",
    "sku_average_volume",
    "sku_length_var",
    "sku_width_var",
    "sku_height_var",
    "sku_length_avg",
    "sku_width_avg",
    "sku_height_avg",
    "max_asr",
    "vehicle_length",
    "vehicle_width",
    "vehicle_height",
    "spare_capacity",
    "sku_concentration",
    "sku_min_length",
    "sku_max_length",
    "sku_std_length",
    "sku_min_width",
    "sku_max_width",
    "sku_std_width",
    "sku_min_height",
    "sku_max_height",
    "sku_std_height",
    "l_to_L_ratio_avg",
    "l_to_L_ratio_min",
    "l_to_L_ratio_max",
    "l_to_L_ratio_std",
    "h_to_H_ratio_avg",
    "h_to_H_ratio_min",
    "h_to_H_ratio_max",
    "h_to_H_ratio_std",
    "w_to_W_ratio_avg",
    "w_to_W_ratio_min",
    "w_to_W_ratio_max",
    "w_to_W_ratio_std",
    "wl_to_vehicle_wl_avg",
    "wl_to_vehicle_wl_min",
    "wl_to_vehicle_wl_max",
    "wl_to_vehicle_wl_std",
    "wl_to_vehicle_wl_total",
]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_var(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    return sum((x - avg) ** 2 for x in values) / (len(values) - 1)


def _sample_std(values: list[float]) -> float:
    return math.sqrt(_sample_var(values))


@dataclass(frozen=True)
class OrderSummary:
    instance_name: str
    order_id: str
    item_count: int
    dim_l: tuple[float, ...]
    dim_m: tuple[float, ...]
    dim_s: tuple[float, ...]
    volumes: tuple[float, ...]
    footprints: tuple[float, ...]

    @property
    def total_volume(self) -> float:
        return sum(self.volumes)


def summarize_items(instance_name: str, order_id: str, items: list[tuple[float, float, float]]) -> OrderSummary:
    dims = [sorted(item) for item in items]
    dim_s = tuple(d[0] for d in dims)
    dim_m = tuple(d[1] for d in dims)
    dim_l = tuple(d[2] for d in dims)
    volumes = tuple(l * m * s for l, m, s in zip(dim_l, dim_m, dim_s))
    footprints = tuple(l * m for l, m in zip(dim_l, dim_m))
    return OrderSummary(
        instance_name=instance_name,
        order_id=order_id,
        item_count=len(items),
        dim_l=dim_l,
        dim_m=dim_m,
        dim_s=dim_s,
        volumes=volumes,
        footprints=footprints,
    )


def read_order_summaries(xml_path: Path) -> list[OrderSummary]:
    root = ET.parse(xml_path).getroot()
    orders_node = root.find("orders")
    if orders_node is None:
        return []

    orders: list[OrderSummary] = []
    for order_el in orders_node.findall("order"):
        items = []
        for item_el in order_el.findall("item"):
            items.append(
                (
                    float(item_el.findtext("p")),
                    float(item_el.findtext("q")),
                    float(item_el.findtext("r")),
                )
            )
        orders.append(summarize_items(xml_path.name, str(order_el.get("id")), items))
    return orders


def make_base40_features(order: OrderSummary, box_l: float, box_w: float, box_h: float) -> dict[str, float]:
    dim_l = list(order.dim_l)
    dim_m = list(order.dim_m)
    dim_s = list(order.dim_s)
    vols = list(order.volumes)
    footprints = list(order.footprints)
    total_vol = sum(vols)
    box_volume = box_l * box_w * box_h
    box_floor = box_l * box_w
    wl_ratios = [fp / box_floor for fp in footprints]

    return {
        "sku_counts": float(order.item_count),
        "sku_average_volume": _mean(vols),
        "sku_length_var": _sample_var(dim_l),
        "sku_width_var": _sample_var(dim_m),
        "sku_height_var": _sample_var(dim_s),
        "sku_length_avg": _mean(dim_l),
        "sku_width_avg": _mean(dim_m),
        "sku_height_avg": _mean(dim_s),
        "max_asr": max(l / (s if s != 0 else 1e-9) for l, s in zip(dim_l, dim_s)),
        "vehicle_length": box_l,
        "vehicle_width": box_w,
        "vehicle_height": box_h,
        "spare_capacity": box_volume - total_vol,
        "sku_concentration": max(vols) / total_vol if total_vol > 0 else 0.0,
        "sku_min_length": min(dim_l),
        "sku_max_length": max(dim_l),
        "sku_std_length": _sample_std(dim_l),
        "sku_min_width": min(dim_m),
        "sku_max_width": max(dim_m),
        "sku_std_width": _sample_std(dim_m),
        "sku_min_height": min(dim_s),
        "sku_max_height": max(dim_s),
        "sku_std_height": _sample_std(dim_s),
        "l_to_L_ratio_avg": _mean([x / box_l for x in dim_l]),
        "l_to_L_ratio_min": min(x / box_l for x in dim_l),
        "l_to_L_ratio_max": max(x / box_l for x in dim_l),
        "l_to_L_ratio_std": _sample_std([x / box_l for x in dim_l]),
        "h_to_H_ratio_avg": _mean([x / box_h for x in dim_s]),
        "h_to_H_ratio_min": min(x / box_h for x in dim_s),
        "h_to_H_ratio_max": max(x / box_h for x in dim_s),
        "h_to_H_ratio_std": _sample_std([x / box_h for x in dim_s]),
        "w_to_W_ratio_avg": _mean([x / box_w for x in dim_m]),
        "w_to_W_ratio_min": min(x / box_w for x in dim_m),
        "w_to_W_ratio_max": max(x / box_w for x in dim_m),
        "w_to_W_ratio_std": _sample_std([x / box_w for x in dim_m]),
        "wl_to_vehicle_wl_avg": _mean(wl_ratios),
        "wl_to_vehicle_wl_min": min(wl_ratios),
        "wl_to_vehicle_wl_max": max(wl_ratios),
        "wl_to_vehicle_wl_std": _sample_std(wl_ratios),
        "wl_to_vehicle_wl_total": sum(footprints) / box_floor,
    }

