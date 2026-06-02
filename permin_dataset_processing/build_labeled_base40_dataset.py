# -*- coding: utf-8 -*-
"""Build package-aware base40 features with OR 2023 3D-BPP MILP labels."""

from __future__ import annotations

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


OR2023_BPP_XML_DIR = Path("or2023_bpp_data/xml")
DEFAULT_XML_RE = re.compile(r".*\.xml$")
S3DBSP_PATH_MARKER = "S3DBSP-main"

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


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_var(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return sum((x - avg) ** 2 for x in values) / (len(values) - 1)


def sample_std(values: list[float]) -> float:
    return math.sqrt(sample_var(values))


def summarize_items(items: list[tuple[float, float, float]]) -> dict[str, float]:
    # Match prepare_permin_data.py: sorted dims are dim_s <= dim_m <= dim_l.
    dims = [sorted(item) for item in items]
    dim_s = [d[0] for d in dims]
    dim_m = [d[1] for d in dims]
    dim_l = [d[2] for d in dims]
    vols = [l * m * s for l, m, s in zip(dim_l, dim_m, dim_s)]
    footprints = [l * m for l, m in zip(dim_l, dim_m)]
    total_vol = sum(vols)
    return {
        "sku_counts": float(len(items)),
        "sku_average_volume": mean(vols),
        "sku_length_var": sample_var(dim_l),
        "sku_width_var": sample_var(dim_m),
        "sku_height_var": sample_var(dim_s),
        "sku_length_avg": mean(dim_l),
        "sku_width_avg": mean(dim_m),
        "sku_height_avg": mean(dim_s),
        "max_asr": max(l / (s if s != 0 else 1e-9) for l, s in zip(dim_l, dim_s)),
        "total_vol": total_vol,
        "sku_concentration": max(vols) / total_vol if total_vol > 0 else 0.0,
        "sku_min_length": min(dim_l),
        "sku_max_length": max(dim_l),
        "sku_std_length": sample_std(dim_l),
        "sku_min_width": min(dim_m),
        "sku_max_width": max(dim_m),
        "sku_std_width": sample_std(dim_m),
        "sku_min_height": min(dim_s),
        "sku_max_height": max(dim_s),
        "sku_std_height": sample_std(dim_s),
        "dim_l": dim_l,
        "dim_m": dim_m,
        "dim_s": dim_s,
        "footprints": footprints,
    }


def reject_bsp_derived_path(path: Path, allow_bsp_derived_data: bool) -> None:
    if allow_bsp_derived_data:
        return
    if S3DBSP_PATH_MARKER.lower() in str(path).replace("\\", "/").lower():
        raise ValueError(
            f"{path} is a S3DBSP/stochastic-BSP data path. "
            "Use the Fontaine & Minner OR 2023 3D-BPP data path instead."
        )


def read_orders(xml_dir: Path, xml_name_regex: str) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    pattern = re.compile(xml_name_regex)
    for xml_path in sorted(xml_dir.glob("*.xml")):
        if not pattern.match(xml_path.name):
            continue
        root = ET.parse(xml_path).getroot()
        orders_node = root.find("orders")
        if orders_node is None:
            continue
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
            out[(xml_path.name, order_el.get("id"))] = summarize_items(items)
    return out


def make_features(summary: dict[str, float], v_l: float, v_w: float, v_h: float) -> dict[str, float]:
    v_cap = v_l * v_w * v_h
    dim_l = summary["dim_l"]
    dim_m = summary["dim_m"]
    dim_s = summary["dim_s"]
    footprints = summary["footprints"]
    wl_ratios = [fp / (v_l * v_w) for fp in footprints]
    res = {k: summary[k] for k in FEATURE_COLS if k in summary}
    res["vehicle_length"] = v_l
    res["vehicle_width"] = v_w
    res["vehicle_height"] = v_h
    res["spare_capacity"] = v_cap - summary["total_vol"]
    res["l_to_L_ratio_avg"] = mean([x / v_l for x in dim_l])
    res["l_to_L_ratio_min"] = min(x / v_l for x in dim_l)
    res["l_to_L_ratio_max"] = max(x / v_l for x in dim_l)
    res["l_to_L_ratio_std"] = sample_std([x / v_l for x in dim_l])
    res["h_to_H_ratio_avg"] = mean([x / v_h for x in dim_s])
    res["h_to_H_ratio_min"] = min(x / v_h for x in dim_s)
    res["h_to_H_ratio_max"] = max(x / v_h for x in dim_s)
    res["h_to_H_ratio_std"] = sample_std([x / v_h for x in dim_s])
    res["w_to_W_ratio_avg"] = mean([x / v_w for x in dim_m])
    res["w_to_W_ratio_min"] = min(x / v_w for x in dim_m)
    res["w_to_W_ratio_max"] = max(x / v_w for x in dim_m)
    res["w_to_W_ratio_std"] = sample_std([x / v_w for x in dim_m])
    res["wl_to_vehicle_wl_avg"] = mean(wl_ratios)
    res["wl_to_vehicle_wl_min"] = min(wl_ratios)
    res["wl_to_vehicle_wl_max"] = max(wl_ratios)
    res["wl_to_vehicle_wl_std"] = sample_std(wl_ratios)
    res["wl_to_vehicle_wl_total"] = sum(footprints) / (v_l * v_w)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", type=Path, default=OR2023_BPP_XML_DIR)
    parser.add_argument(
        "--xml-name-regex",
        default=DEFAULT_XML_RE.pattern,
        help="Regex for OR 2023 BPP XML file names. Defaults to all XML files.",
    )
    parser.add_argument(
        "--allow-bsp-derived-data",
        action="store_true",
        help="Legacy escape hatch: allow S3DBSP/stochastic-BSP paths for audits only.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("permin_dataset_processing/milp_labels/or2023_bpp_package_labels.csv"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("permin_dataset_processing/processed_features/or2023_bpp_labeled_base40_package.csv"),
    )
    args = parser.parse_args()

    reject_bsp_derived_path(args.xml_dir, args.allow_bsp_derived_data)
    order_summaries = read_orders(args.xml_dir, args.xml_name_regex)
    if not order_summaries:
        raise ValueError(f"No order rows found in {args.xml_dir} matching {args.xml_name_regex!r}")
    fieldnames = [
        *FEATURE_COLS,
        "instance_name",
        "order_id",
        "package_id",
        "package_l",
        "package_w",
        "package_h",
        "label_2ori",
        "label_6ori",
        "time_2ori_ms",
        "time_6ori_ms",
    ]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    label_counts = {"label_2ori": {}, "label_6ori": {}}
    rows = 0
    with args.labels_path.open("r", newline="", encoding="utf-8") as src, args.output_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            key = (row["instance_name"], row["order_id"])
            summary = order_summaries[key]
            feats = make_features(
                summary,
                float(row["package_l"]),
                float(row["package_w"]),
                float(row["package_h"]),
            )
            out = {
                **feats,
                "instance_name": row["instance_name"],
                "order_id": row["order_id"],
                "package_id": row["package_id"],
                "package_l": row["package_l"],
                "package_w": row["package_w"],
                "package_h": row["package_h"],
                "label_2ori": row["label_2ori"],
                "label_6ori": row["label_6ori"],
                "time_2ori_ms": row["time_2ori_ms"],
                "time_6ori_ms": row["time_6ori_ms"],
            }
            writer.writerow(out)
            rows += 1
            for label_col in ["label_2ori", "label_6ori"]:
                label_counts[label_col][row[label_col]] = label_counts[label_col].get(row[label_col], 0) + 1
    print(f"Saved {rows} rows x {len(fieldnames)} columns to {args.output_path}")
    print(label_counts)


if __name__ == "__main__":
    main()
