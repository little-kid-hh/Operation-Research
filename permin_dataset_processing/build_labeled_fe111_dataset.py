# -*- coding: utf-8 -*-
"""Build the 111-feature OR 2023 3D-BPP dataset for the feature-engineered SVM."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


OR2023_BPP_XML_DIR = Path("or2023_bpp_data/xml")
DEFAULT_XML_RE = re.compile(r".*\.xml$")
S3DBSP_PATH_MARKER = "S3DBSP-main"
PACK_ROOT = Path("research/loadability_model_pack_20260526_v2")
FE_SCRIPT = PACK_ROOT / "feature_engineering" / "build_active_bank.py"
FE_MODEL_JSON = PACK_ROOT / "models" / "svm_feature_engineered" / "linear_svm_fe_20260526.json"


def load_build_active_bank():
    spec = importlib.util.spec_from_file_location("pack_build_active_bank", FE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {FE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.build_active_bank


def reject_bsp_derived_path(path: Path, allow_bsp_derived_data: bool) -> None:
    if allow_bsp_derived_data:
        return
    if S3DBSP_PATH_MARKER.lower() in str(path).replace("\\", "/").lower():
        raise ValueError(
            f"{path} is a S3DBSP/stochastic-BSP data path. "
            "Use the Fontaine & Minner OR 2023 3D-BPP data path instead."
        )


def read_order_items(xml_dir: Path, xml_name_regex: str) -> pd.DataFrame:
    rows = []
    pattern = re.compile(xml_name_regex)
    for xml_path in sorted(xml_dir.glob("*.xml")):
        if not pattern.match(xml_path.name):
            continue
        root = ET.parse(xml_path).getroot()
        orders_node = root.find("orders")
        if orders_node is None:
            continue
        for order_el in orders_node.findall("order"):
            order_id = order_el.get("id")
            for item_el in order_el.findall("item"):
                l = float(item_el.findtext("p"))
                w = float(item_el.findtext("q"))
                h = float(item_el.findtext("r"))
                dim_s, dim_m, dim_l = sorted([l, w, h])
                rows.append(
                    {
                        "instance_name": xml_path.name,
                        "order_id": str(order_id),
                        "item_length": l,
                        "item_width": w,
                        "item_height": h,
                        "dim_s": dim_s,
                        "dim_m": dim_m,
                        "dim_l": dim_l,
                        "item_volume": l * w * h,
                        "item_footprint": dim_l * dim_m,
                        "item_flatness": dim_s / max(dim_l, 1e-9),
                        "if_fragile": 0.0,
                        "load_parameter": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("permin_dataset_processing/processed_features/or2023_bpp_labeled_base40_package.csv"),
    )
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
        "--output-path",
        type=Path,
        default=Path("permin_dataset_processing/processed_features/or2023_bpp_labeled_fe111_package.csv"),
    )
    args = parser.parse_args()
    reject_bsp_derived_path(args.xml_dir, args.allow_bsp_derived_data)

    base_df = pd.read_csv(args.base_path)
    base_df = base_df.reset_index(drop=True)
    base_df["order_id"] = base_df["order_id"].astype(str)
    base_df["dispatch_id"] = np.arange(len(base_df), dtype=np.int64)
    base_df["vehicle_capacity"] = (
        base_df["vehicle_length"] * base_df["vehicle_width"] * base_df["vehicle_height"]
    )

    order_items = read_order_items(args.xml_dir, args.xml_name_regex)
    if order_items.empty:
        raise ValueError(f"No item rows found in {args.xml_dir} matching {args.xml_name_regex!r}")
    items_df = base_df[
        [
            "dispatch_id",
            "instance_name",
            "order_id",
            "vehicle_capacity",
        ]
    ].merge(order_items, on=["instance_name", "order_id"], how="left")
    if items_df["item_volume"].isna().any():
        missing = int(items_df["item_volume"].isna().sum())
        raise ValueError(f"Missing item rows after merge: {missing}")
    items_df = items_df.drop(columns=["instance_name", "order_id"])

    agg_df = base_df.drop(
        columns=[
            "label_2ori",
            "label_6ori",
            "time_2ori_ms",
            "time_6ori_ms",
        ],
        errors="ignore",
    ).copy()

    build_active_bank = load_build_active_bank()
    active_bank = build_active_bank(agg_df, items_df)
    active_cols = [c for c in active_bank.columns if c != "dispatch_id"]

    out = base_df.merge(active_bank, on="dispatch_id", how="left")
    out[active_cols] = out[active_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out.drop(columns=["dispatch_id", "vehicle_capacity"])

    # Keep the exact packed model feature order where possible, followed by labels/ids.
    import json

    feature_names = json.loads(FE_MODEL_JSON.read_text(encoding="utf-8"))["feature_names"]
    missing = [c for c in feature_names if c not in out.columns]
    if missing:
        raise ValueError(f"Missing expected FE-SVM features: {missing}")
    tail_cols = [c for c in out.columns if c not in feature_names]
    out = out[feature_names + tail_cols]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)
    print(f"Saved {len(out)} rows x {len(out.columns)} columns to {args.output_path}")
    print(f"Feature columns: {len(feature_names)} = 40 base + {len(active_cols)} active-bank")


if __name__ == "__main__":
    main()
