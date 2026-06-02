# -*- coding: utf-8 -*-
"""Prepare OR 2023 BSP XML data for geometry-deduplicated package labeling."""

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path


DEFAULT_BSP_XML_DIR = Path(".codex_training_tmp/opre_sm1/datasets_xml")
DEFAULT_PACKAGES_PATH = Path("or2023_bpp_data/packages.txt")
DEFAULT_OUTPUT_ROOT = Path("or2023_bsp_data")


def geometry_key(order_el: ET.Element) -> tuple[tuple[float, float, float], ...]:
    items: list[tuple[float, float, float]] = []
    for item_el in order_el.findall("item"):
        dims = (
            float(item_el.findtext("p")),
            float(item_el.findtext("q")),
            float(item_el.findtext("r")),
        )
        items.append(dims)
    return tuple(sorted(items))


def append_order(parent: ET.Element, geom_id: int, key: tuple[tuple[float, float, float], ...]) -> None:
    order_el = ET.SubElement(parent, "order", {"id": str(geom_id)})
    for item_id, dims in enumerate(key):
        item_el = ET.SubElement(order_el, "item", {"id": str(item_id)})
        # Preserve each item's original (p, q, r) axis naming so two-orientation labels remain valid.
        for tag, value in zip(("p", "q", "r"), dims):
            child = ET.SubElement(item_el, tag)
            child.text = f"{value:.6f}".rstrip("0").rstrip(".")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsp-xml-dir", type=Path, default=DEFAULT_BSP_XML_DIR)
    parser.add_argument("--packages-path", type=Path, default=DEFAULT_PACKAGES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    xml_files = sorted(args.bsp_xml_dir.glob("BSP_*.xml"))
    if not xml_files:
        raise ValueError(f"No BSP_*.xml files found in {args.bsp_xml_dir}")

    geom_to_id: OrderedDict[tuple[tuple[float, float, float], ...], int] = OrderedDict()
    mapping_rows: list[dict[str, str | int]] = []

    for xml_path in xml_files:
        parts = xml_path.stem.split("_")
        if len(parts) != 4:
            raise ValueError(f"Unexpected BSP XML name: {xml_path.name}")
        _, size, variant, seed = parts
        root = ET.parse(xml_path).getroot()
        orders_node = root.find("orders")
        if orders_node is None:
            raise ValueError(f"Missing <orders> in {xml_path}")
        for order_el in orders_node.findall("order"):
            key = geometry_key(order_el)
            if key not in geom_to_id:
                geom_to_id[key] = len(geom_to_id)
            mapping_rows.append(
                {
                    "instance_name": xml_path.name,
                    "order_id": order_el.get("id", ""),
                    "bsp_size": int(size),
                    "bsp_variant": variant,
                    "bsp_seed": int(seed),
                    "geom_id": geom_to_id[key],
                    "item_count": len(key),
                }
            )

    output_root = args.output_root
    xml_out_dir = output_root / "xml_unique"
    xml_out_dir.mkdir(parents=True, exist_ok=True)

    unique_root = ET.Element("or2023_bsp_unique_orders")
    ET.SubElement(unique_root, "norder").text = str(len(geom_to_id))
    orders_out = ET.SubElement(unique_root, "orders")
    for key, geom_id in geom_to_id.items():
        append_order(orders_out, geom_id, key)
    ET.ElementTree(unique_root).write(xml_out_dir / "or2023_bsp_unique_orders.xml", encoding="utf-8")

    with (output_root / "order_geometry_map.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_name",
                "order_id",
                "bsp_size",
                "bsp_variant",
                "bsp_seed",
                "geom_id",
                "item_count",
            ],
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    shutil.copyfile(args.packages_path, output_root / "packages.txt")
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# OR 2023 BSP Labeling Data",
                "",
                "This directory contains geometry-deduplicated Fontaine-Minner OR 2023 BSP orders for package-feasibility labeling.",
                "",
                "- `xml_unique/or2023_bsp_unique_orders.xml`: canonical unique order geometries.",
                "- `order_geometry_map.csv`: mapping from each BSP XML order to `geom_id`.",
                "- `packages.txt`: the 90 candidate packages used for feasibility labels.",
                "",
                "The expanded BSP label table can be reconstructed by joining `order_geometry_map.csv` to the unique geometry-package labels on `geom_id`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"BSP XML files: {len(xml_files)}")
    print(f"Expanded BSP orders: {len(mapping_rows)}")
    print(f"Unique order geometries: {len(geom_to_id)}")
    print(f"Unique geometry-package tasks @90 packages: {len(geom_to_id) * 90}")


if __name__ == "__main__":
    main()
