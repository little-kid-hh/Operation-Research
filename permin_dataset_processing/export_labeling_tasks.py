# -*- coding: utf-8 -*-
"""Export package-aware OR 2023 3D-BPP labeling tasks.

The formal loadability label target is one row per
(instance_name, order_id, package_id).  The MILP solver should decide whether
all items of that order can be packed into that single candidate package.
"""

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


OR2023_BPP_XML_DIR = Path("or2023_bpp_data/xml")
OR2023_BPP_PACKAGES = Path("or2023_bpp_data/packages.txt")
DEFAULT_XML_RE = re.compile(r".*\.xml$")
S3DBSP_PATH_MARKER = "S3DBSP-main"


def reject_bsp_derived_path(path: Path, allow_bsp_derived_data: bool) -> None:
    if allow_bsp_derived_data:
        return
    if S3DBSP_PATH_MARKER.lower() in str(path).replace("\\", "/").lower():
        raise ValueError(
            f"{path} is a S3DBSP/stochastic-BSP data path. "
            "This project is now scoped to the Fontaine & Minner OR 2023 3D-BPP data. "
            "Pass the OR 2023 BPP e-companion path instead, or use "
            "--allow-bsp-derived-data only for legacy audits."
        )


def read_packages(path):
    packages = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"Invalid package line in {path}: {line!r}")
            package_id, length, width, height = parts
            packages.append(
                {
                    "package_id": int(package_id),
                    "package_l": float(length),
                    "package_w": float(width),
                    "package_h": float(height),
                }
            )
    return packages


def read_orders(xml_path):
    root = ET.parse(xml_path).getroot()
    orders_node = root.find("orders")
    if orders_node is None:
        return []

    orders = []
    for order_el in orders_node.findall("order"):
        items = []
        for item_el in order_el.findall("item"):
            items.append(
                {
                    "p": float(item_el.findtext("p")),
                    "q": float(item_el.findtext("q")),
                    "r": float(item_el.findtext("r")),
                }
            )
        orders.append({"order_id": order_el.get("id"), "items": items})
    return orders


def select_xml_files(xml_dir, xml_name_regex):
    files = sorted(xml_dir.glob("*.xml"))
    pattern = re.compile(xml_name_regex)
    return [path for path in files if pattern.match(path.name)]


def export_tasks(xml_dir, packages_path, output_path, xml_name_regex):
    packages = read_packages(packages_path)
    xml_files = select_xml_files(xml_dir, xml_name_regex)
    if not xml_files:
        raise ValueError(f"No XML files matched {xml_name_regex!r} in {xml_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "instance_name",
                "order_id",
                "package_id",
                "package_l",
                "package_w",
                "package_h",
                "item_count",
                "item_volume_sum",
                "items_json",
            ],
        )
        writer.writeheader()
        for xml_path in xml_files:
            for order in read_orders(xml_path):
                volume_sum = sum(item["p"] * item["q"] * item["r"] for item in order["items"])
                items_json = json.dumps(order["items"], separators=(",", ":"))
                for package in packages:
                    writer.writerow(
                        {
                            "instance_name": xml_path.name,
                            "order_id": order["order_id"],
                            **package,
                            "item_count": len(order["items"]),
                            "item_volume_sum": volume_sum,
                            "items_json": items_json,
                        }
                    )
                    task_count += 1

    return len(xml_files), len(packages), task_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=OR2023_BPP_XML_DIR,
        help="Directory containing Fontaine & Minner OR 2023 3D-BPP XML files.",
    )
    parser.add_argument(
        "--packages-path",
        type=Path,
        default=OR2023_BPP_PACKAGES,
        help="Candidate package list: package_id length width height.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("permin_dataset_processing/milp_labels/labeling_tasks.csv"),
        help="Output CSV consumed by the formal MILP label generator.",
    )
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
    args = parser.parse_args()
    reject_bsp_derived_path(args.xml_dir, args.allow_bsp_derived_data)
    reject_bsp_derived_path(args.packages_path, args.allow_bsp_derived_data)

    xml_count, package_count, task_count = export_tasks(
        args.xml_dir,
        args.packages_path,
        args.output_path,
        args.xml_name_regex,
    )
    print(
        f"Exported {task_count} tasks from {xml_count} XML files and "
        f"{package_count} packages to {args.output_path}"
    )


if __name__ == "__main__":
    main()
