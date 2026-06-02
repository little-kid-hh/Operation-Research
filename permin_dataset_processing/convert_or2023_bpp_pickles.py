# -*- coding: utf-8 -*-
"""Convert OR 2023 BPP e-companion pickle instances to XML label inputs."""

from __future__ import annotations

import argparse
import pickle
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def read_instance(path: Path) -> tuple[list[float], list[float], list[float]]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, (tuple, list)) or len(obj) != 3:
        raise ValueError(f"Unexpected pickle payload in {path}: {type(obj)!r}")
    p, q, r = obj
    if not (len(p) == len(q) == len(r)):
        raise ValueError(f"Dimension list length mismatch in {path}")
    return list(map(float, p)), list(map(float, q)), list(map(float, r))


def write_xml(path: Path, p: list[float], q: list[float], r: list[float]) -> None:
    root = ET.Element("instance")
    ET.SubElement(root, "norder").text = "1"
    orders = ET.SubElement(root, "orders")
    order = ET.SubElement(orders, "order", {"id": "0"})
    for item_id, (pi, qi, ri) in enumerate(zip(p, q, r)):
        item = ET.SubElement(order, "item", {"id": str(item_id)})
        ET.SubElement(item, "p").text = f"{pi:g}"
        ET.SubElement(item, "q").text = f"{qi:g}"
        ET.SubElement(item, "r").text = f"{ri:g}"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(".codex_training_tmp/opre_sm1/codes/BPP"),
        help="Path containing dataset50/ and data/packages.txt from opre.2022.2369.sm1.zip.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("or2023_bpp_data"))
    args = parser.parse_args()

    dataset_dir = args.source_root / "dataset50"
    packages_path = args.source_root / "data" / "packages.txt"
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if not packages_path.exists():
        raise FileNotFoundError(packages_path)

    xml_dir = args.output_root / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(packages_path, args.output_root / "packages.txt")

    converted = 0
    item_counts: dict[int, int] = {}
    for pkl_path in sorted(dataset_dir.glob("H3DBPP_*.pkl")):
        p, q, r = read_instance(pkl_path)
        xml_path = xml_dir / f"{pkl_path.stem}.xml"
        write_xml(xml_path, p, q, r)
        converted += 1
        item_counts[len(p)] = item_counts.get(len(p), 0) + 1

    print(f"Converted {converted} BPP instances to {xml_dir}")
    print(f"Copied packages to {args.output_root / 'packages.txt'}")
    print(f"Item-count distribution: {dict(sorted(item_counts.items()))}")


if __name__ == "__main__":
    main()
