from __future__ import annotations

import copy
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def stable_hash(seed: int, order_id: str) -> str:
    return hashlib.sha256(f"{seed}:{order_id}".encode("utf-8")).hexdigest()


def split_counts(n: int, train_frac: float, dev_frac: float, test_frac: float) -> tuple[int, int, int]:
    total = train_frac + dev_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    if min(train_frac, dev_frac, test_frac) <= 0:
        raise ValueError("split fractions must be positive")
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    n_test = n - n_train - n_dev
    return n_train, n_dev, n_test


def materialize_splits(
    *,
    source_xml: Path,
    out_dir: Path,
    seed: int,
    train_frac: float = 0.6,
    dev_frac: float = 0.2,
    test_frac: float = 0.2,
    limit: int | None = None,
) -> dict[str, Any]:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    orders_node = root.find("orders")
    if orders_node is None:
        raise ValueError(f"missing <orders> node in {source_xml}")

    order_elements = list(orders_node.findall("order"))
    ranked = sorted(
        (
            {
                "order_id": str(order_el.get("id")),
                "hash": stable_hash(seed, str(order_el.get("id"))),
                "item_count": len(order_el.findall("item")),
                "element": order_el,
            }
            for order_el in order_elements
        ),
        key=lambda row: row["hash"],
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        ranked = ranked[:limit]

    n_train, n_dev, n_test = split_counts(len(ranked), train_frac, dev_frac, test_frac)
    split_rows = {
        "train": ranked[:n_train],
        "dev": ranked[n_train : n_train + n_dev],
        "test": ranked[n_train + n_dev :],
    }
    assert len(split_rows["test"]) == n_test

    out_dir.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, str] = {}
    for split_name, rows in split_rows.items():
        split_path = out_dir / f"{source_xml.stem}_{split_name}.xml"
        _write_split_xml(root, rows, split_path)
        split_files[split_name] = str(split_path)

    assignments_path = out_dir / "split_assignments.csv"
    with assignments_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "position", "order_id", "hash", "item_count"])
        writer.writeheader()
        for split_name, rows in split_rows.items():
            for position, row in enumerate(rows):
                writer.writerow(
                    {
                        "split": split_name,
                        "position": position,
                        "order_id": row["order_id"],
                        "hash": row["hash"],
                        "item_count": row["item_count"],
                    }
                )

    manifest = {
        "source_xml": str(source_xml),
        "seed": seed,
        "limit": limit,
        "fractions": {"train": train_frac, "dev": dev_frac, "test": test_frac},
        "counts": {split_name: len(rows) for split_name, rows in split_rows.items()},
        "split_files": split_files,
        "assignments_csv": str(assignments_path),
        "summaries": {split_name: _summary(rows) for split_name, rows in split_rows.items()},
    }
    manifest_path = out_dir / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _write_split_xml(source_root: ET.Element, rows: list[dict[str, Any]], split_path: Path) -> None:
    split_root = ET.Element(source_root.tag, source_root.attrib)
    norder = ET.SubElement(split_root, "norder")
    norder.text = str(len(rows))
    orders = ET.SubElement(split_root, "orders")
    for row in rows:
        orders.append(copy.deepcopy(row["element"]))
    ET.ElementTree(split_root).write(split_path, encoding="utf-8", xml_declaration=False)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    item_hist: dict[str, int] = {}
    for row in rows:
        key = str(row["item_count"])
        item_hist[key] = item_hist.get(key, 0) + 1
    return {
        "orders": len(rows),
        "item_count_histogram": dict(sorted(item_hist.items(), key=lambda kv: int(kv[0]))),
        "first_order_ids": [row["order_id"] for row in rows[:10]],
    }
