#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from box_design_surrogate.candidate_ranker import CANDIDATE_TRACE_COLUMNS, candidate_feature_row
from box_design_surrogate.evaluator import Box
from box_design_surrogate.features import read_order_summaries
from box_design_surrogate.milp_oracle import JavaMilpOracle
from box_design_surrogate.search import BoxMove


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("source_run_id", "")),
        str(row.get("phase", "")),
        str(row.get("stage", "")),
        str(row.get("iteration", "")),
        str(row.get("step", "")),
    )


def as_int(value: Any) -> int:
    return int(float(value))


def as_float(value: Any) -> float:
    return float(value)


def reconstruct_current_boxes(group_rows: list[dict[str, Any]]) -> list[Box]:
    by_id: dict[int, Box] = {}
    for row in group_rows:
        box_id = as_int(row["move_box_id"])
        box = Box(
            box_id=box_id,
            length=as_float(row["current_box_length"]),
            width=as_float(row["current_box_width"]),
            height=as_float(row["current_box_height"]),
        )
        existing = by_id.get(box_id)
        if existing is not None and (
            abs(existing.length - box.length) > 1e-6
            or abs(existing.width - box.width) > 1e-6
            or abs(existing.height - box.height) > 1e-6
        ):
            raise ValueError(f"inconsistent current dimensions for box_id={box_id}")
        by_id[box_id] = box
    return [by_id[box_id] for box_id in sorted(by_id)]


def reconstruct_candidate_boxes(current_boxes: list[Box], row: dict[str, Any]) -> list[Box]:
    move_box_id = as_int(row["move_box_id"])
    out: list[Box] = []
    for box in current_boxes:
        if int(box.box_id) != move_box_id:
            out.append(box)
            continue
        out.append(
            Box(
                box_id=move_box_id,
                length=as_float(row["candidate_box_length"]),
                width=as_float(row["candidate_box_width"]),
                height=as_float(row["candidate_box_height"]),
            )
        )
    return out


def ordered_groups(rows: list[dict[str, Any]]) -> list[tuple[tuple[str, str, str, str, str], list[dict[str, Any]]]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str, str, str, str]] = []
    for row in rows:
        key = group_key(row)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    return [(key, groups[key]) for key in order]


def make_java_oracle(args: argparse.Namespace) -> JavaMilpOracle:
    classpath_parts = [str(args.java_classes)]
    gurobi_jar = os.environ.get("GUROBI_JAR")
    if gurobi_jar:
        classpath_parts.append(gurobi_jar)
    if args.java_classpath:
        classpath_parts.append(args.java_classpath)
    return JavaMilpOracle(
        xml_path=args.xml_path,
        java_classpath=os.pathsep.join(classpath_parts),
        orientation_label=args.orientation_label,
        time_limit_seconds=args.milp_time_limit_seconds,
        allow_bsp_derived_data=args.allow_bsp_derived_data,
        cache_dir=args.oracle_cache_dir,
        orders_offset=args.orders_offset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add assignment-aware candidate-ranker features to an existing exact "
            "candidate trace by re-evaluating each step group's current box set."
        )
    )
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--xml-path", type=Path, required=True)
    parser.add_argument("--orders-offset", type=int, default=0)
    parser.add_argument("--orders-limit", type=int, required=True)
    parser.add_argument("--java-classes", type=Path, default=REPO_ROOT / "MILP_3DBPP/target/classes")
    parser.add_argument("--java-classpath", default=os.environ.get("GUROBI_JAR", ""))
    parser.add_argument("--milp-time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--orientation-label", choices=["label_2ori", "label_6ori"], default="label_6ori")
    parser.add_argument("--allow-bsp-derived-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oracle-cache-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if args.orders_offset < 0:
        raise ValueError("--orders-offset must be non-negative")
    if args.orders_limit <= 0:
        raise ValueError("--orders-limit must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")

    rows = read_rows(args.trace_csv)
    all_orders = read_order_summaries(args.xml_path)
    orders = all_orders[args.orders_offset : args.orders_offset + args.orders_limit]
    if len(orders) != args.orders_limit:
        raise ValueError(f"selected {len(orders)} orders, expected {args.orders_limit}")
    oracle = make_java_oracle(args)

    started = time.perf_counter()
    augmented: list[dict[str, Any]] = []
    groups = ordered_groups(rows)
    for group_idx, (_, group_rows) in enumerate(groups, start=1):
        current_boxes = reconstruct_current_boxes(group_rows)
        current_score = oracle.evaluate(orders, current_boxes)
        for row in group_rows:
            move = BoxMove(
                box_id=as_int(row["move_box_id"]),
                dimension=str(row["move_dimension"]),
                delta=as_float(row["move_delta"]),
            )
            candidate_boxes = reconstruct_candidate_boxes(current_boxes, row)
            updated = dict(row)
            updated.update(
                candidate_feature_row(
                    current_boxes=current_boxes,
                    candidate_boxes=candidate_boxes,
                    move=move,
                    current_score=current_score,
                    step=as_float(row["step"]),
                    stage=as_int(row["stage"]),
                    iteration=as_int(row["iteration"]),
                    candidate_index=as_int(row["candidate_index"]),
                    generated_candidates=as_int(row["generated_candidates"]),
                    orders=orders,
                )
            )
            augmented.append(updated)
        if args.progress_every and group_idx % args.progress_every == 0:
            print(f"processed {group_idx}/{len(groups)} groups", file=sys.stderr)

    input_fields = list(rows[0].keys()) if rows else []
    fieldnames = list(CANDIDATE_TRACE_COLUMNS)
    for field in input_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    write_rows(args.out_csv, augmented, fieldnames)
    summary = {
        "trace_csv": str(args.trace_csv),
        "out_csv": str(args.out_csv),
        "groups": len(groups),
        "rows": len(augmented),
        "orders_offset": args.orders_offset,
        "orders_limit": args.orders_limit,
        "elapsed_seconds": time.perf_counter() - started,
        "oracle_cache": oracle.cache_info(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
