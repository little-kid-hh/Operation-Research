#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from box_design_surrogate.evaluator import Box
from box_design_surrogate.features import read_order_summaries
from box_design_surrogate.kandula_repro import initial_boxes_kmeans
from box_design_surrogate.milp_oracle import (
    JavaMilpOracle,
    MilpBoxSetScore,
    MilpLabelTableOracle,
)
from box_design_surrogate.search import apply_move, coordinate_moves


class BoxSetOracle(Protocol):
    def evaluate(self, orders: list, boxes: list[Box]) -> MilpBoxSetScore:
        ...


def parse_schedule(value: str) -> list[tuple[float, int]]:
    schedule: list[tuple[float, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        step_s, iter_s = item.split(":", 1)
        step = float(step_s)
        iterations = int(iter_s)
        if step <= 0 or iterations < 0:
            raise argparse.ArgumentTypeError("schedule entries must be positive_step:nonnegative_iters")
        schedule.append((step, iterations))
    if not schedule:
        raise argparse.ArgumentTypeError("schedule must contain at least one step:iters entry")
    return schedule


def score_rank(score: MilpBoxSetScore) -> tuple[int, float]:
    return score.uncovered_orders, score.packaging_factor


def boxes_to_rows(boxes: list[Box]) -> list[dict[str, float | int]]:
    return [
        {
            "box_id": box.box_id,
            "length": box.length,
            "width": box.width,
            "height": box.height,
            "volume": box.volume,
        }
        for box in sorted(boxes, key=lambda b: b.box_id)
    ]


def score_to_dict(score: MilpBoxSetScore) -> dict:
    out = asdict(score)
    out["assignments"] = list(score.assignments)
    return out


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def make_oracle(args: argparse.Namespace) -> BoxSetOracle:
    if args.oracle == "labels":
        return MilpLabelTableOracle(
            labels_path=args.labels_path,
            orientation_label=args.orientation_label,
        )

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
    )


def best_single_action(
    *,
    oracle: BoxSetOracle,
    orders: list,
    current: list[Box],
    current_score: MilpBoxSetScore,
    step: float,
) -> tuple[list[Box], MilpBoxSetScore, str]:
    best_boxes = current
    best_score = current_score
    best_action = "noop"
    for move in coordinate_moves(current, step):
        candidate = apply_move(current, move)
        score = oracle.evaluate(orders, candidate)
        if score_rank(score) < score_rank(best_score):
            best_boxes = candidate
            best_score = score
            best_action = f"{move.box_id}:{move.dimension}:{move.delta:+.6f}"
    return best_boxes, best_score, best_action


def run_fixed_step(
    *,
    oracle: BoxSetOracle,
    orders: list,
    boxes: list[Box],
    step: float,
    iterations: int,
) -> tuple[list[Box], MilpBoxSetScore, list[dict]]:
    current = sorted(boxes, key=lambda b: b.box_id)
    current_score = oracle.evaluate(orders, current)
    trace = [
        {
            "phase": "initial",
            "iteration": 0,
            "step": step,
            "action": "init",
            **score_to_dict(current_score),
        }
    ]
    for iteration in range(1, iterations + 1):
        best_boxes, best_score, action = best_single_action(
            oracle=oracle,
            orders=orders,
            current=current,
            current_score=current_score,
            step=step,
        )
        improved = score_rank(best_score) < score_rank(current_score)
        trace.append(
            {
                "phase": "fixed_step",
                "iteration": iteration,
                "step": step,
                "action": action,
                "improved": improved,
                **score_to_dict(best_score),
            }
        )
        if not improved:
            break
        current = best_boxes
        current_score = best_score
    return sorted(current, key=lambda b: b.volume), current_score, trace


def run_staged_greedy(
    *,
    oracle: BoxSetOracle,
    orders: list,
    boxes: list[Box],
    schedule: list[tuple[float, int]],
) -> tuple[list[Box], MilpBoxSetScore, list[dict]]:
    current = sorted(boxes, key=lambda b: b.box_id)
    current_score = oracle.evaluate(orders, current)
    trace = [
        {
            "phase": "initial",
            "iteration": 0,
            "stage": 0,
            "step": schedule[0][0],
            "action": "init",
            **score_to_dict(current_score),
        }
    ]
    global_iteration = 0
    for stage_idx, (step, iterations) in enumerate(schedule, start=1):
        for _ in range(iterations):
            global_iteration += 1
            best_boxes, best_score, action = best_single_action(
                oracle=oracle,
                orders=orders,
                current=current,
                current_score=current_score,
                step=step,
            )
            improved = score_rank(best_score) < score_rank(current_score)
            trace.append(
                {
                    "phase": "staged_greedy",
                    "iteration": global_iteration,
                    "stage": stage_idx,
                    "step": step,
                    "action": action,
                    "improved": improved,
                    **score_to_dict(best_score),
                }
            )
            if not improved:
                break
            current = best_boxes
            current_score = best_score
    return sorted(current, key=lambda b: b.volume), current_score, trace


def write_trace(path: Path, trace: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "stage",
        "iteration",
        "step",
        "action",
        "improved",
        "packaging_factor",
        "mean_box_volume",
        "mean_order_volume",
        "coverage_rate",
        "uncovered_orders",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trace)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MILP-backed OR2023 box search entrypoint. "
            "Algorithms: paper_fixed_step uses the fixed paper action step; "
            "staged_greedy uses a manually adjusted step schedule."
        )
    )
    parser.add_argument("--algorithm", choices=["paper_fixed_step", "staged_greedy"], required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--orders-limit", type=int, default=20)
    parser.add_argument("--fixed-step", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--schedule", type=parse_schedule, default=parse_schedule("0.5:2,0.25:2"))
    parser.add_argument("--oracle", choices=["java", "labels"], default="java")
    parser.add_argument("--orientation-label", choices=["label_2ori", "label_6ori"], default="label_6ori")
    parser.add_argument("--xml-path", type=Path, default=ROOT / "assets/or2023_bsp_data/xml_unique/or2023_bsp_unique_orders.xml")
    parser.add_argument("--labels-path", type=Path, default=ROOT / "assets/milp_labels/or2023_bsp_unique_package_labels.csv")
    parser.add_argument("--java-classes", type=Path, default=REPO_ROOT / "MILP_3DBPP/target/classes")
    parser.add_argument("--java-classpath", default="")
    parser.add_argument("--milp-time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--allow-bsp-derived-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/milp_box_algorithms")
    args = parser.parse_args()

    if args.iterations < 0:
        raise ValueError("--iterations must be non-negative")
    if args.orders_limit is not None and args.orders_limit < args.k:
        raise ValueError("--orders-limit must be >= --k for k-means initialization")

    orders = read_order_summaries(args.xml_path)
    if args.orders_limit is not None:
        orders = orders[: args.orders_limit]
    initial_boxes = initial_boxes_kmeans(orders, args.k, random_state=args.seed)
    oracle = make_oracle(args)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.out_root / args.algorithm / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "algorithm": args.algorithm,
        "dataset": "OR2023 unique orders",
        "orders_limit": args.orders_limit,
        "k": args.k,
        "seed": args.seed,
        "oracle": args.oracle,
        "orientation_label": args.orientation_label,
        "fixed_step": args.fixed_step,
        "iterations": args.iterations,
        "schedule": [{"step": step, "iterations": iters} for step, iters in args.schedule],
        "xml_path": str(args.xml_path),
        "labels_path": str(args.labels_path),
        "java_classes": str(args.java_classes),
        "java_classpath_extra": args.java_classpath,
        "milp_time_limit_seconds": args.milp_time_limit_seconds,
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "initial_boxes.json", boxes_to_rows(initial_boxes))

    if args.algorithm == "paper_fixed_step":
        best_boxes, best_score, trace = run_fixed_step(
            oracle=oracle,
            orders=orders,
            boxes=initial_boxes,
            step=args.fixed_step,
            iterations=args.iterations,
        )
    else:
        best_boxes, best_score, trace = run_staged_greedy(
            oracle=oracle,
            orders=orders,
            boxes=initial_boxes,
            schedule=args.schedule,
        )

    summary = {
        **manifest,
        "initial_score": trace[0],
        "best_score": score_to_dict(best_score),
        "run_dir": str(run_dir),
    }
    write_trace(run_dir / "trace.csv", trace)
    write_json(run_dir / "best_boxes.json", boxes_to_rows(best_boxes))
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
