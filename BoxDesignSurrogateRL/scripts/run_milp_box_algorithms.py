#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
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
from box_design_surrogate.kandula_repro import initial_boxes_kmeans, order_requirement
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


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float value")
    if any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("all float-list values must be positive")
    return values


def score_rank(score: MilpBoxSetScore) -> tuple[int, int, float]:
    return score.uncovered_orders, score.unknown_pairs, score.packaging_factor


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


def boxes_from_json(path: Path) -> list[Box]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of boxes")

    boxes: list[Box] = []
    seen_ids: set[int] = set()
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"{path} box row {idx} must be an object")
        try:
            box_id = int(row["box_id"])
            length = float(row["length"])
            width = float(row["width"])
            height = float(row["height"])
        except KeyError as exc:
            raise ValueError(f"{path} box row {idx} is missing {exc.args[0]!r}") from exc
        if box_id in seen_ids:
            raise ValueError(f"{path} contains duplicate box_id {box_id}")
        if length <= 0.0 or width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path} box_id {box_id} has non-positive dimensions")
        seen_ids.add(box_id)
        boxes.append(Box(box_id=box_id, length=length, width=width, height=height))
    if not boxes:
        raise ValueError(f"{path} must contain at least one box")
    return sorted(boxes, key=lambda b: b.box_id)


def score_to_dict(score: MilpBoxSetScore) -> dict:
    out = asdict(score)
    out["assignments"] = list(score.assignments)
    return out


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def git_info(repo_root: Path) -> dict[str, str | bool | None]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run_git(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


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
        cache_dir=args.oracle_cache_dir,
    )


def replace_box(boxes: list[Box], replacement: Box) -> list[Box]:
    return [replacement if box.box_id == replacement.box_id else box for box in boxes]


def expand_box_for_order(box: Box, order, margin: float) -> Box:
    req_l, req_m, req_s, total_volume = order_requirement(order)
    length = max(box.length, req_l * margin)
    width = max(box.width, req_m * margin)
    height = max(box.height, req_s * margin)
    target_volume = total_volume * (margin**3)
    volume = length * width * height
    if volume < target_volume:
        scale = float((target_volume / volume) ** (1.0 / 3.0))
        length *= scale
        width *= scale
        height *= scale
    return Box(box.box_id, float(length), float(width), float(height))


def same_box_dimensions(left: Box, right: Box, tolerance: float = 1e-9) -> bool:
    return (
        abs(left.length - right.length) <= tolerance
        and abs(left.width - right.width) <= tolerance
        and abs(left.height - right.height) <= tolerance
    )


def repair_coverage_by_expansion(
    *,
    oracle: BoxSetOracle,
    orders: list,
    boxes: list[Box],
    initial_score: MilpBoxSetScore,
    margins: list[float],
    max_rounds: int,
) -> tuple[list[Box], MilpBoxSetScore, list[dict]]:
    current = sorted(boxes, key=lambda b: b.box_id)
    current_score = initial_score
    trace: list[dict] = []
    for repair_round in range(1, max_rounds + 1):
        if current_score.uncovered_orders == 0:
            break

        best_boxes = current
        best_score = current_score
        best_action = "noop"
        candidate_evaluations = 0
        uncovered_indices = [idx for idx, assignment in enumerate(current_score.assignments) if assignment is None]
        for order_idx in uncovered_indices:
            order = orders[order_idx]
            for box in current:
                for margin in margins:
                    replacement = expand_box_for_order(box, order, margin)
                    if same_box_dimensions(box, replacement):
                        continue
                    candidate = replace_box(current, replacement)
                    candidate_evaluations += 1
                    score = oracle.evaluate(orders, candidate)
                    if score_rank(score) < score_rank(best_score):
                        best_boxes = candidate
                        best_score = score
                        best_action = (
                            f"order={order.order_id};box={box.box_id};"
                            f"margin={margin:.6f}"
                        )

        improved = score_rank(best_score) < score_rank(current_score)
        trace.append(
            {
                "phase": "coverage_repair",
                "repair_round": repair_round,
                "iteration": 0,
                "action": best_action,
                "improved": improved,
                "candidate_evaluations": candidate_evaluations,
                **score_to_dict(best_score),
            }
        )
        if not improved:
            break
        current = best_boxes
        current_score = best_score
    return sorted(current, key=lambda b: b.volume), current_score, trace


def best_single_action(
    *,
    oracle: BoxSetOracle,
    orders: list,
    current: list[Box],
    current_score: MilpBoxSetScore,
    step: float,
) -> tuple[list[Box], MilpBoxSetScore, str, int]:
    best_boxes = current
    best_score = current_score
    best_action = "noop"
    candidate_evaluations = 0
    for move in coordinate_moves(current, step):
        candidate = apply_move(current, move)
        candidate_evaluations += 1
        score = oracle.evaluate(orders, candidate)
        if score_rank(score) < score_rank(best_score):
            best_boxes = candidate
            best_score = score
            best_action = f"{move.box_id}:{move.dimension}:{move.delta:+.6f}"
    return best_boxes, best_score, best_action, candidate_evaluations


def run_fixed_step(
    *,
    oracle: BoxSetOracle,
    orders: list,
    boxes: list[Box],
    step: float,
    iterations: int,
    initial_score: MilpBoxSetScore | None = None,
    initial_phase: str = "initial",
) -> tuple[list[Box], MilpBoxSetScore, list[dict]]:
    current = sorted(boxes, key=lambda b: b.box_id)
    current_score = initial_score if initial_score is not None else oracle.evaluate(orders, current)
    trace = [
        {
            "phase": initial_phase,
            "iteration": 0,
            "step": step,
            "action": "init",
            "candidate_evaluations": 0,
            **score_to_dict(current_score),
        }
    ]
    for iteration in range(1, iterations + 1):
        best_boxes, best_score, action, candidate_evaluations = best_single_action(
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
                "candidate_evaluations": candidate_evaluations,
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
    initial_score: MilpBoxSetScore | None = None,
    initial_phase: str = "initial",
) -> tuple[list[Box], MilpBoxSetScore, list[dict]]:
    current = sorted(boxes, key=lambda b: b.box_id)
    current_score = initial_score if initial_score is not None else oracle.evaluate(orders, current)
    trace = [
        {
            "phase": initial_phase,
            "iteration": 0,
            "stage": 0,
            "step": schedule[0][0],
            "action": "init",
            "candidate_evaluations": 0,
            **score_to_dict(current_score),
        }
    ]
    global_iteration = 0
    for stage_idx, (step, iterations) in enumerate(schedule, start=1):
        for _ in range(iterations):
            global_iteration += 1
            best_boxes, best_score, action, candidate_evaluations = best_single_action(
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
                    "candidate_evaluations": candidate_evaluations,
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
        "repair_round",
        "iteration",
        "step",
        "action",
        "improved",
        "packaging_factor",
        "mean_box_volume",
        "mean_order_volume",
        "coverage_rate",
        "uncovered_orders",
        "unknown_pairs",
        "orders_with_unknown",
        "candidate_evaluations",
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
    parser.add_argument(
        "--initial-boxes-json",
        type=Path,
        default=None,
        help="Optional best_boxes.json checkpoint to use instead of k-means initialization.",
    )
    parser.add_argument("--oracle", choices=["java", "labels"], default="java")
    parser.add_argument("--orientation-label", choices=["label_2ori", "label_6ori"], default="label_6ori")
    parser.add_argument("--xml-path", type=Path, default=ROOT / "assets/or2023_bsp_data/xml_unique/or2023_bsp_unique_orders.xml")
    parser.add_argument("--labels-path", type=Path, default=ROOT / "assets/milp_labels/or2023_bsp_unique_package_labels.csv")
    parser.add_argument("--java-classes", type=Path, default=REPO_ROOT / "MILP_3DBPP/target/classes")
    parser.add_argument("--java-classpath", default="")
    parser.add_argument("--milp-time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--allow-bsp-derived-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oracle-cache-dir", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/milp_box_algorithms")
    parser.add_argument(
        "--comparison-label",
        default=None,
        help="Optional label shared by runs that should be paired in one comparison.",
    )
    parser.add_argument(
        "--config-label",
        default=None,
        help="Optional label for this concrete algorithm/configuration, e.g. fixed05_i2.",
    )
    parser.add_argument("--coverage-repair", choices=["none", "geometric_expand"], default="none")
    parser.add_argument(
        "--repair-margins",
        type=parse_float_list,
        default=parse_float_list("1.0,1.05,1.1,1.25,1.5,2.0"),
    )
    parser.add_argument("--repair-max-rounds", type=int, default=5)
    parser.add_argument(
        "--code-version",
        default=None,
        help="Optional manually supplied code version, e.g. the pushed GitHub commit used for this run.",
    )
    args = parser.parse_args()

    if args.iterations < 0:
        raise ValueError("--iterations must be non-negative")
    if args.repair_max_rounds < 0:
        raise ValueError("--repair-max-rounds must be non-negative")
    if args.orders_limit is not None and args.orders_limit < args.k:
        raise ValueError("--orders-limit must be >= --k for k-means initialization")

    orders = read_order_summaries(args.xml_path)
    if args.orders_limit is not None:
        orders = orders[: args.orders_limit]
    if args.initial_boxes_json is None:
        initial_boxes = initial_boxes_kmeans(orders, args.k, random_state=args.seed)
    else:
        initial_boxes = boxes_from_json(args.initial_boxes_json)
        if len(initial_boxes) != args.k:
            raise ValueError(f"--initial-boxes-json contains {len(initial_boxes)} boxes, expected --k={args.k}")
    oracle = make_oracle(args)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.out_root / args.algorithm / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "algorithm": args.algorithm,
        "comparison_label": args.comparison_label,
        "config_label": args.config_label,
        "dataset": "OR2023 unique orders",
        "orders_limit": args.orders_limit,
        "k": args.k,
        "seed": args.seed,
        "oracle": args.oracle,
        "orientation_label": args.orientation_label,
        "fixed_step": args.fixed_step,
        "iterations": args.iterations,
        "schedule": [{"step": step, "iterations": iters} for step, iters in args.schedule],
        "initial_boxes_json": str(args.initial_boxes_json) if args.initial_boxes_json is not None else None,
        "xml_path": str(args.xml_path),
        "labels_path": str(args.labels_path),
        "java_classes": str(args.java_classes),
        "java_classpath_extra": args.java_classpath,
        "milp_time_limit_seconds": args.milp_time_limit_seconds,
        "coverage_repair": args.coverage_repair,
        "repair_margins": args.repair_margins,
        "repair_max_rounds": args.repair_max_rounds,
        "code_version": args.code_version,
        "git": git_info(REPO_ROOT),
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "initial_boxes.json", boxes_to_rows(initial_boxes))

    started = time.perf_counter()
    pre_search_trace: list[dict] = []
    search_boxes = initial_boxes
    search_initial_score: MilpBoxSetScore | None = None
    if args.coverage_repair != "none":
        raw_initial_score = oracle.evaluate(orders, search_boxes)
        pre_search_trace.append(
            {
                "phase": "initial",
                "iteration": 0,
                "action": "init",
                "candidate_evaluations": 0,
                **score_to_dict(raw_initial_score),
            }
        )
        if args.coverage_repair == "geometric_expand":
            search_boxes, search_initial_score, repair_trace = repair_coverage_by_expansion(
                oracle=oracle,
                orders=orders,
                boxes=search_boxes,
                initial_score=raw_initial_score,
                margins=args.repair_margins,
                max_rounds=args.repair_max_rounds,
            )
            pre_search_trace.extend(repair_trace)
        else:
            raise ValueError(f"unknown coverage repair mode: {args.coverage_repair}")

    if args.algorithm == "paper_fixed_step":
        best_boxes, best_score, search_trace = run_fixed_step(
            oracle=oracle,
            orders=orders,
            boxes=search_boxes,
            step=args.fixed_step,
            iterations=args.iterations,
            initial_score=search_initial_score,
            initial_phase="search_initial" if pre_search_trace else "initial",
        )
    else:
        best_boxes, best_score, search_trace = run_staged_greedy(
            oracle=oracle,
            orders=orders,
            boxes=search_boxes,
            schedule=args.schedule,
            initial_score=search_initial_score,
            initial_phase="search_initial" if pre_search_trace else "initial",
        )
    trace = pre_search_trace + search_trace
    elapsed_seconds = time.perf_counter() - started

    repair_rows = [row for row in trace if row.get("phase") == "coverage_repair"]
    search_initial_rows = [row for row in trace if row.get("phase") in {"search_initial", "initial"}]

    summary = {
        **manifest,
        "initial_score": trace[0],
        "search_initial_score": search_initial_rows[-1] if search_initial_rows else trace[0],
        "coverage_repair_score": repair_rows[-1] if repair_rows else None,
        "best_score": score_to_dict(best_score),
        "trace_rows": len(trace),
        "candidate_evaluations": int(sum(row.get("candidate_evaluations", 0) for row in trace)),
        "elapsed_seconds": elapsed_seconds,
        "oracle_cache": oracle.cache_info() if hasattr(oracle, "cache_info") else None,
        "run_dir": str(run_dir),
    }
    write_trace(run_dir / "trace.csv", trace)
    write_json(run_dir / "best_boxes.json", boxes_to_rows(best_boxes))
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
