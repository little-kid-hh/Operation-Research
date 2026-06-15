from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
HYBRID_ROOT = REPO_ROOT / "HybridSVM"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from box_design_surrogate import BatchSurrogateEvaluator, Box, SurrogateEvaluator, read_order_summaries  # noqa: E402
from box_design_surrogate.kandula_repro import (  # noqa: E402
    beam_tree_search,
    evaluate_simple_box_set,
    greedy_coordinate_descent,
    initial_boxes_kmeans,
)
from box_design_surrogate.surrogate_search import surrogate_repair_search  # noqa: E402
from src.ensemble_train import load_ensemble_pipeline  # noqa: E402


SUMMARY_FIELDNAMES = [
    "run_id",
    "stage",
    "k",
    "seed",
    "orders",
    "tau",
    "simplified_pf",
    "simplified_coverage",
    "surrogate_coverage",
    "uncovered_orders",
    "low_margin_assignments",
    "mean_base_cost",
    "mean_adjusted_cost",
    "mean_assigned_probability",
    "mean_box_volume",
    "iterations",
    "evaluated_candidates",
]


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def boxes_to_rows(boxes: list[Box]) -> list[dict[str, float | int]]:
    return [
        {
            "box_id": box.box_id,
            "length": box.length,
            "width": box.width,
            "height": box.height,
            "volume": box.volume,
        }
        for box in boxes
    ]


def write_assignments(path: Path, assignments) -> None:
    fieldnames = [
        "order_key",
        "box_id",
        "probability",
        "base_cost",
        "adjusted_cost",
        "covered",
        "best_box_id",
        "best_probability",
        "low_margin",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for assignment in assignments:
            writer.writerow(
                {
                    "order_key": assignment.order_key,
                    "box_id": "" if assignment.box_id is None else assignment.box_id,
                    "probability": assignment.probability,
                    "base_cost": assignment.base_cost,
                    "adjusted_cost": assignment.adjusted_cost,
                    "covered": int(assignment.covered),
                    "best_box_id": "" if assignment.best_box_id is None else assignment.best_box_id,
                    "best_probability": assignment.best_probability,
                    "low_margin": int(assignment.low_margin),
                }
            )


def add_summary_row(
    rows: list[dict[str, object]],
    *,
    run_id: str,
    stage: str,
    k: int,
    seed: int,
    tau: float,
    orders_count: int,
    boxes: list[Box],
    surrogate_eval,
    simple_score,
    iterations: int = 0,
    evaluated_candidates: int = 1,
) -> dict[str, object]:
    row = {
        "run_id": run_id,
        "stage": stage,
        "k": k,
        "seed": seed,
        "orders": orders_count,
        "tau": tau,
        "simplified_pf": simple_score.packaging_factor,
        "simplified_coverage": simple_score.coverage_rate,
        "surrogate_coverage": surrogate_eval.coverage_rate,
        "uncovered_orders": surrogate_eval.uncovered_orders,
        "low_margin_assignments": surrogate_eval.low_margin_assignments,
        "mean_base_cost": surrogate_eval.total_base_cost / orders_count,
        "mean_adjusted_cost": surrogate_eval.total_adjusted_cost / orders_count,
        "mean_assigned_probability": surrogate_eval.mean_assigned_probability,
        "mean_box_volume": sum(box.volume for box in boxes) / len(boxes),
        "iterations": iterations,
        "evaluated_candidates": evaluated_candidates,
    }
    rows.append(row)
    return row


def init_summary(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()


def append_summary(path: Path, row: dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-limit", type=int, default=2000)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--taus", type=float, nargs="+", default=[0.95, 0.90])
    parser.add_argument("--step-schedule", type=float, nargs="+", default=[0.5, 0.1, 0.05, 0.01])
    parser.add_argument("--kandula-step", type=float, default=0.5)
    parser.add_argument("--kandula-max-iters", type=int, default=20)
    parser.add_argument("--beam-depth", type=int, default=20)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--repair-max-iters", type=int, default=50)
    parser.add_argument("--max-repair-orders", type=int, default=64)
    parser.add_argument("--max-candidates-per-iter", type=int, default=64)
    parser.add_argument("--repair-only", action="store_true")
    parser.add_argument("--tau-high", type=float, default=0.99)
    parser.add_argument("--lambda-risk", type=float, default=1000.0)
    parser.add_argument("--uncovered-penalty", type=float, default=1_000_000.0)
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=ROOT / "assets" / "or2023_bsp_data" / "xml_unique" / "or2023_bsp_unique_orders.xml",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT
        / "assets"
        / "loadability_model_pack"
        / "models"
        / "ensemble"
        / "ensemble_baseline_20260526.json",
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "surrogate_design")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / f"run_{timestamp}"
    boxes_dir = run_dir / "boxes"
    assignments_dir = run_dir / "assignments"
    boxes_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.csv"
    init_summary(summary_path)

    manifest = {
        "timestamp": timestamp,
        "status": "running",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short", "BoxDesignSurrogateRL"]),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": vars(args) | {"xml_path": str(args.xml_path), "model_path": str(args.model_path), "out_root": str(args.out_root)},
        "data": {
            "orders_xml": str(args.xml_path),
            "notes": "Kandula-style reproduction and surrogate repair search on OR2023 BSP unique-order data.",
        },
    }
    write_json(run_dir / "manifest.json", manifest)

    try:
        orders = read_order_summaries(args.xml_path)
        if args.orders_limit:
            orders = orders[: args.orders_limit]
        model = load_ensemble_pipeline(args.model_path)
        rows: list[dict[str, object]] = []

        for tau in args.taus:
            evaluator = BatchSurrogateEvaluator.from_evaluator(
                SurrogateEvaluator(
                    model=model,
                    tau=tau,
                    tau_high=args.tau_high,
                    lambda_risk=args.lambda_risk,
                    uncovered_penalty=args.uncovered_penalty,
                ),
                orders,
            )
            for k in args.ks:
                for seed in args.seeds:
                    run_id = f"k{k}_seed{seed}_tau{tau:g}".replace(".", "p")
                    boxes0 = initial_boxes_kmeans(orders, k, random_state=seed)
                    boxes1, _ = greedy_coordinate_descent(
                        orders,
                        boxes0,
                        step=args.kandula_step,
                        max_iters=args.kandula_max_iters,
                    )
                    boxes2, _ = beam_tree_search(
                        orders,
                        boxes0,
                        step=args.kandula_step,
                        depth=args.beam_depth,
                        beam_width=args.beam_width,
                    )

                    stage_boxes = {
                        "kmeans": boxes0,
                        "local_search": boxes1,
                        "beam_tree": boxes2,
                    }
                    for stage, boxes in stage_boxes.items():
                        surrogate_eval = evaluator.evaluate_prepared(boxes)
                        simple_score = evaluate_simple_box_set(orders, boxes)
                        row = add_summary_row(
                            rows,
                            run_id=run_id,
                            stage=stage,
                            k=k,
                            seed=seed,
                            tau=tau,
                            orders_count=len(orders),
                            boxes=boxes,
                            surrogate_eval=surrogate_eval,
                            simple_score=simple_score,
                        )
                        append_summary(summary_path, row)
                        write_json(boxes_dir / f"{run_id}_{stage}.json", boxes_to_rows(boxes))
                        write_assignments(assignments_dir / f"{run_id}_{stage}.csv", surrogate_eval.assignments)
                        print(
                            f"{run_id} {stage}: "
                            f"uncovered={surrogate_eval.uncovered_orders} "
                            f"low_margin={surrogate_eval.low_margin_assignments} "
                            f"mean_cost={surrogate_eval.total_base_cost / len(orders):.3f}",
                            flush=True,
                        )

                    def report_progress(progress, *, run_id=run_id):
                        print(
                            f"{run_id} repair_iter={progress['iteration']} "
                            f"step={progress['step']} "
                            f"moves={progress['candidate_moves']} "
                            f"evals={progress['evaluated_candidates']} "
                            f"uncovered={progress['uncovered_orders']} "
                            f"low_margin={progress['low_margin_assignments']} "
                            f"elapsed={progress['elapsed_seconds']:.1f}s "
                            f"accepted={int(progress['accepted'])}",
                            flush=True,
                        )

                    repair = surrogate_repair_search(
                        orders,
                        boxes2,
                        evaluator,
                        step_schedule=tuple(args.step_schedule),
                        max_iters=args.repair_max_iters,
                        max_repair_orders=args.max_repair_orders,
                        include_coordinate_moves=not args.repair_only,
                        max_candidates_per_iter=args.max_candidates_per_iter if args.max_candidates_per_iter > 0 else None,
                        progress_callback=report_progress,
                    )
                    simple_repair = evaluate_simple_box_set(orders, repair.boxes)
                    row = add_summary_row(
                        rows,
                        run_id=run_id,
                        stage="surrogate_repair_search",
                        k=k,
                        seed=seed,
                        tau=tau,
                        orders_count=len(orders),
                        boxes=repair.boxes,
                        surrogate_eval=repair.evaluation,
                        simple_score=simple_repair,
                        iterations=repair.iterations,
                        evaluated_candidates=repair.evaluated_candidates,
                    )
                    append_summary(summary_path, row)
                    write_json(boxes_dir / f"{run_id}_surrogate_repair_search.json", boxes_to_rows(repair.boxes))
                    write_assignments(assignments_dir / f"{run_id}_surrogate_repair_search.csv", repair.evaluation.assignments)
                    write_json(run_dir / f"{run_id}_surrogate_repair_manifest.json", repair.manifest)
                    print(
                        f"{run_id}: "
                        f"beam_uncovered={rows[-2]['uncovered_orders']} "
                        f"repair_uncovered={repair.evaluation.uncovered_orders} "
                        f"repair_mean_cost={repair.evaluation.total_base_cost / len(orders):.3f} "
                        f"evals={repair.evaluated_candidates}",
                        flush=True,
                    )

        manifest["status"] = "complete"
        write_json(run_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        write_json(run_dir / "manifest.json", manifest)
        raise

    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()
