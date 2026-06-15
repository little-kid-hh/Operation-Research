from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
HYBRID_ROOT = REPO_ROOT / "HybridSVM"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from box_design_surrogate import SurrogateEvaluator, read_order_summaries  # noqa: E402
from box_design_surrogate.kandula_repro import initial_boxes_kmeans  # noqa: E402
from box_design_surrogate.surrogate_search import surrogate_coordinate_descent  # noqa: E402
from src.ensemble_train import load_ensemble_pipeline  # noqa: E402


def boxes_to_jsonable(boxes):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-limit", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.95)
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
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "results" / "surrogate_local_search_smoke.json",
    )
    args = parser.parse_args()

    orders = read_order_summaries(args.xml_path)
    if args.orders_limit:
        orders = orders[: args.orders_limit]
    boxes0 = initial_boxes_kmeans(orders, args.k, random_state=args.seed)
    model = load_ensemble_pipeline(args.model_path)
    evaluator = SurrogateEvaluator(model=model, tau=args.tau)

    eval0 = evaluator.evaluate(orders, boxes0)
    result = surrogate_coordinate_descent(
        orders,
        boxes0,
        evaluator,
        step=args.step,
        max_iters=args.max_iters,
    )

    payload = {
        "parameters": {
            "orders_limit": args.orders_limit,
            "k": args.k,
            "seed": args.seed,
            "step": args.step,
            "max_iters": args.max_iters,
            "tau": args.tau,
            "model_path": str(args.model_path),
        },
        "initial": {
            "coverage_rate": eval0.coverage_rate,
            "uncovered_orders": eval0.uncovered_orders,
            "mean_base_cost": eval0.total_base_cost / len(orders),
            "mean_adjusted_cost": eval0.total_adjusted_cost / len(orders),
            "mean_assigned_probability": eval0.mean_assigned_probability,
            "low_margin_assignments": eval0.low_margin_assignments,
            "boxes": boxes_to_jsonable(boxes0),
        },
        "surrogate_local_search": {
            "coverage_rate": result.evaluation.coverage_rate,
            "uncovered_orders": result.evaluation.uncovered_orders,
            "mean_base_cost": result.evaluation.total_base_cost / len(orders),
            "mean_adjusted_cost": result.evaluation.total_adjusted_cost / len(orders),
            "mean_assigned_probability": result.evaluation.mean_assigned_probability,
            "low_margin_assignments": result.evaluation.low_margin_assignments,
            "iterations": result.iterations,
            "evaluated_candidates": result.evaluated_candidates,
            "boxes": boxes_to_jsonable(result.boxes),
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        "initial "
        f"coverage={eval0.coverage_rate:.4f} "
        f"uncovered={eval0.uncovered_orders} "
        f"mean_cost={eval0.total_base_cost / len(orders):.3f}"
    )
    print(
        "surrogate_local_search "
        f"coverage={result.evaluation.coverage_rate:.4f} "
        f"uncovered={result.evaluation.uncovered_orders} "
        f"mean_cost={result.evaluation.total_base_cost / len(orders):.3f} "
        f"iters={result.iterations} "
        f"evals={result.evaluated_candidates}"
    )
    print(f"saved={args.output_path}")


if __name__ == "__main__":
    main()

