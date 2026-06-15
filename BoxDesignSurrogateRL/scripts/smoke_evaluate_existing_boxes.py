from __future__ import annotations

import argparse
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

from box_design_surrogate import Box, SurrogateEvaluator, read_order_summaries  # noqa: E402
from src.ensemble_train import load_ensemble_pipeline  # noqa: E402


def read_boxes(packages_path: Path, limit: int | None) -> list[Box]:
    boxes: list[Box] = []
    with packages_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            box_id = int(parts[0])
            boxes.append(Box(box_id, float(parts[1]), float(parts[2]), float(parts[3])))
            if limit is not None and len(boxes) >= limit:
                break
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=ROOT / "assets" / "or2023_bsp_data" / "xml_unique" / "or2023_bsp_unique_orders.xml",
    )
    parser.add_argument(
        "--packages-path",
        type=Path,
        default=ROOT / "assets" / "or2023_bsp_data" / "packages.txt",
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
    parser.add_argument("--orders-limit", type=int, default=100)
    parser.add_argument("--boxes-limit", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.95)
    args = parser.parse_args()

    orders = read_order_summaries(args.xml_path)
    if args.orders_limit:
        orders = orders[: args.orders_limit]
    boxes = read_boxes(args.packages_path, args.boxes_limit)
    model = load_ensemble_pipeline(args.model_path)

    evaluator = SurrogateEvaluator(model=model, tau=args.tau)
    result = evaluator.evaluate(orders, boxes)
    print(f"orders={len(orders)} boxes={len(boxes)}")
    print(f"coverage_rate={result.coverage_rate:.4f}")
    print(f"uncovered_orders={result.uncovered_orders}")
    print(f"total_base_cost={result.total_base_cost:.4f}")
    print(f"total_adjusted_cost={result.total_adjusted_cost:.4f}")
    print(f"mean_assigned_probability={result.mean_assigned_probability:.4f}")
    print(f"low_margin_assignments={result.low_margin_assignments}")


if __name__ == "__main__":
    main()

