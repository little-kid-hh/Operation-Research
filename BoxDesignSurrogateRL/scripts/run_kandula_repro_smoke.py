from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from box_design_surrogate.kandula_repro import (  # noqa: E402
    beam_tree_search,
    evaluate_simple_box_set,
    greedy_coordinate_descent,
    initial_boxes_kmeans,
    read_default_orders,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-limit", type=int, default=2000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--beam-depth", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    orders = read_default_orders(args.orders_limit)
    boxes0 = initial_boxes_kmeans(orders, args.k, random_state=args.random_state)
    score0 = evaluate_simple_box_set(orders, boxes0)
    boxes1, score1 = greedy_coordinate_descent(
        orders,
        boxes0,
        step=args.step,
        max_iters=args.max_iters,
    )
    boxes2, score2 = beam_tree_search(
        orders,
        boxes0,
        step=args.step,
        depth=args.beam_depth,
        beam_width=args.beam_width,
    )

    print(f"orders={len(orders)} k={args.k} step={args.step} max_iters={args.max_iters}")
    print(
        "stage=kmeans "
        f"pf={score0.packaging_factor:.6f} "
        f"mean_box_volume={score0.mean_box_volume:.3f} "
        f"coverage={score0.coverage_rate:.4f} "
        f"uncovered={score0.uncovered_orders}"
    )
    print(
        "stage=local_search "
        f"pf={score1.packaging_factor:.6f} "
        f"mean_box_volume={score1.mean_box_volume:.3f} "
        f"coverage={score1.coverage_rate:.4f} "
        f"uncovered={score1.uncovered_orders}"
    )
    print(
        "stage=beam_tree "
        f"pf={score2.packaging_factor:.6f} "
        f"mean_box_volume={score2.mean_box_volume:.3f} "
        f"coverage={score2.coverage_rate:.4f} "
        f"uncovered={score2.uncovered_orders}"
    )
    print("boxes:")
    for box in boxes2:
        print(f"{box.box_id}\t{box.length:.3f}\t{box.width:.3f}\t{box.height:.3f}\tvol={box.volume:.3f}")


if __name__ == "__main__":
    main()
