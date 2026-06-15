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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from box_design_surrogate.evaluator import Box  # noqa: E402
from box_design_surrogate.kandula_repro import (  # noqa: E402
    SimpleBoxSetScore,
    beam_tree_search,
    evaluate_simple_box_set,
    greedy_coordinate_descent,
    initial_boxes_kmeans,
    read_default_orders,
)


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


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


def score_row(
    *,
    run_id: str,
    stage: str,
    k: int,
    seed: int,
    orders: int,
    step: float,
    score: SimpleBoxSetScore,
) -> dict[str, str | int | float]:
    return {
        "run_id": run_id,
        "stage": stage,
        "k": k,
        "seed": seed,
        "orders": orders,
        "step": step,
        **score.to_dict(),
    }


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-limit", type=int, default=2000)
    parser.add_argument("--ks", type=int, nargs="+", default=[10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--beam-depth", type=int, default=20)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results" / "kandula_repro",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / f"run_{timestamp}"
    boxes_dir = run_dir / "boxes"
    boxes_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": timestamp,
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short", "BoxDesignSurrogateRL"]),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": {
            "orders_limit": args.orders_limit,
            "ks": args.ks,
            "seeds": args.seeds,
            "step": args.step,
            "max_iters": args.max_iters,
            "beam_depth": args.beam_depth,
            "beam_width": args.beam_width,
        },
        "data": {
            "orders_xml": "BoxDesignSurrogateRL/assets/or2023_bsp_data/xml_unique/or2023_bsp_unique_orders.xml",
            "notes": "Kandula-style framework reproduction on local OR2023 BSP unique-order data.",
        },
    }
    write_json(run_dir / "manifest.json", manifest)

    orders = read_default_orders(args.orders_limit)
    rows: list[dict[str, str | int | float]] = []

    for k in args.ks:
        for seed in args.seeds:
            run_id = f"k{k}_seed{seed}"
            boxes0 = initial_boxes_kmeans(orders, k, random_state=seed)
            score0 = evaluate_simple_box_set(orders, boxes0)
            rows.append(
                score_row(
                    run_id=run_id,
                    stage="kmeans",
                    k=k,
                    seed=seed,
                    orders=len(orders),
                    step=args.step,
                    score=score0,
                )
            )
            write_json(boxes_dir / f"{run_id}_kmeans.json", boxes_to_rows(boxes0))

            boxes1, score1 = greedy_coordinate_descent(
                orders,
                boxes0,
                step=args.step,
                max_iters=args.max_iters,
            )
            rows.append(
                score_row(
                    run_id=run_id,
                    stage="local_search",
                    k=k,
                    seed=seed,
                    orders=len(orders),
                    step=args.step,
                    score=score1,
                )
            )
            write_json(boxes_dir / f"{run_id}_local_search.json", boxes_to_rows(boxes1))

            boxes2, score2 = beam_tree_search(
                orders,
                boxes0,
                step=args.step,
                depth=args.beam_depth,
                beam_width=args.beam_width,
            )
            rows.append(
                score_row(
                    run_id=run_id,
                    stage="beam_tree",
                    k=k,
                    seed=seed,
                    orders=len(orders),
                    step=args.step,
                    score=score2,
                )
            )
            write_json(boxes_dir / f"{run_id}_beam_tree.json", boxes_to_rows(boxes2))
            print(
                f"{run_id}: "
                f"kmeans={score0.packaging_factor:.6f} "
                f"local={score1.packaging_factor:.6f} "
                f"beam={score2.packaging_factor:.6f}"
            )

    fieldnames = [
        "run_id",
        "stage",
        "k",
        "seed",
        "orders",
        "step",
        "packaging_factor",
        "mean_box_volume",
        "mean_order_volume",
        "coverage_rate",
        "uncovered_orders",
    ]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()

