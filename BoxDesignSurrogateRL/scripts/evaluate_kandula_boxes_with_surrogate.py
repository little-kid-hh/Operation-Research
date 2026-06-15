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

from box_design_surrogate import Box, SurrogateEvaluator, read_order_summaries  # noqa: E402
from src.ensemble_train import load_ensemble_pipeline  # noqa: E402


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def read_boxes_json(path: Path) -> list[Box]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Box(
            box_id=int(row["box_id"]),
            length=float(row["length"]),
            width=float(row["width"]),
            height=float(row["height"]),
        )
        for row in data
    ]


def parse_box_file_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    for suffix in ("_local_search", "_beam_tree", "_kmeans"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)], suffix[1:]
    return stem, "unknown"


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        required=True,
        help="Kandula-style run directory containing boxes/*.json.",
    )
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
    parser.add_argument("--orders-limit", type=int, default=2000)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--tau-high", type=float, default=0.99)
    parser.add_argument("--lambda-risk", type=float, default=1000.0)
    parser.add_argument("--uncovered-penalty", type=float, default=1_000_000.0)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results" / "surrogate_eval",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": timestamp,
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short", "BoxDesignSurrogateRL"]),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": {
            "source_run_dir": str(args.source_run_dir),
            "xml_path": str(args.xml_path),
            "model_path": str(args.model_path),
            "orders_limit": args.orders_limit,
            "tau": args.tau,
            "tau_high": args.tau_high,
            "lambda_risk": args.lambda_risk,
            "uncovered_penalty": args.uncovered_penalty,
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    orders = read_order_summaries(args.xml_path)
    if args.orders_limit:
        orders = orders[: args.orders_limit]
    model = load_ensemble_pipeline(args.model_path)
    evaluator = SurrogateEvaluator(
        model=model,
        tau=args.tau,
        tau_high=args.tau_high,
        lambda_risk=args.lambda_risk,
        uncovered_penalty=args.uncovered_penalty,
    )

    rows = []
    box_paths = sorted((args.source_run_dir / "boxes").glob("*.json"))
    if not box_paths:
        raise FileNotFoundError(f"No box JSON files found under {args.source_run_dir / 'boxes'}")

    for box_path in box_paths:
        run_id, stage = parse_box_file_name(box_path)
        boxes = read_boxes_json(box_path)
        result = evaluator.evaluate(orders, boxes)
        rows.append(
            {
                "source_run_dir": str(args.source_run_dir),
                "box_file": str(box_path),
                "run_id": run_id,
                "stage": stage,
                "orders": len(orders),
                "boxes": len(boxes),
                "tau": args.tau,
                "coverage_rate": result.coverage_rate,
                "uncovered_orders": result.uncovered_orders,
                "mean_base_cost": result.total_base_cost / len(orders),
                "mean_adjusted_cost": result.total_adjusted_cost / len(orders),
                "mean_assigned_probability": result.mean_assigned_probability,
                "low_margin_assignments": result.low_margin_assignments,
            }
        )
        print(
            f"{run_id} {stage}: "
            f"coverage={result.coverage_rate:.4f} "
            f"uncovered={result.uncovered_orders} "
            f"mean_cost={result.total_base_cost / len(orders):.3f} "
            f"mean_p={result.mean_assigned_probability:.4f}"
        )

    fieldnames = [
        "source_run_dir",
        "box_file",
        "run_id",
        "stage",
        "orders",
        "boxes",
        "tau",
        "coverage_rate",
        "uncovered_orders",
        "mean_base_cost",
        "mean_adjusted_cost",
        "mean_assigned_probability",
        "low_margin_assignments",
    ]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()

