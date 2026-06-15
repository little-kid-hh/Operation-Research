from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from box_design_surrogate import read_order_summaries  # noqa: E402


DEFAULT_SOURCE_RUN = ROOT / "results" / "surrogate_design" / "run_20260613_211826"
DEFAULT_XML = ROOT / "assets" / "or2023_bsp_data" / "xml_unique" / "or2023_bsp_unique_orders.xml"
FIELDNAMES = [
    "candidate_id",
    "source_run_dir",
    "run_id",
    "tau",
    "k",
    "seed",
    "stage",
    "baseline_stage",
    "order_key",
    "order_id",
    "box_id",
    "probability",
    "best_box_id",
    "best_probability",
    "low_margin",
    "source_type",
    "box_length",
    "box_width",
    "box_height",
    "box_volume",
    "order_item_count",
    "order_total_volume",
    "order_max_length",
    "order_max_width",
    "order_max_height",
]


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def covered(row: dict[str, str]) -> bool:
    return row.get("covered") in {"1", "true", "True", "TRUE"}


def truthy(row: dict[str, str], key: str) -> bool:
    return row.get(key) in {"1", "true", "True", "TRUE"}


def parse_run_id(run_id: str) -> tuple[int, int, float]:
    parts = run_id.split("_")
    k = int(parts[0].removeprefix("k"))
    seed = int(parts[1].removeprefix("seed"))
    tau_token = parts[2].removeprefix("tau").replace("p", ".")
    return k, seed, float(tau_token)


def split_assignment_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    for suffix in (
        "_surrogate_repair_search",
        "_local_search",
        "_beam_tree",
        "_kmeans",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)], suffix.removeprefix("_")
    raise ValueError(f"unrecognized assignment filename: {path}")


def load_boxes(path: Path) -> dict[int, dict[str, float]]:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    return {int(row["box_id"]): row for row in rows}


def order_lookup(xml_path: Path, limit: int | None) -> dict[str, object]:
    orders = read_order_summaries(xml_path)
    if limit:
        orders = orders[:limit]
    return {f"{order.instance_name}:{order.order_id}": order for order in orders}


def make_candidate(
    *,
    source_run_dir: Path,
    run_id: str,
    stage: str,
    baseline_stage: str,
    row: dict[str, str],
    boxes: dict[int, dict[str, float]],
    orders: dict[str, object],
    source_type: str,
) -> dict[str, object]:
    k, seed, tau = parse_run_id(run_id)
    box_id = int(row["box_id"])
    box = boxes[box_id]
    order_key = row["order_key"]
    if order_key not in orders:
        raise KeyError(f"order_key not found in source XML: {order_key}")
    order = orders[order_key]
    return {
        "source_run_dir": str(source_run_dir),
        "run_id": run_id,
        "tau": tau,
        "k": k,
        "seed": seed,
        "stage": stage,
        "baseline_stage": baseline_stage,
        "order_key": order_key,
        "order_id": order.order_id,
        "box_id": box_id,
        "probability": float(row["probability"]),
        "best_box_id": row.get("best_box_id", ""),
        "best_probability": row.get("best_probability", ""),
        "low_margin": int(truthy(row, "low_margin")),
        "source_type": source_type,
        "box_length": float(box["length"]),
        "box_width": float(box["width"]),
        "box_height": float(box["height"]),
        "box_volume": float(box["volume"]),
        "order_item_count": order.item_count,
        "order_total_volume": order.total_volume,
        "order_max_length": max(order.dim_l),
        "order_max_width": max(order.dim_m),
        "order_max_height": max(order.dim_s),
    }


def sample_rows(rows: list[dict[str, str]], count: int, rng: random.Random) -> list[dict[str, str]]:
    if count <= 0 or len(rows) <= count:
        return list(rows)
    return rng.sample(rows, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--orders-limit", type=int, default=None)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.95])
    parser.add_argument("--sampling-seed", type=int, default=20260613)
    parser.add_argument("--low-margin-per-run", type=int, default=50)
    parser.add_argument("--random-assigned-per-run", type=int, default=25)
    parser.add_argument("--max-repaired-newly-covered-per-run", type=int, default=0)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    source_run = args.source_run_dir.resolve()
    out_dir = args.out_dir or source_run / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.sampling_seed)
    orders = order_lookup(args.xml_path, args.orders_limit)
    tau_filter = {round(tau, 8) for tau in args.taus}

    assignments_by_run_stage: dict[tuple[str, str], Path] = {}
    for path in sorted((source_run / "assignments").glob("*.csv")):
        run_id, stage = split_assignment_name(path)
        _, _, tau = parse_run_id(run_id)
        if round(tau, 8) in tau_filter:
            assignments_by_run_stage[(run_id, stage)] = path

    candidates: list[dict[str, object]] = []
    counts = defaultdict(int)
    for (run_id, stage), repair_path in sorted(assignments_by_run_stage.items()):
        if stage != "surrogate_repair_search":
            continue
        baseline_stage = "beam_tree"
        baseline_path = assignments_by_run_stage.get((run_id, baseline_stage))
        if baseline_path is None:
            baseline_stage = "local_search"
            baseline_path = assignments_by_run_stage.get((run_id, baseline_stage))
        if baseline_path is None:
            raise FileNotFoundError(f"missing baseline assignment for {run_id}")

        boxes = load_boxes(source_run / "boxes" / f"{run_id}_{stage}.json")
        repair_rows = [row for row in read_csv_rows(repair_path) if covered(row)]
        baseline_by_order = {row["order_key"]: row for row in read_csv_rows(baseline_path)}

        newly_covered = [
            row
            for row in repair_rows
            if not covered(baseline_by_order.get(row["order_key"], {}))
        ]
        low_margin = [row for row in repair_rows if truthy(row, "low_margin")]
        random_assigned = list(repair_rows)

        source_groups = [
            ("repaired_newly_covered", sample_rows(newly_covered, args.max_repaired_newly_covered_per_run, rng)),
            ("low_margin", sample_rows(low_margin, args.low_margin_per_run, rng)),
            ("random_assigned", sample_rows(random_assigned, args.random_assigned_per_run, rng)),
        ]
        for source_type, rows in source_groups:
            for row in rows:
                candidate = make_candidate(
                    source_run_dir=source_run,
                    run_id=run_id,
                    stage=stage,
                    baseline_stage=baseline_stage,
                    row=row,
                    boxes=boxes,
                    orders=orders,
                    source_type=source_type,
                )
                counts[(run_id, source_type)] += 1
                candidates.append(candidate)

    candidates.sort(key=lambda r: (r["run_id"], r["source_type"], r["order_key"], int(r["box_id"])))
    if args.max_total and len(candidates) > args.max_total:
        candidates = rng.sample(candidates, args.max_total)
        candidates.sort(key=lambda r: (r["run_id"], r["source_type"], r["order_key"], int(r["box_id"])))

    for idx, candidate in enumerate(candidates):
        candidate["candidate_id"] = f"vc_{idx:06d}"

    candidates_path = out_dir / "verification_candidates.csv"
    with candidates_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(candidates)

    manifest = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "source_run_dir": str(source_run),
        "output": str(candidates_path),
        "candidate_count": len(candidates),
        "sampling_seed": args.sampling_seed,
        "sample_sizes": {
            "low_margin_per_run": args.low_margin_per_run,
            "random_assigned_per_run": args.random_assigned_per_run,
            "max_repaired_newly_covered_per_run": args.max_repaired_newly_covered_per_run,
            "max_total": args.max_total,
        },
        "taus": args.taus,
        "orders_xml": str(args.xml_path),
        "orders_limit": args.orders_limit,
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short", "BoxDesignSurrogateRL"]),
    }
    (out_dir / "verification_candidates_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {len(candidates)} candidates to {candidates_path}")


if __name__ == "__main__":
    main()
