from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
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
DEFAULT_LABELS = ROOT / "assets" / "milp_labels" / "or2023_bsp_unique_package_labels.csv"
RESULT_FIELDNAMES = [
    "candidate_id",
    "source_type",
    "run_id",
    "tau",
    "k",
    "seed",
    "order_key",
    "order_id",
    "box_id",
    "probability",
    "low_margin",
    "verified_feasible",
    "solver_status",
    "runtime_seconds",
    "failure_reason",
    "backend",
    "label_column",
    "label_package_id",
    "box_length",
    "box_width",
    "box_height",
    "order_item_count",
    "order_total_volume",
]
SUMMARY_FIELDNAMES = [
    "group_type",
    "group_value",
    "candidates",
    "verified_feasible",
    "verified_infeasible",
    "unavailable_or_unknown",
    "pass_rate",
    "false_positive_rate",
]


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def order_lookup(xml_path: Path, limit: int | None) -> dict[str, object]:
    orders = read_order_summaries(xml_path)
    if limit:
        orders = orders[:limit]
    return {f"{order.instance_name}:{order.order_id}": order for order in orders}


def normalized_dims(row: dict[str, str], prefix: str) -> tuple[float, float, float]:
    return tuple(sorted((float(row[f"{prefix}_length"]), float(row[f"{prefix}_width"]), float(row[f"{prefix}_height"])), reverse=True))


def label_dims(row: dict[str, str]) -> tuple[float, float, float]:
    return tuple(sorted((float(row["package_l"]), float(row["package_w"]), float(row["package_h"])), reverse=True))


def dims_close(a: tuple[float, float, float], b: tuple[float, float, float], tolerance: float) -> bool:
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


class VerificationBackend:
    name = "unavailable"
    label_column = ""

    def verify(self, candidate: dict[str, str], order) -> dict[str, object]:
        start = time.perf_counter()
        return {
            "verified_feasible": "",
            "solver_status": "unavailable",
            "runtime_seconds": time.perf_counter() - start,
            "failure_reason": "no exact MILP/packing solver backend configured",
            "backend": self.name,
            "label_column": self.label_column,
            "label_package_id": "",
        }


class LabelLookupBackend(VerificationBackend):
    name = "label_lookup"

    def __init__(self, labels_path: Path, label_column: str, dimension_tolerance: float) -> None:
        self.labels_path = labels_path
        self.label_column = label_column
        self.dimension_tolerance = dimension_tolerance
        self.rows_by_order: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in read_csv_rows(labels_path):
            self.rows_by_order[(row["instance_name"], row["order_id"])].append(row)

    def verify(self, candidate: dict[str, str], order) -> dict[str, object]:
        start = time.perf_counter()
        target_dims = normalized_dims(candidate, "box")
        label_rows = self.rows_by_order.get((order.instance_name, order.order_id), [])
        for row in label_rows:
            if not dims_close(target_dims, label_dims(row), self.dimension_tolerance):
                continue
            label = row.get(self.label_column, "")
            if label not in {"0", "1"}:
                status = "label_unknown"
                feasible = ""
                reason = f"matched label row but {self.label_column}={label!r}"
            else:
                status = "verified"
                feasible = int(label)
                reason = "" if feasible else "MILP label marks candidate infeasible"
            return {
                "verified_feasible": feasible,
                "solver_status": status,
                "runtime_seconds": time.perf_counter() - start,
                "failure_reason": reason,
                "backend": self.name,
                "label_column": self.label_column,
                "label_package_id": row.get("package_id", ""),
            }
        return {
            "verified_feasible": "",
            "solver_status": "not_found",
            "runtime_seconds": time.perf_counter() - start,
            "failure_reason": "no existing MILP label matched this order and box dimensions",
            "backend": self.name,
            "label_column": self.label_column,
            "label_package_id": "",
        }


def make_backend(args: argparse.Namespace) -> VerificationBackend:
    if args.backend == "unavailable":
        return VerificationBackend()
    if args.backend == "label_lookup":
        return LabelLookupBackend(args.labels_path, args.label_column, args.dimension_tolerance)
    raise ValueError(f"unsupported backend: {args.backend}")


def summarize(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    out = []
    for value, group in sorted(groups.items()):
        feasible = sum(1 for row in group if str(row["verified_feasible"]) == "1")
        infeasible = sum(1 for row in group if str(row["verified_feasible"]) == "0")
        known = feasible + infeasible
        total = len(group)
        out.append(
            {
                "group_type": key,
                "group_value": value,
                "candidates": total,
                "verified_feasible": feasible,
                "verified_infeasible": infeasible,
                "unavailable_or_unknown": total - known,
                "pass_rate": feasible / known if known else "",
                "false_positive_rate": infeasible / known if known else "",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=DEFAULT_SOURCE_RUN / "verification" / "verification_candidates.csv",
    )
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--orders-limit", type=int, default=None)
    parser.add_argument("--backend", choices=["unavailable", "label_lookup"], default="unavailable")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--label-column", choices=["label_2ori", "label_6ori"], default="label_6ori")
    parser.add_argument("--dimension-tolerance", type=float, default=1e-6)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    candidates_path = args.candidates_path.resolve()
    out_dir = args.out_dir or candidates_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    orders = order_lookup(args.xml_path, args.orders_limit)
    backend = make_backend(args)

    result_rows: list[dict[str, object]] = []
    for candidate in read_csv_rows(candidates_path):
        order = orders.get(candidate["order_key"])
        if order is None:
            raise KeyError(f"order_key not found in source XML: {candidate['order_key']}")
        verification = backend.verify(candidate, order)
        result_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_type": candidate["source_type"],
                "run_id": candidate["run_id"],
                "tau": candidate["tau"],
                "k": candidate["k"],
                "seed": candidate["seed"],
                "order_key": candidate["order_key"],
                "order_id": candidate["order_id"],
                "box_id": candidate["box_id"],
                "probability": candidate["probability"],
                "low_margin": candidate["low_margin"],
                "box_length": candidate["box_length"],
                "box_width": candidate["box_width"],
                "box_height": candidate["box_height"],
                "order_item_count": candidate["order_item_count"],
                "order_total_volume": candidate["order_total_volume"],
                **verification,
            }
        )

    results_path = out_dir / "verification_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(result_rows)

    summary_rows: list[dict[str, object]] = []
    for key in ("source_type", "run_id", "tau", "k", "seed"):
        summary_rows.extend(summarize(result_rows, key))
    summary_path = out_dir / "verification_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "candidates_path": str(candidates_path),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "candidate_count": len(result_rows),
        "backend": args.backend,
        "solver_backend": backend.name,
        "labels_path": str(args.labels_path) if args.backend == "label_lookup" else "",
        "label_column": args.label_column if args.backend == "label_lookup" else "",
        "dimension_tolerance": args.dimension_tolerance,
        "orders_xml": str(args.xml_path),
        "orders_limit": args.orders_limit,
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short", "BoxDesignSurrogateRL"]),
    }
    (out_dir / "verification_results_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    unavailable = sum(1 for row in result_rows if row["solver_status"] in {"unavailable", "not_found"})
    false_positive = sum(1 for row in result_rows if str(row["verified_feasible"]) == "0")
    print(
        f"wrote {len(result_rows)} results to {results_path}; "
        f"false_positive={false_positive}; unavailable_or_not_found={unavailable}"
    )


if __name__ == "__main__":
    main()
