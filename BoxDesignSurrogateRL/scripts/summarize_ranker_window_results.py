#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


PRIMARY_FIELDS = [
    "milp_validated_candidates",
    "oracle_uncached_boxes",
    "oracle_subprocess_seconds",
    "elapsed_seconds",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("manifest must be a JSON list of window records")
    records: list[dict[str, Any]] = []
    for idx, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"manifest entry {idx} must be an object")
        for field in ("label", "exact_summary", "frontier_summary"):
            if field not in record:
                raise ValueError(f"manifest entry {idx} is missing required field: {field}")
        records.append(record)
    return records


def load_frontier_row(path: Path, row_index: int = 0) -> dict[str, Any]:
    payload = load_json(path)
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"frontier summary does not contain a row list: {path}")
    if not 0 <= row_index < len(rows):
        raise ValueError(f"frontier row_index={row_index} out of range for {path}")
    row = rows[row_index]
    if not isinstance(row, dict):
        raise ValueError(f"frontier row {row_index} must be an object: {path}")
    return row


def score_value(summary: dict[str, Any], field: str) -> float | int | None:
    score = summary.get("best_score") or {}
    value = score.get(field)
    return value if isinstance(value, int | float) else None


def cache_value(summary: dict[str, Any], field: str) -> float | int | None:
    cache = summary.get("oracle_cache") or {}
    value = cache.get(field)
    return value if isinstance(value, int | float) else None


def exact_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "pf": score_value(summary, "packaging_factor"),
        "coverage": score_value(summary, "coverage_rate"),
        "uncovered": score_value(summary, "uncovered_orders"),
        "milp_validated_candidates": summary.get("milp_validated_candidates"),
        "oracle_uncached_boxes": cache_value(summary, "uncached_boxes"),
        "oracle_subprocess_seconds": cache_value(summary, "subprocess_seconds"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "run_dir": summary.get("run_dir"),
    }


def ranker_audit_metrics(frontier_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pf": frontier_row.get("audit_pf"),
        "coverage": frontier_row.get("audit_coverage"),
        "uncovered": frontier_row.get("audit_uncovered"),
        "milp_validated_candidates": frontier_row.get("combined_milp_validated_candidates"),
        "oracle_uncached_boxes": frontier_row.get("combined_oracle_uncached_boxes"),
        "oracle_subprocess_seconds": frontier_row.get("combined_oracle_subprocess_seconds"),
        "elapsed_seconds": frontier_row.get("combined_elapsed_seconds"),
        "run_dir": frontier_row.get("audit_run_dir"),
        "ranker_pf": frontier_row.get("ranker_pf"),
        "ranker_elapsed_seconds": frontier_row.get("ranker_elapsed_seconds"),
        "ranker_stop_reason": frontier_row.get("ranker_stop_reason"),
    }


def percent_reduction(exact: float | int | None, ranker: float | int | None) -> float | None:
    if exact is None or ranker is None or exact == 0:
        return None
    return (float(exact) - float(ranker)) / float(exact) * 100.0


def load_window_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_dir = manifest_path.parent
    for record in load_manifest(manifest_path):
        exact_path = resolve_manifest_path(base_dir, Path(str(record["exact_summary"])))
        frontier_path = resolve_manifest_path(base_dir, Path(str(record["frontier_summary"])))
        row_index = int(record.get("frontier_row_index", 0))
        exact_summary = load_json(exact_path)
        frontier_row = load_frontier_row(frontier_path, row_index=row_index)
        exact = exact_metrics(exact_summary)
        ranker = ranker_audit_metrics(frontier_row)
        rows.append(make_window_row(str(record["label"]), exact_path, frontier_path, exact, ranker))
    return rows


def resolve_manifest_path(base_dir: Path, path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def make_window_row(
    label: str,
    exact_path: Path,
    frontier_path: Path,
    exact: dict[str, Any],
    ranker: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "label": label,
        "exact_pf": exact["pf"],
        "ranker_audit_pf": ranker["pf"],
        "pf_delta": difference(ranker["pf"], exact["pf"]),
        "coverage": ranker["coverage"],
        "exact_coverage": exact["coverage"],
        "ranker_audit_coverage": ranker["coverage"],
        "exact_uncovered": exact["uncovered"],
        "ranker_audit_uncovered": ranker["uncovered"],
        "exact_validations": exact["milp_validated_candidates"],
        "ranker_audit_validations": ranker["milp_validated_candidates"],
        "exact_uncached_boxes": exact["oracle_uncached_boxes"],
        "ranker_audit_uncached_boxes": ranker["oracle_uncached_boxes"],
        "exact_subprocess_seconds": exact["oracle_subprocess_seconds"],
        "ranker_audit_subprocess_seconds": ranker["oracle_subprocess_seconds"],
        "exact_elapsed_seconds": exact["elapsed_seconds"],
        "ranker_audit_elapsed_seconds": ranker["elapsed_seconds"],
        "validation_reduction_pct": percent_reduction(
            exact["milp_validated_candidates"], ranker["milp_validated_candidates"]
        ),
        "uncached_box_reduction_pct": percent_reduction(exact["oracle_uncached_boxes"], ranker["oracle_uncached_boxes"]),
        "subprocess_reduction_pct": percent_reduction(
            exact["oracle_subprocess_seconds"], ranker["oracle_subprocess_seconds"]
        ),
        "elapsed_reduction_pct": percent_reduction(exact["elapsed_seconds"], ranker["elapsed_seconds"]),
        "ranker_pf": ranker["ranker_pf"],
        "ranker_elapsed_seconds": ranker["ranker_elapsed_seconds"],
        "ranker_stop_reason": ranker["ranker_stop_reason"],
        "exact_summary_path": str(exact_path),
        "frontier_summary_path": str(frontier_path),
        "exact_run_dir": exact["run_dir"],
        "ranker_audit_run_dir": ranker["run_dir"],
    }
    return row


def difference(candidate: float | int | None, baseline: float | int | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate) - float(baseline)


def make_total_row(rows: list[dict[str, Any]], *, label: str = "total") -> dict[str, Any]:
    total: dict[str, Any] = {
        "label": label,
        "exact_pf": "window-wise",
        "ranker_audit_pf": "window-wise",
        "pf_delta": max_abs_numeric(row.get("pf_delta") for row in rows),
        "coverage": min_numeric(row.get("ranker_audit_coverage") for row in rows),
        "exact_coverage": min_numeric(row.get("exact_coverage") for row in rows),
        "ranker_audit_coverage": min_numeric(row.get("ranker_audit_coverage") for row in rows),
        "exact_uncovered": sum_numeric(row.get("exact_uncovered") for row in rows),
        "ranker_audit_uncovered": sum_numeric(row.get("ranker_audit_uncovered") for row in rows),
        "ranker_pf": None,
        "ranker_elapsed_seconds": sum_numeric(row.get("ranker_elapsed_seconds") for row in rows),
        "ranker_stop_reason": None,
        "exact_summary_path": None,
        "frontier_summary_path": None,
        "exact_run_dir": None,
        "ranker_audit_run_dir": None,
    }
    pairs = [
        ("milp_validated_candidates", "validations"),
        ("oracle_uncached_boxes", "uncached_boxes"),
        ("oracle_subprocess_seconds", "subprocess_seconds"),
        ("elapsed_seconds", "elapsed_seconds"),
    ]
    for _metric, suffix in pairs:
        exact_key = f"exact_{suffix}"
        ranker_key = f"ranker_audit_{suffix}"
        total[exact_key] = sum_numeric(row.get(exact_key) for row in rows)
        total[ranker_key] = sum_numeric(row.get(ranker_key) for row in rows)

    total["validation_reduction_pct"] = percent_reduction(
        total["exact_validations"], total["ranker_audit_validations"]
    )
    total["uncached_box_reduction_pct"] = percent_reduction(
        total["exact_uncached_boxes"], total["ranker_audit_uncached_boxes"]
    )
    total["subprocess_reduction_pct"] = percent_reduction(
        total["exact_subprocess_seconds"], total["ranker_audit_subprocess_seconds"]
    )
    total["elapsed_reduction_pct"] = percent_reduction(total["exact_elapsed_seconds"], total["ranker_audit_elapsed_seconds"])
    return total


def sum_numeric(values: Any) -> float | int | None:
    total = 0.0
    seen = False
    all_int = True
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        total += float(value)
        seen = True
        all_int = all_int and isinstance(value, int)
    if not seen:
        return None
    return int(total) if all_int else total


def min_numeric(values: Any) -> float | int | None:
    nums = [value for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
    return min(nums) if nums else None


def max_abs_numeric(values: Any) -> float | None:
    nums = [abs(float(value)) for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
    return max(nums) if nums else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "window",
        "exact PF",
        "ranker+audit PF",
        "coverage",
        "exact validations",
        "ranker validations",
        "exact uncached",
        "ranker uncached",
        "exact subprocess s",
        "ranker subprocess s",
        "exact elapsed s",
        "ranker elapsed s",
    ]
    keys = [
        "label",
        "exact_pf",
        "ranker_audit_pf",
        "ranker_audit_coverage",
        "exact_validations",
        "ranker_audit_validations",
        "exact_uncached_boxes",
        "ranker_audit_uncached_boxes",
        "exact_subprocess_seconds",
        "ranker_audit_subprocess_seconds",
        "exact_elapsed_seconds",
        "ranker_audit_elapsed_seconds",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"]
    for row in rows:
        values = [format_number(row.get(key), digits=10 if key.endswith("pf") else 4) for key in keys]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def reduction_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "window",
        "PF delta",
        "validation reduction",
        "uncached-box reduction",
        "subprocess reduction",
        "wall-clock reduction",
    ]
    keys = [
        "label",
        "pf_delta",
        "validation_reduction_pct",
        "uncached_box_reduction_pct",
        "subprocess_reduction_pct",
        "elapsed_reduction_pct",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"]
    for row in rows:
        values: list[str] = []
        for key in keys:
            value = row.get(key)
            if key.endswith("_pct"):
                values.append("" if value is None else f"{float(value):.1f}%")
            elif key == "pf_delta":
                values.append(format_number(value, digits=10))
            else:
                values.append(format_number(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize exact staged vs ranker+exact-audit window results from summary JSON files."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSON list with label/exact_summary/frontier_summary")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--total-label", default="total")
    args = parser.parse_args()

    rows = load_window_rows(args.manifest)
    total_row = make_total_row(rows, label=args.total_label)
    all_rows = rows + [total_row]
    payload = {"windows": rows, "total": total_row}
    if args.out_json is not None:
        write_json(args.out_json, payload)
    if args.out_csv is not None:
        write_csv(args.out_csv, all_rows)

    markdown = markdown_table(all_rows) + "\n\n" + reduction_markdown_table(all_rows) + "\n"
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)


if __name__ == "__main__":
    main()
