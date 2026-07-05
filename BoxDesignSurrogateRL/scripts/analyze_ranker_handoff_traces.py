#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


TRACE_NUMERIC_FIELDS = [
    "packaging_factor",
    "milp_validated_candidates",
    "milp_eval_seconds",
    "ranker_eval_seconds",
    "ranker_tiers_evaluated",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def read_frontier_row(path: Path, row_index: int = 0) -> dict[str, Any]:
    payload = load_json(path)
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"frontier summary must contain a row list: {path}")
    if not 0 <= row_index < len(rows):
        raise ValueError(f"row index {row_index} out of range for: {path}")
    row = rows[row_index]
    if not isinstance(row, dict):
        raise ValueError(f"frontier summary row must be an object: {path}")
    return row


def read_ranker_trace(ranker_run_dir: Path) -> list[dict[str, Any]]:
    trace_path = ranker_run_dir / "trace.csv"
    with trace_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in TRACE_NUMERIC_FIELDS:
            if field in row:
                value = numeric(row[field])
                row[field] = value if value is not None else row[field]
    return rows


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def sum_numeric(rows: list[dict[str, Any]], field: str) -> float:
    return sum(value for row in rows for value in [numeric(row.get(field))] if value is not None)


def ranker_trace_summary(ranker_run_dir: Path, *, tail_iterations: int) -> dict[str, Any]:
    trace = read_ranker_trace(ranker_run_dir)
    ranker_rows = [row for row in trace if row.get("phase") == "ranker_filtered_greedy"]
    improved_rows = [row for row in ranker_rows if is_true(row.get("improved"))]
    pf_values = [value for row in ranker_rows for value in [numeric(row.get("packaging_factor"))] if value is not None]
    tail = ranker_rows[-tail_iterations:] if tail_iterations > 0 else []
    tail_pf_values = [value for row in tail for value in [numeric(row.get("packaging_factor"))] if value is not None]
    last_improvement_iteration = None
    if improved_rows:
        last_improvement_iteration = int(float(improved_rows[-1].get("iteration", 0)))
    last_iteration = int(float(ranker_rows[-1].get("iteration", 0))) if ranker_rows else 0
    return {
        "ranker_trace_path": str(ranker_run_dir / "trace.csv"),
        "ranker_iterations": len(ranker_rows),
        "ranker_improvements": len(improved_rows),
        "last_improvement_iteration": last_improvement_iteration,
        "iterations_since_last_improvement": (
            last_iteration - last_improvement_iteration if last_improvement_iteration is not None else None
        ),
        "trace_start_pf": pf_values[0] if pf_values else None,
        "trace_final_pf": pf_values[-1] if pf_values else None,
        "trace_pf_improvement": (pf_values[0] - pf_values[-1]) if len(pf_values) >= 2 else None,
        "tail_iterations": len(tail),
        "tail_pf_improvement": (
            tail_pf_values[0] - tail_pf_values[-1] if len(tail_pf_values) >= 2 else None
        ),
        "tail_validations": sum_numeric(tail, "milp_validated_candidates"),
        "tail_milp_eval_seconds": sum_numeric(tail, "milp_eval_seconds"),
        "ranker_trace_validations": sum_numeric(ranker_rows, "milp_validated_candidates"),
        "ranker_trace_milp_eval_seconds": sum_numeric(ranker_rows, "milp_eval_seconds"),
    }


def percent_reduction(exact: float | int | None, candidate: float | int | None) -> float | None:
    if exact is None or candidate is None:
        return None
    exact_f = float(exact)
    if exact_f == 0.0:
        return None
    return (exact_f - float(candidate)) / exact_f * 100.0


def labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path_s = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("frontier label cannot be empty")
    return label, Path(path_s)


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    if path.exists():
        return path
    return path


def load_manifest_frontiers(path: Path) -> list[tuple[str, Path]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"manifest must be a JSON list: {path}")
    base_dir = path.parent
    frontiers: list[tuple[str, Path]] = []
    for idx, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"manifest entry {idx} must be an object: {path}")
        if "label" not in record or "frontier_summary" not in record:
            raise ValueError(f"manifest entry {idx} needs label and frontier_summary: {path}")
        label = str(record["label"])
        frontier_path = resolve_path(base_dir, str(record["frontier_summary"]))
        frontiers.append((label, frontier_path))
    return frontiers


def analyze_frontier(label: str, path: Path, *, tail_iterations: int) -> dict[str, Any]:
    row = read_frontier_row(path)
    ranker_run_dir = Path(str(row["ranker_run_dir"]))
    trace = ranker_trace_summary(ranker_run_dir, tail_iterations=tail_iterations)
    baseline_pf = numeric(row.get("baseline_pf"))
    audit_pf = numeric(row.get("audit_pf"))
    exact_uncached = numeric(row.get("baseline_oracle_uncached_boxes"))
    combined_uncached = numeric(row.get("combined_oracle_uncached_boxes"))
    exact_subprocess = numeric(row.get("baseline_oracle_subprocess_seconds"))
    combined_subprocess = numeric(row.get("combined_oracle_subprocess_seconds"))
    exact_elapsed = numeric(row.get("baseline_elapsed_seconds"))
    combined_elapsed = numeric(row.get("combined_elapsed_seconds"))
    exact_validations = numeric(row.get("baseline_milp_validated_candidates"))
    combined_validations = numeric(row.get("combined_milp_validated_candidates"))
    return {
        "label": label,
        "frontier_summary": str(path),
        "ranker_max_elapsed_seconds": row.get("ranker_max_elapsed_seconds"),
        "ranker_stop_reason": row.get("ranker_stop_reason"),
        "ranker_pf": row.get("ranker_pf"),
        "audit_pf": row.get("audit_pf"),
        "audit_pf_gap_vs_baseline": (
            audit_pf - baseline_pf if audit_pf is not None and baseline_pf is not None else None
        ),
        "audit_coverage": row.get("audit_coverage"),
        "audit_uncovered": row.get("audit_uncovered"),
        "ranker_validations": row.get("ranker_milp_validated_candidates"),
        "audit_validations": row.get("audit_milp_validated_candidates"),
        "combined_validations": row.get("combined_milp_validated_candidates"),
        "combined_uncached_boxes": row.get("combined_oracle_uncached_boxes"),
        "combined_subprocess_seconds": row.get("combined_oracle_subprocess_seconds"),
        "combined_elapsed_seconds": row.get("combined_elapsed_seconds"),
        "uncached_reduction_pct": percent_reduction(exact_uncached, combined_uncached),
        "subprocess_reduction_pct": percent_reduction(exact_subprocess, combined_subprocess),
        "elapsed_reduction_pct": percent_reduction(exact_elapsed, combined_elapsed),
        "validation_reduction_pct": percent_reduction(exact_validations, combined_validations),
        **trace,
    }


def fmt(value: Any, precision: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "label",
        "ranker cap",
        "ranker PF",
        "audit PF",
        "PF delta",
        "validations",
        "uncached",
        "subprocess s",
        "elapsed s",
        "ranker iters",
        "last improvement",
        "tail PF improvement",
        "tail validations",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = [
            row.get("label"),
            row.get("ranker_max_elapsed_seconds"),
            fmt(row.get("ranker_pf"), 10),
            fmt(row.get("audit_pf"), 10),
            fmt(row.get("audit_pf_gap_vs_baseline"), 10),
            row.get("combined_validations"),
            row.get("combined_uncached_boxes"),
            fmt(row.get("combined_subprocess_seconds"), 4),
            fmt(row.get("combined_elapsed_seconds"), 4),
            row.get("ranker_iterations"),
            row.get("last_improvement_iteration"),
            fmt(row.get("tail_pf_improvement"), 10),
            fmt(row.get("tail_validations"), 0),
        ]
        lines.append("| " + " | ".join(fmt(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(frontier_summaries: list[tuple[str, Path]], *, tail_iterations: int) -> dict[str, Any]:
    rows = [
        analyze_frontier(label, path, tail_iterations=tail_iterations)
        for label, path in frontier_summaries
    ]
    return {"tail_iterations": tail_iterations, "frontiers": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ranker/audit handoff traces from query-budgeted frontier summaries."
    )
    parser.add_argument("--frontier-summary", action="append", type=labeled_path, default=None)
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=None,
        help="Ranker window manifest containing label/frontier_summary records.",
    )
    parser.add_argument("--tail-iterations", type=int, default=10)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    if args.tail_iterations < 0:
        raise ValueError("--tail-iterations must be non-negative")
    frontier_summaries = list(args.frontier_summary or [])
    for manifest_path in args.manifest or []:
        frontier_summaries.extend(load_manifest_frontiers(manifest_path))
    if not frontier_summaries:
        raise ValueError("at least one --frontier-summary or --manifest is required")
    analysis = analyze(frontier_summaries, tail_iterations=args.tail_iterations)
    if args.out_json is not None:
        write_json(args.out_json, analysis)
    if args.out_md is not None:
        write_markdown(args.out_md, analysis["frontiers"])
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
