#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path_s = value.split("=", 1)
    label = label.strip()
    path_s = path_s.strip()
    if not label or not path_s:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
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
    out: list[tuple[str, Path]] = []
    for idx, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"manifest entry {idx} must be an object: {path}")
        if "label" not in record or "frontier_summary" not in record:
            raise ValueError(f"manifest entry {idx} needs label and frontier_summary: {path}")
        out.append((str(record["label"]), resolve_path(path.parent, str(record["frontier_summary"]))))
    return out


def read_frontier_row(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"frontier summary must be a non-empty list: {path}")
    row = payload[0]
    if not isinstance(row, dict):
        raise ValueError(f"frontier summary row must be an object: {path}")
    return row


def read_ranker_rows(trace_path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    with trace_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"trace is empty: {trace_path}")
    initial = rows[0]
    ranker_rows = [
        row
        for row in rows
        if row.get("phase") == "ranker_filtered_greedy"
        and row.get("action") not in {"init", "time_limit", "adaptive_handoff"}
        and row.get("stop_reason") in {None, ""}
    ]
    return initial, ranker_rows


def as_float(row: dict[str, str], field: str, default: float = 0.0) -> float:
    value = row.get(field)
    if value in {None, ""}:
        return default
    return float(value)


def as_int(row: dict[str, str], field: str, default: int = 0) -> int:
    value = row.get(field)
    if value in {None, ""}:
        return default
    return int(float(value))


def simulate_threshold(
    *,
    initial: dict[str, str],
    rows: list[dict[str, str]],
    threshold: float,
    window: int,
    min_iterations: int,
) -> dict[str, Any]:
    for end_idx, end_row in enumerate(rows):
        observed_iterations = end_idx + 1
        if observed_iterations < min_iterations or observed_iterations < window:
            continue
        if as_int(end_row, "uncovered_orders") > 0 or as_int(end_row, "unknown_pairs") > 0:
            continue
        start_idx = observed_iterations - window
        recent_rows = rows[start_idx : end_idx + 1]
        if start_idx == 0:
            start_pf = as_float(initial, "packaging_factor")
        else:
            start_pf = as_float(rows[start_idx - 1], "packaging_factor")
        end_pf = as_float(end_row, "packaging_factor")
        validations = sum(as_int(row, "milp_validated_candidates") for row in recent_rows)
        if validations <= 0:
            continue
        pf_improvement = max(0.0, start_pf - end_pf)
        pf_per_validation = pf_improvement / float(validations)
        if pf_per_validation < threshold:
            return {
                "triggered": True,
                "iteration": as_int(end_row, "iteration"),
                "ranker_pf_at_trigger": end_pf,
                "recent_pf_improvement": pf_improvement,
                "recent_validations": validations,
                "pf_per_validation": pf_per_validation,
            }
    return {"triggered": False}


def analyze_frontier(
    label: str,
    path: Path,
    *,
    thresholds: list[float],
    window: int,
    min_iterations: int,
) -> dict[str, Any]:
    row = read_frontier_row(path)
    ranker_run_dir = Path(str(row["ranker_run_dir"]))
    trace_path = ranker_run_dir / "trace.csv"
    initial, ranker_rows = read_ranker_rows(trace_path)
    threshold_rows = {
        str(threshold): simulate_threshold(
            initial=initial,
            rows=ranker_rows,
            threshold=threshold,
            window=window,
            min_iterations=min_iterations,
        )
        for threshold in thresholds
    }
    return {
        "label": label,
        "frontier_summary": str(path),
        "ranker_trace": str(trace_path),
        "ranker_stop_reason": row.get("ranker_stop_reason"),
        "ranker_pf": row.get("ranker_pf"),
        "audit_pf": row.get("audit_pf"),
        "audit_pf_gap_vs_baseline": row.get("audit_pf_gap_vs_baseline"),
        "combined_milp_validated_candidates": row.get("combined_milp_validated_candidates"),
        "combined_oracle_uncached_boxes": row.get("combined_oracle_uncached_boxes"),
        "combined_oracle_subprocess_seconds": row.get("combined_oracle_subprocess_seconds"),
        "combined_elapsed_seconds": row.get("combined_elapsed_seconds"),
        "ranker_iterations": len(ranker_rows),
        "thresholds": threshold_rows,
    }


def format_markdown(payload: dict[str, Any]) -> str:
    thresholds = payload["thresholds"]
    lines = [
        "| label | ranker PF | audit PF | stop | ranker iters | threshold | triggered | trigger iter | trigger PF | PF/validation |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for frontier in payload["frontiers"]:
        for threshold in thresholds:
            sim = frontier["thresholds"][str(threshold)]
            lines.append(
                "| {label} | {ranker_pf:.10f} | {audit_pf:.10f} | {stop} | {iters} | {threshold:.8g} | {triggered} | {iteration} | {trigger_pf} | {rate} |".format(
                    label=frontier["label"],
                    ranker_pf=float(frontier["ranker_pf"]),
                    audit_pf=float(frontier["audit_pf"]),
                    stop=frontier["ranker_stop_reason"],
                    iters=frontier["ranker_iterations"],
                    threshold=float(threshold),
                    triggered=str(bool(sim["triggered"])).lower(),
                    iteration=sim.get("iteration", ""),
                    trigger_pf=(
                        f"{float(sim['ranker_pf_at_trigger']):.10f}" if sim.get("triggered") else ""
                    ),
                    rate=(
                        f"{float(sim['pf_per_validation']):.10g}" if sim.get("triggered") else ""
                    ),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate ranker marginal handoff thresholds from trace.csv files.")
    parser.add_argument("--frontier-summary", action="append", type=labeled_path, default=None)
    parser.add_argument("--manifest", action="append", type=Path, default=None)
    parser.add_argument("--threshold", action="append", type=float, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-iterations", type=int, default=20)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    if args.window <= 0:
        raise ValueError("--window must be positive")
    if args.min_iterations < 0:
        raise ValueError("--min-iterations must be non-negative")
    if any(threshold <= 0.0 for threshold in args.threshold):
        raise ValueError("--threshold values must be positive")

    frontier_summaries = list(args.frontier_summary or [])
    for manifest_path in args.manifest or []:
        frontier_summaries.extend(load_manifest_frontiers(manifest_path))
    if not frontier_summaries:
        raise ValueError("at least one --frontier-summary or --manifest is required")

    payload = {
        "thresholds": args.threshold,
        "window": args.window,
        "min_iterations": args.min_iterations,
        "frontiers": [
            analyze_frontier(
                label,
                path,
                thresholds=args.threshold,
                window=args.window,
                min_iterations=args.min_iterations,
            )
            for label, path in frontier_summaries
        ],
    }
    if args.out_json is not None:
        write_json(args.out_json, payload)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
