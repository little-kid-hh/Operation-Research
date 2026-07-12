#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"candidate trace is empty: {path}")
    return rows


def step_key(row: dict[str, str]) -> tuple[str, int, int]:
    return str(row.get("phase", "")), int(float(row["stage"])), int(float(row["iteration"]))


def group_steps(rows: list[dict[str, str]]) -> list[tuple[tuple[str, int, int], list[dict[str, str]]]]:
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    order: list[tuple[str, int, int]] = []
    for row in rows:
        key = step_key(row)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)
    return [(key, grouped[key]) for key in order]


def reconstruct_state_boxes(rows: list[dict[str, str]]) -> list[dict[str, float | int]]:
    by_id: dict[int, dict[str, float | int]] = {}
    for row in rows:
        box_id = int(float(row["move_box_id"]))
        candidate = {
            "box_id": box_id,
            "length": float(row["current_box_length"]),
            "width": float(row["current_box_width"]),
            "height": float(row["current_box_height"]),
        }
        previous = by_id.setdefault(box_id, candidate)
        if previous != candidate:
            raise ValueError(f"inconsistent current dimensions for box_id {box_id}")
    expected = int(float(rows[0]["generated_candidates"])) // 6
    if expected <= 0 or len(by_id) != expected:
        raise ValueError(f"reconstructed {len(by_id)} boxes, expected {expected}")
    return [by_id[box_id] for box_id in sorted(by_id)]


def select_ranked_candidates(rows: list[dict[str, str]], ranks: list[int]) -> list[dict[str, str]]:
    by_rank = {int(float(row["candidate_rank"])): row for row in rows}
    missing = [rank for rank in ranks if rank not in by_rank]
    if missing:
        raise ValueError(f"candidate ranks absent from step: {missing}")
    return [by_rank[rank] for rank in ranks]


def branch_record(
    *,
    branch_id: str,
    state_boxes_path: Path,
    key: tuple[str, int, int],
    row: dict[str, str],
) -> dict[str, Any]:
    phase, stage, iteration = key
    delta = float(row["move_delta"])
    box_id = int(float(row["move_box_id"]))
    dimension = str(row["move_dimension"])
    return {
        "branch_id": branch_id,
        "source_run_id": row.get("source_run_id", ""),
        "phase": phase,
        "stage": stage,
        "iteration": iteration,
        "state_boxes_json": str(state_boxes_path),
        "forced_first_move": f"{box_id}:{dimension}:{delta:.12g}",
        "candidate_rank": int(float(row["candidate_rank"])),
        "candidate_index": int(float(row["candidate_index"])),
        "is_exact_best": int(float(row.get("is_exact_best", "0") or 0)),
        "is_accepted": int(float(row.get("is_accepted", "0") or 0)),
        "current_packaging_factor": float(row["current_packaging_factor"]),
        "one_step_packaging_factor": float(row["candidate_packaging_factor"]),
        "one_step_coverage_rate": float(row["candidate_coverage_rate"]),
        "one_step_uncovered_orders": int(float(row["candidate_uncovered_orders"])),
        "one_step_unknown_pairs": int(float(row["candidate_unknown_pairs"])),
    }


def prepare_branches(
    *,
    trace_path: Path,
    out_dir: Path,
    iterations: list[int],
    candidate_ranks: list[int],
) -> list[dict[str, Any]]:
    rows = read_rows(trace_path)
    selected_steps = [(key, group) for key, group in group_steps(rows) if key[2] in set(iterations)]
    found_iterations = {key[2] for key, _group in selected_steps}
    missing = sorted(set(iterations) - found_iterations)
    if missing:
        raise ValueError(f"iterations absent from candidate trace: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for key, group in selected_steps:
        phase, stage, iteration = key
        state_name = f"state_{phase}_s{stage}_i{iteration}.json"
        state_path = out_dir / state_name
        state_path.write_text(json.dumps(reconstruct_state_boxes(group), indent=2) + "\n", encoding="utf-8")
        for row in select_ranked_candidates(group, candidate_ranks):
            rank = int(float(row["candidate_rank"]))
            branch_id = f"{phase}_s{stage}_i{iteration}_r{rank}"
            records.append(
                branch_record(
                    branch_id=branch_id,
                    state_boxes_path=state_path,
                    key=key,
                    row=row,
                )
            )

    (out_dir / "branches.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    if records:
        with (out_dir / "branches.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare audited counterfactual branch states from an exact candidate trace."
    )
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--iterations", type=parse_int_list, required=True)
    parser.add_argument("--candidate-ranks", type=parse_int_list, default=parse_int_list("1,2,5,10"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    records = prepare_branches(
        trace_path=args.candidate_trace,
        out_dir=args.out_dir,
        iterations=args.iterations,
        candidate_ranks=args.candidate_ranks,
    )
    print(json.dumps({"branches": len(records), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
