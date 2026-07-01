#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def find_summary_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "summary.json":
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("summary.json"))
    return sorted(files)


def load_run(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        summary = json.load(f)
    initial = summary.get("initial_score", {})
    best = summary.get("best_score", {})
    cache = summary.get("oracle_cache") or {}
    git = summary.get("git") or {}
    config_label = summary.get("config_label")
    algorithm = summary.get("algorithm")
    method_label = config_label or algorithm
    return {
        "summary_path": str(path),
        "run_dir": summary.get("run_dir", str(path.parent)),
        "algorithm": algorithm,
        "comparison_label": summary.get("comparison_label"),
        "config_label": config_label,
        "method_label": method_label,
        "code_version": summary.get("code_version"),
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "coverage_repair": summary.get("coverage_repair"),
        "repair_margins": json.dumps(summary.get("repair_margins", []), sort_keys=True),
        "repair_max_rounds": summary.get("repair_max_rounds"),
        "orders_limit": summary.get("orders_limit"),
        "k": summary.get("k"),
        "seed": summary.get("seed"),
        "orientation_label": summary.get("orientation_label"),
        "xml_path": summary.get("xml_path"),
        "milp_time_limit_seconds": summary.get("milp_time_limit_seconds"),
        "fixed_step": summary.get("fixed_step"),
        "schedule": json.dumps(summary.get("schedule", []), sort_keys=True),
        "initial_pf": initial.get("packaging_factor"),
        "best_pf": best.get("packaging_factor"),
        "initial_uncovered": initial.get("uncovered_orders"),
        "best_uncovered": best.get("uncovered_orders"),
        "initial_unknown_pairs": initial.get("unknown_pairs"),
        "best_unknown_pairs": best.get("unknown_pairs"),
        "initial_orders_with_unknown": initial.get("orders_with_unknown"),
        "best_orders_with_unknown": best.get("orders_with_unknown"),
        "mean_box_volume": best.get("mean_box_volume"),
        "mean_order_volume": best.get("mean_order_volume"),
        "coverage_rate": best.get("coverage_rate"),
        "candidate_evaluations": summary.get("candidate_evaluations"),
        "trace_rows": summary.get("trace_rows"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "cache_entries": cache.get("entries"),
        "cache_hits": cache.get("hits"),
        "cache_misses": cache.get("misses"),
        "disk_cache_hits": cache.get("disk_hits"),
    }


def matches_selector(run: dict[str, Any], selector: str) -> bool:
    return selector in {run.get("algorithm"), run.get("config_label"), run.get("method_label")}


def paired_deltas(runs: list[dict[str, Any]], baseline: str, candidate: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for run in runs:
        key = (
            run["comparison_label"],
            run["code_version"],
            run["orders_limit"],
            run["k"],
            run["seed"],
            run["orientation_label"],
            run["xml_path"],
            run["milp_time_limit_seconds"],
            run["coverage_repair"],
            run["repair_margins"],
            run["repair_max_rounds"],
        )
        groups.setdefault(key, []).append(run)

    pairs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        base = [run for run in group if matches_selector(run, baseline)]
        cand = [run for run in group if matches_selector(run, candidate)]
        if len(base) != 1 or len(cand) != 1:
            skipped.append({"key": key, "baseline_count": len(base), "candidate_count": len(cand)})
            continue
        b = base[0]
        c = cand[0]
        pairs.append(
            {
                "comparison_label": key[0],
                "code_version": key[1],
                "orders_limit": key[2],
                "k": key[3],
                "seed": key[4],
                "orientation_label": key[5],
                "xml_path": key[6],
                "milp_time_limit_seconds": key[7],
                "coverage_repair": key[8],
                "repair_margins": key[9],
                "repair_max_rounds": key[10],
                "baseline_method_label": b["method_label"],
                "candidate_method_label": c["method_label"],
                "baseline_pf": b["best_pf"],
                "candidate_pf": c["best_pf"],
                "delta_pf": c["best_pf"] - b["best_pf"],
                "baseline_uncovered": b["best_uncovered"],
                "candidate_uncovered": c["best_uncovered"],
                "delta_uncovered": c["best_uncovered"] - b["best_uncovered"],
                "baseline_unknown_pairs": b["best_unknown_pairs"],
                "candidate_unknown_pairs": c["best_unknown_pairs"],
                "delta_unknown_pairs": c["best_unknown_pairs"] - b["best_unknown_pairs"],
                "baseline_mean_box_volume": b["mean_box_volume"],
                "candidate_mean_box_volume": c["mean_box_volume"],
                "delta_mean_box_volume": c["mean_box_volume"] - b["mean_box_volume"],
                "baseline_run_dir": b["run_dir"],
                "candidate_run_dir": c["run_dir"],
            }
        )
    return pairs, skipped


def paired_stats(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(pair["delta_pf"]) for pair in pairs]
    if not deltas:
        return {"n": 0}
    mean = sum(deltas) / len(deltas)
    if len(deltas) == 1:
        return {"n": 1, "mean_delta_pf": mean, "sd_delta_pf": None, "ci95_low": None, "ci95_high": None}
    sd = math.sqrt(sum((x - mean) ** 2 for x in deltas) / (len(deltas) - 1))
    tcrit = _tcrit_975(len(deltas) - 1)
    half_width = tcrit * sd / math.sqrt(len(deltas))
    return {
        "n": len(deltas),
        "mean_delta_pf": mean,
        "sd_delta_pf": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def algorithm_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        algorithm = str(run.get("algorithm"))
        grouped.setdefault(algorithm, []).append(run)

    return {algorithm: _summarize_runs(group) for algorithm, group in sorted(grouped.items())}


def method_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        method = str(run.get("method_label"))
        grouped.setdefault(method, []).append(run)

    return {method: _summarize_runs(group) for method, group in sorted(grouped.items())}


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    best_pfs = _numeric_values(runs, "best_pf")
    initial_pfs = _numeric_values(runs, "initial_pf")
    uncovered = _numeric_values(runs, "best_uncovered")
    coverage = _numeric_values(runs, "coverage_rate")
    unknown_pairs = _numeric_values(runs, "best_unknown_pairs")
    orders_with_unknown = _numeric_values(runs, "best_orders_with_unknown")
    candidate_evaluations = _numeric_values(runs, "candidate_evaluations")
    elapsed_seconds = _numeric_values(runs, "elapsed_seconds")
    return {
        "n": len(runs),
        "pf_mean": _mean(best_pfs),
        "pf_best": min(best_pfs) if best_pfs else None,
        "pf_worst": max(best_pfs) if best_pfs else None,
        "initial_pf_mean": _mean(initial_pfs),
        "uncovered_mean": _mean(uncovered),
        "uncovered_max": max(uncovered) if uncovered else None,
        "coverage_mean": _mean(coverage),
        "coverage_min": min(coverage) if coverage else None,
        "unknown_pairs_mean": _mean(unknown_pairs),
        "unknown_pairs_max": max(unknown_pairs) if unknown_pairs else None,
        "orders_with_unknown_mean": _mean(orders_with_unknown),
        "orders_with_unknown_max": max(orders_with_unknown) if orders_with_unknown else None,
        "candidate_evaluations_mean": _mean(candidate_evaluations),
        "elapsed_seconds_mean": _mean(elapsed_seconds),
    }


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _tcrit_975(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        24: 2.064,
        29: 2.045,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df < 24:
        return table[20]
    if df < 29:
        return table[24]
    return 1.96


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MILP-backed box-design run summaries.")
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--baseline", default="paper_fixed_step")
    parser.add_argument("--candidate", default="staged_greedy")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries = find_summary_files(args.paths)
    runs = [load_run(path) for path in summaries]
    pairs, skipped = paired_deltas(runs, args.baseline, args.candidate)
    stats = paired_stats(pairs)
    by_algorithm = algorithm_stats(runs)
    by_method = method_stats(runs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if runs:
        write_csv(args.out_dir / "runs.csv", runs)
    if pairs:
        write_csv(args.out_dir / "paired_deltas.csv", pairs)
    with (args.out_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": len(runs),
                "pairs": len(pairs),
                "baseline": args.baseline,
                "candidate": args.candidate,
                "algorithm_stats": by_algorithm,
                "method_stats": by_method,
                "paired_stats": stats,
                "skipped_pairs": skipped,
                "summary_files": [str(path) for path in summaries],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")
    print(
        json.dumps(
            {
                "runs": len(runs),
                "pairs": len(pairs),
                "algorithm_stats": by_algorithm,
                "method_stats": by_method,
                "paired_stats": stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
