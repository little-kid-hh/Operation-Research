#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_milp_box_algorithms.py"


def parse_budget_sequence(value: str) -> str:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("budget sequence must contain at least one top-k value")
    values = [int(part) for part in parts]
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all top-k values must be positive")
    if values != sorted(values):
        raise argparse.ArgumentTypeError("top-k values must be nondecreasing")
    return ",".join(str(value) for value in values)


def sequence_label(sequence: str) -> str:
    return "top" + "_".join(sequence.split(","))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_summary(cmd: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError("runner did not emit a JSON summary") from exc


def cache_value(summary: dict[str, Any], field: str) -> float | int | None:
    cache = summary.get("oracle_cache") or {}
    return cache.get(field)


def score_value(summary: dict[str, Any], field: str) -> float | int | None:
    score = summary.get("best_score") or {}
    return score.get(field)


def add_common_runner_args(
    cmd: list[str],
    args: argparse.Namespace,
    *,
    initial_boxes_json: Path,
    oracle_cache_dir: Path,
    max_elapsed_seconds: float | None = None,
) -> None:
    cmd.extend(
        [
            "--orders-limit",
            str(args.orders_limit),
            "--orders-offset",
            str(args.orders_offset),
            "--k",
            str(args.k),
            "--seed",
            str(args.seed),
            "--schedule",
            args.schedule,
            "--initial-boxes-json",
            str(initial_boxes_json),
            "--xml-path",
            str(args.xml_path),
            "--java-classes",
            str(args.java_classes),
            "--java-classpath",
            args.java_classpath,
            "--milp-time-limit-seconds",
            str(args.milp_time_limit_seconds),
            "--oracle-cache-dir",
            str(oracle_cache_dir),
            "--out-root",
            str(args.out_root),
            "--coverage-repair",
            args.coverage_repair,
            "--code-version",
            args.code_version,
        ]
    )
    if max_elapsed_seconds is not None:
        cmd.extend(["--max-elapsed-seconds", str(max_elapsed_seconds)])
    if args.prefetch_candidate_statuses:
        cmd.append("--prefetch-candidate-statuses")


def make_row(
    *,
    sequence: str,
    ranker_summary: dict[str, Any],
    audit_summary: dict[str, Any] | None,
    baseline_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_pf = score_value(baseline_summary, "packaging_factor") if baseline_summary else None
    ranker_pf = score_value(ranker_summary, "packaging_factor")
    audit_pf = score_value(audit_summary, "packaging_factor") if audit_summary else None
    ranker_generated = ranker_summary.get("generated_candidates")
    ranker_validated = ranker_summary.get("milp_validated_candidates")
    ranker_avoided = ranker_summary.get("milp_candidate_evaluations_avoided")
    audit_generated = audit_summary.get("generated_candidates") if audit_summary else None
    audit_validated = audit_summary.get("milp_validated_candidates") if audit_summary else None
    audit_avoided = audit_summary.get("milp_candidate_evaluations_avoided") if audit_summary else None
    ranker_uncached = cache_value(ranker_summary, "uncached_boxes")
    audit_uncached = cache_value(audit_summary, "uncached_boxes") if audit_summary else None
    ranker_subprocess = cache_value(ranker_summary, "subprocess_seconds")
    audit_subprocess = cache_value(audit_summary, "subprocess_seconds") if audit_summary else None
    ranker_elapsed = ranker_summary.get("elapsed_seconds")
    audit_elapsed = audit_summary.get("elapsed_seconds") if audit_summary else None
    ranker_prefetch_elapsed = ranker_summary.get("prefetch_eval_seconds")
    audit_prefetch_elapsed = audit_summary.get("prefetch_eval_seconds") if audit_summary else None

    return {
        "ranker_budget_sequence": sequence,
        "ranker_config_label": ranker_summary.get("config_label"),
        "ranker_prefetch_candidate_statuses": ranker_summary.get("prefetch_candidate_statuses"),
        "ranker_run_dir": ranker_summary.get("run_dir"),
        "ranker_pf": ranker_pf,
        "ranker_pf_gap_vs_baseline": ranker_pf - baseline_pf if ranker_pf is not None and baseline_pf is not None else None,
        "ranker_coverage": score_value(ranker_summary, "coverage_rate"),
        "ranker_uncovered": score_value(ranker_summary, "uncovered_orders"),
        "ranker_generated_candidates": ranker_generated,
        "ranker_milp_validated_candidates": ranker_validated,
        "ranker_candidate_avoidance_rate": ranker_summary.get("milp_avoidance_rate"),
        "ranker_safety_policy": ranker_summary.get("ranker_safety_policy"),
        "ranker_safety_max_candidates": ranker_summary.get("ranker_safety_max_candidates"),
        "ranker_safety_candidates": ranker_summary.get("ranker_safety_candidates"),
        "ranker_eval_seconds": ranker_summary.get("ranker_eval_seconds"),
        "ranker_prefetch_eval_seconds": ranker_prefetch_elapsed,
        "ranker_oracle_uncached_boxes": ranker_uncached,
        "ranker_oracle_subprocess_seconds": ranker_subprocess,
        "ranker_elapsed_seconds": ranker_elapsed,
        "ranker_stop_reason": ranker_summary.get("stop_reason"),
        "ranker_max_elapsed_seconds": ranker_summary.get("max_elapsed_seconds"),
        "audit_config_label": audit_summary.get("config_label") if audit_summary else None,
        "audit_prefetch_candidate_statuses": audit_summary.get("prefetch_candidate_statuses") if audit_summary else None,
        "audit_run_dir": audit_summary.get("run_dir") if audit_summary else None,
        "audit_pf": audit_pf,
        "audit_pf_gap_vs_baseline": audit_pf - baseline_pf if audit_pf is not None and baseline_pf is not None else None,
        "audit_coverage": score_value(audit_summary, "coverage_rate") if audit_summary else None,
        "audit_uncovered": score_value(audit_summary, "uncovered_orders") if audit_summary else None,
        "audit_generated_candidates": audit_generated,
        "audit_milp_validated_candidates": audit_validated,
        "audit_candidate_avoidance_rate": audit_summary.get("milp_avoidance_rate") if audit_summary else None,
        "audit_oracle_uncached_boxes": audit_uncached,
        "audit_oracle_subprocess_seconds": audit_subprocess,
        "audit_prefetch_eval_seconds": audit_prefetch_elapsed,
        "audit_elapsed_seconds": audit_elapsed,
        "audit_stop_reason": audit_summary.get("stop_reason") if audit_summary else None,
        "audit_max_elapsed_seconds": audit_summary.get("max_elapsed_seconds") if audit_summary else None,
        "combined_oracle_uncached_boxes": (
            ranker_uncached + audit_uncached
            if ranker_uncached is not None and audit_uncached is not None
            else ranker_uncached
        ),
        "combined_generated_candidates": (
            ranker_generated + audit_generated
            if ranker_generated is not None and audit_generated is not None
            else ranker_generated
        ),
        "combined_milp_validated_candidates": (
            ranker_validated + audit_validated
            if ranker_validated is not None and audit_validated is not None
            else ranker_validated
        ),
        "combined_milp_candidate_evaluations_avoided": (
            ranker_avoided + audit_avoided
            if ranker_avoided is not None and audit_avoided is not None
            else ranker_avoided
        ),
        "combined_oracle_subprocess_seconds": (
            ranker_subprocess + audit_subprocess
            if ranker_subprocess is not None and audit_subprocess is not None
            else ranker_subprocess
        ),
        "combined_prefetch_eval_seconds": (
            ranker_prefetch_elapsed + audit_prefetch_elapsed
            if ranker_prefetch_elapsed is not None and audit_prefetch_elapsed is not None
            else ranker_prefetch_elapsed
        ),
        "combined_elapsed_seconds": (
            ranker_elapsed + audit_elapsed if ranker_elapsed is not None and audit_elapsed is not None else ranker_elapsed
        ),
        "baseline_pf": baseline_pf,
        "baseline_prefetch_candidate_statuses": baseline_summary.get("prefetch_candidate_statuses")
        if baseline_summary
        else None,
        "baseline_run_dir": baseline_summary.get("run_dir") if baseline_summary else None,
        "baseline_oracle_uncached_boxes": cache_value(baseline_summary, "uncached_boxes") if baseline_summary else None,
        "baseline_oracle_subprocess_seconds": cache_value(baseline_summary, "subprocess_seconds")
        if baseline_summary
        else None,
        "baseline_elapsed_seconds": baseline_summary.get("elapsed_seconds") if baseline_summary else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run query-budgeted ranker local search across one or more top-k "
            "budgets, optionally followed by an exact staged-greedy audit."
        )
    )
    parser.add_argument("--xml-path", type=Path, required=True)
    parser.add_argument("--initial-boxes-json", type=Path, required=True)
    parser.add_argument("--candidate-ranker-path", type=Path, required=True)
    parser.add_argument("--exact-baseline-summary", type=Path, default=None)
    parser.add_argument("--ranker-budget-sequence", action="append", type=parse_budget_sequence, default=None)
    parser.add_argument(
        "--ranker-safety-policy",
        choices=["none", "all_expansions", "targeted_expansion_capture"],
        default="none",
        help="Optional ranker safety set passed to ranker_filtered_greedy.",
    )
    parser.add_argument(
        "--ranker-safety-max-candidates",
        type=int,
        default=None,
        help="Optional maximum safety candidates per iteration for targeted ranker safety policies.",
    )
    parser.add_argument("--run-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ranker-max-elapsed-seconds",
        type=float,
        default=None,
        help="Optional graceful wall-clock budget passed only to the ranker run.",
    )
    parser.add_argument(
        "--audit-max-elapsed-seconds",
        type=float,
        default=None,
        help="Optional graceful wall-clock budget passed only to the exact audit run.",
    )
    parser.add_argument("--orders-limit", type=int, default=500)
    parser.add_argument(
        "--orders-offset",
        type=int,
        default=0,
        help="Number of orders to skip before applying --orders-limit; enables non-overlapping held-out windows.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schedule", default="0.25:1000")
    parser.add_argument("--java-classes", type=Path, default=ROOT.parent / "MILP_3DBPP/target/classes")
    parser.add_argument("--java-classpath", default=os.environ.get("GUROBI_JAR", ""))
    parser.add_argument("--milp-time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--coverage-repair", choices=["none", "geometric_expand"], default="none")
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/query_budgeted_ranker_frontier")
    parser.add_argument("--oracle-cache-dir", type=Path, default=None)
    parser.add_argument("--prefetch-candidate-statuses", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--code-version", default="")
    args = parser.parse_args()

    if args.ranker_max_elapsed_seconds is not None and args.ranker_max_elapsed_seconds <= 0.0:
        raise ValueError("--ranker-max-elapsed-seconds must be positive when supplied")
    if args.audit_max_elapsed_seconds is not None and args.audit_max_elapsed_seconds <= 0.0:
        raise ValueError("--audit-max-elapsed-seconds must be positive when supplied")
    if args.ranker_safety_max_candidates is not None and args.ranker_safety_max_candidates <= 0:
        raise ValueError("--ranker-safety-max-candidates must be positive when supplied")
    if args.orders_offset < 0:
        raise ValueError("--orders-offset must be non-negative")

    sequences = args.ranker_budget_sequence or ["10,30"]
    run_id = datetime.now().strftime("frontier_%Y%m%d_%H%M%S")
    frontier_dir = args.out_root / run_id
    oracle_cache_root = args.oracle_cache_dir or (frontier_dir / "oracle_cache")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    baseline_summary = load_json(args.exact_baseline_summary) if args.exact_baseline_summary else None
    rows: list[dict[str, Any]] = []
    raw_summaries: list[dict[str, Any]] = []
    for sequence in sequences:
        label = sequence_label(sequence)
        sequence_cache_dir = oracle_cache_root / label
        ranker_label = f"query_budgeted_ranker_{label}"
        ranker_cmd = [sys.executable, str(RUNNER), "--algorithm", "ranker_filtered_greedy"]
        add_common_runner_args(
            ranker_cmd,
            args,
            initial_boxes_json=args.initial_boxes_json,
            oracle_cache_dir=sequence_cache_dir,
            max_elapsed_seconds=args.ranker_max_elapsed_seconds,
        )
        ranker_cmd.extend(
            [
                "--candidate-ranker-path",
                str(args.candidate_ranker_path),
                "--ranker-adaptive-top-k",
                sequence,
                "--ranker-safety-policy",
                args.ranker_safety_policy,
                "--no-ranker-noop-fallback",
                "--config-label",
                ranker_label,
            ]
        )
        if args.ranker_safety_max_candidates is not None:
            ranker_cmd.extend(["--ranker-safety-max-candidates", str(args.ranker_safety_max_candidates)])
        ranker_summary = run_summary(ranker_cmd, env=env)
        raw_summaries.append(ranker_summary)

        audit_summary = None
        if args.run_audit:
            ranker_run_dir = Path(str(ranker_summary["run_dir"]))
            audit_label = f"exact_audit_after_{ranker_label}"
            audit_cmd = [sys.executable, str(RUNNER), "--algorithm", "staged_greedy"]
            add_common_runner_args(
                audit_cmd,
                args,
                initial_boxes_json=ranker_run_dir / "best_boxes.json",
                oracle_cache_dir=sequence_cache_dir,
                max_elapsed_seconds=args.audit_max_elapsed_seconds,
            )
            audit_cmd.extend(["--config-label", audit_label])
            audit_summary = run_summary(audit_cmd, env=env)
            raw_summaries.append(audit_summary)

        rows.append(
            make_row(
                sequence=sequence,
                ranker_summary=ranker_summary,
                audit_summary=audit_summary,
                baseline_summary=baseline_summary,
            )
        )

    write_csv(frontier_dir / "frontier_summary.csv", rows)
    write_json(frontier_dir / "frontier_summary.json", rows)
    write_json(frontier_dir / "raw_summaries.json", raw_summaries)
    print(json.dumps({"frontier_dir": str(frontier_dir), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
