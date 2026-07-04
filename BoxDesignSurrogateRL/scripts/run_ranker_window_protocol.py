#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MILP_RUNNER = ROOT / "scripts/run_milp_box_algorithms.py"
FRONTIER_RUNNER = ROOT / "scripts/run_query_budgeted_ranker_frontier.py"
WINDOW_SUMMARIZER = ROOT / "scripts/summarize_ranker_window_results.py"


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return values


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_json_command(cmd: list[str], *, env: dict[str, str]) -> dict[str, Any]:
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
        raise RuntimeError("child command did not emit a JSON summary") from exc


def window_label(offset: int, limit: int) -> str:
    return f"test[{offset},{offset + limit})"


def safe_slug(value: str) -> str:
    return (
        value.replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(":", "_")
        .replace("/", "_")
    )


def exact_command(
    *,
    args: argparse.Namespace,
    seed: int,
    offset: int,
    exact_out_root: Path,
    exact_cache_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(MILP_RUNNER),
        "--algorithm",
        "staged_greedy",
        "--orders-offset",
        str(offset),
        "--orders-limit",
        str(args.orders_limit),
        "--k",
        str(args.k),
        "--seed",
        str(seed),
        "--schedule",
        args.schedule,
        "--xml-path",
        str(args.xml_path),
        "--java-classes",
        str(args.java_classes),
        "--java-classpath",
        args.java_classpath,
        "--milp-time-limit-seconds",
        str(args.milp_time_limit_seconds),
        "--coverage-repair",
        args.coverage_repair,
        "--oracle-cache-dir",
        str(exact_cache_dir),
        "--out-root",
        str(exact_out_root),
        "--code-version",
        args.code_version,
        "--config-label",
        f"{args.config_prefix}_exact_seed{seed}_offset{offset}",
    ]
    if args.initial_boxes_json is not None:
        cmd.extend(["--initial-boxes-json", str(args.initial_boxes_json)])
    return cmd


def frontier_command(
    *,
    args: argparse.Namespace,
    seed: int,
    offset: int,
    initial_boxes_json: Path,
    exact_summary: Path,
    frontier_out_root: Path,
    frontier_cache_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(FRONTIER_RUNNER),
        "--xml-path",
        str(args.xml_path),
        "--initial-boxes-json",
        str(initial_boxes_json),
        "--candidate-ranker-path",
        str(args.candidate_ranker_path),
        "--exact-baseline-summary",
        str(exact_summary),
        "--schedule",
        args.schedule,
        "--orders-offset",
        str(offset),
        "--orders-limit",
        str(args.orders_limit),
        "--k",
        str(args.k),
        "--seed",
        str(seed),
        "--java-classes",
        str(args.java_classes),
        "--java-classpath",
        args.java_classpath,
        "--milp-time-limit-seconds",
        str(args.milp_time_limit_seconds),
        "--coverage-repair",
        args.coverage_repair,
        "--ranker-max-elapsed-seconds",
        str(args.ranker_max_elapsed_seconds),
        "--out-root",
        str(frontier_out_root),
        "--oracle-cache-dir",
        str(frontier_cache_dir),
        "--code-version",
        args.code_version,
    ]
    for sequence in args.ranker_budget_sequence:
        cmd.extend(["--ranker-budget-sequence", sequence])
    return cmd


def summarize_command(*, manifest_path: Path, summary_dir: Path, total_label: str) -> list[str]:
    return [
        sys.executable,
        str(WINDOW_SUMMARIZER),
        "--manifest",
        str(manifest_path),
        "--out-json",
        str(summary_dir / "summary.json"),
        "--out-csv",
        str(summary_dir / "summary.csv"),
        "--out-md",
        str(summary_dir / "summary.md"),
        "--total-label",
        total_label,
    ]


def resolve_initial_boxes_for_ranker(exact_summary: dict[str, Any]) -> Path:
    run_dir = exact_summary.get("run_dir")
    if not run_dir:
        raise ValueError("exact summary did not include run_dir")
    path = Path(str(run_dir)) / "initial_boxes.json"
    if not path.exists():
        raise FileNotFoundError(f"exact run initial boxes not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact staged baseline and ranker+exact-audit over one or more "
            "order windows/seeds, then write a summarizer manifest."
        )
    )
    parser.add_argument("--xml-path", type=Path, required=True)
    parser.add_argument("--candidate-ranker-path", type=Path, required=True)
    parser.add_argument(
        "--initial-boxes-json",
        type=Path,
        default=None,
        help=(
            "Optional fixed initial boxes. Omit this for true seed-specific "
            "k-means initial boxes generated by the exact baseline run."
        ),
    )
    parser.add_argument("--orders-offsets", type=parse_int_list, required=True)
    parser.add_argument("--orders-limit", type=int, default=100)
    parser.add_argument("--seeds", type=parse_int_list, default=[0])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--schedule", default="0.25:1000")
    parser.add_argument("--java-classes", type=Path, default=ROOT.parent / "MILP_3DBPP/target/classes")
    parser.add_argument("--java-classpath", default=os.environ.get("GUROBI_JAR", ""))
    parser.add_argument("--milp-time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--coverage-repair", choices=["none", "geometric_expand"], default="geometric_expand")
    parser.add_argument("--ranker-budget-sequence", action="append", default=None)
    parser.add_argument("--ranker-max-elapsed-seconds", type=float, default=180.0)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/ranker_window_protocol")
    parser.add_argument("--code-version", default="")
    parser.add_argument("--config-prefix", default="window_protocol")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.orders_limit <= 0:
        raise ValueError("--orders-limit must be positive")
    if any(offset < 0 for offset in args.orders_offsets):
        raise ValueError("--orders-offsets must be non-negative")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be non-negative")
    if args.ranker_max_elapsed_seconds <= 0.0:
        raise ValueError("--ranker-max-elapsed-seconds must be positive")
    args.ranker_budget_sequence = args.ranker_budget_sequence or ["10,20,30,40,50"]

    protocol_id = datetime.now().strftime("protocol_%Y%m%d_%H%M%S")
    protocol_dir = args.out_root / protocol_id
    records: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    for seed in args.seeds:
        for offset in args.orders_offsets:
            label = f"seed{seed}:{window_label(offset, args.orders_limit)}"
            slug = f"seed{seed}_{safe_slug(window_label(offset, args.orders_limit))}"
            exact_out_root = protocol_dir / slug / "exact"
            frontier_out_root = protocol_dir / slug / "ranker_frontier"
            exact_cache_dir = protocol_dir / slug / "oracle_cache_exact"
            frontier_cache_dir = protocol_dir / slug / "oracle_cache_ranker"

            exact_cmd = exact_command(
                args=args,
                seed=seed,
                offset=offset,
                exact_out_root=exact_out_root,
                exact_cache_dir=exact_cache_dir,
            )
            commands.append({"label": label, "phase": "exact", "cmd": exact_cmd})
            if args.dry_run:
                continue

            exact_summary = run_json_command(exact_cmd, env=env)
            exact_summary_path = Path(str(exact_summary["run_dir"])) / "summary.json"
            initial_boxes_json = resolve_initial_boxes_for_ranker(exact_summary)

            frontier_cmd = frontier_command(
                args=args,
                seed=seed,
                offset=offset,
                initial_boxes_json=initial_boxes_json,
                exact_summary=exact_summary_path,
                frontier_out_root=frontier_out_root,
                frontier_cache_dir=frontier_cache_dir,
            )
            commands.append({"label": label, "phase": "ranker_frontier", "cmd": frontier_cmd})
            frontier_summary = run_json_command(frontier_cmd, env=env)
            frontier_summary_path = Path(str(frontier_summary["frontier_dir"])) / "frontier_summary.json"
            records.append(
                {
                    "label": label,
                    "exact_summary": str(exact_summary_path),
                    "frontier_summary": str(frontier_summary_path),
                }
            )

    write_json(protocol_dir / "commands.json", commands)
    if args.dry_run:
        print(json.dumps({"protocol_dir": str(protocol_dir), "commands": commands}, ensure_ascii=False, indent=2))
        return

    manifest_path = protocol_dir / "ranker_window_manifest.json"
    write_json(manifest_path, records)
    total_label = f"seeds={','.join(str(seed) for seed in args.seeds)} window total"
    summary_dir = protocol_dir / "summary"
    summary_cmd = summarize_command(manifest_path=manifest_path, summary_dir=summary_dir, total_label=total_label)
    commands.append({"label": "summary", "phase": "summary", "cmd": summary_cmd})
    write_json(protocol_dir / "commands.json", commands)
    subprocess.run(summary_cmd, check=True, env=env)

    print(
        json.dumps(
            {
                "protocol_dir": str(protocol_dir),
                "manifest": str(manifest_path),
                "summary_dir": str(summary_dir),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
