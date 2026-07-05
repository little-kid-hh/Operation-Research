#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FRONTIER_RUNNER = ROOT / "scripts/run_query_budgeted_ranker_frontier.py"
WINDOW_SUMMARIZER = ROOT / "scripts/summarize_ranker_window_results.py"

LABEL_RE = re.compile(r"seed(?P<seed>\d+):test\[(?P<offset>\d+),(?P<end>\d+)\)")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


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


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("manifest must be a JSON list")
    records: list[dict[str, Any]] = []
    for idx, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"manifest entry {idx} must be an object")
        if "label" not in record or "exact_summary" not in record:
            raise ValueError(f"manifest entry {idx} needs label and exact_summary")
        records.append(record)
    return records


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def parse_record_window(record: dict[str, Any]) -> tuple[int, int, int]:
    if "seed" in record and "orders_offset" in record and "orders_limit" in record:
        seed = int(record["seed"])
        offset = int(record["orders_offset"])
        limit = int(record["orders_limit"])
        return seed, offset, limit
    label = str(record["label"])
    match = LABEL_RE.fullmatch(label)
    if not match:
        raise ValueError(
            "record must include seed/orders_offset/orders_limit or a label like seed2:test[400,500): "
            f"{label}"
        )
    seed = int(match.group("seed"))
    offset = int(match.group("offset"))
    end = int(match.group("end"))
    if end <= offset:
        raise ValueError(f"label has non-positive window length: {label}")
    return seed, offset, end - offset


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
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError("child command did not emit a JSON summary") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("child command emitted non-object JSON")
    return payload


def resolve_initial_boxes(exact_summary_path: Path) -> Path:
    summary = load_json(exact_summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"exact summary must be a JSON object: {exact_summary_path}")
    run_dir = summary.get("run_dir")
    if not run_dir:
        raise ValueError(f"exact summary did not include run_dir: {exact_summary_path}")
    initial_boxes = Path(str(run_dir)) / "initial_boxes.json"
    if not initial_boxes.exists():
        raise FileNotFoundError(f"initial boxes not found for exact summary: {initial_boxes}")
    return initial_boxes


def frontier_command(
    *,
    args: argparse.Namespace,
    seed: int,
    offset: int,
    limit: int,
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
        str(limit),
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
        "--ranker-safety-policy",
        args.ranker_safety_policy,
        "--out-root",
        str(frontier_out_root),
        "--oracle-cache-dir",
        str(frontier_cache_dir),
        "--code-version",
        args.code_version,
    ]
    if args.ranker_max_elapsed_seconds is not None:
        cmd.extend(["--ranker-max-elapsed-seconds", str(args.ranker_max_elapsed_seconds)])
    if args.audit_max_elapsed_seconds is not None:
        cmd.extend(["--audit-max-elapsed-seconds", str(args.audit_max_elapsed_seconds)])
    if args.prefetch_candidate_statuses:
        cmd.append("--prefetch-candidate-statuses")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse an existing exact-staged manifest and run only a new "
            "query-budgeted ranker frontier plus exact audit for each window."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--xml-path", type=Path, required=True)
    parser.add_argument("--candidate-ranker-path", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--schedule", default="0.25:1000")
    parser.add_argument("--java-classes", type=Path, default=ROOT.parent / "MILP_3DBPP/target/classes")
    parser.add_argument("--java-classpath", default=os.environ.get("GUROBI_JAR", ""))
    parser.add_argument("--milp-time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--coverage-repair", choices=["none", "geometric_expand"], default="geometric_expand")
    parser.add_argument("--ranker-budget-sequence", action="append", type=parse_budget_sequence, default=None)
    parser.add_argument(
        "--ranker-safety-policy",
        choices=["none", "all_expansions"],
        default="none",
    )
    parser.add_argument("--ranker-max-elapsed-seconds", type=float, default=None)
    parser.add_argument("--audit-max-elapsed-seconds", type=float, default=None)
    parser.add_argument("--prefetch-candidate-statuses", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/ranker_frontier_from_exact_manifest")
    parser.add_argument("--oracle-cache-dir", type=Path, default=None)
    parser.add_argument("--code-version", default="")
    parser.add_argument("--total-label", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.milp_time_limit_seconds <= 0.0:
        raise ValueError("--milp-time-limit-seconds must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index")
    if args.ranker_max_elapsed_seconds is not None and args.ranker_max_elapsed_seconds <= 0.0:
        raise ValueError("--ranker-max-elapsed-seconds must be positive")
    if args.audit_max_elapsed_seconds is not None and args.audit_max_elapsed_seconds <= 0.0:
        raise ValueError("--audit-max-elapsed-seconds must be positive")
    args.ranker_budget_sequence = args.ranker_budget_sequence or ["10,20,30,40,50"]

    input_manifest = load_manifest(args.manifest)
    selected_manifest = input_manifest[args.start_index : args.end_index]
    if not selected_manifest:
        raise ValueError("selected manifest slice is empty")
    protocol_id = datetime.now().strftime("manifest_frontier_%Y%m%d_%H%M%S")
    protocol_dir = args.out_root / protocol_id
    cache_root = args.oracle_cache_dir or (protocol_dir / "oracle_cache")
    manifest_base_dir = args.manifest.parent
    output_manifest = protocol_dir / "ranker_window_manifest.json"
    records: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    for idx, record in enumerate(selected_manifest, start=args.start_index):
        label = str(record["label"])
        seed, offset, limit = parse_record_window(record)
        exact_summary = resolve_path(manifest_base_dir, str(record["exact_summary"]))
        if not exact_summary.exists():
            raise FileNotFoundError(f"exact summary not found for {label}: {exact_summary}")
        initial_boxes_json = resolve_initial_boxes(exact_summary)
        slug = f"{idx:02d}_s{seed}_o{offset}"
        frontier_out_root = protocol_dir / slug / "ranker"
        frontier_cache_dir = cache_root / f"s{seed}_o{offset}"
        cmd = frontier_command(
            args=args,
            seed=seed,
            offset=offset,
            limit=limit,
            initial_boxes_json=initial_boxes_json,
            exact_summary=exact_summary,
            frontier_out_root=frontier_out_root,
            frontier_cache_dir=frontier_cache_dir,
        )
        commands.append({"label": label, "phase": "ranker_frontier", "cmd": cmd})
        write_json(protocol_dir / "commands.json", commands)
        if args.dry_run:
            continue
        frontier_summary = run_json_command(cmd, env=env)
        frontier_summary_path = Path(str(frontier_summary["frontier_dir"])) / "frontier_summary.json"
        records.append(
            {
                "label": label,
                "exact_summary": str(exact_summary),
                "frontier_summary": str(frontier_summary_path),
            }
        )
        write_json(output_manifest, records)

    write_json(protocol_dir / "commands.json", commands)
    if args.dry_run:
        print(json.dumps({"protocol_dir": str(protocol_dir), "commands": commands}, ensure_ascii=False, indent=2))
        return

    write_json(output_manifest, records)
    summary_dir = protocol_dir / "summary"
    total_label = args.total_label or "manifest window total"
    summary_cmd = summarize_command(manifest_path=output_manifest, summary_dir=summary_dir, total_label=total_label)
    commands.append({"label": "summary", "phase": "summary", "cmd": summary_cmd})
    write_json(protocol_dir / "commands.json", commands)
    subprocess.run(summary_cmd, check=True, env=env)
    print(
        json.dumps(
            {
                "protocol_dir": str(protocol_dir),
                "manifest": str(output_manifest),
                "summary_dir": str(summary_dir),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
