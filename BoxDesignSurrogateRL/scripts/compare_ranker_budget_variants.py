#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


COST_FIELDS = [
    ("validations", "milp_validated_candidates"),
    ("uncached_boxes", "oracle_uncached_boxes"),
    ("subprocess_seconds", "oracle_subprocess_seconds"),
    ("elapsed_seconds", "elapsed_seconds"),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def numeric(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def difference(candidate: Any, baseline: Any) -> float | None:
    candidate_n = numeric(candidate)
    baseline_n = numeric(baseline)
    if candidate_n is None or baseline_n is None:
        return None
    return float(candidate_n) - float(baseline_n)


def percent_reduction(baseline: Any, candidate: Any) -> float | None:
    baseline_n = numeric(baseline)
    candidate_n = numeric(candidate)
    if baseline_n is None or candidate_n is None or float(baseline_n) == 0.0:
        return None
    return (float(baseline_n) - float(candidate_n)) / float(baseline_n) * 100.0


def load_window_row(summary_path: Path, window_label: str) -> dict[str, Any]:
    payload = load_json(summary_path)
    windows = payload.get("windows") if isinstance(payload, dict) else None
    if not isinstance(windows, list):
        raise ValueError(f"window summary must contain a windows list: {summary_path}")
    matches = [row for row in windows if isinstance(row, dict) and row.get("label") == window_label]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one window labeled {window_label!r} in {summary_path}; found {len(matches)}")
    return matches[0]


def load_frontier_row(path: Path, row_index: int = 0) -> dict[str, Any]:
    payload = load_json(path)
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"frontier summary must contain a row list: {path}")
    if not 0 <= row_index < len(rows):
        raise ValueError(f"frontier row_index={row_index} out of range for {path}")
    row = rows[row_index]
    if not isinstance(row, dict):
        raise ValueError(f"frontier row {row_index} must be an object: {path}")
    return row


def exact_variant(window: dict[str, Any], *, summary_path: Path) -> dict[str, Any]:
    return {
        "label": "exact_staged_baseline",
        "kind": "exact",
        "source_path": str(summary_path),
        "pf": window.get("exact_pf"),
        "coverage": window.get("exact_coverage"),
        "uncovered": window.get("exact_uncovered"),
        "validations": window.get("exact_validations"),
        "uncached_boxes": window.get("exact_uncached_boxes"),
        "subprocess_seconds": window.get("exact_subprocess_seconds"),
        "elapsed_seconds": window.get("exact_elapsed_seconds"),
        "ranker_pf": None,
        "ranker_elapsed_seconds": None,
        "ranker_stop_reason": None,
        "ranker_handoff_policy": None,
        "ranker_handoff_threshold": None,
        "ranker_cap_seconds": None,
    }


def fixed_ranker_variant(window: dict[str, Any], *, summary_path: Path, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "kind": "ranker_audit",
        "source_path": str(summary_path),
        "pf": window.get("ranker_audit_pf"),
        "coverage": window.get("ranker_audit_coverage"),
        "uncovered": window.get("ranker_audit_uncovered"),
        "validations": window.get("ranker_audit_validations"),
        "uncached_boxes": window.get("ranker_audit_uncached_boxes"),
        "subprocess_seconds": window.get("ranker_audit_subprocess_seconds"),
        "elapsed_seconds": window.get("ranker_audit_elapsed_seconds"),
        "ranker_pf": window.get("ranker_pf"),
        "ranker_elapsed_seconds": window.get("ranker_elapsed_seconds"),
        "ranker_stop_reason": window.get("ranker_stop_reason"),
        "ranker_handoff_policy": "none",
        "ranker_handoff_threshold": None,
        "ranker_cap_seconds": window.get("ranker_elapsed_seconds"),
    }


def frontier_variant(label: str, path: Path, *, row_index: int = 0) -> dict[str, Any]:
    row = load_frontier_row(path, row_index=row_index)
    return {
        "label": label,
        "kind": "ranker_audit",
        "source_path": str(path),
        "pf": row.get("audit_pf"),
        "coverage": row.get("audit_coverage"),
        "uncovered": row.get("audit_uncovered"),
        "validations": row.get("combined_milp_validated_candidates"),
        "uncached_boxes": row.get("combined_oracle_uncached_boxes"),
        "subprocess_seconds": row.get("combined_oracle_subprocess_seconds"),
        "elapsed_seconds": row.get("combined_elapsed_seconds"),
        "ranker_pf": row.get("ranker_pf"),
        "ranker_elapsed_seconds": row.get("ranker_elapsed_seconds"),
        "ranker_stop_reason": row.get("ranker_stop_reason"),
        "ranker_handoff_policy": row.get("ranker_handoff_policy"),
        "ranker_handoff_threshold": row.get("ranker_handoff_min_pf_improvement_per_validation"),
        "ranker_cap_seconds": row.get("ranker_max_elapsed_seconds"),
    }


def add_comparisons(
    variant: dict[str, Any],
    *,
    exact: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    out = dict(variant)
    out["pf_delta_vs_exact"] = difference(variant.get("pf"), exact.get("pf"))
    out["pf_delta_vs_reference"] = difference(variant.get("pf"), reference.get("pf"))
    for short, _source_field in COST_FIELDS:
        out[f"{short}_reduction_vs_exact_pct"] = percent_reduction(exact.get(short), variant.get(short))
        out[f"{short}_reduction_vs_reference_pct"] = percent_reduction(reference.get(short), variant.get(short))
    return out


def labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path_s = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("variant label cannot be empty")
    return label, Path(path_s)


def compare(
    *,
    window_summary_json: Path,
    window_label: str,
    fixed_label: str,
    frontier_summaries: list[tuple[str, Path]],
) -> dict[str, Any]:
    window = load_window_row(window_summary_json, window_label)
    exact = exact_variant(window, summary_path=window_summary_json)
    fixed = fixed_ranker_variant(window, summary_path=window_summary_json, label=fixed_label)
    variants = [fixed] + [frontier_variant(label, path) for label, path in frontier_summaries]
    compared = [add_comparisons(variant, exact=exact, reference=fixed) for variant in variants]
    exact_compared = add_comparisons(exact, exact=exact, reference=fixed)
    best_elapsed = min(
        (row for row in compared if numeric(row.get("elapsed_seconds")) is not None),
        key=lambda row: float(numeric(row["elapsed_seconds"]) or 0.0),
        default=None,
    )
    best_validations = min(
        (row for row in compared if numeric(row.get("validations")) is not None),
        key=lambda row: float(numeric(row["validations"]) or 0.0),
        default=None,
    )
    return {
        "window_label": window_label,
        "comparison_reference": fixed_label,
        "exact": exact_compared,
        "variants": compared,
        "summary": {
            "best_elapsed_variant": None if best_elapsed is None else best_elapsed["label"],
            "best_validation_variant": None if best_validations is None else best_validations["label"],
            "n_variants": len(compared),
            "n_pf_regressions_vs_exact": sum(
                1
                for row in compared
                if (delta := numeric(row.get("pf_delta_vs_exact"))) is not None and float(delta) > 1e-12
            ),
            "n_pf_improvements_vs_exact": sum(
                1
                for row in compared
                if (delta := numeric(row.get("pf_delta_vs_exact"))) is not None and float(delta) < -1e-12
            ),
        },
    }


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = [payload["exact"]] + payload["variants"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    value_n = numeric(value)
    if value_n is None:
        return str(value)
    return f"{float(value_n):.{digits}f}"


def format_pct(value: Any) -> str:
    value_n = numeric(value)
    if value_n is None:
        return ""
    return f"{float(value_n):.1f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [payload["exact"]] + payload["variants"]
    metric_rows = []
    for row in all_rows:
        metric_rows.append(
            [
                str(row.get("label")),
                str(row.get("kind")),
                format_value(row.get("pf"), 10),
                format_value(row.get("pf_delta_vs_exact"), 10),
                format_value(row.get("coverage"), 4),
                format_value(row.get("uncovered"), 0),
                format_value(row.get("validations"), 0),
                format_value(row.get("uncached_boxes"), 0),
                format_value(row.get("subprocess_seconds"), 4),
                format_value(row.get("elapsed_seconds"), 4),
                str(row.get("ranker_stop_reason") or ""),
            ]
        )
    reduction_rows = []
    for row in payload["variants"]:
        reduction_rows.append(
            [
                str(row.get("label")),
                format_pct(row.get("validations_reduction_vs_exact_pct")),
                format_pct(row.get("uncached_boxes_reduction_vs_exact_pct")),
                format_pct(row.get("subprocess_seconds_reduction_vs_exact_pct")),
                format_pct(row.get("elapsed_seconds_reduction_vs_exact_pct")),
                format_pct(row.get("validations_reduction_vs_reference_pct")),
                format_pct(row.get("uncached_boxes_reduction_vs_reference_pct")),
                format_pct(row.get("subprocess_seconds_reduction_vs_reference_pct")),
                format_pct(row.get("elapsed_seconds_reduction_vs_reference_pct")),
            ]
        )
    lines = [
        f"# Ranker budget comparison: {payload['window_label']}",
        "",
        "This is a focused diagnostic comparison on one hard validation window. It is not a replacement for the multi-window main result.",
        "",
        markdown_table(
            [
                "variant",
                "kind",
                "final PF",
                "PF delta vs exact",
                "coverage",
                "uncovered",
                "validations",
                "uncached",
                "subprocess s",
                "elapsed s",
                "ranker stop",
            ],
            metric_rows,
        ),
        "",
        f"Cost reductions use `{payload['comparison_reference']}` as the fixed-ranker reference.",
        "",
        markdown_table(
            [
                "variant",
                "val vs exact",
                "uncached vs exact",
                "subprocess vs exact",
                "elapsed vs exact",
                "val vs reference",
                "uncached vs reference",
                "subprocess vs reference",
                "elapsed vs reference",
            ],
            reduction_rows,
        ),
        "",
        "## Interpretation",
        "",
        "- Exact audit keeps the final PF and coverage measurable under the original MILP oracle.",
        "- Fixed short ranker caps can be too short on hard windows: they may preserve or improve PF after audit but still increase some costs.",
        "- Adaptive handoff is a budget-control diagnostic: it tests whether spending more ranker time before exact audit reduces the downstream exact workload on the same hard window.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare exact, fixed ranker+audit, and adaptive ranker budget variants on one window."
    )
    parser.add_argument("--window-summary-json", type=Path, required=True)
    parser.add_argument("--window-label", required=True)
    parser.add_argument("--fixed-label", default="fixed_ranker180_audit")
    parser.add_argument("--frontier-summary", action="append", type=labeled_path, default=[])
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    payload = compare(
        window_summary_json=args.window_summary_json,
        window_label=args.window_label,
        fixed_label=args.fixed_label,
        frontier_summaries=args.frontier_summary,
    )
    if args.out_json is not None:
        write_json(args.out_json, payload)
    if args.out_csv is not None:
        write_csv(args.out_csv, payload)
    if args.out_md is not None:
        write_markdown(args.out_md, payload)
    if args.out_json is None and args.out_csv is None and args.out_md is None:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
