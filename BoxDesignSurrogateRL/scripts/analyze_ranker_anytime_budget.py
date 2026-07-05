#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
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


def normalize_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    text = text.replace("\\", "/")
    prefixes = [
        "C:/Users/Lenovo/Downloads/Operation-Research/",
        "BoxDesignSurrogateRL/results/",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.strip("/")


def parse_budget_fractions(value: str) -> list[float]:
    out = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one budget fraction")
    if any(item <= 0.0 for item in out):
        raise argparse.ArgumentTypeError("budget fractions must be positive")
    return out


def numeric(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def integer(value: Any) -> int:
    num = numeric(value)
    return int(num) if num is not None else 0


def short_label(label: str) -> str:
    match = re.fullmatch(r"seed(\d+):test\[(\d+),(\d+)\)", label)
    if not match:
        return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return f"seed{match.group(1)}_o{match.group(2)}"


class ZipTraceStore:
    def __init__(self, paths: list[Path]):
        self.archives: list[zipfile.ZipFile] = []
        self.members: dict[str, tuple[zipfile.ZipFile, str]] = {}
        for path in paths:
            archive = zipfile.ZipFile(path)
            self.archives.append(archive)
            for member in archive.namelist():
                normalized = member.replace("\\", "/").strip("/")
                self.members[normalized] = (archive, member)

    def read_text(self, result_path: str | None) -> str | None:
        rel = normalize_path(result_path)
        if rel is None:
            return None
        for candidate in [rel, rel.removeprefix("results/")]:
            item = self.members.get(candidate)
            if item is None:
                continue
            archive, member = item
            with archive.open(member) as f:
                return f.read().decode("utf-8-sig")
        return None


@dataclass(frozen=True)
class TraceEvent:
    cumulative_validations: int
    packaging_factor: float
    phase: str
    action: str


def read_csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def trace_events(rows: list[dict[str, str]], *, start_validations: int = 0, phase_label: str) -> list[TraceEvent]:
    cumulative = int(start_validations)
    events: list[TraceEvent] = []
    for row in rows:
        cumulative += integer(row.get("milp_validated_candidates"))
        pf = numeric(row.get("packaging_factor"))
        if pf is None:
            continue
        events.append(
            TraceEvent(
                cumulative_validations=cumulative,
                packaging_factor=pf,
                phase=phase_label,
                action=str(row.get("action", "")),
            )
        )
    return events


def pf_at_budget(events: list[TraceEvent], budget: int) -> float | None:
    candidate: float | None = None
    for event in events:
        if event.cumulative_validations <= budget:
            candidate = event.packaging_factor
        else:
            break
    return candidate


def first_budget_reaching(events: list[TraceEvent], target_pf: float, *, tolerance: float) -> int | None:
    for event in events:
        if event.packaging_factor <= target_pf + tolerance:
            return event.cumulative_validations
    return None


def read_flat_trace(directory: Path | None, filename: str) -> str | None:
    if directory is None:
        return None
    path = directory / filename
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8-sig")


def load_frontier_ranker_run_dir(store: ZipTraceStore, row: dict[str, Any]) -> str | None:
    text = store.read_text(str(row.get("frontier_summary_path")))
    if text is None:
        return None
    payload = json.loads(text)
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        return None
    frontier_row = rows[0]
    if not isinstance(frontier_row, dict):
        return None
    return str(frontier_row.get("ranker_run_dir"))


def trace_texts_for_window(
    *,
    row: dict[str, Any],
    store: ZipTraceStore,
    exact_trace_dir: Path | None,
    extra_trace_dir: Path | None,
) -> tuple[str | None, str | None, str | None, list[str]]:
    label = str(row["label"])
    short = short_label(label)
    missing: list[str] = []

    exact_text = store.read_text(f"{row.get('exact_run_dir')}/trace.csv")
    exact_text = exact_text or read_flat_trace(exact_trace_dir, f"{short}_exact_trace.csv")
    if exact_text is None:
        missing.append("exact_trace")

    ranker_run_dir = load_frontier_ranker_run_dir(store, row)
    ranker_text = store.read_text(f"{ranker_run_dir}/trace.csv" if ranker_run_dir else None)
    ranker_text = ranker_text or read_flat_trace(extra_trace_dir, f"{short}_ranker_trace.csv")
    if ranker_text is None:
        missing.append("ranker_trace")

    audit_text = store.read_text(f"{row.get('ranker_audit_run_dir')}/trace.csv")
    audit_text = audit_text or read_flat_trace(extra_trace_dir, f"{short}_audit_trace.csv")
    if audit_text is None:
        missing.append("audit_trace")

    return exact_text, ranker_text, audit_text, missing


def analyze_window(
    *,
    row: dict[str, Any],
    store: ZipTraceStore,
    exact_trace_dir: Path | None,
    extra_trace_dir: Path | None,
    budget_fractions: list[float],
    pf_tolerance: float,
) -> dict[str, Any]:
    label = str(row["label"])
    exact_text, ranker_text, audit_text, missing = trace_texts_for_window(
        row=row,
        store=store,
        exact_trace_dir=exact_trace_dir,
        extra_trace_dir=extra_trace_dir,
    )
    out: dict[str, Any] = {"label": label, "trace_complete": not missing, "missing": missing}
    if missing or exact_text is None or ranker_text is None or audit_text is None:
        return out

    exact_events = trace_events(read_csv_rows(exact_text), phase_label="exact")
    ranker_events = trace_events(read_csv_rows(ranker_text), phase_label="ranker")
    audit_events = trace_events(
        read_csv_rows(audit_text),
        start_validations=ranker_events[-1].cumulative_validations if ranker_events else 0,
        phase_label="audit",
    )
    combined_events = ranker_events + audit_events
    if not exact_events or not combined_events:
        out.update({"trace_complete": False, "missing": ["empty_trace"]})
        return out

    exact_total = exact_events[-1].cumulative_validations
    ranker_total = combined_events[-1].cumulative_validations
    exact_final_pf = exact_events[-1].packaging_factor
    ranker_final_pf = combined_events[-1].packaging_factor
    budget_rows = []
    for fraction in budget_fractions:
        budget = max(0, int(round(exact_total * fraction)))
        exact_pf = pf_at_budget(exact_events, budget)
        ranker_pf = pf_at_budget(combined_events, budget)
        budget_rows.append(
            {
                "fraction_of_exact_validations": fraction,
                "budget_validations": budget,
                "exact_pf": exact_pf,
                "ranker_audit_pf": ranker_pf,
                "pf_delta_ranker_minus_exact": (
                    None if exact_pf is None or ranker_pf is None else ranker_pf - exact_pf
                ),
            }
        )

    out.update(
        {
            "exact_total_validations": exact_total,
            "ranker_audit_total_validations": ranker_total,
            "validation_reduction_pct": (
                (exact_total - ranker_total) / exact_total * 100.0 if exact_total > 0 else None
            ),
            "exact_final_pf": exact_final_pf,
            "ranker_audit_final_pf": ranker_final_pf,
            "final_pf_delta_ranker_minus_exact": ranker_final_pf - exact_final_pf,
            "exact_validations_to_exact_final_pf": first_budget_reaching(
                exact_events, exact_final_pf, tolerance=pf_tolerance
            ),
            "ranker_audit_validations_to_exact_final_pf": first_budget_reaching(
                combined_events, exact_final_pf, tolerance=pf_tolerance
            ),
            "budget_rows": budget_rows,
        }
    )
    return out


def summarize_budget_rows(windows: list[dict[str, Any]], budget_fractions: list[float], *, tolerance: float) -> list[dict]:
    complete = [row for row in windows if row.get("trace_complete")]
    summary = []
    for fraction in budget_fractions:
        deltas = []
        better = tied = worse = missing = 0
        for row in complete:
            match = [
                budget_row
                for budget_row in row["budget_rows"]
                if abs(float(budget_row["fraction_of_exact_validations"]) - fraction) <= 1e-12
            ]
            if not match:
                missing += 1
                continue
            delta = numeric(match[0].get("pf_delta_ranker_minus_exact"))
            if delta is None:
                missing += 1
                continue
            deltas.append(delta)
            if delta < -tolerance:
                better += 1
            elif delta > tolerance:
                worse += 1
            else:
                tied += 1
        summary.append(
            {
                "fraction_of_exact_validations": fraction,
                "n": len(deltas),
                "ranker_better": better,
                "ranker_tied": tied,
                "ranker_worse": worse,
                "missing": missing,
                "mean_pf_delta_ranker_minus_exact": sum(deltas) / len(deltas) if deltas else None,
                "min_pf_delta_ranker_minus_exact": min(deltas) if deltas else None,
                "max_pf_delta_ranker_minus_exact": max(deltas) if deltas else None,
            }
        )
    return summary


def summarize_time_to_final(windows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (row["exact_validations_to_exact_final_pf"], row["ranker_audit_validations_to_exact_final_pf"])
        for row in windows
        if row.get("trace_complete")
        and row.get("exact_validations_to_exact_final_pf") is not None
        and row.get("ranker_audit_validations_to_exact_final_pf") is not None
    ]
    exact_sum = sum(float(exact) for exact, _ranker in pairs)
    ranker_sum = sum(float(ranker) for _exact, ranker in pairs)
    return {
        "n": len(pairs),
        "exact_sum": exact_sum if pairs else None,
        "ranker_audit_sum": ranker_sum if pairs else None,
        "aggregate_reduction_pct": ((exact_sum - ranker_sum) / exact_sum * 100.0) if exact_sum else None,
        "windows_reduced": sum(1 for exact, ranker in pairs if ranker < exact),
        "windows_tied": sum(1 for exact, ranker in pairs if ranker == exact),
        "windows_increased": sum(1 for exact, ranker in pairs if ranker > exact),
    }


def analyze(
    *,
    summary_json: Path,
    zip_paths: list[Path],
    exact_trace_dir: Path | None,
    extra_trace_dir: Path | None,
    budget_fractions: list[float],
    pf_tolerance: float,
) -> dict[str, Any]:
    payload = load_json(summary_json)
    rows = payload.get("windows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"summary JSON must contain windows list: {summary_json}")
    store = ZipTraceStore(zip_paths)
    windows = [
        analyze_window(
            row=row,
            store=store,
            exact_trace_dir=exact_trace_dir,
            extra_trace_dir=extra_trace_dir,
            budget_fractions=budget_fractions,
            pf_tolerance=pf_tolerance,
        )
        for row in rows
        if isinstance(row, dict)
    ]
    complete = [row for row in windows if row.get("trace_complete")]
    return {
        "summary_json": str(summary_json),
        "zip_paths": [str(path) for path in zip_paths],
        "exact_trace_dir": str(exact_trace_dir) if exact_trace_dir else None,
        "extra_trace_dir": str(extra_trace_dir) if extra_trace_dir else None,
        "budget_fractions": budget_fractions,
        "pf_tolerance": pf_tolerance,
        "n_windows": len(windows),
        "trace_complete_windows": len(complete),
        "missing_windows": [row for row in windows if not row.get("trace_complete")],
        "budget_summary": summarize_budget_rows(windows, budget_fractions, tolerance=pf_tolerance),
        "time_to_exact_final_pf": summarize_time_to_final(windows),
        "windows": windows,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    time_to_final = payload["time_to_exact_final_pf"]
    lines = [
        "# Ranker-Audit Anytime Budget Analysis",
        "",
        f"- Windows in summary: {payload['n_windows']}",
        f"- Windows with complete traces: {payload['trace_complete_windows']}",
        f"- PF tolerance: `{payload['pf_tolerance']}`",
        "",
        "## Validation Budget Comparison",
        "",
        "| exact validation budget | n | ranker better/tied/worse | mean PF delta | min PF delta | max PF delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["budget_summary"]:
        counts = f"{row['ranker_better']}/{row['ranker_tied']}/{row['ranker_worse']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['fraction_of_exact_validations']) * 100:.0f}%",
                    str(row["n"]),
                    counts,
                    fmt(row["mean_pf_delta_ranker_minus_exact"], digits=10),
                    fmt(row["min_pf_delta_ranker_minus_exact"], digits=10),
                    fmt(row["max_pf_delta_ranker_minus_exact"], digits=10),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Time To Exact Final PF",
            "",
            f"- Windows: {time_to_final['n']}",
            f"- Exact validations: {fmt(time_to_final['exact_sum'], digits=0)}",
            f"- Ranker-audit validations: {fmt(time_to_final['ranker_audit_sum'], digits=0)}",
            f"- Aggregate reduction: {fmt(time_to_final['aggregate_reduction_pct'], digits=2)}%",
            (
                "- Windows reduced/tied/increased: "
                f"{time_to_final['windows_reduced']}/{time_to_final['windows_tied']}/"
                f"{time_to_final['windows_increased']}"
            ),
            "",
            "## Per-Window Final Validation Cost",
            "",
            "| window | exact validations | ranker-audit validations | reduction | final PF delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["windows"]:
        if not row.get("trace_complete"):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["exact_total_validations"]),
                    str(row["ranker_audit_total_validations"]),
                    f"{float(row['validation_reduction_pct']):.1f}%",
                    fmt(row["final_pf_delta_ranker_minus_exact"], digits=10),
                ]
            )
            + " |"
        )
    if payload["missing_windows"]:
        lines.extend(["", "## Missing Trace Windows", ""])
        for row in payload["missing_windows"]:
            lines.append(f"- {row['label']}: {', '.join(row['missing'])}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay exact and ranker-audit traces under matched exact-validation budgets."
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_paths", action="append", type=Path, default=None)
    parser.add_argument("--exact-trace-dir", type=Path, default=None)
    parser.add_argument("--extra-trace-dir", type=Path, default=None)
    parser.add_argument("--budget-fractions", type=parse_budget_fractions, default=parse_budget_fractions("0.25,0.5,0.75,1.0"))
    parser.add_argument("--pf-tolerance", type=float, default=1e-12)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    if args.pf_tolerance < 0.0:
        raise ValueError("--pf-tolerance must be non-negative")
    if not args.zip_paths:
        raise ValueError("at least one --zip trace bundle is required")

    payload = analyze(
        summary_json=args.summary_json,
        zip_paths=args.zip_paths,
        exact_trace_dir=args.exact_trace_dir,
        extra_trace_dir=args.extra_trace_dir,
        budget_fractions=args.budget_fractions,
        pf_tolerance=args.pf_tolerance,
    )
    if args.out_json:
        write_json(args.out_json, payload)
    if args.out_md:
        write_markdown(args.out_md, payload)
    if not args.out_json and not args.out_md:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
