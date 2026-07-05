#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


COST_METRICS = {
    "validations": ("exact_validations", "ranker_audit_validations"),
    "uncached_boxes": ("exact_uncached_boxes", "ranker_audit_uncached_boxes"),
    "subprocess_seconds": ("exact_subprocess_seconds", "ranker_audit_subprocess_seconds"),
    "elapsed_seconds": ("exact_elapsed_seconds", "ranker_audit_elapsed_seconds"),
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if pct <= 0.0:
        return sorted_values[0]
    if pct >= 100.0:
        return sorted_values[-1]
    idx = (len(sorted_values) - 1) * pct / 100.0
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return sorted_values[lower]
    weight = idx - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def percentile_interval(values: list[float], *, low_pct: float = 2.5, high_pct: float = 97.5) -> dict[str, float | None]:
    sorted_values = sorted(values)
    return {
        "low": percentile(sorted_values, low_pct),
        "high": percentile(sorted_values, high_pct),
    }


def sign_test_two_sided_p(positive: int, negative: int) -> float | None:
    n = positive + negative
    if n == 0:
        return None
    k = min(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def reduction_pct(exact: float, ranker: float) -> float | None:
    if exact == 0.0:
        return None
    return (exact - ranker) / exact * 100.0


def bootstrap_metric(
    windows: list[dict[str, Any]],
    *,
    exact_key: str,
    ranker_key: str,
    iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    valid_rows = [
        (exact, ranker)
        for row in windows
        for exact, ranker in [(numeric(row.get(exact_key)), numeric(row.get(ranker_key)))]
        if exact is not None and ranker is not None
    ]
    if not valid_rows:
        return {"iterations": 0, "mean_reduction_pct_ci95": {"low": None, "high": None}}
    if iterations <= 0:
        return {"iterations": 0, "mean_reduction_pct_ci95": {"low": None, "high": None}}

    samples: list[float] = []
    n = len(valid_rows)
    for _ in range(iterations):
        exact_sum = 0.0
        ranker_sum = 0.0
        for _sample_idx in range(n):
            exact, ranker = valid_rows[rng.randrange(n)]
            exact_sum += exact
            ranker_sum += ranker
        pct = reduction_pct(exact_sum, ranker_sum)
        if pct is not None:
            samples.append(pct)
    ci = percentile_interval(samples)
    return {
        "iterations": iterations,
        "mean_reduction_pct_ci95": ci,
    }


def bootstrap_pf_delta(
    windows: list[dict[str, Any]],
    *,
    iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    deltas = [delta for row in windows for delta in [numeric(row.get("pf_delta"))] if delta is not None]
    if not deltas or iterations <= 0:
        return {"iterations": 0, "mean_delta_ci95": {"low": None, "high": None}}
    samples: list[float] = []
    n = len(deltas)
    for _ in range(iterations):
        samples.append(sum(deltas[rng.randrange(n)] for _idx in range(n)) / n)
    return {
        "iterations": iterations,
        "mean_delta_ci95": percentile_interval(samples),
    }


def summarize_pf(windows: list[dict[str, Any]], *, tolerance: float, iterations: int, rng: random.Random) -> dict[str, Any]:
    deltas = [delta for row in windows for delta in [numeric(row.get("pf_delta"))] if delta is not None]
    improved = sum(1 for delta in deltas if delta < -tolerance)
    regressed = sum(1 for delta in deltas if delta > tolerance)
    matched = sum(1 for delta in deltas if abs(delta) <= tolerance)
    return {
        "n": len(deltas),
        "matched": matched,
        "improved": improved,
        "regressed": regressed,
        "mean_delta": mean(deltas),
        "sd_delta": sample_sd(deltas),
        "max_regression": max(deltas) if deltas else None,
        "max_improvement": min(deltas) if deltas else None,
        "sign_test_two_sided_p_excluding_ties": sign_test_two_sided_p(improved, regressed),
        "bootstrap": bootstrap_pf_delta(windows, iterations=iterations, rng=rng),
    }


def summarize_coverage(windows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_coverage = [value for row in windows for value in [numeric(row.get("exact_coverage"))] if value is not None]
    ranker_coverage = [
        value for row in windows for value in [numeric(row.get("ranker_audit_coverage"))] if value is not None
    ]
    exact_uncovered = [value for row in windows for value in [numeric(row.get("exact_uncovered"))] if value is not None]
    ranker_uncovered = [
        value for row in windows for value in [numeric(row.get("ranker_audit_uncovered"))] if value is not None
    ]
    return {
        "min_exact_coverage": min(exact_coverage) if exact_coverage else None,
        "min_ranker_audit_coverage": min(ranker_coverage) if ranker_coverage else None,
        "total_exact_uncovered": sum(exact_uncovered) if exact_uncovered else None,
        "total_ranker_audit_uncovered": sum(ranker_uncovered) if ranker_uncovered else None,
    }


def summarize_cost_metric(
    windows: list[dict[str, Any]],
    *,
    exact_key: str,
    ranker_key: str,
    iterations: int,
    rng: random.Random,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    pairs = [
        (exact, ranker)
        for row in windows
        for exact, ranker in [(numeric(row.get(exact_key)), numeric(row.get(ranker_key)))]
        if exact is not None and ranker is not None
    ]
    reductions = [pct for exact, ranker in pairs for pct in [reduction_pct(exact, ranker)] if pct is not None]
    exact_sum = sum(exact for exact, _ranker in pairs)
    ranker_sum = sum(ranker for _exact, ranker in pairs)
    aggregate_reduction = reduction_pct(exact_sum, ranker_sum) if pairs else None
    reduced = sum(1 for exact, ranker in pairs if ranker < exact - tolerance)
    increased = sum(1 for exact, ranker in pairs if ranker > exact + tolerance)
    tied = len(pairs) - reduced - increased
    return {
        "n": len(pairs),
        "exact_sum": exact_sum if pairs else None,
        "ranker_audit_sum": ranker_sum if pairs else None,
        "absolute_delta": (ranker_sum - exact_sum) if pairs else None,
        "aggregate_reduction_pct": aggregate_reduction,
        "window_reduction_pct_mean": mean(reductions),
        "window_reduction_pct_sd": sample_sd(reductions),
        "windows_reduced": reduced,
        "windows_tied": tied,
        "windows_increased": increased,
        "sign_test_two_sided_p_excluding_ties": sign_test_two_sided_p(reduced, increased),
        "bootstrap": bootstrap_metric(
            windows,
            exact_key=exact_key,
            ranker_key=ranker_key,
            iterations=iterations,
            rng=rng,
        ),
    }


def load_windows(summary_path: Path) -> list[dict[str, Any]]:
    payload = load_json(summary_path)
    windows = payload.get("windows") if isinstance(payload, dict) else None
    if not isinstance(windows, list):
        raise ValueError(f"summary JSON must contain a windows list: {summary_path}")
    return [row for row in windows if isinstance(row, dict)]


def analyze_summary(
    summary_path: Path,
    *,
    bootstrap_iterations: int = 10000,
    seed: int = 20260705,
    pf_tolerance: float = 1e-12,
) -> dict[str, Any]:
    windows = load_windows(summary_path)
    rng = random.Random(seed)
    cost = {
        name: summarize_cost_metric(
            windows,
            exact_key=exact_key,
            ranker_key=ranker_key,
            iterations=bootstrap_iterations,
            rng=rng,
        )
        for name, (exact_key, ranker_key) in COST_METRICS.items()
    }
    return {
        "summary_path": str(summary_path),
        "n_windows": len(windows),
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": seed,
        "pf_tolerance": pf_tolerance,
        "pf": summarize_pf(windows, tolerance=pf_tolerance, iterations=bootstrap_iterations, rng=rng),
        "coverage": summarize_coverage(windows),
        "cost": cost,
        "notes": [
            "Bootstrap confidence intervals resample windows with replacement and are descriptive, not a formal test of independent random samples.",
            "Sign tests exclude ties and should be interpreted cautiously when windows share data construction and code paths.",
        ],
    }


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_ci(ci: dict[str, Any] | None, *, digits: int = 2, suffix: str = "") -> str:
    if not ci:
        return ""
    low = ci.get("low")
    high = ci.get("high")
    if low is None or high is None:
        return ""
    return f"[{float(low):.{digits}f}{suffix}, {float(high):.{digits}f}{suffix}]"


def markdown_table(analysis: dict[str, Any]) -> str:
    cost = analysis["cost"]
    lines = [
        "| metric | aggregate reduction | bootstrap 95% CI | windows reduced/tied/increased | sign-test p |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in COST_METRICS:
        row = cost[name]
        ci = row["bootstrap"]["mean_reduction_pct_ci95"]
        counts = f"{row['windows_reduced']}/{row['windows_tied']}/{row['windows_increased']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{row['aggregate_reduction_pct']:.2f}%",
                    format_ci(ci, digits=2, suffix="%"),
                    counts,
                    format_float(row["sign_test_two_sided_p_excluding_ties"], digits=4),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    pf = analysis["pf"]
    coverage = analysis["coverage"]
    lines = [
        "# Ranker Window Statistical Summary",
        "",
        f"- Windows: {analysis['n_windows']}",
        f"- Bootstrap iterations: {analysis['bootstrap_iterations']}",
        f"- Bootstrap seed: {analysis['bootstrap_seed']}",
        "- Confidence intervals are descriptive window-bootstrap intervals, not formal iid statistical guarantees.",
        "",
        "## Quality",
        "",
        f"- PF matched/improved/regressed: {pf['matched']}/{pf['improved']}/{pf['regressed']}",
        f"- Mean PF delta: {format_float(pf['mean_delta'], digits=10)}",
        f"- Max PF regression: {format_float(pf['max_regression'], digits=10)}",
        f"- Max PF improvement: {format_float(pf['max_improvement'], digits=10)}",
        f"- Mean PF delta bootstrap 95% CI: {format_ci(pf['bootstrap']['mean_delta_ci95'], digits=10)}",
        "",
        "## Coverage",
        "",
        f"- Min exact coverage: {format_float(coverage['min_exact_coverage'], digits=4)}",
        f"- Min ranker+audit coverage: {format_float(coverage['min_ranker_audit_coverage'], digits=4)}",
        f"- Total exact uncovered: {format_float(coverage['total_exact_uncovered'], digits=0)}",
        f"- Total ranker+audit uncovered: {format_float(coverage['total_ranker_audit_uncovered'], digits=0)}",
        "",
        "## Cost",
        "",
        markdown_table(analysis),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired exact vs ranker+audit window summary statistics.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260705)
    parser.add_argument("--pf-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    if args.bootstrap_iterations < 0:
        raise ValueError("--bootstrap-iterations must be non-negative")
    if args.pf_tolerance < 0.0:
        raise ValueError("--pf-tolerance must be non-negative")

    analysis = analyze_summary(
        args.summary_json,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
        pf_tolerance=args.pf_tolerance,
    )
    if args.out_json:
        write_json(args.out_json, analysis)
    if args.out_md:
        write_markdown(args.out_md, analysis)
    if not args.out_json and not args.out_md:
        print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
