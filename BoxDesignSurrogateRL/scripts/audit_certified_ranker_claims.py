#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_ranker_window_statistics import COST_METRICS, analyze_summary  # noqa: E402


def optional_float(text: str) -> float | None:
    if text.lower() in {"none", "off", "disabled"}:
        return None
    return float(text)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "missing"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pct(value: Any) -> str:
    if value is None:
        return "missing"
    return f"{float(value):.2f}%"


def fmt_ci(ci: dict[str, Any] | None) -> str:
    if not ci:
        return "missing"
    low = ci.get("low")
    high = ci.get("high")
    if low is None or high is None:
        return "missing"
    return f"[{float(low):.2f}%, {float(high):.2f}%]"


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def audit_claims(
    summary_json: Path,
    *,
    min_windows: int = 15,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 20260705,
    pf_tolerance: float = 1e-12,
    min_coverage: float = 1.0,
    coverage_tolerance: float = 1e-12,
    max_uncovered: float = 0.0,
    min_aggregate_cost_reduction_pct: float = 0.0,
    max_cost_increased_windows: int = 0,
    min_bootstrap_low_pct: float | None = 0.0,
    max_sign_test_p: float | None = 0.05,
) -> dict[str, Any]:
    if min_windows <= 0:
        raise ValueError("min_windows must be positive")
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be non-negative")
    if pf_tolerance < 0.0:
        raise ValueError("pf_tolerance must be non-negative")
    if coverage_tolerance < 0.0:
        raise ValueError("coverage_tolerance must be non-negative")
    if max_uncovered < 0.0:
        raise ValueError("max_uncovered must be non-negative")
    if max_cost_increased_windows < 0:
        raise ValueError("max_cost_increased_windows must be non-negative")

    analysis = analyze_summary(
        summary_json,
        bootstrap_iterations=bootstrap_iterations,
        seed=bootstrap_seed,
        pf_tolerance=pf_tolerance,
    )
    checks: list[dict[str, Any]] = []

    n_windows = int(analysis["n_windows"])
    add_check(
        checks,
        "paired_window_count",
        n_windows >= min_windows,
        f"windows={n_windows}, required>={min_windows}",
    )

    pf = analysis["pf"]
    max_regression = numeric(pf.get("max_regression"))
    pf_n = int(pf.get("n", 0))
    add_check(
        checks,
        "pf_complete",
        pf_n == n_windows,
        f"PF deltas={pf_n}, windows={n_windows}",
    )
    add_check(
        checks,
        "no_pf_regression_after_exact_audit",
        int(pf["regressed"]) == 0 and max_regression is not None and max_regression <= pf_tolerance,
        (
            f"matched/improved/regressed={pf['matched']}/{pf['improved']}/{pf['regressed']}, "
            f"max_regression={fmt(max_regression, digits=10)}, tolerance={pf_tolerance:g}"
        ),
    )

    coverage = analysis["coverage"]
    min_exact_coverage = numeric(coverage.get("min_exact_coverage"))
    min_ranker_coverage = numeric(coverage.get("min_ranker_audit_coverage"))
    exact_uncovered = numeric(coverage.get("total_exact_uncovered"))
    ranker_uncovered = numeric(coverage.get("total_ranker_audit_uncovered"))
    add_check(
        checks,
        "exact_baseline_feasible",
        (
            min_exact_coverage is not None
            and min_exact_coverage >= min_coverage - coverage_tolerance
            and exact_uncovered is not None
            and exact_uncovered <= max_uncovered
        ),
        f"min_coverage={fmt(min_exact_coverage)}, uncovered={fmt(exact_uncovered, digits=0)}",
    )
    add_check(
        checks,
        "ranker_audit_feasible",
        (
            min_ranker_coverage is not None
            and min_ranker_coverage >= min_coverage - coverage_tolerance
            and ranker_uncovered is not None
            and ranker_uncovered <= max_uncovered
        ),
        f"min_coverage={fmt(min_ranker_coverage)}, uncovered={fmt(ranker_uncovered, digits=0)}",
    )

    for metric_name in COST_METRICS:
        row = analysis["cost"][metric_name]
        n_metric = int(row["n"])
        aggregate = numeric(row.get("aggregate_reduction_pct"))
        increased = int(row["windows_increased"])
        sign_p = numeric(row.get("sign_test_two_sided_p_excluding_ties"))
        ci = row["bootstrap"]["mean_reduction_pct_ci95"]
        ci_low = numeric(ci.get("low")) if isinstance(ci, dict) else None

        metric_passed = (
            n_metric == n_windows
            and aggregate is not None
            and aggregate > min_aggregate_cost_reduction_pct
            and increased <= max_cost_increased_windows
        )
        if min_bootstrap_low_pct is not None:
            metric_passed = metric_passed and ci_low is not None and ci_low > min_bootstrap_low_pct
        if max_sign_test_p is not None:
            metric_passed = metric_passed and sign_p is not None and sign_p <= max_sign_test_p

        add_check(
            checks,
            f"cost_{metric_name}",
            metric_passed,
            (
                f"n={n_metric}, aggregate_reduction={fmt_pct(aggregate)}, "
                f"bootstrap95={fmt_ci(ci)}, reduced/tied/increased="
                f"{row['windows_reduced']}/{row['windows_tied']}/{row['windows_increased']}, "
                f"sign_p={fmt(sign_p, digits=6)}"
            ),
        )

    passed = all(check["passed"] for check in checks)
    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "summary_json": str(summary_json),
        "thresholds": {
            "min_windows": min_windows,
            "pf_tolerance": pf_tolerance,
            "min_coverage": min_coverage,
            "coverage_tolerance": coverage_tolerance,
            "max_uncovered": max_uncovered,
            "min_aggregate_cost_reduction_pct": min_aggregate_cost_reduction_pct,
            "max_cost_increased_windows": max_cost_increased_windows,
            "min_bootstrap_low_pct": min_bootstrap_low_pct,
            "max_sign_test_p": max_sign_test_p,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "checks": checks,
        "analysis": analysis,
    }


def escape_md(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    analysis = audit["analysis"]
    pf = analysis["pf"]
    coverage = analysis["coverage"]
    lines = [
        "# Certified Ranker-Audit Claim Audit",
        "",
        f"- Verdict: {audit['verdict']}",
        f"- Summary JSON: `{audit['summary_json']}`",
        f"- Windows: {analysis['n_windows']}",
        f"- Bootstrap iterations: {analysis['bootstrap_iterations']}",
        f"- Bootstrap seed: {analysis['bootstrap_seed']}",
        "- Bootstrap intervals are descriptive window resampling checks, not iid guarantees.",
        "",
        "## Acceptance Checks",
        "",
        "| check | result | detail |",
        "| --- | ---: | --- |",
    ]
    for check in audit["checks"]:
        result = "PASS" if check["passed"] else "FAIL"
        lines.append(f"| {escape_md(check['name'])} | {result} | {escape_md(check['detail'])} |")

    lines.extend(
        [
            "",
            "## Quality And Coverage",
            "",
            f"- PF matched/improved/regressed: {pf['matched']}/{pf['improved']}/{pf['regressed']}",
            f"- Mean PF delta: {fmt(pf['mean_delta'], digits=10)}",
            f"- Max PF regression: {fmt(pf['max_regression'], digits=10)}",
            f"- Max PF improvement: {fmt(pf['max_improvement'], digits=10)}",
            f"- Min exact coverage: {fmt(coverage['min_exact_coverage'])}",
            f"- Min ranker+audit coverage: {fmt(coverage['min_ranker_audit_coverage'])}",
            f"- Total exact uncovered: {fmt(coverage['total_exact_uncovered'], digits=0)}",
            f"- Total ranker+audit uncovered: {fmt(coverage['total_ranker_audit_uncovered'], digits=0)}",
            "",
            "## Cost Metrics",
            "",
            "| metric | aggregate reduction | bootstrap 95% CI | reduced/tied/increased | sign-test p |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric_name in COST_METRICS:
        row = analysis["cost"][metric_name]
        ci = row["bootstrap"]["mean_reduction_pct_ci95"]
        counts = f"{row['windows_reduced']}/{row['windows_tied']}/{row['windows_increased']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    metric_name,
                    fmt_pct(row["aggregate_reduction_pct"]),
                    fmt_ci(ci),
                    counts,
                    fmt(row["sign_test_two_sided_p_excluding_ties"], digits=6),
                ]
            )
            + " |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether a paired exact-vs-ranker summary supports the certified ranker-audit claim."
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--min-windows", type=int, default=15)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260705)
    parser.add_argument("--pf-tolerance", type=float, default=1e-12)
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--coverage-tolerance", type=float, default=1e-12)
    parser.add_argument("--max-uncovered", type=float, default=0.0)
    parser.add_argument("--min-aggregate-cost-reduction-pct", type=float, default=0.0)
    parser.add_argument("--max-cost-increased-windows", type=int, default=0)
    parser.add_argument("--min-bootstrap-low-pct", type=optional_float, default=0.0)
    parser.add_argument("--max-sign-test-p", type=optional_float, default=0.05)
    args = parser.parse_args()

    audit = audit_claims(
        args.summary_json,
        min_windows=args.min_windows,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        pf_tolerance=args.pf_tolerance,
        min_coverage=args.min_coverage,
        coverage_tolerance=args.coverage_tolerance,
        max_uncovered=args.max_uncovered,
        min_aggregate_cost_reduction_pct=args.min_aggregate_cost_reduction_pct,
        max_cost_increased_windows=args.max_cost_increased_windows,
        min_bootstrap_low_pct=args.min_bootstrap_low_pct,
        max_sign_test_p=args.max_sign_test_p,
    )
    if args.out_json:
        write_json(args.out_json, audit)
    if args.out_md:
        write_markdown(args.out_md, audit)
    if not args.out_json and not args.out_md:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
