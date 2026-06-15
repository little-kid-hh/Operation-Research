from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run_dir",
    "run_type",
    "mode",
    "status",
    "orders_limit",
    "k",
    "step_size",
    "tau",
    "initial_objective",
    "final_or_best_objective",
    "objective_delta",
    "initial_uncovered",
    "final_or_best_uncovered",
    "initial_low_margin",
    "final_or_best_low_margin",
    "initial_mean_probability",
    "final_or_best_mean_probability",
]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_last_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def first_existing(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def as_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_run(run_dir: Path) -> dict[str, object]:
    manifest = read_json(run_dir / "manifest.json")
    summary = read_json(run_dir / "summary.json")
    train_final = read_last_csv_row(run_dir / "train_summary.csv")
    params = manifest.get("parameters", {})

    if summary:
        run_type = "paas"
        initial_objective = summary.get("initial_objective", "")
        final_objective = summary.get("best_objective", "")
        initial_uncovered = summary.get("initial_uncovered", "")
        final_uncovered = summary.get("best_uncovered", "")
        initial_low_margin = summary.get("initial_low_margin", "")
        final_low_margin = summary.get("best_low_margin", "")
        initial_mean_probability = summary.get("initial_mean_probability", "")
        final_mean_probability = summary.get("best_mean_probability", "")
        mode = summary.get("mode", params.get("mode", ""))
    else:
        run_type = "policy"
        initial_objective = train_final.get("initial_objective", "")
        final_objective = train_final.get("final_objective", "")
        initial_uncovered = train_final.get("initial_uncovered", "")
        final_uncovered = train_final.get("final_uncovered", "")
        initial_low_margin = train_final.get("initial_low_margin", "")
        final_low_margin = train_final.get("final_low_margin", "")
        initial_mean_probability = train_final.get("initial_mean_probability", "")
        final_mean_probability = train_final.get("final_mean_probability", "")
        mode = train_final.get("mode", params.get("mode", ""))

    start = as_float(initial_objective)
    end = as_float(final_objective)
    delta = "" if start is None or end is None else end - start
    return {
        "run_dir": str(run_dir),
        "run_type": run_type,
        "mode": mode,
        "status": manifest.get("status", ""),
        "orders_limit": params.get("orders_limit", ""),
        "k": params.get("k", ""),
        "step_size": params.get("step_size", ""),
        "tau": params.get("tau", ""),
        "initial_objective": initial_objective,
        "final_or_best_objective": final_objective,
        "objective_delta": delta,
        "initial_uncovered": initial_uncovered,
        "final_or_best_uncovered": final_uncovered,
        "initial_low_margin": initial_low_margin,
        "final_or_best_low_margin": final_low_margin,
        "initial_mean_probability": initial_mean_probability,
        "final_or_best_mean_probability": final_mean_probability,
    }


def expand_inputs(paths: list[Path]) -> list[Path]:
    out = []
    for path in paths:
        if (path / "manifest.json").exists():
            out.append(path)
            continue
        out.extend(sorted(child for child in path.rglob("run_*") if (child / "manifest.json").exists()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [summarize_run(path) for path in expand_inputs(args.paths)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"runs={len(rows)} saved={args.out}")


if __name__ == "__main__":
    main()
