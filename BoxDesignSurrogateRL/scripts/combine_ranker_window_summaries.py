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

from scripts.summarize_ranker_window_results import (  # noqa: E402
    make_total_row,
    markdown_table,
    reduction_markdown_table,
    write_csv,
    write_json,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_windows(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    windows = payload.get("windows") if isinstance(payload, dict) else None
    if not isinstance(windows, list):
        raise ValueError(f"summary JSON must contain a windows list: {path}")
    out = [row for row in windows if isinstance(row, dict)]
    if not out:
        raise ValueError(f"summary JSON has no window rows: {path}")
    return out


def combine_summaries(summary_paths: list[Path], *, total_label: str) -> dict[str, Any]:
    if not summary_paths:
        raise ValueError("at least one summary path is required")
    windows: list[dict[str, Any]] = []
    for path in summary_paths:
        windows.extend(load_windows(path))
    return {
        "source_summaries": [str(path) for path in summary_paths],
        "windows": windows,
        "total": make_total_row(windows, label=total_label),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    all_rows = payload["windows"] + [payload["total"]]
    lines = [
        "# Combined Ranker Window Summary",
        "",
        "Source summaries:",
        "",
    ]
    lines.extend(f"- `{source}`" for source in payload["source_summaries"])
    lines.extend(["", markdown_table(all_rows), "", reduction_markdown_table(all_rows), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine ranker window summary JSON files.")
    parser.add_argument("--summary-json", action="append", type=Path, required=True)
    parser.add_argument("--total-label", default="combined total")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    payload = combine_summaries(args.summary_json, total_label=args.total_label)
    if args.out_json:
        write_json(args.out_json, payload)
    if args.out_csv:
        write_csv(args.out_csv, payload["windows"] + [payload["total"]])
    if args.out_md:
        write_markdown(args.out_md, payload)
    if not args.out_json and not args.out_csv and not args.out_md:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
