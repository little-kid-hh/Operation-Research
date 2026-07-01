#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from box_design_surrogate.splits import materialize_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize deterministic hash splits for OR2023 XML orders.")
    parser.add_argument(
        "--source-xml",
        type=Path,
        default=ROOT / "assets/or2023_bsp_data/xml_unique/or2023_bsp_unique_orders.xml",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--dev-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = materialize_splits(
        source_xml=args.source_xml,
        out_dir=args.out_dir,
        seed=args.seed,
        train_frac=args.train_frac,
        dev_frac=args.dev_frac,
        test_frac=args.test_frac,
        limit=args.limit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
