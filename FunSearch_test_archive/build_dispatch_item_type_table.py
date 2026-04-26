# -*- coding: utf-8 -*-
"""
Build dispatch-level item type counts from:
  - training_2orientations.csv (source of truth for dispatch ids and if_loaded)
  - 物品信息和dblf信息.csv (per-SKU dimensions)

Output CSV schema (long format, one row per dispatch x canonical size type):
  dispatch_id   : 发车号 (int)
  if_loaded     : from training (0/1)
  n_types       : number of distinct size types for this dispatch (same on all rows of dispatch)
  dim_s, dim_m, dim_l : sorted SKU box dimensions (unordered triple -> ascending canonical form)
  n_items       : count of items of this type in this dispatch

Reading:
  - pandas: pd.read_csv(..., dtype={...})
  - numpy: np.genfromtxt or loadtxt (skip header)
  - torch: torch.tensor(pd.read_csv(...).values) or from_numpy
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
TRAINING = ROOT / "training_2orientations.csv"
ITEMS = ROOT / "物品信息和dblf信息.csv"
OUT = ROOT / "training_dispatch_item_types.csv"

CHUNKSIZE = 300_000


def main() -> None:
    train = pd.read_csv(
        TRAINING,
        usecols=["发车号", "if_loaded"],
        encoding="utf-8",
        dtype={"发车号": np.int64, "if_loaded": np.int8},
    )
    if train["发车号"].duplicated().any():
        train = train.drop_duplicates(subset=["发车号"], keep="first")
    train_ids = set(train["发车号"].tolist())
    train = train.set_index("发车号")

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        ITEMS,
        usecols=["发车号", "SKU长度", "SKU宽度", "SKU高度"],
        encoding="utf-8",
        chunksize=CHUNKSIZE,
        dtype={"发车号": np.int64, "SKU长度": np.int32, "SKU宽度": np.int32, "SKU高度": np.int32},
    ):
        sub = chunk[chunk["发车号"].isin(train_ids)]
        if sub.empty:
            continue
        dims = sub[["SKU长度", "SKU宽度", "SKU高度"]].to_numpy()
        sorted_dims = np.sort(dims, axis=1)
        sub = sub[["发车号"]].copy()
        sub["dim_s"] = sorted_dims[:, 0]
        sub["dim_m"] = sorted_dims[:, 1]
        sub["dim_l"] = sorted_dims[:, 2]
        parts.append(sub)

    if not parts:
        raise RuntimeError("No item rows matched training dispatch ids.")

    items = pd.concat(parts, ignore_index=True)

    grouped = (
        items.groupby(["发车号", "dim_s", "dim_m", "dim_l"], sort=True)
        .size()
        .reset_index(name="n_items")
    )

    n_types = grouped.groupby("发车号", sort=False).size().reset_index(name="n_types")
    out = grouped.merge(n_types, on="发车号", how="left")

    out["if_loaded"] = out["发车号"].map(train["if_loaded"]).astype(np.int8)
    if out["if_loaded"].isna().any():
        bad = out[out["if_loaded"].isna()]["发车号"].unique()
        raise RuntimeError(f"Missing if_loaded for dispatch ids: {bad[:10]}")

    out = out.rename(columns={"发车号": "dispatch_id"})
    out = out[
        ["dispatch_id", "if_loaded", "n_types", "dim_s", "dim_m", "dim_l", "n_items"]
    ]
    out = out.sort_values(["dispatch_id", "dim_s", "dim_m", "dim_l"], kind="mergesort").reset_index(
        drop=True
    )

    out.to_csv(OUT, index=False, encoding="utf-8")

    n_dispatch = out["dispatch_id"].nunique()
    assert n_dispatch == len(train_ids), (n_dispatch, len(train_ids))
    print(f"Wrote {OUT}")
    print(f"  rows: {len(out)}, dispatches: {n_dispatch}")


if __name__ == "__main__":
    main()
