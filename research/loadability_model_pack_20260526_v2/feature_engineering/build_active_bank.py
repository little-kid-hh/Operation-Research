# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent
TRIALS_ROOT = ROOT / "accepted_trials"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # Inject dependencies that some candidates might miss
    mod.pd = pd
    mod.np = np
    spec.loader.exec_module(mod)
    return mod


def build_active_bank(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for trial_dir in sorted(TRIALS_ROOT.glob("iter_*")):
        mod = _load_module(trial_dir / "feature_candidate.py")
        out = mod.build_candidate_features(agg_df, items_df)
        if "dispatch_id" not in out.columns:
            raise ValueError(f"{trial_dir} did not return dispatch_id")
        parts.append(out.reset_index(drop=True))
    bank = parts[0]
    for nxt in parts[1:]:
        bank = bank.merge(nxt, on="dispatch_id", how="inner")
    return bank

