# XGBoost-Guided Feature Hypotheses for HybridSVM

> Status: exploratory only. This note mixes later active-bank features into the
> teacher analysis and should not be used as the formal base40-only guidance
> source for the main reproducible route.

This note is the XGBoost-specific guidance input for the `HybridSVM` feature
search route. It is separate from `TREE_INSPIRED_FEATURE_HYPOTHESES.md`.

## Why this note exists

The tree-guided route already showed that nonlinear structure can be translated
into useful explicit features for the linear SVM. The next question is narrower:

> if the current strongest single model is XGBoost, what parts of its advantage
> over the linear SVM look most convertible into compact, interpretable
> per-dispatch features?

This file is intended to be used as a formal prompt input through:

- `HybridSVM/scripts/run_feature_search_agent.py --tree-guidance-path ...`

For this workflow, the argument name is legacy. The actual guidance source here
is XGBoost, not generic tree notes.

## Reproducible evidence source

Teacher-model evidence for this note comes from these local artifacts:

- fixed active bank:
  `HybridSVM/experiments_feature_search/by_model/glm-5.1/exp_20260513_235323/active_feature_bank.csv`
- XGBoost H21 hyperparameter search:
  `Ensemble_baseline/experiments/xgb_search_20260514_143602_base40_plus_h21/summary.json`
- XGBoost guidance package:
  `research/xgb_guidance_20260514/summary.json`
  `research/xgb_guidance_20260514/base40_plus_h21_gain_importance.csv`
  `research/xgb_guidance_20260514/xgb_vs_svm_recovery_contrast_h21.csv`

All of them use the same fixed split:

- data: `FunSearch_test/training_2orientations.csv`
- test size: `0.25`
- random state: `42`

## Current evidence

On the fixed split:

- linear `svm` with current `base40 + h21`: AUC `0.9765`, TPR@1% `0.7458`, ACC `0.9364`
- tuned `xgb` on `base40`: AUC `0.9868`, TPR@1% `0.8289`, ACC `0.9532`
- tuned `xgb` on `base40 + h21`: AUC `0.9880`, TPR@1% `0.8274`, ACC `0.9560`

So XGBoost still keeps a meaningful gap over the current linear-SVM route even
after the existing 21 handcrafted / LLM-searched features are added.

In direct test-set contrast for `base40 + h21`:

- rows where `xgb` is right and `svm` is wrong: `77`
- rows where `svm` is right and `xgb` is wrong: `28`

This suggests there is still nonlinear structure left to linearize.

## High-importance current custom features inside XGBoost

Among the current H21 active-bank features, the strongest XGBoost signals are:

- `spare_x_count`
- `tight_bin_large_piece_interaction`
- `max_face_area_load_over_floor`
- `long_wide_item_share`
- `near_limit_mid_share`
- `two_dim_large_share`
- `vol_cv`
- `vol_top3_concentration`

Interpretation:

- XGBoost values explicit low-slack x large-piece interaction features
- count/share style threshold features matter more than smooth averages
- concentration and upper-tail occupancy still carry information not fully
  captured linearly

## Features with strong contrast on XGBoost-recovered rows

Rows recovered by XGBoost over the current SVM show notable contrast on:

- `spare_x_avgvol`
- `spare_x_wlstd`
- `spare_x_count`
- `spare_capacity`
- `spare_x_p90long`
- `spare_x_volcv`
- `sku_average_volume`
- `wl_to_vehicle_wl_total`
- `wl_to_vehicle_wl_max`
- `wl_to_vehicle_wl_avg`
- `wl_to_vehicle_wl_std`
- `sku_length_var`

Interpretation:

- the unresolved gap is still heavily tied to slack-conditioned size pressure
- both tail size and heterogeneity under low slack appear important
- XGBoost likely benefits from regime switches such as:
  - low slack + many medium/large pieces
  - low slack + high length-width pressure
  - low slack + high item-size dispersion

## Feature directions to test

Priority directions for the next LLM-guided search:

1. second-order slack-pressure features
   - interactions between `spare_x_*` family and current threshold features
   - examples: low-slack multiplied by near-limit share, top-volume concentration,
     or face-area pressure
2. thresholded dispersion under low slack
   - counts/shares of items beyond dispersion-aware cutoffs
   - examples: many pieces near upper-tail `dim_l` with high `wl` spread
3. piece-count x awkward-size regime indicators
   - separate dense-small-piece pressure from few-large-piece pressure
   - count and share features are preferred over generic means
4. pressure-concentration hybrids
   - combine local concentration with vehicle-relative pressure
   - examples: dominant bulky subgroup share under small remaining slack
5. breakpoints around current strong features
   - create explicit threshold features around `spare_x_count`,
     `spare_x_p90long`, `spare_x_wlstd`, `vol_top3_concentration`,
     `near_limit_mid_share`

## What not to over-invest in

Less promising directions:

- repeating the same `spare_x_*` family with only mild rescaling
- generic averages without thresholding or interaction structure
- wide feature bursts that make attribution and ablation unclear
- rebuilding features already present in the H21 active bank

## Immediate workflow implication

The next `HybridSVM` feature-search run should keep the search tight:

- prefer incremental additions on top of the current H21 bank
- add only a few interpretable features per iteration
- bias toward threshold features and interaction features
- use XGBoost-recovered error patterns as hypothesis targets

The goal is not to mimic XGBoost mechanically. The goal is to expose a small
part of its regime-switching advantage in a form the linear SVM can exploit and
we can still explain.
