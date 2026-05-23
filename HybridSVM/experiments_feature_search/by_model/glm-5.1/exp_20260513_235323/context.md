# Context

Task: improve the current linear-SVM route with a small number of interpretable,
per-dispatch numeric features. The model family and split protocol are fixed.

## Data Scope
- dispatch rows after deduplication: `10000`
- train rows: `7500`
- test rows: `2500`
- positive rate: `0.8083`
- matched item rows for these dispatches: `100317`
- item-count per dispatch: mean=`10.032`, median=`10.0`, max=`22`

## Fixed Baseline Metrics
- Accuracy: `0.9276`
- Precision: `0.9444`
- Recall: `0.9680`
- ROC AUC: `0.9651`
- TPR@FPR=1%: `0.6347`

## Existing Aggregate Feature Columns
`sku_counts`, `sku_average_volume`, `sku_length_var`, `sku_width_var`, `sku_height_var`, `sku_length_avg`, `sku_width_avg`, `sku_height_avg`, `max_asr`, `vehicle_length`, `vehicle_width`, `vehicle_height`, `spare_capacity`, `sku_concentration`, `sku_min_length`, `sku_max_length`, `sku_std_length`, `sku_min_width`, `sku_max_width`, `sku_std_width`, `sku_min_height`, `sku_max_height`, `sku_std_height`, `l_to_L_ratio_avg`, `l_to_L_ratio_min`, `l_to_L_ratio_max`, `l_to_L_ratio_std`, `h_to_H_ratio_avg`, `h_to_H_ratio_min`, `h_to_H_ratio_max`, `h_to_H_ratio_std`, `w_to_W_ratio_avg`, `w_to_W_ratio_min`, `w_to_W_ratio_max`, `w_to_W_ratio_std`, `wl_to_vehicle_wl_avg`, `wl_to_vehicle_wl_min`, `wl_to_vehicle_wl_max`, `wl_to_vehicle_wl_std`, `wl_to_vehicle_wl_total`

## Item-Level Table Schema
`dispatch_id`, `item_length`, `item_width`, `item_height`, `if_fragile`, `load_parameter`, `vehicle_capacity`, `dim_s`, `dim_m`, `dim_l`, `item_volume`, `item_flatness`

Notes:
- `agg_df` already contains per-dispatch aggregate features and vehicle dimensions.
- `items_df` contains one row per item with both raw dimensions and sorted dimensions:
  `dim_s <= dim_m <= dim_l`.
- Candidate code must return one row per dispatch with ASCII feature names.

## Current SVM Linear Insights
The classifier is **linear** on MinMax-scaled features: `decision = w·x_scaled + b`, then label 1 if decision ≥ 0.
- Intercept `b` = **11.5265**
- **|w_j|** large → SVM is sensitive to that feature (in scaled space).
- **w_j > 0** → higher scaled value pushes toward **feasible (1)**; **w_j < 0** → toward **not feasible (0)**.

| rank | feature | w_j (on scaled x) | |w_j| |
|------|---------|---------------------|------|
| 1 | `sku_average_volume` | -16.4102 | 16.4102 |
| 2 | `spare_capacity` | 13.5627 | 13.5627 |
| 3 | `sku_counts` | -10.5919 | 10.5919 |
| 4 | `wl_to_vehicle_wl_avg` | 7.62511 | 7.62511 |
| 5 | `wl_to_vehicle_wl_std` | -4.26032 | 4.26032 |
| 6 | `l_to_L_ratio_avg` | -3.0218 | 3.0218 |
| 7 | `sku_length_avg` | -3.0218 | 3.0218 |
| 8 | `wl_to_vehicle_wl_min` | -2.94932 | 2.94932 |
| 9 | `sku_std_length` | -2.5518 | 2.5518 |
| 10 | `l_to_L_ratio_std` | -2.5518 | 2.5518 |
| 11 | `sku_height_var` | -2.39459 | 2.39459 |
| 12 | `wl_to_vehicle_wl_total` | 2.1109 | 2.1109 |
| 13 | `l_to_L_ratio_min` | -1.66505 | 1.66505 |
| 14 | `sku_min_length` | -1.66505 | 1.66505 |
| 15 | `w_to_W_ratio_avg` | -1.59533 | 1.59533 |
| 16 | `sku_width_avg` | -1.59533 | 1.59533 |
| 17 | `wl_to_vehicle_wl_max` | 1.3037 | 1.3037 |
| 18 | `sku_max_width` | 1.24877 | 1.24877 |
| 19 | `w_to_W_ratio_max` | 1.24877 | 1.24877 |
| 20 | `w_to_W_ratio_min` | -1.10221 | 1.10221 |
| 21 | `sku_min_width` | -1.10221 | 1.10221 |
| 22 | `w_to_W_ratio_std` | -1.03794 | 1.03794 |
| 23 | `sku_std_width` | -1.03794 | 1.03794 |
| 24 | `sku_width_var` | -0.833531 | 0.833531 |
| 25 | `sku_height_avg` | -0.736217 | 0.736217 |

_(15 more features omitted; smallest |w| omitted.)_

When designing `apply_rule_patch`, raw feature values are **not** scaled like `x_scaled`; combine this table with the **Feature scales** section to reason about conflicts on hard cases.

## Tree-Model Contrast Hypotheses
- Purely linear-style combinations did not close the gap to the stronger tree baselines;
  the likely missing piece is explicit threshold / interaction structure rather than more
  smooth averages of the same aggregates.
- In local checks, tree models appear to recover some cases that linear SVM misses,
  especially around `spare_capacity`, `sku_average_volume`, `wl_to_vehicle_wl_max`,
  `wl_to_vehicle_wl_total`, `sku_length_avg`, and item-dimension variance patterns.
- Translate those advantages into explicit features such as: upper-tail dimension pressure,
  low-slack x large-piece interactions, near-limit piece counts/shares, repetition versus
  heterogeneity signals, and footprint or wall-pressure proxies.
- Treat this as a search hint: propose thresholded counts, tail-pressure features,
  and interaction terms that might linearize those nonlinear regimes for the SVM.

## Search Bias
- prioritize compact, explainable feature sets
- use item-distribution shape, tails, concentration, and bottleneck signals
- explicitly test threshold-style and interaction-style hypotheses suggested by trees
- avoid re-encoding what the existing aggregate means and variances already say

## Tree-Guidance Snapshot
The following markdown is a frozen guidance snapshot derived from the
current tree-model analysis workflow. It is part of the formal prompt
input for this run and should be treated as a hypothesis source rather
than ground truth.

# Tree-Inspired Feature Hypotheses for HybridSVM

This note connects the standalone `Ensemble_baseline/` findings back to the
`HybridSVM` feature-engineering route.

## Why this note exists

The ensemble ablations show that the main gain does not come from stacking
linear models together. It comes mostly from tree learners such as:

- `rf`
- `xgb` / `gbdt_fallback`

So the practical question for the linear-SVM route is:

> what structure are trees exploiting that we might convert into explicit,
> interpretable features?

In the current feature-search workflow, this file is not just background notes.
It is treated as a formal guidance input:

- `scripts/run_feature_search_agent.py` reads it through `--tree-guidance-path`
- each run copies it into the experiment directory as `tree_guidance.md`
- its contents are embedded into `context.md`, which is part of the frozen LLM prompt

## Current evidence

On the fixed split:

- linear `svm`: AUC about `0.965`, TPR@1% about `0.635`
- tree baselines: AUC about `0.982` to `0.985`, TPR@1% about `0.806` to `0.813`
- `svm+lr` without tree models does not materially improve over the linear range

In a direct error contrast on the same split:

- `svm_only_errors = 86`
- `hgb_only_errors = 38`

Features with notable distribution contrast between `svm-only` and
`hgb-only` error sets included:

- `spare_capacity`
- `wl_to_vehicle_wl_total`
- `sku_average_volume`
- `wl_to_vehicle_wl_max`
- `sku_length_avg`
- `sku_width_var`

This suggests the trees are benefiting from thresholded or interaction-heavy
structure rather than just better linear weighting of the same aggregates.

## Feature directions to test

Priority directions for LLM-guided feature search:

1. tail pressure:
   - high quantiles of `dim_l`, `dim_m`
   - counts/shares above tight vehicle-relative thresholds
2. slack-pressure interactions:
   - low `spare_capacity` combined with large-piece ratios
   - low slack combined with high face-area load
3. local awkward-pattern counts:
   - number/share of items whose long edge is near vehicle limits
   - number/share of items with thin-flat or long-thick geometry
4. heterogeneity vs repetition:
   - distinct size-type count
   - dominant type share
   - type concentration or long-tail diversity
5. footprint and wall-pressure proxies:
   - summed face-area load against floor-like or side-like dimensions
   - extreme local occupancy rather than only total volume

## What not to over-invest in

Less promising directions:

- adding more smooth mean/std variants of features already present
- stacking linear-style aggregates without new interaction structure
- very large opaque feature sets that cannot be ablated cleanly

## Immediate workflow implication

The `HybridSVM/src/feature_search.py` seed policy should keep biasing toward:

- compact feature sets
- explicit threshold features
- interpretable interaction features
- ablation-friendly candidates

The target is not to imitate trees mechanically. The target is to expose part of
their nonlinear advantage in a form a linear SVM can use and a human can still
interpret.
