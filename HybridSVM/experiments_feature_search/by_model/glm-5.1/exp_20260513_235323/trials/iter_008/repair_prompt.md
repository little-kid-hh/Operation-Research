You produced a feature-search response that failed validation.

        Fix the candidate and return a new response in the exact same format:

        ## POLICY_UPDATE
        ...

        ## FEATURE_CODE
        ```python
        ...
        ```

        ## RATIONALE
        ...

        Failure summary:
        - error: TypeError: Candidate must return a pandas DataFrame

        Original response:
        ```text
        Let me analyze the current state:

**Current best metrics:**
- AUC: 0.9758
- TPR@FPR=1%: 0.7665
- Accuracy: 0.9356

**Active feature bank (18 features):**
- dominant_type_share, p90_long_over_bin_long, p90_mid_over_bin_mid, thin_item_share
- max_face_area_load_over_floor, tight_bin_large_piece_interaction, spare_x_count
- near_limit_long_share, vol_cv, height_pressure_share, long_wide_item_share
- spare_x_volcv, two_dim_large_share, spare_x_p90long, cube_item_share
- near_limit_mid_share, spare_x_avgvol, long_thin_item_share

**What's been tried and failed:**
- iter 1: volume_top3_share, multi_dim_near_limit_share, side_wall_area_load (rejected)
- iter 4: spare_x_wlmax, total_footprint_ratio, cross_dim_pressure_count (rejected)
- iter 7: failed (ValueError)

**What's been accepted:**
- iter 3: height_pressure_share, long_wide_item_share, spare_x_volcv
- iter 5: two_dim_large_share, spare_x_p90long, cube_item_share
- iter 6: near_limit_mid_share, spare_x_avgvol, long_thin_item_share

**Key observations from SVM weights:**
1. sku_average_volume is the strongest negative weight (-16.4) - high average volume pushes toward infeasible
2. spare_capacity is the strongest positive weight (+13.6) - more slack pushes toward feasible
3. sku_counts is strongly negative (-10.6) - more items push toward infeasible
4. wl_to_vehicle_wl_avg is positive (+7.6) - interesting
5. wl_to_vehicle_wl_std is negative (-4.3) - variation in wl ratio hurts

**What's missing from the active bank:**

Looking at the tree guidance and what trees exploit:
1. **Short-dimension pressure** - We have near_limit_long_share and near_limit_mid_share, but no near_limit_short_share for the smallest dimension
2. **Wall pressure on the height dimension specifically** - height_pressure_share exists but maybe we need more specific height-related signals
3. **Fragile item interaction** - if_fragile is in items_df but never used
4. **Load parameter** - load_parameter is in items_df but never used
5. **Item stacking compatibility** - items that can't stack create voids
6. **Dimension ordering chaos** - how many items have very different shape profiles

Let me think about what could genuinely help:

1. **fragile_load_stress** - Count/share of fragile items weighted by load parameter. Fragile items can't be stacked on, creating dead space. This is a completely new signal from unused columns.

2. **spare_x_nearlimit_count** - Interaction between low spare capacity and the number of items near any limit. Trees would split on spare_capacity first, then on near-limit counts. This interaction isn't in the bank.

3. **height_wall_pressure** - Sum of item heights relative to vehicle height, measuring vertical stacking pressure. Different from height_pressure_share which is about items near height limit.

Wait, let me reconsider. The bank already has:
- spare_x_count, spare_x_volcv, spare_x_p90long, spare_x_avgvol - many spare interactions
- near_limit_long_share, near_limit_mid_share - dimension limit pressures

What's truly novel and unused:
1. **if_fragile** column - completely unused
2. **load_parameter** column - completely unused
3. **dim_s** patterns - we have thin_item_share (items flat in one dim) but no feature about the smallest dimension specifically
4. **Three-dimension near-limit** - items near limit in ALL three dimensions simultaneously (not just long or mid)

Let me think about what would be most impactful:

**Feature 1: fragile_void_share** - Share of items that are fragile. Fragile items can't have items stacked on top, creating vertical voids. This directly relates to packing difficulty and uses an unused column.

**Feature 2: all_dim_near_limit_share** - Share of items near the vehicle limit in ALL three sorted dimensions simultaneously. These are the most awkward items - they fill the bin in every direction. This is different from multi_dim_near_limit_share (which was rejected in iter 1, but that might have been because it was combined with weaker features).

Wait, multi_dim_near_limit_share was in iter 1 which was rejected. But iter 1 also had volume_top3_share and side_wall_area_load. The whole set was rejected, not necessarily because of multi_dim_near_limit_share specifically. But I should be cautious.

**Feature 3: spare_x_fragile** - Interaction between spare capacity and fragile item share. When spare capacity is low AND there are fragile items, packing becomes especially hard because you can't stack on fragile items.

Actually, let me think more carefully about what the trees are exploiting that we haven't captured:

The SVM weights show:
- sku_average_volume (-16.4) and spare_capacity (+13.6) are the two biggest
- sku_counts (-10.6) is third

The accepted features mostly interact spare_capacity with other things (spare_x_count, spare_x_volcv, spare_x_p90long, spare_x_avgvol). 

What about interactions with sku_average_volume? The bank has spare_x_avgvol which is spare × avg_volume interaction. But what about avgvol-related thresholds?

Let me think about dimension-specific stacking pressure:

**Feature idea: vertical_stack_pressure** - For each dispatch, compute the sum of item heights (in the vehicle's height orientation) divided by vehicle height. This captures whether items can physically stack to fill the height. But wait, we don't know orientation...

Actually, with sorted dimensions (dim_s, dim_m, dim_l), we could think about it differently. The height of an item when placed optimally would typically be dim_s (shortest dimension vertical for stability). But that's an assumption.

Let me think about something more concrete:

**Feature: load_param_max** - Maximum load_parameter across items in a dispatch. High load parameter items are heavy and constrain placement.

**Feature: fragile_count** - Number of fragile items. Simple, interpretable, uses unused column.

**Feature: spare_x_fragile_count** - Interaction of spare capacity with fragile item count.

Actually, let me look at this more carefully. The items_df has:
- if_fragile: binary
- load_parameter: numeric
- vehicle_capacity: numeric (same for all items in a dispatch?)
- dim_s, dim_m, dim_l: sorted dimensions
- item_volume, item_flatness

item_flatness is in items_df but I don't see it used in any feature. Let me check... thin_item_share is in the bank - that might be based on flatness.

Let me focus on genuinely novel signals:

1. **fragile_share** - Share of fragile items. Unused column, directly affects packing (can't stack on fragile items).

2. **max_load_param** - Maximum load parameter in the dispatch. High load items constrain placement and stacking.

3. **spare_x_fragile** - Interaction: low spare capacity × fragile items = very hard to pack.

Wait, I need to be more careful. Let me think about what physical mechanism makes packing infeasible:

- Items that are fragile can't be stacked upon → vertical space waste
- Items with high load parameter → weight constraints on what can go below
- When spare capacity is low AND fragile items exist → double pressure

But also:
- **Short-dimension bottleneck**: Items where dim_s is relatively large (cube-like) waste space because they can't nest. We have cube_item_share already.
- **Extreme aspect ratio items**: Very long and thin items are hard to place. We have long_thin_item_share already.

What about:
- **p90_short_over_bin_short** - We have p90_long and p90_mid quantiles but not p90 for the short dimension. Items with large short dimension relative to vehicle's shortest dimension could be a bottleneck.

Wait, looking at the active bank: p90_long_over_bin_long and p90_mid_over_bin_mid exist, but there's no p90_short_over_bin_short. This is a gap!

- **near_limit_short_share** - We have near_limit_long_share and near_limit_mid_share but not near_limit_short_share.

These seem like obvious gaps. Let me add:

1. **p90_short_over_bin_short** - 90th percentile of dim_s / min(vehicle_length, vehicle_width, vehicle_height)
2. **near_limit_short_share** - Share of items where dim_s > 0.7 * min(vehicle dimensions)

And then something from the unused columns:

3. **fragile_share** - Share of fragile items

Let me also think about whether these would actually help. The short dimension is typically the height when placed, so items with large dim_s relative to vehicle height would be hard to stack. But we already have height_pressure_share...

Hmm, height_pressure_share might be about items whose height (not necessarily dim_s) is near the vehicle height. If items are placed with dim_s as height (optimal orientation), then dim_s near vehicle height would mean the item takes up most of the vertical space.

Actually, I think the key insight is: we don't know the orientation. In 3D bin packing, items can be rotated. So the relevant question is about sorted dimensions relative to sorted vehicle dimensions.

Let me reconsider. The vehicle has length, width, height. Items have dim_s ≤ dim_m ≤ dim_l. In optimal packing:
- dim_l goes along the longest vehicle dimension
- dim_m goes along the middle vehicle dimension  
- dim_s goes along the shortest vehicle dimension

So the relevant ratio for the short dimension would be dim_s / min(vehicle_length, vehicle_width, vehicle_height).

We have p90_long_over_bin_long (dim_l / vehicle_length presumably) and p90_mid_over_bin_mid (dim_m / vehicle_width presumably), but not the short dimension equivalent.

Let me go with:

1. **p90_short_over_bin_short** - 90th percentile of dim_s / min(vehicle dimensions). Captures vertical stacking bottleneck.
2. **near_limit_short_share** - Share of items where dim_s > 0.7 * min(vehicle_length, vehicle_width, vehicle_height). Thresholded count for short dimension.
3. **fragile_share** - Share of fragile items. Uses completely unused column.

Actually wait - I need to check what "bin" dimensions mean in the existing features. p90_long_over_bin_long uses dim_l / vehicle_length, and p90_mid_over_bin_mid uses dim_m / vehicle_width. So the "bin" dimensions are the actual vehicle dimensions, not sorted.

For the short dimension, it should be dim_s / vehicle_height (since height is typically the shortest vehicle dimension). But that's an assumption. Let me use min(vehicle_length, vehicle_width, vehicle_height) to be safe.

Hmm, but looking at the existing features more carefully:
- l_to_L_ratio = item_length / vehicle_length (unsorted)
- h_to_H_ratio = item_height / vehicle_height
- w_to_W_ratio = item_width / vehicle_width

And the accepted features:
- p90_long_over_bin_long = p90(dim_l / vehicle_length)
- p90_mid_over_bin_mid = p90(dim_m / vehicle_width)

So it seems like the convention is:
- dim_l (longest sorted) → vehicle_length
- dim_m (middle sorted) → vehicle_width
- dim_s (shortest sorted) → vehicle_height

This makes sense for typical bin packing where length ≥ width ≥ height.

So p90_short_over_bin_short = p90(dim_s / vehicle_height).

And near_limit_short_share = share of items where dim_s / vehicle_height > 0.7.

Let me finalize my 3 features:

1. **p90_short_over_bin_short** - p90 of dim_s / vehicle_height. Captures vertical stacking pressure from the short dimension tail.

2. **near_limit_short_share** - Share of items where dim_s > 0.7 * vehicle_height. Thresholded count for items consuming most of vertical space.

3. **fragile_share** - Share of fragile items. Novel signal from unused column.

Wait, but I want to make sure these are truly new. Let me check the active bank again:
- dominant_type_share, p90_long_over_bin_long, p90_mid_over_bin_mid, thin_item_share
- max_face_area_load_over_floor, tight_bin_large_piece_interaction, spare_x_count
- near_limit_long_share, vol_cv, height_pressure_share, long_wide_item_share
- spare_x_volcv, two_dim_large_share, spare_x_p90long, cube_item_share
- near_limit_mid_share, spare_x_avgvol, long_thin_item_share

height_pressure_share might overlap with near_limit_short_share. Let me think... height_pressure_share was from iter 3. It's probably about items whose height dimension is near the vehicle height limit. If it's defined as share of items where item_height / vehicle_height > threshold, that's similar to what I'm proposing but using unsorted height vs sorted dim_s.

The key difference: dim_s is the SORTED shortest dimension, while item_height is the ORIGINAL height dimension. In 3D bin packing with rotation, the sorted dimension is more meaningful because items can be rotated. So dim_s / vehicle_height captures the minimum vertical footprint regardless of orientation, while item_height / vehicle_height assumes a fixed orientation.

So near_limit_short_share would be genuinely different from height_pressure_share if height_pressure_share uses unsorted dimensions.

OK, I'm fairly confident these are new. But let me also think about whether 3 features is enough or if I should add more.

Given the acceptance rule requires AUC improvement > 0.0005 or TPR@1% improvement > 0.005, and the current best is already quite high (0.9758 AUC, 0.7665 TPR@1%), I need features that capture genuinely new information.

The fragile_share feature is the most novel since it uses a completely unused column. The short-dimension features fill a gap in the dimension coverage.

Let me also consider one more feature:

4. **spare_x_fragile_share** - Interaction between spare capacity and fragile share. When both are adverse (low spare, high fragile), packing is especially difficult because you can't use vertical stacking efficiently.

This would be a natural extension of the spare_x_* interaction family that's been successful.

Actually, I think 3 features is better for a compact hypothesis. Let me go with:

1. **p90_short_over_bin_short** - Vertical stacking tail pressure
2. **near_limit_short_share** - Thresholded vertical space consumers  
3. **fragile_share** - Novel signal from unused column

Hmm, but I'm worried that p90_short_over_bin_short and near_limit_short_share might be too correlated (both measure short dimension pressure). Let me replace one with something else.

How about:

1. **near_limit_short_share** - Thresholded vertical space consumers (fills the gap in near_limit_* coverage)
2. **fragile_share** - Novel signal from unused column
3. **spare_x_fragile** - Interaction between low spare capacity and fragile items

This gives:
- One gap-filling dimension feature (near_limit_short_share)
- One novel column feature (fragile_share)
- One interaction feature (spare_x_fragile)

This is a good mix. Let me code it up.

Wait, for spare_x_fragile, I need to be careful. spare_capacity is in agg_df. fragile_share I compute from items_df. Then I multiply them. But I should make sure the interaction is meaningful.

spare_x_fragile = spare_capacity * (1 - fragile_share) perhaps? Or spare_capacity * fragile_share?

If spare_capacity is low (bad) and fragile_share is high (bad), the product spare_capacity * fragile_share would be low, which should push toward infeasible. In the SVM, spare_capacity has positive weight (more spare = more feasible). So spare_capacity * fragile_share would be: when spare is high and fragile is high, the product is high (more feasible, which is wrong - fragile items should make it harder). When spare is low and fragile is high, product is low (less feasible, which is correct).

Hmm, the interaction isn't quite right as a simple product. Let me think...

Actually, the standard interaction for "both bad" is: (1 - spare_capacity_normalized) * fragile_share. But we don't normalize in the feature function.

Alternatively: spare_capacity * (1 - fragile_share) - this would be high when spare is high AND few fragile items (good situation), low when spare is low OR many fragile items. This captures the "both good" scenario.

Or just use: -spare_capacity * fragile_share. This would be negative (pushing infeasible) when both spare is high and fragile is high. That's wrong.

Let me think about this differently. The SVM will learn the weight. If I create spare_x_fragile = spare_capacity * fragile_share, the SVM can assign a negative weight to it, meaning "even if spare capacity is decent, having fragile items reduces the effective slack." That makes physical sense.

Actually, looking at the existing spare_x_* features:
- spare_x_count: spare_capacity * sku_counts → SVM gives it some weight
- spare_x_volcv: spare_capacity * vol_cv
- spare_x_p90long: spare_capacity * p90_long_over_bin_long
- spare_x_avgvol: spare_capacity * sku_average_volume

These are all simple products. The SVM learns the appropriate sign. So spare_x_fragile = spare_capacity * fragile_share is fine.

But wait - if_fragile is binary (0 or 1), so fragile_share is between 0 and 1. And spare_capacity is a fraction. So the product will be between 0 and 1. The SVM should be able to use this.

Let me finalize:

1. **near_limit_short_share** - Share of items where dim_s > 0.7 * vehicle_height
2. **fragile_share** - Mean of if_fragile across items per dispatch
3. **spare_x_fragile** - spare_capacity * fragile_share (interaction)

Actually, I realize I should double-check: is vehicle_height always the shortest vehicle dimension? In truck loading, typically length > width > height. But I shouldn't assume. Let me use vehicle_height directly since that's the standard mapping for dim_s in bin packing.

Let me code this up properly.

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # Feature 1: near_limit_short_share
    # Share of items where the shortest sorted dimension > 70% of vehicle height
    items = items_df.copy()
    items['short_ratio'] = items['dim_s'] / items['vehicle_capacity']  # wait, vehicle_capacity is not vehicle_height

    # I need vehicle_height from agg_df, merged into items_df
    # items_df has vehicle_capacity column - what is this?
```

Hmm, I need to check the schema. items_df has `vehicle_capacity` column. Is this the same
        ```

        Original prompt:
        ```text
        You are improving a fixed linear-SVM baseline for 3D bin-packing feasibility.

        Produce:
        1. a short policy update for the next iteration;
        2. one Python candidate that adds interpretable per-dispatch features.

        Hard rules:
        - Keep the model family fixed: the candidate only generates features.
        - Use only `agg_df` and `items_df` passed into the function.
        - No file I/O, no API calls, no labels, no target leakage.
        - Return one row per dispatch with `dispatch_id` plus numeric feature columns.
        - Use ASCII feature names only.
        - Prefer at most `6` new features.
        - Use only pandas and numpy.
        - Keep feature computations in pandas objects; do not call `.values` unless you
          immediately wrap the result back into a `pd.Series` or `pd.DataFrame`.
        - Do not call `.rename(...)` on numpy arrays.
        - Favor tree-inspired signals: threshold counts, tail ratios, pressure-slack interactions,
          and local awkward-pattern shares.
        - You are in an iterative search loop with a cumulative active feature bank.
        - The evaluated model uses: base aggregate features + active feature bank + your new features.
        - Propose only new incremental features to add on top of the active feature bank.
        - Do not repeat, rename, or rewrite any feature already in the active bank.
        - Make a small, explicit local change: usually add 1-3 new feature ideas, not a reset.
        - Propose a candidate only if you believe the cumulative feature set can beat
          the current active bank under the acceptance rule below.

        Current best accepted target to beat:
        - AUC: `0.9758`
        - TPR@FPR=1%: `0.7665`
        - Accuracy: `0.9356`

        Acceptance rule:
        - accept if AUC improves by more than `0.0005`
        - otherwise require TPR@FPR=1% improvement larger than `0.005`
        - if still tied, require Accuracy improvement larger than `0.0005`

        Candidate function signature:

        ```python
        def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
            ...
        ```

        Response format:

        ## POLICY_UPDATE
        <markdown bullets>

        ## FEATURE_CODE
        ```python
        ...
        ```

        ## RATIONALE
        - bullet 1
        - bullet 2

        Current iteration: `8`

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


        # Feature Search Policy

Objective:
Improve the linear SVM baseline by adding a small number of interpretable,
per-dispatch numeric features.

Hard constraints:
- Do not change the train/test split.
- Do not change the model family: keep the stage-1 model as linear SVM.
- Do not use labels or any target-derived statistics inside feature generation.
- Do not read files or call external services in candidate feature code.
- Return one row per dispatch and keep feature names ASCII.

Preferred feature families:
- item-level quantiles / tails rather than only mean/std
- thresholded local bottleneck counts, not only global averages
- slack / spare-capacity interactions with large-piece pressure
- repeated-type structure and concentration
- extreme-piece bottlenecks against vehicle dimensions
- face-area / edge-pressure style packing stress signals
- heterogeneity / multimodality / long-tail measures
- features that are easy to explain to a human reviewer

Avoid:
- duplicating obvious existing aggregates unless the new version captures a
  different shape signal
- opaque embeddings
- huge feature sets; prefer a compact hypothesis with clear semantics

Tree-inspired search bias:
- assume trees are winning partly because they exploit local thresholds and
  interactions on features like `spare_capacity`, `wl_to_vehicle_wl_max`,
  `wl_to_vehicle_wl_total`, `sku_average_volume`, and dimension variances
- try to convert those nonlinear effects into explicit numeric features that
  a linear SVM can use

## Iteration 2
- Focus on slack-pressure interactions: the two strongest SVM features (spare_capacity +13.6, sku_counts -10.6) suggest a nonlinear interaction when both are in adverse regimes simultaneously.
- Add a thresholded near-limit count share for the long dimension, complementing the existing p90 quantile with a hard-threshold perspective.
- Add volume heterogeneity (CV) to capture the diversity vs. repetition axis that trees exploit but the linear SVM cannot without an explicit feature.
- These three features are compact, interpretable, and target the three main tree-advantage directions: interaction structure, threshold counts, and distribution shape.

## Iteration 5
- Target the two-dimensional coupling gap: current bank has single-dimension pressure (near_limit_long_share, p90_long) but no feature capturing items large in **both** long and mid dimensions simultaneously.
- Add a spare × p90_long interaction to linearize the nonlinear regime where low slack coincides with extreme long-dimension tail pressure — trees exploit this threshold interaction.
- Add a cube-item share signal: items with dim_s/dim_l > 0.5 resist nesting and create voids, a shape class not covered by thin_item_share or long_wide_item_share.


        # Memory

Baseline reference:
- AUC `0.9651`
- TPR@FPR=1% `0.6347`
- Accuracy `0.9276`

## Accepted Trials
- iter `6`: AUC `0.9758` (Δ `+0.0106`), TPR@1% `0.7665` (Δ `+0.1318`), features=near_limit_mid_share, spare_x_avgvol, long_thin_item_share
- iter `5`: AUC `0.9758` (Δ `+0.0106`), TPR@1% `0.7463` (Δ `+0.1116`), features=two_dim_large_share, spare_x_p90long, cube_item_share
- iter `3`: AUC `0.9753` (Δ `+0.0101`), TPR@1% `0.7198` (Δ `+0.0851`), features=height_pressure_share, long_wide_item_share, spare_x_volcv

## Rejected Trials
- iter `1`: AUC `0.9719` (Δ `+0.0067`), TPR@1% `0.7070` (Δ `+0.0723`), features=volume_top3_share, multi_dim_near_limit_share, side_wall_area_load
- iter `4`: AUC `0.9749` (Δ `+0.0097`), TPR@1% `0.7119` (Δ `+0.0772`), features=spare_x_wlmax, total_footprint_ratio, cross_dim_pressure_count

## Failed Trials
- iter `7`: ValueError: Candidate code must define build_candidate_features

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale


        ## Current Active Feature Bank
- source accepted trials: `0`, `2`, `3`, `5`, `6`
- active feature count: `18`
- active feature names: `dominant_type_share`, `p90_long_over_bin_long`, `p90_mid_over_bin_mid`, `thin_item_share`, `max_face_area_load_over_floor`, `tight_bin_large_piece_interaction`, `spare_x_count`, `near_limit_long_share`, `vol_cv`, `height_pressure_share`, `long_wide_item_share`, `spare_x_volcv`, `two_dim_large_share`, `spare_x_p90long`, `cube_item_share`, `near_limit_mid_share`, `spare_x_avgvol`, `long_thin_item_share`
- active bank AUC: `0.9758`
- active bank TPR@FPR=1%: `0.7665`
- active bank Accuracy: `0.9356`

Only propose new incremental features to add on top of this bank.
Do not re-emit, rename, or overwrite any active-bank feature.

        ```

        Repair rules:
        - Keep the same overall hypothesis if possible.
        - Preserve the exact function signature `build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame`.
        - Return one row per dispatch with `dispatch_id` and only numeric feature columns.
        - Avoid `.values` unless absolutely necessary.
        - Never call `.rename(...)` on numpy arrays.
        - Prefer pandas Series / DataFrame objects all the way through.
        - Make the code syntactically valid and executable.
