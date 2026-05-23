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
        - error: KeyError: "Label(s) ['vehicle_length'] do not exist"

        Original response:
        ```text
        Let me analyze the current state and figure out what features to propose next.

Current active feature bank (18 features):
1. dominant_type_share
2. p90_long_over_bin_long
3. p90_mid_over_bin_mid
4. thin_item_share
5. max_face_area_load_over_floor
6. tight_bin_large_piece_interaction
7. spare_x_count
8. near_limit_long_share
9. vol_cv
10. height_pressure_share
11. long_wide_item_share
12. spare_x_volcv
13. two_dim_large_share
14. spare_x_p90long
15. cube_item_share
16. near_limit_mid_share
17. spare_x_avgvol
18. long_thin_item_share

Current best: AUC 0.9758, TPR@FPR=1% 0.7665, Accuracy 0.9356

Tree baselines achieve AUC 0.982-0.985, TPR@1% 0.806-0.813. There's still a gap.

Looking at what's already covered:
- Tail pressure: p90_long_over_bin_long, p90_mid_over_bin_mid, near_limit_long_share, near_limit_mid_share
- Slack-pressure interactions: spare_x_count, spare_x_volcv, spare_x_p90long, spare_x_avgvol, tight_bin_large_piece_interaction
- Shape classes: thin_item_share, long_wide_item_share, cube_item_share, long_thin_item_share
- Heterogeneity: vol_cv, dominant_type_share
- Footprint/wall: max_face_area_load_over_floor
- Two-dim coupling: two_dim_large_share

What's NOT covered yet:
1. **Short dimension pressure** - near_limit_short_share or p90_short_over_bin_short - the smallest dimension relative to vehicle height could matter for stacking
2. **Three-dimension pressure** - items large in ALL three dimensions simultaneously (not just two)
3. **Fragile item interactions** - if_fragile is available but never used
4. **Load parameter** - load_parameter is available but never used
5. **Floor occupancy ratio** - total footprint of items vs floor area (different from max_face_area_load)
6. **Stacking stress** - items with large footprint but also tall (column-like items that resist stacking on top)
7. **Packing order difficulty** - items that must go first because they're large in multiple dimensions
8. **Extreme spare capacity threshold** - binary-like feature for very low spare capacity

Let me think about what the tree model might be exploiting that we haven't captured:

The SVM weights show:
- sku_average_volume is the strongest negative feature (-16.4)
- spare_capacity is the strongest positive feature (+13.6)
- sku_counts is strongly negative (-10.6)

The tree guidance mentions:
1. Tail pressure - partially covered
2. Slack-pressure interactions - well covered
3. Local awkward-pattern counts - partially covered
4. Heterogeneity vs repetition - partially covered
5. Footprint and wall-pressure proxies - partially covered

What I think is missing:
1. **Floor saturation** - total item footprint on the floor vs vehicle floor area. This is different from max_face_area_load which looks at the max single item face. Floor saturation captures whether items collectively cover the floor.
2. **Three-dim extreme share** - items that are near vehicle limits in ALL three dimensions. This is a much stronger bottleneck signal than two_dim_large_share.
3. **Fragile load interaction** - fragile items constrain stacking; the combination of fragile items with tight space could be a nonlinear signal.

Wait, let me reconsider. The items_df has `if_fragile` and `load_parameter`. Let me think about what's physically meaningful.

Actually, let me focus on what's most likely to help based on the gap analysis:

The gap between SVM (0.9758 AUC) and trees (0.982-0.985 AUC) is about 0.006-0.009. The TPR@1% gap is larger: 0.7665 vs 0.806-0.813, about 0.04-0.05.

For the next iteration, I should focus on:
1. Something that captures a genuinely different signal
2. Threshold/interaction features that linearize tree-like splits

Let me think about what physical packing constraints are not yet captured:

**Floor footprint saturation**: The total area of items when placed on the floor. Items with dim_s × dim_m as their footprint (assuming they sit on their smallest face). This captures whether the floor is overwhelmed even if volume fits.

**Three-dim pressure share**: Items that are near limits in all three sorted dimensions simultaneously. These are the hardest to pack.

**Stacking penalty**: Items that are tall (dim_l/dim_m ratio high) AND wide (dim_m close to vehicle width) resist stacking anything on top.

Let me propose:

1. **floor_footprint_saturation** - sum of (dim_s * dim_m) for all items / (vehicle_width * vehicle_length). This captures floor crowding which is a different constraint from volume.

2. **three_dim_near_limit_share** - fraction of items where dim_l > 0.7*vehicle_length AND dim_m > 0.5*vehicle_width AND dim_s > 0.4*vehicle_height. These are items that are large in all dimensions and extremely hard to pack.

3. **tall_wide_item_share** - fraction of items with dim_l/dim_m > 2 (tall column) AND dim_m > 0.4*vehicle_width. These items resist stacking and take up vertical space inefficiently.

Actually, wait. Let me reconsider. I need to be more careful about what signals are truly new.

Looking at the active bank again:
- floor_footprint_saturation is NOT covered (max_face_area_load_over_floor is the max single item face, not total)
- three_dim_near_limit_share is NOT covered (two_dim_large_share covers 2D)
- tall_wide_item_share is somewhat related to long_thin_item_share but different

Let me think about which of these is most likely to help...

The floor footprint saturation is interesting because it captures a 2D packing constraint that's fundamentally different from volume. Even if total volume fits, if the floor is overwhelmed, packing fails.

The three-dim pressure is also interesting because it captures the hardest items.

Let me also consider: **spare_x_near_limit_long** - interaction between spare capacity and near_limit_long_share. Wait, we already have spare_x_p90long which is similar.

What about: **fragile_count** or **fragile_share**? The if_fragile column is available but unused. Fragile items constrain stacking.

Actually, let me look at this differently. The accepted features that worked well were:
- iter 6: near_limit_mid_share, spare_x_avgvol, long_thin_item_share (big TPR@1% jump from 0.7463 to 0.7665)
- iter 5: two_dim_large_share, spare_x_p90long, cube_item_share
- iter 3: height_pressure_share, long_wide_item_share, spare_x_volcv

The pattern is: threshold counts + slack interactions + shape classes.

For iteration 7, I should try something genuinely new. Let me propose:

1. **floor_saturation** - total item footprint / vehicle floor area. Physical meaning: even if volume fits, floor crowding prevents arrangement.

2. **three_dim_large_share** - fraction of items where dim_l > 0.6*L AND dim_m > 0.5*W AND dim_s > 0.4*H. These are the worst-case items for packing.

3. **spare_x_floor_sat** - interaction of spare_capacity with floor_saturation. When both are tight, packing is very difficult.

Wait, but I should be careful not to propose too many interaction features with spare_capacity since we already have spare_x_count, spare_x_volcv, spare_x_p90long, spare_x_avgvol. Adding more spare_x interactions might not help much.

Let me reconsider. Maybe I should focus on:

1. **floor_saturation** - captures a genuinely different constraint (2D floor packing vs 3D volume)
2. **three_dim_large_share** - captures the hardest items to pack
3. **fragile_share** or **fragile_load_penalty** - fragile items constrain stacking, a signal not yet captured

Actually, the `load_parameter` column could be very informative. Let me think about what it means... It's likely a parameter indicating how items should be loaded (orientation constraints, stacking limits, etc.).

Let me propose:

1. **floor_saturation** - sum(dim_s * dim_m) / (vehicle_width * vehicle_length) per dispatch
2. **three_dim_large_share** - fraction of items large in all three sorted dimensions relative to vehicle
3. **max_load_param** or **avg_load_param** - captures loading constraints from items_df

Hmm, but I don't know what load_parameter means exactly. Let me be safe and use it as-is.

Actually, let me think more carefully. The key insight from tree models is that they exploit threshold effects and interactions. The current bank has good coverage of:
- Single-dimension thresholds (near_limit_long_share, near_limit_mid_share)
- Two-dimension thresholds (two_dim_large_share)
- Shape classes (thin, long_wide, cube, long_thin)
- Slack interactions (spare_x_count, spare_x_volcv, spare_x_p90long, spare_x_avgvol)
- Distribution shape (vol_cv, dominant_type_share, p90 quantiles)

What's clearly missing:
1. **Floor-level 2D constraint** - not the max single face, but total floor coverage
2. **Three-dimension extreme items** - the worst bottleneck
3. **Stacking difficulty** - items that prevent other items from being stacked on top

Let me go with:

1. **floor_saturation** = sum(dim_s × dim_m) / (vehicle_width × vehicle_length) - captures whether the floor plane is overwhelmed
2. **three_dim_large_share** = fraction of items where dim_l > 0.6*vehicle_length AND dim_m > 0.5*vehicle_width AND dim_s > 0.3*vehicle_height - the hardest items
3. **spare_x_floor_sat** = spare_capacity × floor_saturation - interaction capturing when both volume and floor are tight

Wait, I said I shouldn't add more spare_x interactions. Let me replace the third one.

3. **p90_short_over_bin_height** = p90 of dim_s / vehicle_height - the short dimension tail pressure, complementing p90_long and p90_mid

Actually, we don't have p90_short_over_bin_height. We have p90_long_over_bin_long and p90_mid_over_bin_mid. Adding the short dimension quantile would complete the set and could capture stacking difficulty in the vertical dimension.

But wait, the short dimension (dim_s) is the smallest sorted dimension. For packing, dim_s would be the height when the item is laid flat. If dim_s is large relative to vehicle height, that's a stacking constraint.

Let me finalize my proposal:

1. **floor_saturation** - total item footprint on floor / vehicle floor area
2. **three_dim_large_share** - fraction of items near vehicle limits in all three dimensions
3. **p90_short_over_bin_height** - 90th percentile of dim_s / vehicle_height, capturing vertical stacking pressure

These three are compact, interpretable, and capture genuinely new signals.

Let me write the code:

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # Vehicle dimensions from items_df (same per dispatch)
    veh = items_df.groupby('dispatch_id').agg(
        vehicle_length=('vehicle_length', 'first'),  # Wait, vehicle_capacity is there, not vehicle_length
    )

    # Hmm, items_df has vehicle_capacity but not vehicle_length/width/height directly
    # But agg_df has vehicle_length, vehicle_width, vehicle_height

    # Floor saturation
    floor_area = items_df.groupby('dispatch_id').apply(
        lambda g: (g['dim_s'] * g['dim_m']).sum()
    )
    veh_floor = agg_df.set_index('dispatch_id')['vehicle_length'] * agg_df.set_index('dispatch_id')['vehicle_width']
    result['floor_saturation'] = (floor_area / veh_floor).values

    # Three-dim large share
    def three_dim_large(g):
        vl = g['vehicle_capacity']  # Wait, this isn't right
        ...

    # I need to get vehicle dimensions. They're in agg_df.
    # Let me merge them into items_df

    veh_dims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    items_with_veh = items_df.merge(veh_dims, on='dispatch_id', how='left')

    # Floor saturation
    items_with_veh['footprint'] = items_with_veh['dim_s'] * items_with_veh['dim_m']
    floor_sat = items_with_veh.groupby('dispatch_id')['footprint'].sum()
    veh_floor_area = agg_df.set_index('dispatch_id')['vehicle_length'] * agg_df.set_index('dispatch_id')['vehicle_width']
    result['floor_saturation'] = result['dispatch_id'].map(floor_sat / veh_floor_area)

    # Three-dim large share
    items_with_veh['is_three_large'] = (
        (items_with_veh['dim_l'] > 0.6 * items_with_veh['vehicle_length']) &
        (items_with_veh['dim_m'] > 0.5 * items_with_veh['vehicle_width']) &
        (items_with_veh['dim_s'] > 0.3 * items_with_veh['vehicle_height'])
    ).astype(int)
    three_large = items_with_veh.groupby('dispatch_id')['is_three_large'].mean()
    result['three_dim_large_share'] = result['dispatch_id'].map(three_large)

    # P90 short over bin height
    p90_short = items_with_veh.groupby('dispatch_id').apply(
        lambda g: g['dim_s'].quantile(0.9) / g['vehicle_height'].iloc[0]
    )
    result['p90_short_over_bin_height'] = result['dispatch_id'].map(p90_short)

    return result
```

Wait, I need to be more careful with the pandas operations. The rules say:
- Keep feature computations in pandas objects; do not call `.values` unless you immediately wrap the result back into a pd.Series or pd.DataFrame.
- Do not call `.rename(...)` on numpy arrays.

Let me rewrite more carefully:

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # Merge vehicle dimensions into items
    veh_dims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']]
    items_v = items_df.merge(veh_dims, on='dispatch_id', how='left')

    # 1. Floor saturation: total item footprint / vehicle floor area
    items_v['footprint'] = items_v['dim_s'] * items_v['dim_m']
    floor_sum = items_v.groupby('dispatch_id')['footprint'].sum()
    veh_floor = agg_df.set_index('dispatch_id')['vehicle_length'] * agg_df.set_index('dispatch_id')['vehicle_width']
    floor_sat = floor_sum / veh_floor
    result = result.merge(floor_sat.rename('floor_saturation').reset_index(), on='dispatch_id', how='left')

    # 2. Three-dim large share
    items_v['is_three_large'] = (
        (items_v['dim_l'] > 0.6 * items_v['vehicle_length']) &
        (items_v['dim_m'] > 0.5 * items_v['vehicle_width']) &
        (items_v['dim_s'] > 0.3 * items_v['vehicle_height'])
    ).astype(int)
    three_share = items_v.groupby('dispatch_id')['is_three_large'].mean()
    result = result.merge(three_share.rename('three_dim_large_share').reset_index(), on='dispatch_id', how='left')

    # 3. P90 short dim over bin height
    p90_short = items_v.groupby('dispatch_id').apply(
        lambda g: g['dim_s'].quantile(0.9) / g['vehicle_height'].iloc[0]
    )
    result = result.merge(p90_short.rename('p90_short_over_bin_height').reset_index(), on='dispatch_id', how='left')

    return result
```

Hmm, but the groupby.apply with lambda might be slow and could cause issues. Let me use a more vectorized approach:

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # Merge vehicle dimensions into items
    veh_dims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']]
    items_v = items_df.merge(veh_dims, on='dispatch_id', how='left')

    # 1. Floor saturation: total item footprint / vehicle floor area
    items_v['footprint'] = items_v['dim_s'] * items_v['dim_m']
    floor_sum = items_v.groupby('dispatch_id')['footprint'].sum()
    veh_floor = agg_df.set_index('dispatch_id')['vehicle_length'] * agg_df.set_index('dispatch_id')['vehicle_width']
    floor_sat = floor_sum / veh_floor
    result = result.merge(floor_sat.rename('floor_saturation').reset_index(), on='dispatch_id', how='left')

    # 2. Three-dim large share
    is_three_large = (
        (items_v['dim_l'] > 0.6 * items_v['vehicle_length']) &
        (items_v['dim_m'] > 0.5 * items_v['vehicle_width']) &
        (items_v['dim_s'] > 0.3 * items_v['vehicle_height'])
    ).astype(int)
    items_v['is_three_large'] = is_three_large
    three_share = items_v.groupby('dispatch_id')['is_three_large'].mean()
    result = result.merge(three_share.rename('three_dim_large_share').reset_index(), on='dispatch_id', how='left')

    # 3. P90 short dim over bin height
    # Compute ratio first, then take p90
    items_v['short_over_height'] = items_v['dim_s'] / items_v['vehicle_height']
    p90_short = items_v.groupby('dispatch_id')['short_over_height'].quantile(0.9)
    result = result.merge(p90_short.rename('p90_short_over_bin_height').reset_index(), on='dispatch_id', how='left')

    return result
```

This is cleaner.
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

        Current iteration: `7`

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
