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
        - error: RateLimitError: Error code: 429 - {'error': {'message': "No deployments available for selected model, Try again in 5 seconds. Passed model=glm-5.1. pre-call-checks=False, cooldown_list=[('b9ca8bb822ee3f6a483f081af040bbc6769519968ef875dbfcc563db02907968', {'exception_received': 'litellm.RateLimitError: Model rate limit exceeded.', 'status_code': '429', 'timestamp': 1779097158.235544, 'cooldown_time': 5}), ('ee7f63fe7b5a6e0c0f610a8773fbc7d2b61ec00beadff16b9c535feb3698db62', {'exception_received': 'litellm.InternalServerError: InternalServerError: Hosted_vllmException - Internal Server Error', 'status_code': '500', 'timestamp': 1779097158.2137609, 'cooldown_time': 5})]", 'type': 'None', 'param': 'None', 'code': '429'}}

        Original response:
        ```text
        Let me analyze the current situation:

1. Current best: AUC 0.9681, TPR@FPR=1% 0.6672, Accuracy 0.9292
2. Active features: spare_cap_sq, spare_cap_x_conc, spare_cap_x_wl_total, vol_top3_share, n_high_asr_items
3. Need to beat: AUC improvement > 0.0005, or TPR@FPR=1% improvement > 0.005, or Accuracy improvement > 0.0005

From the XGB guidance, the key areas are:
- spare_capacity nonlinear transforms (already have spare_cap_sq)
- spare_capacity interactions (already have spare_cap_x_conc, spare_cap_x_wl_total)
- Utilization-ratio interactions (wl_total_x_wl_max_ratio, wl_max_residual)
- Dimensional-ratio dispersion features
- SKU-width distribution features
- Item-level tail/threshold features

What's missing from the active bank:
1. We have volume top3 share and high ASR item counts, but no threshold-based features on dimension ratios
2. No interaction between spare_capacity and sku_average_volume (top-2 XGB feature)
3. No features around wl_to_vehicle_wl_max (XGB rank 4 by gain)
4. No height-related bottleneck features
5. No features capturing whether load is concentrated vs distributed (wl_total/wl_max ratio)

Let me think about what's most promising:

From the XGB analysis:
- spare_capacity is dominant (already exploited with sq and interactions)
- sku_average_volume is #2 by gain - we haven't interacted it with spare_capacity
- wl_to_vehicle_wl_total is #3 - already interacted with spare_capacity
- wl_to_vehicle_wl_max is #4 - not yet featured
- l_to_L_ratio_std has high gain-per-split
- h_to_H_ratio_max has high gain-per-split

Key hypotheses for iteration 3:
1. `spare_cap_x_avg_vol` - spare_capacity × sku_average_volume interaction (top-1 × top-2 XGB features)
2. `wl_max_residual` - wl_to_vehicle_wl_max - wl_to_vehicle_wl_total (captures concentration vs distribution of footprint pressure)
3. `h_to_H_max_sq` - quadratic of h_to_H_ratio_max (height bottleneck threshold effect)

Wait, but I need to use only agg_df and items_df. Let me check what's available:

From agg_df: all the base40 features including spare_capacity, sku_average_volume, wl_to_vehicle_wl_max, wl_to_vehicle_wl_total, h_to_H_ratio_max, l_to_L_ratio_std, etc.

From items_df: dispatch_id, item_length, item_width, item_height, if_fragile, load_parameter, vehicle_capacity, dim_s, dim_m, dim_l, item_volume, item_flatness

Let me think about item-level features that would be most valuable:

1. Items with dim_l close to vehicle length - these are the bottleneck pieces
2. Items with large footprint (dim_s * dim_m) relative to vehicle floor
3. Height pressure - items stacking tall

Let me propose 3-4 features:

1. `spare_cap_x_avg_vol` - from agg_df: spare_capacity * sku_average_volume (interaction of top-1 and top-2 XGB features by gain)
2. `wl_max_residual` - from agg_df: wl_to_vehicle_wl_max - wl_to_vehicle_wl_total (peak vs total footprint utilization gap)
3. `n_long_items` - from items_df: count of items where dim_l > 0.7 * vehicle_length (bottleneck long pieces that constrain placement)
4. `h_max_pressure` - from agg_df: h_to_H_ratio_max * sku_counts (height bottleneck × item count interaction)

Wait, let me reconsider. The active bank already has 5 features. I should add a small number (1-3) of new features that are most likely to help.

Looking at the XGB guidance more carefully:

The biggest gap is TPR@FPR=1% (0.6672 vs XGB's 0.8289). This means we need features that help identify infeasible cases at high confidence (low FPR region).

The key insight is that XGB uses spare_capacity in a multi-threshold way. We already have spare_cap_sq which captures some curvature. But we might need more specific threshold effects.

Let me think about what features would help the SVM separate the hard cases:

1. **spare_cap_x_avg_vol**: The interaction of spare capacity and average volume. When spare capacity is low AND average volume is high, infeasibility is very likely. This is a direct encoding of the XGB top-2 interaction.

2. **wl_max_residual**: The gap between peak and total footprint utilization. When wl_max is much larger than wl_total, it means one item dominates the footprint, creating a bottleneck. This captures the XGB's use of both wl_total and wl_max.

3. **n_long_items**: Count of items where the longest dimension exceeds a threshold relative to vehicle length. These items are hard to place and create constraints.

Actually, let me think more carefully. The items_df gives us per-item data. Let me compute some item-level features that capture bottleneck patterns:

1. **n_long_items**: Count of items where dim_l > 0.5 * vehicle_length. Long items are hard to orient and constrain the bin.

2. **max_dim_l_to_L**: The maximum dim_l / vehicle_length across items. This is similar to l_to_L_ratio_max which already exists in base40.

Wait, l_to_L_ratio_max already exists. Let me think of something different.

How about:
1. **spare_cap_x_avg_vol** - from agg_df, interaction of top-1 and top-2 XGB features
2. **wl_max_residual** - from agg_df, captures concentration vs distribution
3. **n_tall_items** - from items_df, count of items where dim_l (sorted, so largest dim) exceeds some fraction of vehicle height. Actually wait, we need to be careful about orientation. In 3D bin packing with 2 orientations, the item can be rotated. So the relevant thing is whether any orientation fits.

Let me think about this differently. The items_df has dim_s <= dim_m <= dim_l (sorted dimensions). The vehicle has length, width, height. With 2 orientations, the item can be placed with dim_l along vehicle_length or vehicle_width (assuming 2 rotations in the L-W plane).

A "long item" is one where dim_l is close to the vehicle's longer dimension. This creates a bottleneck because it must be placed in a specific orientation.

Let me propose:

1. **spare_cap_x_avg_vol**: spare_capacity * sku_average_volume - captures the interaction between available space and item size
2. **wl_max_residual**: wl_to_vehicle_wl_max - wl_to_vehicle_wl_total - captures footprint concentration
3. **n_long_items**: count of items where dim_l > 0.6 * vehicle_length (from items_df, needs vehicle_length from agg_df or items_df)

Wait, items_df has vehicle_capacity but not vehicle_length/width/height. Let me check the schema again:

items_df columns: dispatch_id, item_length, item_width, item_height, if_fragile, load_parameter, vehicle_capacity, dim_s, dim_m, dim_l, item_volume, item_flatness

Hmm, vehicle_capacity is there but not the individual dimensions. I can get vehicle dimensions from agg_df though.

For item-level features, I need to join agg_df info with items_df. Let me think about how to do this cleanly.

Actually, for n_long_items, I can compute it from items_df using dim_l and then join with agg_df for vehicle_length. Or I can compute it purely from items_df if I can derive the vehicle length somehow.

Wait, looking more carefully at items_df, it has `vehicle_capacity` which is likely volume (L*W*H). That doesn't directly give me L.

Alternative approach: compute features from items_df that don't need vehicle dimensions, then merge with agg_df features.

Item-level features from items_df alone:
- Count of items with dim_l above some absolute threshold (but this doesn't account for vehicle size)
- Variance of dim_l across items
- Ratio of max dim_l to min dim_l
- Count of "flat" items (item_flatness > threshold)

Or I can merge vehicle dimensions from agg_df into items_df for computation.

Let me design the features:

1. **spare_cap_x_avg_vol** (from agg_df): `agg_df['spare_capacity'] * agg_df['sku_average_volume']`
   - Motivation: XGB top-1 × top-2 interaction. When both spare capacity is low and items are large, infeasibility is very likely.

2. **wl_max_residual** (from agg_df): `agg_df['wl_to_vehicle_wl_max'] - agg_df['wl_to_vehicle_wl_total']`
   - Motivation: Captures whether footprint pressure is concentrated (one big item) vs distributed. XGB uses both features.

3. **n_long_items** (from items_df + agg_df): Count of items per dispatch where dim_l > 0.6 * vehicle_length
   - Motivation: Long items constrain placement options and create bottlenecks. This is a threshold count feature.

4. **h_max_x_counts** (from agg_df): `agg_df['h_to_H_ratio_max'] * agg_df['sku_counts']`
   - Motivation: Height bottleneck × item count. More items with tall ones = harder to pack.

Wait, let me reconsider. I should keep it to at most 6 features, preferably 1-3 new ones.

Let me go with 3 features:

1. **spare_cap_x_avg_vol**: The most promising interaction based on XGB evidence
2. **wl_max_residual**: Captures concentration signal
3. **n_long_items**: Item-level threshold count

Actually, I realize I should double-check that none of these overlap with the active bank:
- spare_cap_sq: spare_capacity^2
- spare_cap_x_conc: spare_capacity * sku_concentration
- spare_cap_x_wl_total: spare_capacity * wl_to_vehicle_wl_total
- vol_top3_share: volume share of top 3 items
- n_high_asr_items: count of high aspect-ratio items

My proposed features don't overlap. Good.

Let me write the code:

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # Feature 1: spare_capacity × sku_average_volume interaction
    result['spare_cap_x_avg_vol'] = (
        agg_df['spare_capacity'] * agg_df['sku_average_volume']
    )

    # Feature 2: wl_to_vehicle_wl_max - wl_to_vehicle_wl_total (footprint concentration gap)
    result['wl_max_residual'] = (
        agg_df['wl_to_vehicle_wl_max'] - agg_df['wl_to_vehicle_wl_total']
    )

    # Feature 3: count of long items (dim_l > 60% of vehicle_length)
    veh_dims = agg_df[['dispatch_id', 'vehicle_length']].copy()
    items_with_veh = items_df.merge(veh_dims, on='dispatch_id', how='left')
    items_with_veh['is_long'] = (items_with_veh['dim_l'] > 0.6 * items_with_veh['vehicle_length']).astype(int)
    n_long = items_with_veh.groupby('dispatch_id')['is_long'].sum().reset_index()
    n_long.columns = ['dispatch_id', 'n_long_items']
    result = result.merge(n_long, on='dispatch_id', how='left')
    result['n_long_items'] = result['n_long_items'].fillna(0)

    return result
```

Wait, I need to be careful about the `.rename()` rule. The rule says "Do not call `.rename(...)` on numpy arrays." I'm using `.columns` assignment on a DataFrame, which should be fine. But let me use a cleaner approach.

Actually, let me reconsider the n_long_items feature. The threshold of 0.6 is somewhat arbitrary. Let me think about what threshold makes physical sense.

In 3D bin packing with 2 orientations, an item with dim_l > vehicle_length can't be placed with dim_l along the length. If dim_l > vehicle_width too, it can't be placed at all (infeasible). But if dim_l < vehicle_length and dim_l < vehicle_width, it can go either way.

A threshold of 0.5 means dim_l > half the vehicle length. This is a reasonable threshold for "long items that constrain placement."

Actually, let me think about this more carefully. The key insight from the tree model is about threshold effects and interactions. Let me focus on the most promising features.

Given the XGB guidance strongly emphasizes spare_capacity interactions and the wl_max/wl_total cluster, let me prioritize:

1. **spare_cap_x_avg_vol**: Direct encoding of XGB top-1 × top-2 interaction
2. **wl_max_residual**: Captures the concentration signal between wl_max and wl_total
3. **n_long_items**: Item-level threshold count for bottleneck pieces

These 3 features are compact, interpretable, and directly motivated by the XGB evidence.

Let me also consider: should I add a feature related to height pressure? The XGB guidance mentions h_to_H_ratio_max (gain 2.00). But sku_height_avg and sku_height_var are already in base40, and the active bank doesn't have height-specific features. However, I think the 3 features above are more strongly motivated.

Let me finalize the code. I need to make sure I'm not using .values improperly and not calling .rename on numpy arrays.

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # Feature 1: spare_capacity × sku_average_volume
    # XGB top-1 (gain 16.9) × top-2 (gain 4.76) interaction
    result['spare_cap_x_avg_vol'] = (
        agg_df['spare_capacity'] * agg_df['sku_average_volume']
    )

    # Feature 2: footprint concentration gap
    # wl_to_vehicle_wl_max - wl_to_vehicle_wl_total
    # XGB uses both (gain 2.49 and 3.19); gap captures whether
    # one item dominates the footprint vs distributed load
    result['wl_max_residual'] = (
        agg_df['wl_to_vehicle_wl_max'] - agg_df['wl_to_vehicle_wl_total']
    )

    # Feature 3: count of long items per dispatch
    # Items where sorted longest dim > 50% of vehicle length
    # are placement-constrained bottlenecks
    veh_lens = agg_df[['dispatch_id', 'vehicle_length']].copy()
    items_merged = items_df.merge(veh_lens, on='dispatch_id', how='left')
    items_merged['is_long'] = (
        items_merged['dim_l'] > 0.5 * items_merged['vehicle_length']
    ).astype(int)
    n_long = items_merged.groupby('dispatch_id')['is_long'].sum()
    n_long = n_long.reset_index()
    n_long.columns = ['dispatch_id', 'n_long_items']
    result = result.merge(n_long, on='dispatch_id', how='left')
    result['n_long_items'] = result['n_long_items'].fillna(0)

    return result
```

Hmm, actually I want to reconsider. The `n_long_items` with threshold 0.5 might not be the best. Let me think about what threshold captures the most signal.

In the tree model, threshold effects are learned from data. Since I can't use labels, I need to pick a physically meaningful threshold. 

A threshold of 0.5 means the item's longest dimension is more than half the vehicle length. This is meaningful because:
- Such items are hard to stack side by side along the length
- They consume a large portion of the length dimension
- Multiple such items create severe placement constraints

But maybe I should use a higher threshold like 0.6 or 0.7 to capture truly constraining items. Or I could use multiple thresholds.

Actually, let me think about this differently. Instead of a single threshold, I could compute the fraction of items that are "long" (dim_l > 0.5 * vehicle_length). This normalizes by the number of items and might be more informative.

But wait, sku_counts is already in the model, so the SVM can learn the interaction. A count might be more directly interpretable.

Let me go with 0.5 threshold for n_long_items. It's a reasonable physical threshold.

Actually, I just realized I should also consider: is there a feature that captures the "awkward pattern" of having both long and tall items? That would be an interaction between length and height bottlenecks.

But let me keep it simple with 3 features for now.

One more thought: the `wl_max_residual` could be negative (if wl_total > wl_max, which shouldn't happen since wl_max is the max ratio and wl_total is the sum). Wait, let me think about what these features mean:

- `wl_to_vehicle_wl_max`: the maximum (over items) of (item_width * item_length) / (vehicle_width * vehicle_length). This is the footprint ratio of the largest-footprint item.
- `wl_to_vehicle_wl_total`: the sum (over items) of (item_width * item_length) / (vehicle_width * vehicle_length). This is the total footprint utilization.

So wl_max_residual = wl_max - wl_total. Since wl_total is a sum of positive ratios and wl_max is just one of them, wl_total >= wl_max always (assuming at least one item). So wl_max_residual would be negative or zero.

Hmm, that doesn't seem right. Let me reconsider.

Actually, looking at the base40 features more carefully:
- `wl_to_vehicle_wl_avg`: average of (item_wl / vehicle_wl) across items
- `wl_to_vehicle_wl_min`: minimum
- `wl_to_vehicle_wl_max`: maximum
- `wl_to_vehicle_wl_std`: standard deviation
- `wl_to_vehicle_wl_total`: sum

So wl_total = avg * count, and wl_max is the max single item's footprint ratio.

wl_max_residual = wl_max - wl_total would be negative when there are multiple items (since total = sum of all ratios >= max). This is always <= 0 for n_items >= 1.

That's not very useful as a "residual." Let me reconsider.

A better feature might be: wl_max / wl_total (the share of total footprint from the largest item). This is a concentration measure.

Or: wl_max - wl_avg (the gap between the largest item and the average). This captures how much the largest item dominates.

Let me use `wl_max_to_total_ratio` = wl_max / (
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
        - AUC: `0.9681`
        - TPR@FPR=1%: `0.6672`
        - Accuracy: `0.9292`

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

        Current iteration: `4`

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

# XGB-Guided Feature Hypotheses for HybridSVM (Base40)

**Route:** `xgb_guided_hybridsvm_base40`  
**Output file:** `XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

---

## 1. Reproducible Evidence Source

| Artifact | Path |
|---|---|
| Training data | `repro_bundle/data/training_2orientations.csv` |
| XGB config & summary | `repro_bundle/reference/Ensemble_baseline/experiments/xgb_search_20260511_202834_base40/summary.json` |
| Analysis directory | `repro_bundle/teacher_analysis/xgb_guidance_base40_20260514` |
| Analysis summary | `repro_bundle/teacher_analysis/xgb_guidance_base40_20260514/summary.json` |
| Train/test split | `test_size=0.25`, `random_state=42` |
| Feature scope | `base40` only — no active-bank features permitted |
| Target workflow entry | `code/HybridSVM/scripts/run_feature_search_agent.py --tree-guidance-path guidance/XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md` |

---

## 2. Current Evidence

### 2.1 Baseline Performance (base40, same split)

| Metric | SVM (base40) | XGB (base40) | Δ (XGB − SVM) |
|---|---|---|---|
| Accuracy | 0.9276 | 0.9532 | +0.0256 |
| AUC | 0.9651 | 0.9868 | +0.0217 |
| TPR@FPR1% | 0.6347 | 0.8289 | +0.1942 |
| FPR | 0.2489 | 0.1609 | −0.0880 |
| FNR | 0.0320 | 0.0206 | −0.0114 |
| Precision | 0.9444 | 0.9637 | +0.0193 |

### 2.2 Recovery Analysis (n_test=2500)

| Category | Count |
|---|---|
| SVM wrong, XGB right | 92 |
| SVM right, XGB wrong | 28 |
| Net recoverable | 64 |

**Interpretation:** XGB correctly classifies 92 cases that SVM misses; only 28 cases go the other way. The dominant gap is at the low-FPR operating point (TPR@FPR1%: 63.5% vs 82.9%), indicating SVM's linear boundary underperforms in the high-confidence region where XGB's tree splits capture nonlinear structure.

### 2.3 XGB Feature Importance (top 12 by gain, base40)

| Rank | Feature | Gain | Split Count | Weight |
|---|---|---|---|---|
| 1 | `spare_capacity` | 16.90 | 1860 | 1860 |
| 2 | `sku_average_volume` | 4.76 | 1306 | 1306 |
| 3 | `wl_to_vehicle_wl_total` | 3.19 | 1058 | 1058 |
| 4 | `wl_to_vehicle_wl_max` | 2.49 | 816 | 816 |
| 5 | `sku_counts` | 2.28 | 99 | 99 |
| 6 | `l_to_L_ratio_std` | 2.27 | 14 | 14 |
| 7 | `sku_max_width` | 2.03 | 184 | 184 |
| 8 | `h_to_H_ratio_max` | 2.00 | 32 | 32 |
| 9 | `sku_concentration` | 1.94 | 1073 | 1073 |
| 10 | `w_to_W_ratio_avg` | 1.86 | 93 | 93 |
| 11 | `w_to_W_ratio_min` | 1.81 | 23 | 23 |
| 12 | `sku_min_width` | 1.71 | 194 | 194 |

**Key structural observation:** `spare_capacity` dominates with 16.9 gain — 3.6× the second feature. It is the primary axis along which XGB partitions the space. Features ranked 2–4 (`sku_average_volume`, `wl_to_vehicle_wl_total`, `wl_to_vehicle_wl_max`) form a secondary cluster. `sku_concentration` has high split count (1073) but moderate gain (1.94), suggesting it serves as a frequent secondary splitter.

---

## 3. Feature Directions to Test

All proposed features must be derivable from base40 columns only. Each direction is a hypothesis supported by XGB split patterns, not a guaranteed improvement.

### 3.1 Nonlinear transforms of `spare_capacity` (priority: HIGH)

XGB uses `spare_capacity` as the dominant splitter with 1860 splits, implying multiple threshold regions. SVM's linear kernel cannot replicate this with a single coefficient.

- **`spare_capacity_sq`** — quadratic term; captures curvature in the spare-capacity–label relationship.
- **`spare_capacity_log1p`** — `log(1 + spare_capacity)`; compresses the right tail if XGB's splits concentrate at low values.
- **`spare_capacity_bucket_*`** — indicator features for `spare_capacity` falling in empirically chosen bins (e.g., quartile boundaries from training data); approximates XGB's step-function behavior.
- **`spare_capacity_is_low`** — binary: `spare_capacity < p25(train)`; directly tests whether a threshold effect exists at the lower end.

### 3.2 Pairwise interactions involving `spare_capacity` (priority: HIGH)

XGB's tree structure implies that `spare_capacity` is split conditionally on other features (and vice versa). These interactions are invisible to a linear SVM unless explicitly encoded.

- **`spare_capacity_x_sku_concentration`** — `spare_capacity * sku_concentration`; both are high-frequency splitters (1860 and 1073 splits).
- **`spare_capacity_x_wl_to_vehicle_wl_total`** — interaction of the top-1 and top-3 gain features.
- **`spare_capacity_x_sku_average_volume`** — interaction of the top-1 and top-2 gain features.
- **`spare_capacity_x_sku_counts`** — spare capacity may behave differently at high vs. low SKU counts.

### 3.3 Utilization-ratio interactions (priority: MEDIUM)

`wl_to_vehicle_wl_total` (gain 3.19) and `wl_to_vehicle_wl_max` (gain 2.49) are correlated utilization features. XGB uses both, suggesting they capture complementary information.

- **`wl_total_x_wl_max_ratio`** — `wl_to_vehicle_wl_total / (wl_to_vehicle_wl_max + ε)`; captures whether load is concentrated or distributed.
- **`wl_total_sq`** — quadratic term for total utilization.
- **`wl_max_residual`** — `wl_to_vehicle_wl_max - wl_to_vehicle_wl_total`; the gap between peak and average utilization.

### 3.4 Dimensional-ratio dispersion features (priority: MEDIUM)

`l_to_L_ratio_std` (gain 2.27, only 14 splits but high per-split gain) and `h_to_H_ratio_max` (gain 2.00, 32 splits) have low split counts but high gain-per-split, suggesting sharp decision boundaries.

- **`l_to_L_ratio_std_x_h_to_H_ratio_max`** — interaction of two high-gain-per-split features.
- **`h_to_H_ratio_max_sq`** — quadratic; tests for threshold effects in height utilization.
- **`l_to_L_ratio_std_is_zero`** — binary indicator for zero std (all items same length ratio); may isolate a homogeneous-load subset.

### 3.5 SKU-width distribution features (priority: LOW–MEDIUM)

`sku_max_width` (gain 2.03), `sku_min_width` (gain 1.71), and `w_to_W_ratio_avg` (gain 1.86) appear in the top 12.

- **`sku_width_range`** — `sku_max_width - sku_min_width`; spread of SKU widths.
- **`w_to_W_ratio_avg_x_sku_concentration`** — width-fit quality interacting with concentration.
- **`w_to_W_ratio_min_sq`** — quadratic of the minimum width ratio; tests for a floor effect.

### 3.6 Concentration-interaction cluster (priority: LOW–MEDIUM)

`sku_concentration` has high split count (1073) but moderate gain, functioning as a secondary splitter.

- **`sku_concentration_x_sku_counts`** — concentration may have different implications at different SKU counts.
- **`sku_concentration_x_sku_average_volume`** — concentration × average volume; captures whether concentrated loads are also large.

---

## 4. What Not to Over-Invest In

### 4.1 Base40 features with negligible XGB gain

Features outside the top 12 by gain have collectively minimal influence on XGB's decisions. Engineering interactions among low-gain features is unlikely to close the SVM–XGB gap. Prioritize directions in §3 before exploring features derived from low-rank base40 columns.

### 4.2 Exact replication of XGB tree paths

Do not attempt to hand-encode specific XGB split sequences (e.g., "if spare_capacity < 0.3 and sku_counts > 5 then …"). These are overfit to the tree structure and brittle. Instead, approximate the implied nonlinear surfaces with smooth transforms and interactions.

### 4.3 Excessive polynomial expansion

Avoid generating all pairwise or all cubic terms of base40. The 92 recoverable cases do not justify a combinatorial explosion. Each candidate feature should be motivated by a specific XGB split pattern (gain or split-count signal).

### 4.4 Active-bank features

This guidance is scoped to `base40_only`. Do not introduce features that depend on columns not present in the base40 set. Any feature derived from the active bank will be invalid under the current teacher scope and will contaminate the reproducibility chain.

### 4.5 Marginal gains on accuracy alone

The primary gap is at TPR@FPR1% (Δ = +0.194), not accuracy (Δ = +0.026). Features that improve overall accuracy but do not shift the high-confidence region will not address the main deficiency. Evaluate new features with attention to the low-FPR operating point.

---

## 5. Immediate Workflow Implication

1. **Generate features from §3.1 and §3.2 first.** The `spare_capacity` dominance (16.9 gain, 3.6× runner-up) makes it the single most promising axis for nonlinear encoding. The 92 SVM-wrong/XGB-right cases are disproportionately driven by XGB's multi-threshold splitting on this feature.

2. **Validate incrementally.** Add 1–3 features per iteration, not all at once. This preserves attribution and avoids collinearity masking. Use the same split (`test_size=0.25, random_state=42`) for comparability.

3. **Monitor TPR@FPR1% as the primary metric.** The 19.4-point gap is the clearest signal of recoverable performance. AUC and accuracy are secondary checkpoints.

4. **Cap exploration at ~15–20 candidate features total.** The net recoverable pool is 64 cases. Beyond a certain point, additional features will overfit to noise in this small residual. Stop if three consecutive iterations yield no TPR@FPR1% improvement.

5. **Log every candidate feature with its motivation.** For reproducibility, each feature's definition must reference which XGB evidence (specific gain value, split count, or interaction hypothesis) motivated it. This ensures the guidance-to-feature chain is auditable.

6. **Save all artifacts.** The prompt, raw LLM response, and final markdown must be persisted under the analysis directory (`repro_bundle/teacher_analysis/xgb_guidance_base40_20260514/`) for full reproducibility.


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



        # Memory

Baseline reference:
- AUC `0.9651`
- TPR@FPR=1% `0.6347`
- Accuracy `0.9276`

## Accepted Trials
- iter `1`: AUC `0.9681` (Δ `+0.0030`), TPR@1% `0.6672` (Δ `+0.0324`), features=spare_cap_sq, spare_cap_x_conc, spare_cap_x_wl_total, vol_top3_share, n_high_asr_items

## Rejected Trials
- iter `3`: AUC `0.9682` (Δ `+0.0030`), TPR@1% `0.6608` (Δ `+0.0261`), features=spare_cap_x_avg_vol, wl_max_residual, n_long_items

## Failed Trials
- iter `2`: RateLimitError: Error code: 429 - {'error': {'message': "No deployments available for selected model, Try again in 5 seconds. Passed model=glm-5.1. pre-call-checks=False, cooldown_list=[('b9ca8bb822ee3f6a483f081af040bbc6769519968ef875dbfcc563db02907968', {'exception_received': 'litellm.RateLimitError: Model rate limit exceeded.', 'status_code': '429', 'timestamp': 1779097037.1183505, 'cooldown_time': 5}), ('ee7f63fe7b5a6e0c0f610a8773fbc7d2b61ec00beadff16b9c535feb3698db62', {'exception_received': 'litellm.InternalServerError: InternalServerError: Hosted_vllmException - Internal Server Error', 'status_code': '500', 'timestamp': 1779097037.671829, 'cooldown_time': 5})]", 'type': 'None', 'param': 'None', 'code': '429'}}

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale


        ## Current Active Feature Bank
- source accepted trials: `1`
- active feature count: `5`
- active feature names: `spare_cap_sq`, `spare_cap_x_conc`, `spare_cap_x_wl_total`, `vol_top3_share`, `n_high_asr_items`
- active bank AUC: `0.9681`
- active bank TPR@FPR=1%: `0.6672`
- active bank Accuracy: `0.9292`

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
