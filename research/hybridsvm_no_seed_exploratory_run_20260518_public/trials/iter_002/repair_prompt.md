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
        - error: NameError: name 'near_len' is not defined

        Original response:
        ```text
        Let me analyze the current situation:

1. Current active feature bank has 5 features: `spare_cap_sq`, `spare_cap_x_conc`, `spare_cap_x_wl_total`, `vol_top3_share`, `n_high_asr_items`
2. Current best metrics: AUC 0.9681, TPR@FPR=1% 0.6672, Accuracy 0.9292
3. Need to beat: AUC improvement > 0.0005, or TPR@1% improvement > 0.005, or Accuracy improvement > 0.0005

The XGB guidance strongly suggests:
- spare_capacity nonlinear transforms (already have spare_cap_sq)
- spare_capacity interactions (already have spare_cap_x_conc, spare_cap_x_wl_total)
- Utilization-ratio interactions (wl_total_x_wl_max_ratio, wl_max_residual)
- Dimensional-ratio dispersion features (l_to_L_ratio_std_x_h_to_H_ratio_max, h_to_H_ratio_max_sq)
- Item-level tail features

Looking at what's missing from the active bank:
1. We don't have features related to `wl_to_vehicle_wl_max` interactions or residuals
2. We don't have height-ratio features (h_to_H_ratio_max is important in XGB)
3. We don't have features capturing the concentration of footprint pressure
4. We don't have features about large-piece bottlenecks against vehicle dimensions

Let me think about what features to propose. The key insight from the tree model is:
- `spare_capacity` dominates with 16.9 gain
- `wl_to_vehicle_wl_total` (3.19) and `wl_to_vehicle_wl_max` (2.49) are important
- `h_to_H_ratio_max` (2.00) has high gain per split
- `l_to_L_ratio_std` (2.27) has high gain per split

The current active bank already has:
- spare_cap_sq (quadratic of spare_capacity)
- spare_cap_x_conc (spare_capacity × concentration)
- spare_cap_x_wl_total (spare_capacity × wl_total)
- vol_top3_share (volume share of top 3 items)
- n_high_asr_items (count of high aspect ratio items)

What's still missing that could help:
1. **wl_max_residual** - `wl_to_vehicle_wl_max - wl_to_vehicle_wl_total` (captures whether peak footprint is concentrated vs distributed)
2. **h_to_H_ratio_max_sq** - quadratic of max height ratio (threshold effect)
3. **spare_cap_x_wl_max** - spare_capacity × wl_max interaction (top-1 and top-4 gain features)
4. **n_large_dim_items** - count of items where any sorted dimension exceeds a threshold relative to vehicle (bottleneck pieces)
5. **l_ratio_std_x_h_ratio_max** - interaction of two high-gain-per-split features

Wait, I need to be careful. The features should use items_df for item-level computations. Let me think about what I can compute.

From items_df, I have: dispatch_id, item_length, item_width, item_height, if_fragile, load_parameter, vehicle_capacity, dim_s, dim_m, dim_l, item_volume, item_flatness

From agg_df, I have all the base aggregate features.

Let me design features that:
1. Use item-level data for threshold counts and tail features
2. Use agg_df for interaction features
3. Are physically interpretable
4. Target the TPR@FPR=1% gap

Feature ideas:

1. **spare_cap_x_wl_max** - Interaction of spare capacity with max footprint ratio. XGB uses both heavily. This captures whether high spare capacity helps more when max footprint pressure is also high.

2. **wl_max_residual** - `wl_to_vehicle_wl_max - wl_to_vehicle_wl_total` from agg_df. Captures concentration of footprint pressure vs distributed load.

3. **n_dim_bottleneck** - Count of items where the largest sorted dimension (dim_l) exceeds 70% of the corresponding vehicle dimension. These are the pieces that create packing bottlenecks.

4. **h_ratio_max_sq** - Quadratic of h_to_H_ratio_max. XGB shows high gain per split on this feature, suggesting threshold effects.

5. **vol_tail_ratio** - Ratio of volume in the largest item to total volume. Captures whether a single item dominates the load.

Actually, let me reconsider. I want features that are likely to improve TPR@FPR=1% specifically. The gap there is huge (0.6672 vs 0.8289 for XGB). This means the SVM is missing high-confidence infeasible cases.

For TPR@FPR=1%, we need features that help identify infeasible cases with high confidence. These would be:
- Features that signal extreme packing difficulty
- Features that create sharp boundaries

Let me think about what makes a dispatch clearly infeasible:
- Very large items relative to vehicle
- Many items with high dimension ratios
- Concentrated footprint pressure
- Low spare capacity with awkward items

Feature candidates:

1. **spare_cap_x_wl_max** - When spare capacity is low AND max footprint is high, it's very likely infeasible. This interaction should create a sharp signal.

2. **wl_max_residual** - The gap between max and total footprint utilization. A large residual means one item dominates the footprint, creating a bottleneck.

3. **n_dim_bottleneck** - Count of items where dim_l > 0.7 * vehicle_length (or similar). These are the awkward pieces that trees can isolate with threshold splits.

4. **h_ratio_max_sq** - Quadratic of max height ratio. Height constraints are often the binding constraint.

5. **l_ratio_std_x_h_ratio_max** - Interaction of length-ratio dispersion and max height ratio. When items vary a lot in length AND one is tall, packing is hard.

Actually, I need to be more careful about using items_df. Let me check what vehicle dimensions are available in items_df. The schema shows `vehicle_capacity` but not vehicle_length, vehicle_width, vehicle_height. So I might need to get vehicle dimensions from agg_df.

Wait, looking more carefully at items_df schema: `dispatch_id`, `item_length`, `item_width`, `item_height`, `if_fragile`, `load_parameter`, `vehicle_capacity`, `dim_s`, `dim_m`, `dim_l`, `item_volume`, `item_flatness`

It has `vehicle_capacity` but not individual vehicle dimensions. For computing bottleneck items relative to vehicle dimensions, I'd need to join with agg_df.

Let me design the features:

1. **spare_cap_x_wl_max** from agg_df: `agg_df['spare_capacity'] * agg_df['wl_to_vehicle_wl_max']`

2. **wl_max_residual** from agg_df: `agg_df['wl_to_vehicle_wl_max'] - agg_df['wl_to_vehicle_wl_total']`

3. **h_ratio_max_sq** from agg_df: `agg_df['h_to_H_ratio_max'] ** 2`

4. **n_tall_items** from items_df: Count of items where dim_l (largest sorted dimension) is very large relative to... hmm, I need vehicle dimensions. Let me use item_flatness or dim_s/dim_m/dim_l directly.

Actually, let me think about this differently. I can compute item-level features using items_df and then aggregate per dispatch.

5. **max_item_l_to_veh_L** - I can compute this by joining items_df with agg_df to get vehicle_length, then compute max(dim_l / vehicle_length) per dispatch. This is similar to l_to_L_ratio_max which already exists in agg_df.

Hmm, many of these are already in the base features. Let me focus on what's truly new.

Let me reconsider. The active bank already has:
- spare_cap_sq: spare_capacity^2
- spare_cap_x_conc: spare_capacity * sku_concentration
- spare_cap_x_wl_total: spare_capacity * wl_to_vehicle_wl_total
- vol_top3_share: volume share of top 3 items (item-level)
- n_high_asr_items: count of high aspect ratio items (item-level)

What's still missing:
1. spare_cap_x_wl_max - the interaction with the MAX footprint, not total
2. wl_max_residual - concentration signal
3. h_ratio_max_sq - height threshold effect
4. Something from item-level about tall/awkward pieces
5. l_ratio_std_x_h_ratio_max - interaction of two high-gain features

Let me pick 3-4 that are most likely to help:

1. **spare_cap_x_wl_max** - Directly addresses the XGB finding that spare_capacity and wl_max interact. Currently we have spare_cap_x_wl_total but not wl_max.

2. **wl_max_over_total** - `wl_to_vehicle_wl_max / (wl_to_vehicle_wl_total + eps)`. Captures whether the max footprint piece dominates. High values = concentrated bottleneck.

3. **h_ratio_max_sq** - Quadratic of max height ratio. Height is often the binding constraint and XGB shows high per-split gain here.

4. **n_large_l_items** - Count of items where dim_l > some threshold. From items_df, I can compute this using the sorted dimensions. I'll use a threshold relative to the item's own dimensions or an absolute threshold.

Wait, for n_large_l_items, I need a reference. Let me use the vehicle dimensions from agg_df. I can merge items_df with agg_df[['dispatch_id', 'vehicle_length']] to get the vehicle length, then count items where dim_l > 0.7 * vehicle_length.

Actually, this is basically what l_to_L_ratio_max already captures in aggregate. But a COUNT of such items is different from the MAX ratio. The count tells you HOW MANY items are near the vehicle length limit, which is a different signal.

Let me finalize my feature set:

1. **spare_cap_x_wl_max** - `spare_capacity * wl_to_vehicle_wl_max` from agg_df
2. **wl_max_over_total** - `wl_to_vehicle_wl_max / (wl_to_vehicle_wl_total + 1e-8)` from agg_df  
3. **h_ratio_max_sq** - `h_to_H_ratio_max ** 2` from agg_df
4. **n_near_ceil_items** - Count of items where dim_l > 0.7 * vehicle_length (need merge with agg_df)

Actually, let me reconsider n_near_ceil_items. The "ceiling" in 3D packing is the height. Let me count items where the item height (or the medium sorted dimension, which often corresponds to height when placed) is close to the vehicle height.

But I don't have vehicle_height in items_df. I need to merge.

Let me design the code:

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # Feature 1: spare_capacity × wl_to_vehicle_wl_max
    result['spare_cap_x_wl_max'] = agg_df['spare_capacity'] * agg_df['wl_to_vehicle_wl_max']

    # Feature 2: wl_max concentration ratio
    result['wl_max_over_total'] = agg_df['wl_to_vehicle_wl_max'] / (agg_df['wl_to_vehicle_wl_total'] + 1e-8)

    # Feature 3: h_to_H_ratio_max squared (threshold effect)
    result['h_ratio_max_sq'] = agg_df['h_to_H_ratio_max'] ** 2

    # Feature 4: count of items with dim_l near vehicle length limit
    merged = items_df.merge(agg_df[['dispatch_id', 'vehicle_length']], on='dispatch_id', how='left')
    near_len_mask = merged['dim_l'] > 0.7 * merged['vehicle_length']
    n_near_len = near_len_mask.groupby(merged['dispatch_id']).sum().astype(float)
    result = result.merge(near_len.rename('n_near_len_items'), left_on='dispatch_id', right_index=True, how='left')
    result['n_near_len_items'] = result['n_near_len_items'].fillna(0)

    return result
```

Wait, I should double-check: is `wl_to_vehicle_wl_total` potentially zero? It's a ratio of item wl to vehicle wl, summed over items. It should be positive if there are items. The epsilon is just for safety.

Also, let me reconsider whether wl_max_over_total is the right formulation. If wl_to_vehicle_wl_total is the sum of (item_wl / vehicle_wl) for all items, and wl_to_vehicle_wl_max is the max of (item_wl / vehicle_wl), then:
- wl_max_over_total = max / sum = the share of the largest item's footprint in the total footprint
- This is similar to a concentration measure but for footprint specifically

Actually, this might be somewhat correlated with sku_concentration. Let me think...

sku_concentration is probably Herfindahl-like (sum of squared shares). wl_max_over_total is just the max share. They're related but different.

Hmm, let me also think about whether I should include l_ratio_std_x_h_ratio_max. The XGB guidance specifically mentions this as a high-priority interaction.

Let me revise:

1. **spare_cap_x_wl_max** - spare_capacity × wl_to_vehicle_wl_max (HIGH priority from XGB guidance §3.2)
2. **h_ratio_max_sq** - h_to_H_ratio_max^2 (MEDIUM priority from XGB guidance §3.4)
3. **n_near_len_items** - count of items with dim_l > 0.7 * vehicle_length (tree-style threshold count)
4. **l_ratio_std_x_h_ratio_max** - l_to_L_ratio_std × h_to_H_ratio_max (MEDIUM priority from XGB guidance §3.4)

That's 4 features. Let me check for potential issues:

- spare_cap_x_wl_max: Both are from agg_df, straightforward
- h_ratio_max_sq: From agg_df, straightforward
- n_near_len_items: Requires merge of items_df with agg_df, then groupby
- l_ratio_std_x_h_ratio_max: Both from agg_df, straightforward

Actually, wait. I should check if l_ratio_std_x_h_ratio_max might be redundant. l_to_L_ratio_std is already in the base features, and h_to_H_ratio_max is in the base features. The SVM can learn their interaction if both are present... but with a linear kernel, it cannot learn the product. So this is a valid new feature.

But is it better than wl_max_over_total? Let me think about which is more likely to help TPR@FPR=1%.

For TPR@FPR=1%, we need features that strongly signal infeasibility. The strongest signals would be:
- Low spare capacity with high max footprint pressure → spare_cap_x_wl_max
- Items that barely fit in height → h_ratio_max_sq
- Many items near the length limit → n_near_len_items
- Interaction of length dispersion and height pressure → l_ratio_std_x_h_ratio_max

I think all 4 are good. Let me go with these 4.

Actually, let me reconsider n_near_len_items. The threshold of 0.7 is somewhat arbitrary. Let me think about what threshold makes physical sense.

In 3D bin packing, an item with dim_l > 0.7 * vehicle_length is quite long relative to the vehicle. Such items are hard to place because they constrain the layout significantly. But maybe I should use a higher threshold like 0.8 or 0.85 to capture truly bottleneck items.

Actually, let me use 0.75 as a compromise. Or better yet, let me use the vehicle height instead, since height is often the most constraining dimension in practice.

Hmm, but I'm using dim_l (the largest sorted dimension of the item). When placed, this could be oriented along any vehicle axis. The key bottleneck is when any item dimension is close to any vehicle dimension.

Let me simplify: count items where the largest sorted dimension (dim_l) exceeds 75% of the vehicle's largest dimension (vehicle_length). This captures items that are long in at least one dimension relative to the vehicle's longest axis.

Actually, I realize I should be more careful. In the problem, items can be rotated (2 orientations based on the filename "training_2orientations"). So dim_l could be placed along vehicle_length or vehicle_width. The bottleneck is when dim_l is close to the smaller of vehicle_length and vehicle_width (since it needs to fit in at least one orientation).

But let me not overthink this. Let me just count items where dim_l > 0.75 * vehicle_length as a simple threshold count. This is a tree-style feature that captures "how many items are really long."

Let me also consider: should I add a feature about the volume of the largest item relative to vehicle volume? This could capture "elephant in the room" cases where one huge item dominates.

Actually, vol_top3_share already captures this to some degree. Let me not duplicate.

Let me finalize with 4 features:

1. spare_cap_x_wl_max
2. h_ratio_max_sq
3. n_near_len_items (threshold count from items_df)
4. l_ratio_std_x_h_ratio_max

Wait, I want to make sure I'm not going over 6 features as per the rules. 4 is fine.

Let me also consider if I should add wl_max_residual (wl_max - wl_total) instead of one of these. The residual captures whether the max footprint piece is much larger than average, which is a concentration signal.

Actually, wl_max_residual might be more interpretable and less correlated with existing features than l_ratio_std_x_h_ratio_max. Let me swap.

Final set:
1. **spare_cap_x_wl_max** - spare_capacity × wl_to_vehicle_wl_max
2. **h_ratio_max_sq** - h_to_H_ratio_max^2
3. **n_near_len_items** - count of items with dim_l > 0.75 * vehicle_length
4. **wl_max_residual** - wl_to_vehicle_wl_max - wl_to_vehicle_wl_total

Hmm, but wl_max_residual might be negative (if total > max, which... wait, that can't happen because total is a sum and max is the maximum single value, so total >= max always). So wl_max_residual = max - total <= 0 always. That's not great.

Let me reconsider. wl_to_vehicle_wl_total is the SUM of (item_wl / vehicle_wl) over all items. wl_to_vehicle_wl_max is the MAX of (item_wl / vehicle_wl). So total >= max, and the residual max - total <= 0.

A better formulation would be wl_total_minus_wl_max = total - max, which is always >= 0 and represents the footprint contribution of all items except the largest. Or wl_max_share = max / total, which is the share of the largest item.

Let me use wl_max_share instead:
**wl_max_share** = wl_to_vehicle_wl_max / (wl_to
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

        Current iteration: `2`

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
