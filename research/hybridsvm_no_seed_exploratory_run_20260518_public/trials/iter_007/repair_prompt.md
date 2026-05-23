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
        - error: RateLimitError: Error code: 429 - {'error': {'message': "No deployments available for selected model, Try again in 5 seconds. Passed model=glm-5.1. pre-call-checks=False, cooldown_list=[('b9ca8bb822ee3f6a483f081af040bbc6769519968ef875dbfcc563db02907968', {'exception_received': 'litellm.RateLimitError: Model rate limit exceeded.', 'status_code': '429', 'timestamp': 1779097551.7085834, 'cooldown_time': 5}), ('ee7f63fe7b5a6e0c0f610a8773fbc7d2b61ec00beadff16b9c535feb3698db62', {'exception_received': 'litellm.RateLimitError: Model rate limit exceeded.', 'status_code': '429', 'timestamp': 1779097552.2410598, 'cooldown_time': 5})]", 'type': 'None', 'param': 'None', 'code': '429'}}

        Original response:
        ```text
        Let me analyze the current situation carefully.

Current best metrics (active bank):
- AUC: 0.9689
- TPR@FPR=1%: 0.6593
- Accuracy: 0.9292

Active features (9 total):
1. spare_cap_sq
2. spare_cap_x_conc
3. spare_cap_x_wl_total
4. vol_top3_share
5. n_high_asr_items
6. spare_cap_log1p
7. h_to_H_max_sq
8. l_std_x_h_max
9. max_fp_ratio

Acceptance rule:
- AUC improvement > 0.0005
- Otherwise TPR@FPR=1% improvement > 0.005
- Otherwise Accuracy improvement > 0.0005

The main gap is still TPR@FPR=1% (0.6593 vs XGB's 0.8289). The XGB guidance emphasizes spare_capacity nonlinear transforms and interactions, which we've partially explored. 

Looking at what's NOT in the active bank yet:
- No threshold/bucket features for spare_capacity
- No wl_total_x_wl_max_ratio (concentrated vs distributed load)
- No item-level tail features based on sorted dimensions (dim_s, dim_m, dim_l)
- No features using `if_fragile`, `load_parameter`, `item_flatness` from items_df
- No features using `dim_s`, `dim_m`, `dim_l` directly

Key insights from the SVM weights and XGB guidance:
1. spare_capacity is still the dominant feature - we've added sq, log1p, and interactions with conc and wl_total
2. wl_to_vehicle_wl_total and wl_to_vehicle_wl_max are important
3. sku_concentration is a frequent secondary splitter
4. The items_df has sorted dimensions (dim_s, dim_m, dim_l) that we haven't used at all

Let me think about what new features to propose:

1. **wl_total_over_wl_max** - ratio of total footprint utilization to max footprint utilization. This captures whether the load is concentrated (few items taking up most space) vs distributed. If wl_total is high but wl_max is also high, it means one item dominates the footprint. This is a "concentration of footprint pressure" signal.

2. **n_items_near_height_limit** - count of items where h_to_H ratio is close to 1 (say > 0.7). This captures height bottleneck items that are hard to stack around. Using items_df with dim_s/dim_m/dim_l and vehicle dimensions.

3. **spare_cap_x_wl_max** - interaction of spare_capacity with wl_to_vehicle_wl_max. The XGB guidance says spare_capacity interacts with utilization features. We have spare_cap_x_wl_total but not spare_cap_x_wl_max.

Wait, let me check - spare_cap_x_wl_total is already in the bank. But spare_cap_x_wl_max is NOT. The XGB guidance ranks wl_to_vehicle_wl_max as 4th by gain (2.49). This interaction could capture: "even with spare capacity, if the maximum footprint utilization is high, packing is harder."

4. **dim_l_tail_share** - share of items whose longest sorted dimension exceeds some threshold relative to vehicle length. This is a tree-inspired threshold count.

5. **fragile_load_interaction** - using if_fragile and load_parameter from items_df. Fragile items constrain stacking.

6. **flat_item_share** - share of items with high flatness (item_flatness), which affects stacking patterns.

Let me focus on the most promising 2-3 features:

1. **spare_cap_x_wl_max** - High priority interaction from XGB guidance. spare_capacity (rank 1 by gain) × wl_to_vehicle_wl_max (rank 4 by gain). We have spare_cap_x_wl_total but not this one.

2. **n_tall_items** - count of items where dim_s (shortest sorted dim) is a large fraction of vehicle height, meaning they're hard to stack. Or better: count where h_to_H ratio > 0.5. Actually, we need to compute this from items_df.

Wait, items_df has `item_height` but also sorted dims `dim_s`, `dim_m`, `dim_l`. For packing feasibility, what matters is whether any orientation of the item has a dimension close to the vehicle height. If dim_l (longest sorted dim) is close to vehicle height, the item is tall in some orientation.

Actually, let me think more carefully. The items_df has vehicle_capacity but not vehicle dimensions directly. Wait, it doesn't have vehicle_length, vehicle_width, vehicle_height. Those are in agg_df.

Hmm, but items_df has `load_parameter` and `if_fragile` and `dim_s`, `dim_m`, `dim_l` and `item_flatness`. These are item-level features we haven't exploited at all.

Let me think about what's most likely to help:

The biggest gap is TPR@FPR=1%. We need features that help identify infeasible cases at high confidence. These are typically cases where:
- Spare capacity looks OK but there's a hidden bottleneck
- Items have awkward dimensions that don't fit well despite volume being OK

Feature ideas:

1. **spare_cap_x_wl_max** - This directly addresses the XGB guidance priority. spare_capacity × wl_to_vehicle_wl_max interaction. When both are at certain levels, it creates a nonlinear decision boundary.

2. **n_dim_l_near_vehicle_L** - count of items where dim_l > 0.6 * vehicle_length. These are long items that constrain placement. This uses items_df sorted dimensions.

3. **flat_item_frac** - fraction of items with high flatness (item_flatness > some threshold). Flat items can be stacked but create floor pressure.

Actually, let me reconsider. I should be careful about what's available in items_df vs agg_df.

items_df columns: dispatch_id, item_length, item_width, item_height, if_fragile, load_parameter, vehicle_capacity, dim_s, dim_m, dim_l, item_volume, item_flatness

agg_df has all the aggregate features including vehicle dimensions.

For feature 1 (spare_cap_x_wl_max), I only need agg_df columns: spare_capacity and wl_to_vehicle_wl_max. Easy.

For feature 2, I need items_df dim_l and agg_df vehicle_length. I'd need to merge or compute per-dispatch.

Let me think about what's most likely to move the needle:

The rejected trial 3 had spare_cap_x_avg_vol, wl_max_residual, n_long_items. It was rejected because AUC only improved by 0.0030 (same as iter 1) and TPR@1% improved by 0.0261 (less than iter 1's 0.0324). So wl_max_residual and n_long_items didn't add enough on top of what iter 1 already had.

But iter 5 added spare_cap_log1p, h_to_H_max_sq, l_std_x_h_max, max_fp_ratio and improved further.

So the active bank now has:
- spare_cap_sq, spare_cap_x_conc, spare_cap_x_wl_total (from iter 1)
- spare_cap_log1p, h_to_H_max_sq, l_std_x_h_max, max_fp_ratio (from iter 5)
- vol_top3_share, n_high_asr_items (from iter 1)

What's still missing:
- spare_cap_x_wl_max (interaction not yet tried)
- wl_total_over_wl_max ratio (concentration signal)
- Item-level features using dim_s, dim_m, dim_l from items_df
- Features using if_fragile, load_parameter, item_flatness

Let me propose:

1. **spare_cap_x_wl_max** - spare_capacity × wl_to_vehicle_wl_max. This is a high-priority interaction from XGB guidance that hasn't been tried yet.

2. **wl_total_over_wl_max** - wl_to_vehicle_wl_total / (wl_to_vehicle_wl_max + eps). Captures footprint concentration vs distribution. High ratio = distributed load, low ratio = one item dominates.

3. **n_fragile_items** - count of fragile items per dispatch from items_df. Fragile items constrain stacking and may be a hidden infeasibility factor.

Wait, let me reconsider. The item_flatness feature in items_df is interesting - flat items (high flatness) are easier to stack but create floor pressure. Let me think about what tree models might capture.

Actually, let me focus on the most physically meaningful features:

1. **spare_cap_x_wl_max** - Key interaction. When spare capacity is low AND max footprint utilization is high, packing is very constrained. This nonlinear combination should help the linear SVM.

2. **dim_l_max_to_vehicle_L** - the ratio of the largest item's longest sorted dimension to vehicle length. This is a bottleneck signal - if the biggest item is almost as long as the vehicle, placement is very constrained. I can compute this from items_df grouped by dispatch_id.

3. **load_param_max** - maximum load_parameter across items in a dispatch. Higher load parameters may indicate items that are harder to handle/pack.

Hmm, but I need to be careful. Let me check what load_parameter means - it's likely a weight or stacking constraint parameter. If it's per-item, the max or sum could be informative.

Let me finalize my proposal:

1. **spare_cap_x_wl_max** - spare_capacity × wl_to_vehicle_wl_max interaction
2. **wl_total_over_wl_max** - footprint concentration ratio
3. **max_dim_l_to_vL** - max(dim_l) / vehicle_length per dispatch, from items_df

These are 3 features, all interpretable, all addressing different aspects of the packing feasibility problem.

Actually wait - for feature 3, I need to get vehicle_length from agg_df. Let me think about how to compute this efficiently.

I can group items_df by dispatch_id, compute max(dim_l), then merge with agg_df to get vehicle_length. Or I can compute the ratio directly if vehicle_length is available.

Actually, looking at items_df schema again: it has `vehicle_capacity` but not vehicle_length/width/height. So I'd need to merge with agg_df to get vehicle dimensions.

Let me reconsider. To keep it simple and avoid complex merges:

1. **spare_cap_x_wl_max** - from agg_df only
2. **wl_total_over_wl_max** - from agg_df only  
3. **max_dim_l_ratio** - from items_df: group by dispatch_id, take max(dim_l), then merge with agg_df for vehicle_length

Or I could use items_df more directly:

3. **max_dim_l_to_vL** - merge max dim_l per dispatch with vehicle_length from agg_df

Let me write the code:

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # Feature 1: spare_capacity × wl_to_vehicle_wl_max interaction
    result['spare_cap_x_wl_max'] = (
        agg_df['spare_capacity'] * agg_df['wl_to_vehicle_wl_max']
    )

    # Feature 2: footprint concentration ratio
    result['wl_total_over_wl_max'] = (
        agg_df['wl_to_vehicle_wl_total'] / (agg_df['wl_to_vehicle_wl_max'] + 1e-9)
    )

    # Feature 3: max item longest-dim ratio to vehicle length
    max_dim_l = items_df.groupby('dispatch_id')['dim_l'].max().reset_index()
    max_dim_l.columns = ['dispatch_id', 'max_dim_l']
    merged = result.merge(max_dim_l, on='dispatch_id', how='left')
    result['max_dim_l_to_vL'] = merged['max_dim_l'] / (agg_df['vehicle_length'] + 1e-9)

    return result
```

Wait, I need to be careful with the merge. Let me ensure the order is preserved.

Actually, let me reconsider the features. The max_dim_l_to_vL is essentially capturing the same thing as l_to_L_ratio_max which is already in the base features. Let me check... yes, `l_to_L_ratio_max` is in the base features. So max_dim_l / vehicle_length would be redundant.

Let me think of something else from items_df that's not captured by existing aggregates.

What about:
- **fragile_item_count** - count of fragile items per dispatch (using if_fragile)
- **avg_flatness** - average item flatness per dispatch
- **dim_l_tail_count** - count of items where dim_l > 0.8 * vehicle_length (threshold count)

Actually, dim_l_tail_count is different from l_to_L_ratio_max because it's a COUNT of items near the limit, not just the maximum ratio. This is a tree-inspired threshold count.

But wait, I need vehicle_length from agg_df to compute this threshold. Let me think...

I could do:
1. Merge items_df with agg_df on dispatch_id to get vehicle_length
2. Compute per-item: dim_l / vehicle_length
3. Count items where this ratio > threshold

Let me also think about what's most likely to help TPR@FPR=1%. The key is finding features that strongly separate the hard cases.

Looking at the SVM weights:
- spare_capacity has large positive weight (13.56) - more spare = more feasible
- wl_to_vehicle_wl_total has positive weight (2.11) - this seems odd... higher utilization = more feasible? Maybe because it correlates with efficient packing
- wl_to_vehicle_wl_max has positive weight (1.30) - similar

Wait, that's interesting. wl_to_vehicle_wl_total and wl_to_vehicle_wl_max both have positive weights. This might be because in the linear model, they're correlated with the number of items or some other confounder. The interaction spare_cap_x_wl_max would capture the case where spare capacity is low (negative for feasibility) AND max footprint is high (should be negative for feasibility but currently has positive weight).

Hmm, let me think about this differently. The interaction spare_cap_x_wl_max would be high when both spare_capacity and wl_to_vehicle_wl_max are high. If spare_capacity is high, that's good for feasibility. If wl_to_vehicle_wl_max is high, that means one item takes up a lot of footprint. The interaction captures: "high spare capacity but also a dominant item" - which could go either way.

Actually, maybe I should think about what specific nonlinear patterns the SVM is missing. The XGB model recovers 92 cases that SVM misses. These are likely cases where:
- Spare capacity is moderate (not clearly low or high) but there's a hidden bottleneck
- The combination of features creates a nonlinear boundary

Let me try a different approach. Instead of spare_cap_x_wl_max, let me try:

1. **spare_cap_x_wl_max** - still worth trying as it's a top-1 × top-4 interaction
2. **wl_total_over_wl_max** - concentration signal
3. **n_fragile** - fragile item count from items_df

The fragile item count is interesting because fragile items constrain stacking - you can't put heavy items on top of them. This is a physical constraint that's not captured by dimension-based features.

Let me also consider: what about using item_flatness from items_df? Flat items (large footprint, small height) are easy to stack on but take up floor space. A dispatch with many flat items might have floor-space pressure.

**flat_item_floor_pressure** - sum of (item_flatness * item_volume) / vehicle_volume, representing how much floor space is consumed by flat items relative to available space.

Hmm, that might be too complex. Let me keep it simple.

Final proposal:

1. **spare_cap_x_wl_max** - spare_capacity × wl_to_vehicle_wl_max (top-1 × top-4 XGB gain interaction)
2. **wl_total_over_wl_max** - footprint concentration ratio (total / max)
3. **n_fragile_items** - count of fragile items per dispatch

These are 3 interpretable features that address:
1. Nonlinear interaction in the spare-capacity regime
2. Load concentration vs distribution
3. Stacking constraints from fragility

Let me write clean code:

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # Feature 1: spare_capacity × max footprint utilization interaction
    result['spare_cap_x_wl_max'] = (
        agg_df['spare_capacity'] * agg_df['wl_to_vehicle_wl_max']
    )

    # Feature 2: footprint concentration ratio (distributed vs peaked)
    result['wl_total_over_wl_max'] = (
        agg_df['wl_to_vehicle_wl_total'] / (agg_df['wl_to_vehicle_wl_max'] + 1e-9)
    )

    # Feature 3: number of fragile items per dispatch
    fragile_counts = (
        items_df.groupby('dispatch_id')['if_fragile']
        .sum()
        .reset_index()
    )
    fragile_counts.columns = ['dispatch_id', 'n_fragile_items']
    result = result.merge(fragile_counts, on='dispatch_id', how='left')
    result['n_fragile_items'] = result['n_fragile_items'].fillna(0)

    return result
```

Wait, I need to make sure the result has the right columns and order. Let me also double-check that none of these features are already in the active bank.

Active bank: spare_cap_sq, spare_cap_x_conc, spare_cap_x_wl_total, vol_top3_share, n_high_asr_items, spare_cap_log1p, h_to_H_max_sq, l_std_x_h_max, max_fp_ratio

None of my proposed features are in the active bank. Good.

Let me also check that wl_total_over_wl_max is not equivalent to any existing feature. The base features have wl_to_vehicle_wl_total and wl_to_vehicle_wl_max separately, but not their ratio. The ratio captures a different signal - how concentrated the footprint utilization is.

Actually, I realize I should also consider whether n_fragile_items would be useful. If most dispatches have 0 fragile items, this feature would be very sparse and might not help much. Let me think about alternatives.

What about **avg_item_flatness** - average flatness across items in a dispatch? This uses items_df's item_flatness column and captures a different aspect of packing difficulty.

Or **dim_m_avg_to_vW** - average of dim_m (middle sorted dimension) relative to vehicle width. This captures whether items are wide in their middle dimension, which constrains placement along the vehicle width.

Actually, let me think about what's most likely to help. The XGB guidance says the main gap is TPR@FPR=1%. The features that help there are ones that correctly identify infeasible cases at high
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
        - AUC: `0.9695`
        - TPR@FPR=1%: `0.6647`
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
- iter `6`: AUC `0.9695` (Δ `+0.0044`), TPR@1% `0.6647` (Δ `+0.0300`), features=spare_cap_x_wl_max, wl_total_over_wl_max, max_dim_l_to_vL
- iter `5`: AUC `0.9689` (Δ `+0.0037`), TPR@1% `0.6593` (Δ `+0.0246`), features=spare_cap_log1p, h_to_H_max_sq, l_std_x_h_max, max_fp_ratio
- iter `1`: AUC `0.9681` (Δ `+0.0030`), TPR@1% `0.6672` (Δ `+0.0324`), features=spare_cap_sq, spare_cap_x_conc, spare_cap_x_wl_total, vol_top3_share, n_high_asr_items

## Rejected Trials
- iter `3`: AUC `0.9682` (Δ `+0.0030`), TPR@1% `0.6608` (Δ `+0.0261`), features=spare_cap_x_avg_vol, wl_max_residual, n_long_items

## Failed Trials
- iter `2`: RateLimitError: Error code: 429 - {'error': {'message': "No deployments available for selected model, Try again in 5 seconds. Passed model=glm-5.1. pre-call-checks=False, cooldown_list=[('b9ca8bb822ee3f6a483f081af040bbc6769519968ef875dbfcc563db02907968', {'exception_received': 'litellm.RateLimitError: Model rate limit exceeded.', 'status_code': '429', 'timestamp': 1779097037.1183505, 'cooldown_time': 5}), ('ee7f63fe7b5a6e0c0f610a8773fbc7d2b61ec00beadff16b9c535feb3698db62', {'exception_received': 'litellm.InternalServerError: InternalServerError: Hosted_vllmException - Internal Server Error', 'status_code': '500', 'timestamp': 1779097037.671829, 'cooldown_time': 5})]", 'type': 'None', 'param': 'None', 'code': '429'}}
- iter `4`: ValueError: Repair response did not include FEATURE_CODE python block

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale


        ## Current Active Feature Bank
- source accepted trials: `1`, `5`, `6`
- active feature count: `12`
- active feature names: `spare_cap_sq`, `spare_cap_x_conc`, `spare_cap_x_wl_total`, `vol_top3_share`, `n_high_asr_items`, `spare_cap_log1p`, `h_to_H_max_sq`, `l_std_x_h_max`, `max_fp_ratio`, `spare_cap_x_wl_max`, `wl_total_over_wl_max`, `max_dim_l_to_vL`
- active bank AUC: `0.9695`
- active bank TPR@FPR=1%: `0.6647`
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
