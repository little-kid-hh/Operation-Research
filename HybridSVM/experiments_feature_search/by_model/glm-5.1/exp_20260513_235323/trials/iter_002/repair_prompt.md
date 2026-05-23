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
        - error: ValueError: Candidate code must define build_candidate_features

        Original response:
        ```text
        Let me analyze the current situation:

1. Current best: AUC 0.9721, TPR@FPR=1% 0.7139, Accuracy 0.9332
2. Active features: dominant_type_share, p90_long_over_bin_long, p90_mid_over_bin_mid, thin_item_share, max_face_area_load_over_floor, tight_bin_large_piece_interaction
3. Rejected trial (iter 1): volume_top3_share, multi_dim_near_limit_share, side_wall_area_load - these didn't beat the current best

The tree guidance suggests focusing on:
- Tail pressure (high quantiles, counts above thresholds)
- Slack-pressure interactions (low spare_capacity + large pieces)
- Local awkward-pattern counts
- Heterogeneity vs repetition
- Footprint and wall-pressure proxies

The SVM insights show the most important features are:
- sku_average_volume (negative weight - larger average volume → harder to pack)
- spare_capacity (positive weight - more spare → easier)
- sku_counts (negative - more items → harder)
- wl_to_vehicle_wl_avg, wl_to_vehicle_wl_std, l_to_L_ratio_avg, etc.

The rejected features (volume_top3_share, multi_dim_near_limit_share, side_wall_area_load) didn't help. Let me think about what might help differently.

Key observations:
1. The current active features already cover: dominant type share, p90 ratios for long/mid dimensions, thin item share, face area load, and tight bin × large piece interaction.
2. The rejected features tried: volume concentration (top3 share), multi-dimension near-limit share, and side wall area load.

What's missing that trees might be exploiting:
1. **Spare capacity × dimension pressure interaction** - The SVM shows spare_capacity has large positive weight and various dimension ratios have negative weights. An interaction between low spare capacity and high dimension pressure could help.
2. **Long-thick item share** - Items that are both long and thick (not thin/flat) create different packing challenges. The current features have thin_item_share but not long-thick.
3. **Extreme tail of item volume** - Not just p90 of dimensions, but p90/p95 of item volume relative to bin volume.
4. **Count of items near vehicle limit on multiple dimensions simultaneously** - Items that are tight on 2+ dimensions.
5. **Variance of item volumes** - High variance in item volumes might indicate awkward packing.
6. **Floor coverage ratio** - Sum of item footprints (s×m) vs vehicle floor area.

Let me think about what's most likely to help:

The rejected `multi_dim_near_limit_share` tried to capture items tight on multiple dimensions but didn't work. The rejected `side_wall_area_load` tried wall pressure but didn't work.

Let me focus on:
1. **spare_capacity × large_piece_pressure interaction** - This directly addresses the tree guidance about slack-pressure interactions. The idea is that when spare capacity is low AND there are large pieces, packing becomes especially hard. This is a multiplicative interaction that linearizes a nonlinear effect.

2. **long_thick_share** - Share of items where dim_l/dim_s > some threshold (not thin/flat) but dim_l is also relatively large. These are "rod-like" items that are hard to pack.

3. **p95_item_vol_over_bin_vol** - The 95th percentile of item volume relative to bin volume captures extreme large items better than p90.

Actually, let me reconsider. The current features already have:
- p90_long_over_bin_long and p90_mid_over_bin_mid (dimension ratios at p90)
- thin_item_share (flat items)
- tight_bin_large_piece_interaction (interaction between tight bin and large pieces)
- max_face_area_load_over_floor
- dominant_type_share

What could add incremental value?

Looking at the SVM weights more carefully:
- spare_capacity has weight +13.56 (most positive after sku_average_volume's -16.41)
- wl_to_vehicle_wl_total has weight +2.11

The trees seem to benefit from threshold effects around spare_capacity. Let me think about:

1. **low_spare_x_high_vol_pressure** - An interaction: (spare_capacity < threshold) × (some volume pressure metric). But we need to compute this from items_df and agg_df.

Wait, I need to be careful. The function receives agg_df and items_df. agg_df already has spare_capacity and vehicle dimensions. items_df has per-item data.

Let me design features that use both:

1. **spare_vol_x_avg_item_vol_ratio** - Interaction of spare_capacity with the ratio of average item volume to bin volume. When spare capacity is low AND average items are large, packing is very hard. This captures a nonlinear interaction.

Actually, I should be more creative. Let me think about what the trees are doing differently:

The trees can create splits like:
- IF spare_capacity < 0.15 AND sku_average_volume > X THEN infeasible
- IF wl_to_vehicle_wl_max > 0.9 AND sku_counts > 12 THEN infeasible

So I need features that capture these kinds of threshold interactions.

Let me propose:

1. **spare_cap_x_big_item_count** - spare_capacity × (count of items whose dim_l > 0.7 × vehicle_length). When spare capacity is low and there are many long items, packing is very constrained.

2. **long_item_share** - Share of items where dim_l > 0.5 × vehicle_length. This is different from p90_long_over_bin_long which is a quantile of the ratio; this is a count-based measure.

Wait, but tight_bin_large_piece_interaction might already be capturing something similar. Let me think about what's truly different.

Actually, let me look at this from a different angle. The rejected features were:
- volume_top3_share (concentration of volume in top 3 items)
- multi_dim_near_limit_share (items near limit on multiple dims)
- side_wall_area_load (wall pressure)

These were rejected because they didn't improve enough. So I need different hypotheses.

Let me try:

1. **spare_cap_x_dim_var** - Interaction of spare_capacity with dimension variance (e.g., sku_height_var). The SVM shows sku_height_var has a negative weight (-2.39), meaning more height variance → harder to pack. When spare capacity is also low, this effect amplifies. This interaction could help the linear SVM capture this.

2. **item_vol_p95_over_bin_vol** - 95th percentile of item volume / bin volume. This captures extreme tail items better than the existing p90 dimension ratios.

3. **heterogeneity_index** - A measure of how many distinct "size types" there are relative to total items. Different from dominant_type_share which measures concentration. This would be something like: number of distinct (rounded) size buckets / total item count.

Actually, let me reconsider what's most likely to work based on the tree guidance:

The tree guidance specifically mentions:
1. "low spare_capacity combined with large-piece ratios"
2. "low slack combined with high face-area load"  
3. "number/share of items whose long edge is near vehicle limits"
4. "number/share of items with thin-flat or long-thick geometry"
5. "distinct size-type count"
6. "summed face-area load against floor-like or side-like dimensions"

Current active features already cover some of these. Let me focus on the gaps:

- Gap 1: Interaction of spare capacity with dimension pressure (not just the individual features)
- Gap 2: Long-thick items (we have thin items but not long-thick)
- Gap 3: Distinct size-type count vs dominant type share (we have concentration but not diversity)

Let me propose:

1. **spare_cap_x_long_dim_pressure** - spare_capacity × (1 - p90_long_over_bin_long). When spare capacity is low AND the longest items take up much of the bin length, this product is very low (pushing toward infeasible). This interaction helps the SVM capture the nonlinear regime.

Actually wait - I can use agg_df columns directly. Let me check what's in agg_df:
- spare_capacity, vehicle dimensions, all the aggregate features

And items_df has per-item data.

Let me design:

1. **spare_cap_x_avg_vol_ratio** - spare_capacity × (1 / sku_average_volume_scaled). This creates an interaction where low spare AND high average volume gives very low values. But I need to be careful about how to compute this.

Actually, let me just use the agg_df columns for the interaction since they're already there:

```python
# spare_capacity is in agg_df
# sku_average_volume is in agg_df
# We can create: spare_capacity / sku_average_volume (or some interaction)
```

But wait - the rules say "Use only agg_df and items_df passed into the function." So I can use both.

Let me think about this more carefully. The key insight is:

The SVM has:
- spare_capacity weight = +13.56 (more spare → more feasible)
- sku_average_volume weight = -16.41 (larger avg volume → less feasible)

These are the two strongest features with opposite signs. A tree can split on combinations of these. To linearize this, I need an interaction feature.

Feature 1: **spare_cap_div_avg_vol** - spare_capacity / max(sku_average_volume, eps). When spare is high and avg vol is low, this is high (feasible). When spare is low and avg vol is high, this is very low (infeasible). This captures the interaction.

Feature 2: **long_thick_share** - Share of items where dim_l/dim_m > 3 AND dim_m/dim_s > 2 (or similar). These are "rod-like" items that are hard to orient and pack.

Actually, let me simplify. Let me think about what specific nonlinear patterns the trees capture:

Pattern 1: When spare_capacity is very low (< some threshold), even moderate-sized items make packing infeasible. The linear SVM can't capture this threshold effect.

Pattern 2: When there are many items with dimensions near the vehicle limits, the packing becomes infeasible even if average dimensions look okay.

For Pattern 1, I could create:
- **low_spare_flag** - Binary or soft threshold: max(0, 0.15 - spare_capacity) / 0.15. This is 0 when spare > 15%, and increases as spare decreases below 15%.

But this is a simple threshold on an existing feature, which the SVM should already be able to approximate with the linear term.

The key is INTERACTIONS. Let me focus on:

1. **spare_x_vol_pressure** - spare_capacity × (1 - total_item_vol / vehicle_vol). Wait, this is just spare_capacity × spare_capacity, which is quadratic and redundant.

Let me think differently. What if I create:

1. **spare_cap_x_wl_total** - spare_capacity × wl_to_vehicle_wl_total. Both are in agg_df. When spare is high AND wl_total is high, this is high (feasible - lots of space even with large footprint items). When spare is low AND wl_total is high, this is low (infeasible - tight space with large footprint items).

Actually, wl_to_vehicle_wl_total has positive weight (+2.11), meaning higher wl_total → more feasible. And spare_capacity has positive weight (+13.56). So their product would be high when both are high (feasible) and low when either is low. This might help capture the interaction regime.

But actually, I want to capture the case where BOTH are in a bad regime simultaneously. The product naturally does this: it's very low when either is low.

Hmm, but for a linear SVM, adding the product of two already-included features might not add much because the SVM already has both linear terms. The product adds the interaction term, which helps capture the nonlinear effect that the combination of low spare AND low wl_total is especially bad.

Wait, wl_to_vehicle_wl_total has POSITIVE weight, meaning higher → more feasible. So "low wl_total" is bad. And low spare_capacity is bad. So their product being low means at least one is bad. The interaction captures that when BOTH are bad simultaneously, it's worse than the sum of individual effects.

Let me also think about item-level features:

2. **near_limit_long_count** - Count of items where dim_l > 0.8 × vehicle_length. This is a threshold count that the trees can easily split on.

3. **max_item_vol_over_spare** - Maximum item volume / spare capacity volume. If the largest item takes up most of the spare space, packing is very constrained.

Actually, let me be more practical. Let me propose 3 features that are clearly different from existing ones:

1. **spare_cap_x_vol_ratio** - Interaction: (spare_capacity from agg_df) × (some volume ratio). This linearizes the tree's ability to split on combinations of spare capacity and volume pressure.

Let me compute this properly:
- spare_capacity is already in agg_df
- I need a volume pressure metric from items

Actually, I can compute: total_item_volume / vehicle_volume = 1 - spare_capacity (approximately). So that's redundant.

Let me use a different interaction:
- spare_capacity × sku_counts: When spare is low AND there are many items, packing is very hard. sku_counts has weight -10.59 (negative). So the product of a positive (spare) and negative-correlated (counts) feature creates an interesting interaction.

spare_capacity × sku_counts: High when lots of spare and many items (feasible - lots of space), low when little spare and many items (infeasible - tight with many items). This captures the "pressure" interaction.

2. **near_limit_item_share** - Share of items where dim_l > 0.75 × vehicle_length. This is a threshold count that directly captures the "items near vehicle limits" hypothesis.

3. **vol_heterogeneity** - Coefficient of variation of item volumes (std/mean). High heterogeneity means a mix of very large and very small items, which can be harder to pack efficiently.

Let me refine these:

Feature 1: **spare_x_count** - spare_capacity × sku_counts (from agg_df). Simple interaction.

Wait, but sku_counts is already in agg_df and has a strong negative weight. The interaction spare × counts would be: when spare is high and counts are high, the product is high (feasible); when spare is low and counts are high, the product is low (infeasible). This captures the nonlinear interaction.

Feature 2: **near_limit_long_share** - From items_df: share of items where dim_l > 0.75 × vehicle_length (need vehicle_length from agg_df or items_df).

Feature 3: **vol_cv** - From items_df: coefficient of variation of item_volume per dispatch.

Actually, I realize I should check - does items_df have vehicle_length? Looking at the schema: `dispatch_id`, `item_length`, `item_width`, `item_height`, `if_fragile`, `load_parameter`, `vehicle_capacity`, `dim_s`, `dim_m`, `dim_l`, `item_volume`, `item_flatness`

No vehicle_length directly, but there's vehicle_capacity. I can get vehicle dimensions from agg_df which has vehicle_length, vehicle_width, vehicle_height.

Let me design the code:

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # Feature 1: spare_capacity × sku_counts interaction
    result['spare_x_count'] = agg_df['spare_capacity'] * agg_df['sku_counts']

    # Feature 2: near_limit_long_share - share of items with dim_l > 0.75 * vehicle_length
    # Need to merge vehicle_length into items_df
    veh_dims = agg_df[['dispatch_id', 'vehicle_length']].copy()
    items_with_veh = items_df.merge(veh_dims, on='dispatch_id', how='left')
    items_with_veh['is_near_limit_long'] = (items_with_veh['dim_l'] > 0.75 * items_with_veh['vehicle_length']).astype(int)
    near_limit_share = items_with_veh.groupby('dispatch_id')['is_near_limit_long'].mean()
    result['near_limit_long_share'] = result['dispatch_id'].map(near_limit_share).fillna(0)

    # Feature 3: vol_cv - coefficient of variation of item volumes
    vol_stats = items_df.groupby('dispatch_id')['item_volume'].agg(['mean', 'std'])
    vol_stats['vol_cv'] = vol_stats['std'] / vol_stats['mean'].clip(lower=1e-9)
    result['vol_cv'] = result['dispatch_id'].map(vol_stats['vol_cv']).fillna(0)

    return result
```

Wait, but I should think about whether these features are truly incremental and not redundant with existing features.

- spare_x_count: This is an interaction of two existing features (spare_capacity and sku_counts). Neither the product nor any similar interaction is in the active bank. This should be new.

- near_limit_long_share: This is a threshold count (share of items above 75% of vehicle length). The existing p90_long_over_bin_long is a quantile of the ratio, not a count-based measure. The rejected multi_dim_near_limit_share was about items near limit on MULTIPLE dimensions, which is different. This should be new.

- vol_cv: This is the coefficient of variation of item volumes. The existing features don't include volume heterogeneity. The rejected volume_top3_share was about concentration, not CV. This should be new.

Let me also consider adding one more feature:

4. **spare_cap_x_avg_vol** - spare_capacity × sku_average_volume. This captures the interaction between available space and average item size. When spare is low and avg vol is high, this is very low (hard to pack). When spare is high and avg vol is low, this is moderate (easy to pack).

Actually, wait. sku_average_volume has a LARGE negative weight (-16.41). So higher avg vol → less feasible. And spare_capacity has a large positive weight (+13.56). Their product: when both are high, the product is high. But high avg vol means infeasible, and high spare means feasible. The product doesn't clearly capture the interaction I want.

Let me think again. The interaction I want is: when spare is LOW and avg vol is HIGH → very infeasible. This is captured by: (1 - spare_capacity) × sku_average_volume, or equivalently: sku_average_volume - spare_capacity × sku_average_volume.

Actually, the simplest interaction that captures "low spare AND high avg vol" is:
- sku_average_volume / spare_capacity: Very high when avg vol is high and spare is low (infeasible)

But division can be unstable. Let me use:
- sku_average_volume × (1 - spare_capacity): This is high when avg vol is high and spare is low.

But (1 - spare_capacity) is essentially the volume utilization ratio, which might be correlated with existing features.

Let me just go with the three features I have. They're clean and interpretable.

Actually, let me reconsider. Let me think about what the trees are really exploiting:

The error contrast shows features with notable
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
        - AUC: `0.9721`
        - TPR@FPR=1%: `0.7139`
        - Accuracy: `0.9332`

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



        # Memory

Baseline reference:
- AUC `0.9651`
- TPR@FPR=1% `0.6347`
- Accuracy `0.9276`

## Accepted Trials
- iter `0`: AUC `0.9721` (Δ `+0.0070`), TPR@1% `0.7139` (Δ `+0.0792`), features=dominant_type_share, p90_long_over_bin_long, p90_mid_over_bin_mid, thin_item_share, max_face_area_load_over_floor, tight_bin_large_piece_interaction

## Rejected Trials
- iter `1`: AUC `0.9719` (Δ `+0.0067`), TPR@1% `0.7070` (Δ `+0.0723`), features=volume_top3_share, multi_dim_near_limit_share, side_wall_area_load

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale


        ## Current Active Feature Bank
- source accepted trials: `0`
- active feature count: `6`
- active feature names: `dominant_type_share`, `p90_long_over_bin_long`, `p90_mid_over_bin_mid`, `thin_item_share`, `max_face_area_load_over_floor`, `tight_bin_large_piece_interaction`
- active bank AUC: `0.9721`
- active bank TPR@FPR=1%: `0.7139`
- active bank Accuracy: `0.9332`

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
