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
        - error: ValueError: LLM response did not include FEATURE_CODE python block

        Original response:
        ```text
        Let me analyze the current situation carefully.

Current active feature bank (12 features):
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

Current best metrics:
- AUC: 0.9753
- TPR@FPR=1%: 0.7198
- Accuracy: 0.9324

I need to beat this. The acceptance rule requires:
- AUC improvement > 0.0005, OR
- TPR@FPR=1% improvement > 0.005, OR
- Accuracy improvement > 0.0005

Rejected trial 4 had: spare_x_wlmax, total_footprint_ratio, cross_dim_pressure_count → AUC 0.9749, TPR@1% 0.7119

So that combination didn't work well. Let me think about what's missing.

Looking at the tree guidance and SVM insights:
- Trees exploit threshold/interaction structure
- Key features where trees win: spare_capacity, wl_to_vehicle_wl_total, sku_average_volume, wl_to_vehicle_wl_max, sku_length_avg, sku_width_var

Current active features already cover:
- Type concentration (dominant_type_share)
- Tail pressure on long/mid dimensions (p90_long_over_bin_long, p90_mid_over_bin_mid)
- Thin item share
- Face area load (max_face_area_load_over_floor)
- Spare x count interaction
- Near-limit long share
- Volume CV
- Height pressure share
- Long-wide item share
- Spare x vol_cv interaction

What's NOT yet covered that trees might exploit:
1. **Wall pressure / side pressure** - rejected iter 1 had side_wall_area_load which didn't help, but maybe a different formulation
2. **Footprint stacking pressure** - how much floor area is consumed when considering stacking
3. **Cross-dimension coupling** - items that are large in multiple dimensions simultaneously (not just long-wide)
4. **Extreme item bottleneck** - the single largest item's footprint vs floor area
5. **Multi-modal dimension distribution** - items clustering into different size groups
6. **Slack x wl_to_vehicle_wl_max interaction** - iter 4 tried spare_x_wlmax but it didn't work

Let me think about what new directions might help:

1. **max_item_footprint_ratio** - The largest single item's footprint (dim_l * dim_m) relative to the vehicle floor (vehicle_length * vehicle_width). A single item consuming most of the floor is a strong bottleneck signal that trees would pick up via thresholding.

2. **cross_dim_near_limit_share** - Share of items where BOTH dim_l and dim_m exceed 50% of their respective vehicle dimensions. This captures "blocky" items that are hard to pack because they're large in two dimensions.

3. **floor_occupancy_stress** - Total footprint of all items divided by floor area, but weighted by stacking impossibility. This is different from total_footprint_ratio (which was rejected) because it considers that items can't always stack.

Wait, let me reconsider. The rejected trial 4 had total_footprint_ratio and cross_dim_pressure_count. Maybe the formulations were different from what I'd propose.

Let me think about what's really different:

The current features focus on:
- Volume-based measures (vol_cv, spare_x_volcv)
- Single-dimension pressure (near_limit_long_share, p90_long_over_bin_long)
- Item shape classification (thin_item_share, long_wide_item_share)
- Floor face area (max_face_area_load_over_floor)
- Type structure (dominant_type_share)

Missing areas:
1. **Extreme single-item dominance** - one item that's almost as big as the bin in multiple dimensions
2. **Dimensional coupling at the extreme** - items large in 2+ dimensions simultaneously
3. **Stacking pressure** - height utilization pattern

Let me propose:

1. **max_item_diml_over_vehL** - The single largest item's longest dimension relative to vehicle length. This is a bottleneck signal: if one item nearly spans the vehicle, remaining items must fit around it.

2. **big_two_dim_share** - Share of items where dim_l > 0.5 * vehicle_length AND dim_m > 0.5 * vehicle_width. These "big two-dimension" items are particularly hard to pack because they consume floor space in both directions.

3. **height_slack_x_tall_share** - Interaction between height slack (spare_capacity related to height dimension) and share of tall items. This captures the regime where height is the binding constraint.

Wait, I need to be careful. Let me think about what would genuinely help a linear SVM.

The SVM already has:
- spare_capacity (w=+13.6, positive → feasible)
- sku_counts (w=-10.6, negative → not feasible)
- wl_to_vehicle_wl_avg (w=+7.6)
- wl_to_vehicle_wl_std (w=-4.3)

The interaction spare_x_volcv is already in the bank. What other interactions might help?

Looking at the error analysis: trees win on cases around spare_capacity, wl_to_vehicle_wl_total, sku_average_volume, wl_to_vehicle_wl_max.

Let me think about wl_to_vehicle_wl_max specifically. It has w=+1.3 in the SVM. But trees might exploit a nonlinear pattern: when wl_to_vehicle_wl_max is very high AND spare_capacity is low, that's particularly bad. The linear SVM can't capture this interaction.

So: **spare_x_wlmax** was tried in iter 4 and rejected. But maybe the formulation was wrong, or it conflicted with other features in that batch.

Let me try a different angle. What about:

1. **max_item_floor_ratio** - The single item with the largest footprint (dim_l × dim_m) relative to vehicle floor area. This captures extreme bottleneck items.

2. **count_above_half_two_dims** - Number of items where dim_l > 0.5 * vehicle_length AND dim_m > 0.5 * vehicle_width. Captures "chunky" items that are hard to nest.

3. **spare_x_nearlimit_count** - Interaction of spare_capacity with near_limit_long_share (already in bank). This captures the regime where both low slack and many near-limit items coincide.

Wait, near_limit_long_share is already in the bank. And spare_x_count is in the bank. So spare_x_nearlimit_count might be too correlated with spare_x_count.

Let me think more carefully...

The key insight from tree analysis is that trees exploit:
1. Threshold effects (items near vehicle limits)
2. Interactions (low spare × high pressure)
3. Distribution shape (heterogeneity, tails)

Current bank covers these but maybe not deeply enough. Let me focus on:

1. **max_single_footprint_ratio** - Largest item's floor footprint ratio. This is a pure bottleneck signal that's different from averages or sums. A single item consuming 80%+ of the floor is a hard constraint.

2. **two_dim_large_share** - Share of items large in 2 sorted dimensions simultaneously (dim_l > 0.5*veh_L AND dim_m > 0.5*veh_W). These are the hardest items to pack.

3. **spare_x_p90long** - Interaction between spare_capacity and p90_long_over_bin_long. When spare is low AND the 90th percentile item is near the vehicle limit, packing becomes much harder.

Actually, p90_long_over_bin_long is already in the bank. So spare_x_p90long would be an interaction of an existing bank feature with spare_capacity. That's a valid new feature.

Let me finalize my proposal:

1. **max_single_footprint_ratio** - Max item footprint (dim_l × dim_m) / (vehicle_length × vehicle_width). Captures the worst-case single-item floor bottleneck.

2. **two_dim_large_share** - Fraction of items where dim_l > 0.5*vehicle_length AND dim_m > 0.5*vehicle_width. Captures "chunky" hard-to-pack items.

3. **spare_x_p90long** - spare_capacity × p90_long_over_bin_long. Interaction capturing when both low slack and extreme long-dimension pressure coincide.

Wait, but I should be careful. The existing `tight_bin_large_piece_interaction` from iter 0 might already be similar to some of these. Let me check - I don't know its exact definition, but the name suggests it's about tight bin (low spare) × large pieces. And `spare_x_count` is spare × count. And `spare_x_volcv` is spare × volume CV.

So spare_x_p90long would be a new interaction axis: spare × extreme dimension pressure. This is different from spare × count or spare × volume variability.

Let me also think about whether max_single_footprint_ratio might be too correlated with existing features. The existing `max_face_area_load_over_floor` might be similar... but "face area" could mean the largest face of any item, while "footprint" specifically means the bottom face when placed. Actually, for packing, the footprint is dim_l × dim_m (the two largest dimensions form the base). The face area could be any face. So they might be different.

Actually, looking more carefully, `max_face_area_load_over_floor` probably already captures something similar to max_single_footprint_ratio. Let me think of something different.

How about:

1. **two_dim_large_share** - Share of items where dim_l > 0.5*vehicle_length AND dim_m > 0.5*vehicle_width.

2. **spare_x_p90long** - spare_capacity × p90_long_over_bin_long interaction.

3. **dim_l_range_ratio** - (max(dim_l) - min(dim_l)) / vehicle_length. Captures the spread of longest dimensions relative to the bin. High spread means very heterogeneous item sizes along the longest axis, which can make packing harder (or easier, if there's room to nest).

Actually, let me reconsider. I want features that are most likely to help. Let me focus on the strongest hypotheses:

**Hypothesis 1**: Items that are large in TWO dimensions simultaneously are disproportionately hard to pack. The existing features capture single-dimension pressure (near_limit_long_share) and shape classification (thin_item_share, long_wide_item_share), but not the two-dimensional coupling.

**Hypothesis 2**: The interaction between spare capacity and extreme dimension pressure (p90 long) captures a nonlinear regime that trees exploit.

**Hypothesis 3**: The range/spread of item dimensions matters - when items have very different sizes, some packing algorithms struggle, but also some combinations are easier to nest.

Let me go with:

1. **two_dim_large_share** - Fraction of items where dim_l > 0.5*vehicle_length AND dim_m > 0.5*vehicle_width
2. **spare_x_p90long** - spare_capacity × p90_long_over_bin_long  
3. **max_diml_gap_ratio** - (max(dim_l) - second_max(dim_l)) / vehicle_length - captures whether there's one dominant large item

Actually, max_diml_gap_ratio might be too noisy. Let me think again.

How about focusing on just 2-3 very clean features:

1. **two_dim_large_share** - Share of items large in both long and mid dimensions relative to vehicle
2. **spare_x_p90long** - Interaction of spare capacity with extreme long-dimension pressure
3. **dim_l_p75_over_vehL** - 75th percentile of dim_l / vehicle_length. This is different from p90 (already in bank) and captures a different part of the distribution.

Wait, p90_long_over_bin_long is already in the bank. Adding p75 might be too similar. Let me avoid that.

Let me try:

1. **two_dim_large_share** - Share of items large in 2+ dimensions
2. **spare_x_p90long** - spare × p90_long interaction
3. **floor_contention_ratio** - sum of item footprints (dim_l × dim_m) / (vehicle_length × vehicle_width) but only counting items with dim_l > 0.3*vehicle_length. This captures floor contention from non-trivial-sized items.

Hmm, floor_contention_ratio might overlap with existing features. Let me simplify.

Final proposal - 3 features:

1. **two_dim_large_share** - Share of items where dim_l > 0.5 * vehicle_length AND dim_m > 0.5 * vehicle_width. These "chunky" items can't be nested easily and create floor contention.

2. **spare_x_p90long** - spare_capacity × p90_long_over_bin_long. Captures the nonlinear interaction between low slack and extreme long-dimension pressure.

3. **max_diml_to_vehL** - Maximum dim_l across items / vehicle_length. The single longest item's dimension relative to the bin. This is a pure bottleneck signal - different from p90 which is a quantile, this is the absolute max.

Wait, is max_diml_to_vehL too similar to l_to_L_ratio_max which is already in the baseline? Let me check... yes, `l_to_L_ratio_max` is in the existing aggregate features. So I shouldn't add this.

OK let me reconsider. What about:

1. **two_dim_large_share** - as above
2. **spare_x_p90long** - as above  
3. **dim_m_p90_over_vehW** - 90th percentile of dim_m / vehicle_width. We have p90_long_over_bin_long but NOT p90 of the mid dimension over vehicle width. The mid dimension matters for floor packing too.

Wait, we have p90_mid_over_bin_mid in the bank already. So this is covered.

Let me think about what's truly missing...

Looking at the rejected trial 4: spare_x_wlmax, total_footprint_ratio, cross_dim_pressure_count → AUC 0.9749, TPR@1% 0.7119

This was WORSE than the current best (AUC 0.9753, TPR@1% 0.7198). So those specific features didn't help.

The key question: what nonlinear structure are trees still exploiting that the current 12-feature bank doesn't capture?

Let me think about this differently. The current bank has:
- Type structure: dominant_type_share
- Tail pressure: p90_long_over_bin_long, p90_mid_over_bin_mid
- Shape: thin_item_share, long_wide_item_share
- Floor pressure: max_face_area_load_over_floor
- Interactions: tight_bin_large_piece_interaction, spare_x_count, spare_x_volcv
- Near-limit: near_limit_long_share
- Volume shape: vol_cv
- Height: height_pressure_share

What's NOT covered:
1. **Coupling between dimensions at the item level** - items that are simultaneously large in multiple dimensions
2. **Stacking constraints** - can tall items be stacked on top of each other?
3. **Waste/void space indicators** - items with awkward proportions that create voids
4. **Permutation/arrangement difficulty** - how many ways can items be arranged?

I think the strongest bet is:

1. **two_dim_large_share** - Items large in 2+ sorted dimensions. This is genuinely new - not covered by near_limit_long_share (which only looks at dim_l) or long_wide_item_share (which might use different thresholds or raw dimensions).

2. **spare_x_p90long** - The interaction between spare capacity and the p90 long-dimension ratio. This is a specific nonlinear interaction that trees would pick up.

3. **awkward_shape_share** - Share of items where the aspect ratio (dim_l/dim_s) is very high (>8) OR the item is both non-thin AND non-long (cube-like with all dimensions > 0.3*vehicle). These items create packing difficulties because they can't be slotted into gaps.

Actually, let me simplify. I'll propose 3 clean features:

1. **two_dim_large_share** - Share of items where dim_l > 0.5*vehicle_length AND dim_m > 0.5*vehicle_width
2. **spare_x_p90long** - spare_capacity × p90_long_over_bin_long
3. **cube_item_share** - Share of items where dim_s/dim_l > 0.5 (cube-like items that resist nesting)

Let me code this up:

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    # Merge vehicle dimensions into items_df
    items = items_df.merge(
        agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height', 'spare_capacity']],
        on='dispatch_id',
        how='left'
    )

    # Feature 1: two_dim_large_share
    # Share of items where dim_l > 0.5*vehicle_length AND dim_m > 0.5*vehicle_width
    items['is_two_dim_large'] = (
        (items['dim_l'] > 0.5 * items['vehicle_length']) &
        (items['dim_m'] > 0.5 * items['vehicle_width'])
    ).astype(int)

    two_dim_large = items.groupby('dispatch_id')['is_two_dim_large'].mean().reset_index()
    two_dim_large.columns = ['dispatch_id', 'two_dim_large_share']

    # Feature 2: spare_x_p90long
    # Need p90_long_over_bin_long from items, then interact with spare_capacity
    p90_long = items.groupby('dispatch_id').apply(
        lambda g: g['dim_l'].quantile(0.9) / g['vehicle_length'].iloc[0]
    ).reset_index()
    p90_long.columns = ['dispatch_id', 'p90_long_ratio']

    spare_cap = agg_df[['dispatch_id', 'spare_capacity']]
    spare_x_p90long = p90_long.merge(spare_cap, on='dispatch_id')
    spare_x_p90long['spare_x_p90long'] = spare_x_p90long['spare_capacity'] * spare_x_p90long['p90_long_ratio']
    spare_x_p90long = spare_x_p90long[['dispatch_id', 'spare_x_p90long']]

    # Feature 3: cube_item_share
    # Share of items where dim_s/dim_l > 0.5 (cube-like items)
    items['is_cube'] = (items['dim_s'] / items['dim_l'].clip(lower=1e-9) > 0.5).astype(int)
    cube_share = items.groupby('dispatch_id')['is_cube'].mean().reset_index()
    cube_share.columns = ['dispatch_id', 'cube_item_share']

    # Merge all features
    result = two_dim_large.merge(spare_x_p90long, on='dispatch_id').merge(cube
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
        - AUC: `0.9753`
        - TPR@FPR=1%: `0.7198`
        - Accuracy: `0.9324`

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

        Current iteration: `5`

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


        # Memory

Baseline reference:
- AUC `0.9651`
- TPR@FPR=1% `0.6347`
- Accuracy `0.9276`

## Accepted Trials
- iter `3`: AUC `0.9753` (Δ `+0.0101`), TPR@1% `0.7198` (Δ `+0.0851`), features=height_pressure_share, long_wide_item_share, spare_x_volcv
- iter `2`: AUC `0.9740` (Δ `+0.0089`), TPR@1% `0.7080` (Δ `+0.0733`), features=spare_x_count, near_limit_long_share, vol_cv
- iter `0`: AUC `0.9721` (Δ `+0.0070`), TPR@1% `0.7139` (Δ `+0.0792`), features=dominant_type_share, p90_long_over_bin_long, p90_mid_over_bin_mid, thin_item_share, max_face_area_load_over_floor, tight_bin_large_piece_interaction

## Rejected Trials
- iter `1`: AUC `0.9719` (Δ `+0.0067`), TPR@1% `0.7070` (Δ `+0.0723`), features=volume_top3_share, multi_dim_near_limit_share, side_wall_area_load
- iter `4`: AUC `0.9749` (Δ `+0.0097`), TPR@1% `0.7119` (Δ `+0.0772`), features=spare_x_wlmax, total_footprint_ratio, cross_dim_pressure_count

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale


        ## Current Active Feature Bank
- source accepted trials: `0`, `2`, `3`
- active feature count: `12`
- active feature names: `dominant_type_share`, `p90_long_over_bin_long`, `p90_mid_over_bin_mid`, `thin_item_share`, `max_face_area_load_over_floor`, `tight_bin_large_piece_interaction`, `spare_x_count`, `near_limit_long_share`, `vol_cv`, `height_pressure_share`, `long_wide_item_share`, `spare_x_volcv`
- active bank AUC: `0.9753`
- active bank TPR@FPR=1%: `0.7198`
- active bank Accuracy: `0.9324`

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
