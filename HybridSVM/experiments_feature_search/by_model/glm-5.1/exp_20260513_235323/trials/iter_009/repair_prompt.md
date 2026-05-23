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
        Let me analyze the current state:

**Current best metrics:**
- AUC: 0.9758
- TPR@FPR=1%: 0.7665
- Accuracy: 0.9356

**Active feature bank (18 features):**
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

**Key observations from SVM weights:**
- sku_average_volume is the strongest negative feature (-16.4)
- spare_capacity is the strongest positive feature (+13.6)
- sku_counts is strongly negative (-10.6)
- wl_to_vehicle_wl_avg is positive (+7.6)
- wl_to_vehicle_wl_std is negative (-4.3)

**What's missing from the active bank:**
Looking at the tree guidance and current features, I see we have:
- Threshold counts: near_limit_long_share, near_limit_mid_share
- Interactions: spare_x_count, spare_x_volcv, spare_x_p90long, spare_x_avgvol
- Shape shares: thin_item_share, long_wide_item_share, long_thin_item_share, cube_item_share
- Distribution: dominant_type_share, vol_cv
- Quantiles: p90_long_over_bin_long, p90_mid_over_bin_mid
- Pressure: height_pressure_share, max_face_area_load_over_floor
- Two-dim: two_dim_large_share, tight_bin_large_piece_interaction

**What I haven't tried yet:**
1. **Short dimension pressure** - near_limit for the shortest dimension (dim_s near vehicle height). We tried near_limit_short_share in iter 8 but it didn't improve. However, the short dimension is often the height constraint which is critical for stacking.

2. **Three-dimension extreme items** - items that are near-limit in ALL three dimensions simultaneously (very tight fits)

3. **Dominant dimension coupling** - when one item dominates the footprint, leaving little room for others

4. **Floor occupancy** - how much of the vehicle floor area is consumed by the largest few items' footprints

5. **Stacking difficulty** - items with incompatible heights that prevent efficient stacking

6. **Edge-case: very large single item share** - when one item takes up a huge fraction of volume

Let me think about what trees are exploiting that we haven't captured:

Looking at the rejected trials:
- iter 1: volume_top3_share, multi_dim_near_limit_share, side_wall_area_load - didn't beat
- iter 4: spare_x_wlmax, total_footprint_ratio, cross_dim_pressure_count - didn't beat
- iter 8: near_limit_short_share, fragile_share, spare_x_fragile - matched but didn't beat

The fact that iter 8 matched but didn't beat suggests we're close to a plateau. We need something genuinely new.

Let me think about what physical packing constraints matter:

1. **Footprint conflict** - When items have large footprints (dim_l × dim_m), they compete for floor space. Multiple large-footprint items create a bottleneck that volume-only measures miss.

2. **Height stacking inefficiency** - Items with different heights waste vertical space. The sum of max-height-per-stack vs total height is a key signal.

3. **Extreme aspect ratio items** - Very long and thin items (like rods) or very flat items (like sheets) are awkward to pack regardless of volume.

4. **Near-limit in multiple dimensions** - Items close to vehicle limits in 2+ dimensions simultaneously are extremely hard to place.

Wait, we already have two_dim_large_share. But what about items near limit in ALL three dimensions?

Let me think about what's different about the tree model's advantage:

The tree model has AUC ~0.982-0.985 vs our 0.9758. That's still a significant gap. The tree is capturing nonlinear interactions that our linear model can't see.

Key insight: The SVM is heavily relying on spare_capacity and sku_average_volume. But the relationship between these and feasibility is nonlinear - there are threshold effects. When spare_capacity is very low AND there are many items, that's much worse than either condition alone.

We already have spare_x_count and spare_x_avgvol. What about:

1. **spare_capacity squared** or **spare_capacity threshold** - directly capturing the nonlinear effect
2. **Very large item bottleneck** - count of items whose volume exceeds some fraction of vehicle volume
3. **Packing density heterogeneity** - mix of very large and very small items

Let me think about specific new features:

1. **big_item_count** - Number of items where dim_l > 0.7 * vehicle_length. This is a hard threshold count that captures "how many items are genuinely large in the long dimension" - different from near_limit_long_share which might use a tighter threshold.

Actually wait, near_limit_long_share already captures items near the vehicle long dimension limit. Let me think differently.

2. **short_dim_near_height_share** - Items where dim_s is close to vehicle height. This matters because if an item's shortest dimension is near the vehicle height, it can't be laid on its side - it must go in with a specific orientation.

Hmm, but iter 8 tried near_limit_short_share and it didn't help.

3. **floor_footprint_top1_ratio** - The largest item's floor footprint (dim_l × dim_m) divided by vehicle floor area. A single item dominating the floor leaves little room for others.

4. **floor_footprint_top3_ratio** - Top 3 items' floor footprints / vehicle floor area.

5. **spare_x_near_limit_long** - Interaction between low spare capacity and having items near the long limit. This is different from spare_x_p90long because near_limit_long_share is a count/share measure rather than a quantile.

Wait, we already have near_limit_long_share and spare_x_p90long. But we don't have spare_x_near_limit_long. That could be a useful interaction.

6. **three_dim_near_limit_share** - Items near limit in all three dimensions simultaneously. This is the hardest to pack.

Let me look at what iter 1 tried: multi_dim_near_limit_share - this was rejected. So maybe this isn't useful.

7. **large_volume_item_share** - Share of items whose volume exceeds some threshold relative to average or to vehicle volume.

8. **dim_l_max_ratio** - The maximum dim_l / vehicle_length ratio. How close is the longest item to the vehicle length?

Actually, we have l_to_L_ratio_max in the base features already.

Let me think about what's really missing...

The tree model advantage is about 0.006-0.009 AUC. Our active bank already improved AUC by 0.0106 from baseline. To get another 0.0005 improvement, we need features that capture genuinely new information.

Key insight from the SVM weights: wl_to_vehicle_wl_std is negative (-4.3), meaning variability in the width-length product relative to vehicle is bad. This suggests that when items have very different footprint sizes, packing becomes harder.

But we already have vol_cv which captures volume heterogeneity. What about footprint heterogeneity specifically?

Let me think about physical packing more carefully:

**Floor coverage problem**: When you place items in a bin, you're essentially tiling the floor. Items with large footprints (dim_l × dim_m) consume floor area. If the sum of footprints exceeds the vehicle floor area, you need to stack. But stacking requires compatible heights.

**Feature idea: floor_overshoot_ratio** = sum(dim_l × dim_m for all items) / (vehicle_length × vehicle_width). If this ratio > 1, you MUST stack. The further above 1, the more stacking is needed, and the harder packing becomes.

This is related to but different from volume-based measures because it specifically captures the 2D floor tiling constraint.

Wait, iter 4 tried total_footprint_ratio and it was rejected. Let me think about why...

Maybe total_footprint_ratio is too correlated with volume-based measures. What about something more nuanced?

**Feature idea: stacking_pressure** = max(0, floor_overshoot_ratio - 1) × some measure of height incompatibility.

Or simpler: **floor_overshoot** = max(0, sum(item footprints) / vehicle floor - 1). This is 0 when items fit on one layer and positive when stacking is required.

**Feature idea: height_diversity** - Number of distinct height "bands" among items. If items have very different heights, stacking is inefficient.

Let me think about what specific features to propose:

1. **floor_overshoot** - max(0, sum(dim_l × dim_m) / (VL × VW) - 1). Captures when 2D floor constraint forces stacking.

2. **spare_x_near_limit_long** - spare_capacity × near_limit_long_share interaction. When spare capacity is low AND there are items near the vehicle length limit, packing is especially hard.

3. **max_item_vol_ratio** - Maximum item volume / vehicle volume. A single very large item creates a bottleneck.

Wait, I need to be careful. near_limit_long_share is already in the active bank. So spare_x_near_limit_long would be an interaction between an active bank feature and a base feature. That should be fine as a new feature.

Let me also think about:

4. **tall_item_share** - Items where dim_l (the longest sorted dimension) is close to vehicle height. Wait, this doesn't make sense because dim_l is the longest dimension of the item, not necessarily the height.

Actually, in packing, items can be rotated. So the sorted dimensions (dim_s ≤ dim_m ≤ dim_l) represent the item's dimensions regardless of orientation. The key constraint is that no dimension of the item can exceed the corresponding vehicle dimension.

For packing feasibility, the critical question is: can we orient and place all items so they fit? This depends on:
- Total volume vs vehicle volume (spare_capacity captures this)
- Individual item dimensions vs vehicle dimensions (near_limit shares capture this)
- The interaction between these (spare_x_... captures some of this)
- The combinatorial difficulty of fitting items together (hard to capture)

What about features that capture combinatorial difficulty?

5. **dominant_footprint_share** - The share of total floor area consumed by the largest item. Similar to dominant_type_share but for footprints.

6. **height_range_ratio** - (max dim_s - min dim_s) / vehicle_height. Large range in shortest dimensions means stacking is inefficient.

Let me settle on a focused set of 3 features:

1. **floor_overshoot** - max(0, sum(dim_l × dim_m) / (VL × VW) - 1). This captures the 2D packing pressure that volume alone doesn't capture. When floor area demand exceeds supply, stacking becomes mandatory, and the degree of overshoot measures how much stacking pressure exists.

2. **spare_x_near_limit_long** - spare_capacity × near_limit_long_share. This is a direct interaction between the two most important SVM features' regimes: low spare capacity combined with items near the vehicle length limit.

3. **max_vol_item_ratio** - Maximum item volume / total item volume. When one item dominates the volume, it constrains placement of all others.

Wait, let me reconsider. The total_footprint_ratio was tried in iter 4 and rejected. But floor_overshoot is different - it's a thresholded version (only the excess above 1). This is more tree-like because it creates a piecewise linear feature.

Actually, I realize I should think more carefully about what features would genuinely help. Let me look at the error patterns:

The SVM makes 86 errors while HGB makes 38. The difference is 48 cases. These are cases where nonlinear structure matters.

The strongest SVM features are:
1. sku_average_volume (negative) - more volume per item → harder
2. spare_capacity (positive) - more spare → easier
3. sku_counts (negative) - more items → harder

The interaction between these is key. We already have spare_x_count and spare_x_avgvol. 

What about sku_counts × sku_average_volume? More items AND larger average volume is a double whammy. But this is essentially total_volume / vehicle_volume which is related to spare_capacity.

Let me think about what's NOT captured:

- **The distribution of item sizes within a dispatch**: We have vol_cv and dominant_type_share, but not the skewness or the specific pattern of a few large items + many small ones vs uniform medium items.

- **The "biggest item bottleneck"**: How much does the single largest item constrain the packing? We don't have this directly.

- **The floor-area competition**: Multiple items with large footprints competing for the same floor space.

Let me try these features:

1. **floor_overshoot** = max(0, sum(dim_l × dim_m for all items) / (VL × VW) - 1)
   - Captures 2D floor pressure beyond what volume captures
   - Thresholded at 0, so it only activates when stacking is forced

2. **max_item_vol_share** = max(item_volume) / sum(item_volume) per dispatch
   - When one item dominates, it constrains all placement decisions
   - Different from dominant_type_share which is about repeated identical items

3. **spare_x_near_limit_long** = spare_capacity × near_limit_long_share
   - Interaction: low spare AND long items near limit = very hard
   - Linearizes the tree's threshold interaction

Actually, wait. Let me reconsider max_item_vol_share. If one item is very large relative to others, that's actually sometimes easier (you place it first, then fit smaller items around it). The hard case is when there are several large items that compete for space.

What about **top3_vol_share** = sum of top 3 item volumes / total volume? iter 1 tried volume_top3_share and it was rejected. Hmm.

Let me think about this differently. What specific packing failure modes exist?

1. **One item too large** - Already captured by near_limit shares and l_to_L_ratio_max
2. **Too many medium-large items** - Partially captured by sku_counts and sku_average_volume
3. **Incompatible shapes** - Partially captured by thin_item_share, cube_item_share, etc.
4. **Floor area competition** - Not well captured
5. **Height stacking incompatibility** - Not well captured
6. **Orientation constraints** - Not captured (fragile items, etc.)

For floor area competition, let me think about a better feature:

**floor_pressure_top2** = (footprint of largest item + footprint of 2nd largest item) / vehicle floor area. If two items both have large footprints, they compete for the same floor space.

Or: **large_footprint_pair_count** = number of pairs of items where both have footprint > some threshold. This captures the combinatorial competition.

Actually, that's complex. Let me keep it simple.

Let me try a different approach. What about features related to the SHORT dimension (height in typical orientation)?

**short_dim_pressure** = sum(dim_s) / vehicle_height. This captures how many "layers" of items you need vertically. If the sum of shortest dimensions exceeds vehicle height, you can't stack them all vertically.

Wait, that's not quite right either. Items can be oriented differently.

Let me think about this more carefully. In 3D packing:
- Items have sorted dimensions dim_s ≤ dim_m ≤ dim_l
- The vehicle has dimensions VL, VW, VH
- For an item to fit, we need to find an orientation where each item dimension ≤ corresponding vehicle dimension
- Since dim_s ≤ dim_m ≤ dim_l, the item fits if dim_l ≤ max(VL,VW,VH) AND dim_m ≤ mid(VL,VW,VH) AND dim_s ≤ min(VL,VW,VH)

The existing features like l_to_L_ratio_avg etc. capture the average ratio of item long dimension to vehicle long dimension.

What about the **cumulative** dimension pressure? If you line up all items along the vehicle length, the sum of their long dimensions would be sum(dim_l). The ratio sum(dim_l) / VL tells you about the 1D packing pressure along the longest axis.

**Feature: long_dim_sum_ratio** = sum(dim_l) / VL. This captures 1D pressure along the vehicle length.

Similarly: **mid_dim_sum_ratio** = sum(dim_m) / VW, **short_dim_sum_ratio** = sum(dim_s) / VH.

These are related to but different from the average ratios already in the base features. The average ratio is sum(dim_l / VL) / n = sum(dim_l) / (n × VL). The sum ratio is sum(dim_l) / VL = n × avg_ratio. So it's essentially sku_counts × l_to_L_ratio_avg, which is an interaction we could create.

But we already have spare_x_count. And wl_to_vehicle_wl_total is in the base features.

Hmm, let me think about what's truly novel...

**Key insight**: The tree model captures the fact that certain COMBINATIONS of conditions make packing infeasible. For example:
- Low spare capacity + items near length limit + items near width limit = very hard
- Low spare capacity + many items + high volume variance = very hard

We've been adding pairwise interactions (spare × count, spare × volcv, spare × p90long, spare × avgvol). What about three-way interactions? Or what about interactions between two active bank features?

**Feature: near_limit_long_x_near_limit_mid** = near_limit_long_share × near_limit_mid_share. When items are near limit in BOTH long and mid dimensions, they're extremely hard to pack.

**Feature: two_dim_large_x_spare** = two_dim_large_share × spare_capacity. Interaction between two-dimensional large items and slack.

**Feature: vol_cv_x_count** = vol_cv × sku_counts. High volume variance with many items means complex packing.

Let me think about which of these is most promising...

The near_limit_long_share × near_limit_mid_share interaction captures: "items that are large in both the long and mid dimensions relative to the vehicle." This is a direct nonlinear interaction that trees would capture through splits.

But wait, we already have two_dim_large_share which captures items large in two dimensions. The interaction near_limit_long × near_limit_mid is different - it's about the SHARE of items near the long limit TIMES the SHARE near the mid limit, not about individual items being large in both dimensions.

Actually, that's a dispatch-level interaction: if many items are near the long limit AND many items are near the mid limit (possibly different items), the overall packing is very constrained.

This could be powerful. Let me include it.

Let me finalize my 3 features:

1. **floor_overshoot** = max(0, sum(dim_l × dim_m
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

        Current iteration: `9`

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
- iter `8`: AUC `0.9758` (Δ `+0.0106`), TPR@1% `0.7665` (Δ `+0.1318`), features=near_limit_short_share, fragile_share, spare_x_fragile

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
