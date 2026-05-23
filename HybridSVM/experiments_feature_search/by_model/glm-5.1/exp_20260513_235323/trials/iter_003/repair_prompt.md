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
        Let me analyze the current situation:

**Current best metrics (active bank):**
- AUC: 0.9740
- TPR@FPR=1%: 0.7080
- Accuracy: 0.9316

**Active features (9 total):**
1. `dominant_type_share` - from iter 0
2. `p90_long_over_bin_long` - from iter 0
3. `p90_mid_over_bin_mid` - from iter 0
4. `thin_item_share` - from iter 0
5. `max_face_area_load_over_floor` - from iter 0
6. `tight_bin_large_piece_interaction` - from iter 0
7. `spare_x_count` - from iter 2
8. `near_limit_long_share` - from iter 2
9. `vol_cv` - from iter 2

**Rejected features from iter 1:**
- `volume_top3_share`
- `multi_dim_near_limit_share`
- `side_wall_area_load`

**Key insights from SVM weights:**
- `sku_average_volume` is the strongest negative weight (-16.4) → larger average volume pushes toward infeasible
- `spare_capacity` is the strongest positive weight (+13.6) → more spare capacity pushes toward feasible
- `sku_counts` is strongly negative (-10.6) → more items pushes toward infeasible
- `wl_to_vehicle_wl_avg` is positive (+7.6) → higher width-length footprint ratio pushes toward feasible
- `wl_to_vehicle_wl_std` is negative (-4.3) → more variation in footprint ratios pushes toward infeasible
- `l_to_L_ratio_avg` is negative (-3.0) → higher length-to-vehicle-length ratio pushes toward infeasible

**What's already covered:**
- Tail pressure: p90_long_over_bin_long, p90_mid_over_bin_mid
- Threshold counts: near_limit_long_share
- Slack-pressure interaction: spare_x_count, tight_bin_large_piece_interaction
- Distribution shape: vol_cv, dominant_type_share
- Footprint/wall pressure: max_face_area_load_over_floor
- Awkward patterns: thin_item_share

**What's still missing based on tree guidance:**

1. **Height dimension pressure** - We have long and mid dimension pressure (p90), but no explicit height-related pressure features. The SVM shows `h_to_H_ratio` features are present but not among top weights. A thresholded height pressure could help.

2. **Multi-dimensional near-limit** - This was rejected in iter 1, but maybe a different formulation would work. The idea is items that are near the limit in multiple dimensions simultaneously (cubic-like items near bin limits).

3. **Face-area / wall pressure on different faces** - We have max_face_area_load_over_floor, but not side-wall pressure. The rejected `side_wall_area_load` didn't work, but maybe a different formulation.

4. **Heterogeneity in specific dimensions** - We have vol_cv (volume coefficient of variation), but not dimension-specific heterogeneity. Items with very different lengths might cause different packing issues than items with very different volumes.

5. **Extreme piece bottleneck** - The single largest item's dimensions relative to the bin, particularly when it's close to the limit in multiple dimensions.

6. **Repetition vs. heterogeneity in sorted dimensions** - Are items similar in their sorted dimensions (good for stacking) or very different?

Let me think about what would be most impactful:

Looking at the SVM weights more carefully:
- `wl_to_vehicle_wl_total` has weight +2.1 (positive) - total footprint ratio
- `wl_to_vehicle_wl_max` has weight +1.3 (positive) - max footprint ratio
- `wl_to_vehicle_wl_std` has weight -4.3 (negative) - variation in footprint ratios
- `wl_to_vehicle_wl_min` has weight -2.9 (negative) - min footprint ratio

This suggests that when the minimum footprint ratio is high (small items taking up little floor space), it pushes toward infeasible. And when there's high variation in footprint ratios, it's also infeasible. But total footprint ratio being high pushes toward feasible.

This is interesting - it suggests that items with small footprints but significant height might be problematic (they don't use floor space efficiently but still take up volume).

**Feature ideas for iteration 3:**

1. **`height_pressure_share`** - Share of items where `dim_s / vehicle_height > 0.5` (or some threshold). Items that are tall relative to the bin height create vertical pressure.

2. **`long_x_tall_share`** - Share of items that are near-limit in both length AND height dimensions simultaneously. These are the most awkward to pack.

3. **`dim_l_cv`** - Coefficient of variation of the longest sorted dimension across items. Captures whether items have similar or very different lengths, which affects stacking.

4. **`floor_occupancy_gap`** - Difference between total footprint ratio and max footprint ratio, capturing whether items have diverse floor needs. This might capture the interaction between wl_to_vehicle_wl_total (positive) and wl_to_vehicle_wl_max (positive but smaller).

Wait, let me reconsider. The existing features already cover a lot. Let me focus on what's clearly missing:

1. **Height dimension is underrepresented** - We have p90 for long and mid, but nothing specific for height pressure or tall items.

2. **Multi-dimensional bottleneck** - Items near limits in multiple dimensions simultaneously are the hardest to pack. We have `near_limit_long_share` but not a multi-dimensional version.

3. **Dimension-specific heterogeneity** - `vol_cv` captures volume heterogeneity, but not whether items have consistent vs. inconsistent shapes (e.g., all similar boxes vs. mix of long thin and short wide items).

Let me propose:

1. **`tall_item_share`** - Share of items where `dim_l / vehicle_height > 0.6` (items that are tall relative to bin height). This captures vertical pressure that isn't well captured by existing features.

2. **`multi_near_limit_share`** - Share of items near the limit in 2+ dimensions simultaneously. An item is "near limit" in a dimension if its sorted dim exceeds 0.6 of the corresponding vehicle dim. This captures cubic-like items that are hard to pack.

Wait, `multi_dim_near_limit_share` was rejected in iter 1. Let me think about why it might have failed and how to reformulate.

The rejected version might have used different thresholds or a different definition. Let me try a different approach - maybe count items that are near the limit in both the longest AND the shortest dimension, which would capture items that are both long AND tall (or wide).

3. **`dim_l_heterogeneity`** - Coefficient of variation of `dim_l` across items. When items have very different longest dimensions, packing becomes harder because you can't align them efficiently.

Actually, let me reconsider the rejected features:
- `volume_top3_share` - share of volume in top 3 largest items
- `multi_dim_near_limit_share` - share near limit in multiple dimensions
- `side_wall_area_load` - side wall area pressure

These were rejected because they didn't improve enough. The acceptance rule requires AUC improvement > 0.0005, or TPR@1% improvement > 0.005, or Accuracy improvement > 0.0005.

Current best: AUC 0.9740, TPR@1% 0.7080, Accuracy 0.9316

Iter 1 got: AUC 0.9719 (worse), TPR@1% 0.7070 (worse), so it was clearly worse.

Let me think about what new features could push past the current best:

The tree models achieve AUC ~0.983-0.985 and TPR@1% ~0.806-0.813. So there's still a significant gap.

Key areas not yet explored:
1. **Height-specific pressure** - No height-related threshold features
2. **Sorted dimension interactions** - How do dim_s, dim_m, dim_l relate to each other within items?
3. **Packing stress on specific faces** - We have floor load, but what about stacking stress?
4. **Item aspect ratio patterns** - Are items cube-like or flat? This affects packing.

Let me propose 3 new features:

1. **`tall_item_share`** - Share of items where `dim_l / vehicle_height > 0.55`. Captures items that are tall relative to the bin, creating vertical packing pressure. This is different from `near_limit_long_share` which focuses on the long dimension relative to vehicle length.

2. **`cube_like_near_limit_share`** - Share of items where the ratio `dim_s / dim_l > 0.5` AND `dim_l / max(vehicle_length, vehicle_width, vehicle_height) > 0.5`. These are items that are both cube-like (not flat) and near the bin limits, making them the hardest to pack.

3. **`dim_s_max_ratio`** - Maximum of `dim_s / vehicle_height` across items. The thickest item relative to bin height creates a hard constraint on vertical stacking.

Wait, let me reconsider. I want features that are clearly different from what exists and that target the tree advantage.

Looking at the SVM weights again:
- The model strongly uses `spare_capacity` (+13.6) and `sku_average_volume` (-16.4)
- `sku_counts` is -10.6
- `wl_to_vehicle_wl_avg` is +7.6

The interaction `spare_x_count` (spare_capacity * sku_counts) is already in the bank. But what about:

1. **Low spare capacity with large average volume** - This is a dangerous combination. When spare capacity is low AND average volume is high, packing is very constrained.

2. **Height bottleneck** - The height dimension might be the binding constraint in some cases. If many items are tall relative to bin height, even if length/width are fine, packing fails.

3. **Item flatness concentration** - We have `thin_item_share` but not a measure of how concentrated the flatness is. Are all items similarly flat, or is there a mix?

Let me go with these 3:

1. **`spare_x_avgvol`** - Interaction of low spare capacity with high average volume: `(1 - spare_capacity) * sku_average_volume`. When spare capacity is low (high 1-spare) AND average volume is high, this feature is large, indicating high packing stress. This directly captures the nonlinear interaction between the two strongest SVM features.

Wait, but `spare_x_count` already exists which is `spare_capacity * sku_counts`. Adding another spare_capacity interaction might be redundant.

Let me think differently. What unique information is missing?

Looking at the tree guidance:
1. tail pressure - covered by p90 features
2. slack-pressure interactions - covered by spare_x_count and tight_bin_large_piece_interaction
3. local awkward-pattern counts - covered by near_limit_long_share and thin_item_share
4. heterogeneity vs repetition - covered by dominant_type_share and vol_cv
5. footprint and wall-pressure proxies - covered by max_face_area_load_over_floor

What's NOT covered:
- **Height-specific pressure** - No features targeting the height dimension specifically
- **Multi-dimensional bottleneck items** - Items that are hard to place because they're large in multiple dimensions
- **Stacking difficulty** - Items that can't be stacked on top of each other

Let me propose:

1. **`height_near_limit_share`** - Share of items where `dim_l > 0.6 * vehicle_height`. Items whose longest dimension exceeds 60% of bin height are likely to be height-constrained.

Actually, wait. The sorted dimensions are `dim_s <= dim_m <= dim_l`. The longest dimension `dim_l` could align with any vehicle axis. But in practice, for packing, the item's longest dimension would typically be placed along the vehicle's longest dimension. However, if `dim_l` is close to `vehicle_height`, it means the item is very tall.

Let me reconsider. The vehicle has length, width, height. Items have sorted dim_s, dim_m, dim_l. In optimal packing, dim_l would go along the vehicle's longest dimension (length), dim_m along width, dim_s along height. But this isn't always possible.

A key insight: if an item's `dim_m` (middle dimension) is large relative to `vehicle_width`, it creates a width constraint. If `dim_s` (smallest dimension) is large relative to `vehicle_height`, it creates a height constraint.

So:
1. **`width_pressure_share`** - Share of items where `dim_m / vehicle_width > 0.5`. Items whose middle dimension takes up a lot of the vehicle width.
2. **`height_pressure_share`** - Share of items where `dim_s / vehicle_height > 0.5`. Items whose smallest dimension is still significant relative to vehicle height.

But wait, `p90_mid_over_bin_mid` already captures the 90th percentile of `dim_m / vehicle_width` (approximately). So a thresholded share might add different information.

Let me look at what `p90_mid_over_bin_mid` actually is. From the iter 0 features, it's the p90 of `dim_m / vehicle_width`. This captures the tail of the middle dimension distribution. A thresholded share would capture how many items are in that tail, which is different information.

OK, let me finalize my proposal. I'll add 3 features targeting underexplored areas:

1. **`height_pressure_share`** - Share of items where `dim_s / vehicle_height > 0.5`. Captures vertical stacking pressure - items that are thick relative to bin height limit how many can be stacked.

2. **`long_and_wide_share`** - Share of items where `dim_l / vehicle_length > 0.5` AND `dim_m / vehicle_width > 0.4`. Items that are large in both length and width dimensions simultaneously, creating 2D floor pressure.

3. **`dim_l_skew`** - Skewness of the `dim_l` distribution across items. Positive skew means a few very long items among shorter ones, which creates packing challenges. This captures distribution shape beyond CV.

Wait, skewness might be noisy. Let me think of something more robust.

How about:
3. **`spare_vol_interaction`** - `(1 - spare_capacity) * (1 - dominant_type_share)`. When spare capacity is low AND items are diverse (not dominated by one type), packing is harder because you can't use repetition to fill space efficiently.

Hmm, but this uses agg_df features which are already in the model. The interaction might still help the linear SVM though.

Let me reconsider. The key insight from the tree guidance is that trees exploit **thresholds** and **interactions**. The current active bank has some of these but there are still gaps.

Let me focus on the most promising unexplored direction: **height dimension pressure** and **multi-dimensional constraint items**.

Final proposal (3 features):

1. **`height_pressure_share`** - Share of items where `dim_s / vehicle_height > 0.5`. Captures items that are thick relative to bin height, limiting vertical stacking.

2. **`long_wide_item_share`** - Share of items where `dim_l / vehicle_length > 0.5` AND `dim_m / vehicle_width > 0.4`. Items large in both horizontal dimensions create floor occupancy pressure.

3. **`spare_x_volcv`** - Interaction: `(1 - spare_capacity) * vol_cv`. When space is tight AND items have diverse volumes, packing becomes much harder. This captures the interaction between the two most important regimes.

Wait, `vol_cv` is already in the active bank. So `spare_x_volcv` would be an interaction between an existing aggregate feature and an active bank feature. That should be fine since it's a new feature.

Actually, let me reconsider whether this is a good idea. The SVM can already learn some interaction between spare_capacity and vol_cv through their individual weights, but an explicit product feature would allow it to capture the nonlinear interaction directly.

Let me go with a different set that's more clearly adding new information:

1. **`height_pressure_share`** - Share of items where `dim_s / vehicle_height > 0.5`. 
2. **`long_wide_item_share`** - Share of items where `dim_l / vehicle_length > 0.5` AND `dim_m / vehicle_width > 0.4`.
3. **`max_dim_s_over_height`** - Maximum `dim_s / vehicle_height` across items. The single thickest item relative to bin height creates a hard constraint.

Actually, `max_dim_s_over_height` might be too correlated with `height_pressure_share`. Let me replace it with something else.

3. **`spare_x_volcv`** - `(1 - spare_capacity) * vol_cv`. Interaction between tight space and volume diversity.

Hmm, but I'm not sure this will beat the current best. Let me think about what the trees are really exploiting that we're missing.

The trees achieve TPR@1% of ~0.81 while we're at 0.708. That's a big gap in the low-FPR regime. This means trees are better at identifying the truly hard cases (actual infeasible packs that look borderline feasible).

For low-FPR recall, we need features that strongly signal infeasibility even when other features are ambiguous. These would be features that capture "hidden" packing constraints.

What creates hidden constraints?
1. **Items that must be placed in specific orientations** - e.g., tall items that can only go one way
2. **Items that block multiple dimensions** - e.g., items large in both width and height
3. **Fragile items that can't be stacked on** - load_parameter constraints
4. **Items with incompatible dimensions** - can't be placed next to each other efficiently

The `load_parameter` and `if_fragile` columns in items_df haven't been used at all! This could be a significant missing signal.

Let me add:
1. **`fragile_share`** - Share of items that are fragile. Fragile items can't have other items stacked on top, reducing effective packing space.
2. **`height_pressure_share`** - Share of items where `dim_s / vehicle_height > 0.5`.
3. **`long_wide_item_share`** - Share of items where `dim_l / vehicle_length > 0.5` AND `dim_m / vehicle_width > 0.4`.

Wait, but I should check if `if_fragile` is actually informative. The problem says the positive rate is 0.8083, meaning most dispatches are feasible. Fragile items might be rare and not very predictive.

Let me think about this more carefully. The items_df has `if_fragile` and `load_parameter`. These are constraints that trees could exploit through splits but that our current features completely ignore.

Actually, let me check: the
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
        - AUC: `0.9740`
        - TPR@FPR=1%: `0.7080`
        - Accuracy: `0.9316`

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

        Current iteration: `3`

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
- iter `2`: AUC `0.9740` (Δ `+0.0089`), TPR@1% `0.7080` (Δ `+0.0733`), features=spare_x_count, near_limit_long_share, vol_cv
- iter `0`: AUC `0.9721` (Δ `+0.0070`), TPR@1% `0.7139` (Δ `+0.0792`), features=dominant_type_share, p90_long_over_bin_long, p90_mid_over_bin_mid, thin_item_share, max_face_area_load_over_floor, tight_bin_large_piece_interaction

## Rejected Trials
- iter `1`: AUC `0.9719` (Δ `+0.0067`), TPR@1% `0.7070` (Δ `+0.0723`), features=volume_top3_share, multi_dim_near_limit_share, side_wall_area_load

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale


        ## Current Active Feature Bank
- source accepted trials: `0`, `2`
- active feature count: `9`
- active feature names: `dominant_type_share`, `p90_long_over_bin_long`, `p90_mid_over_bin_mid`, `thin_item_share`, `max_face_area_load_over_floor`, `tight_bin_large_piece_interaction`, `spare_x_count`, `near_limit_long_share`, `vol_cv`
- active bank AUC: `0.9740`
- active bank TPR@FPR=1%: `0.7080`
- active bank Accuracy: `0.9316`

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
