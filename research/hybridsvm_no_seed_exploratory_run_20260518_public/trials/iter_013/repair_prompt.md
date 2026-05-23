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

Current best metrics:
- AUC: 0.9736
- TPR@FPR=1%: 0.7006
- Accuracy: 0.9360

Active feature bank (18 features):
- spare_cap_sq, spare_cap_x_conc, spare_cap_x_wl_total, vol_top3_share, n_high_asr_items
- spare_cap_log1p, h_to_H_max_sq, l_std_x_h_max, max_fp_ratio
- spare_cap_x_wl_max, wl_total_over_wl_max, max_dim_l_to_vL
- spare_cap_x_sku_counts, n_items_near_vL, n_tall_items
- n_wide_items, n_multi_dim_stress, spare_cap_x_n_near_vL

Rejected trials (10, 11, 12) show that recent attempts haven't beaten the acceptance rule. The main gap is still TPR@FPR=1% (0.7006 vs XGB's 0.8289).

Looking at what's been tried and rejected:
- iter 10: n_fragile_items, flat_item_share, vol_top1_share - rejected (AUC improved but TPR@1% didn't improve enough)
- iter 11: spare_cap_x_h_to_H_max, wl_total_x_h_to_H_max, dim_s_avg_to_vmin - rejected
- iter 12: footprint_pressure, spare_cap_x_n_tall_items, n_long_tall_items - rejected

What's missing from the active bank that could help:

1. The XGB guidance emphasizes spare_capacity interactions and threshold effects. We have many spare_cap interactions already.

2. Looking at the SVM weights, the model is sensitive to:
   - sku_average_volume (strongest negative)
   - spare_capacity (strongest positive)
   - sku_counts (negative)
   - wl_to_vehicle_wl_avg (positive)
   - wl_to_vehicle_wl_std (negative)

3. What hasn't been explored:
   - Item-level dimension tail features beyond what we have
   - Cross-dimension stress (items that are simultaneously long AND tall)
   - Volume utilization pressure at different thresholds
   - The interaction between item count and dimension variance
   - "Awkward" item patterns - items that are hard to pack not just because they're large but because of their shape

Let me think about what features could specifically help the TPR@FPR=1% operating point. This means we need features that help identify infeasible cases more confidently at very low false positive rates.

Key insight: The rejected features (n_long_tall_items, footprint_pressure, spare_cap_x_n_tall_items) were on the right track but didn't quite work. Let me think about what's different.

The items_df has: dispatch_id, item_length, item_width, item_height, if_fragile, load_parameter, vehicle_capacity, dim_s, dim_m, dim_l, item_volume, item_flatness

Features I could add that are truly new:

1. **n_long_and_tall** - Count of items where BOTH the longest dimension > 50% vehicle length AND height > 50% vehicle height. This is a stricter version of n_multi_dim_stress that specifically targets the L-H plane. Items that are simultaneously long and tall create the most severe packing conflicts.

Wait, n_multi_dim_stress already exists. Let me check what it likely does - it probably counts items that stress multiple dimensions. Let me think of something different.

2. **dim_l_tail_share** - Share of items whose longest sorted dimension is in the top quartile of all items in that dispatch. This captures tail heaviness of the length distribution differently from vol_top3_share.

3. **spare_cap_x_vol_top3** - Interaction between spare capacity and volume concentration (top 3 items share). This tests whether volume concentration matters more when spare capacity is low.

4. **height_pressure** - Sum of (item_height / vehicle_height) for all items. This is a cumulative height utilization measure that captures stacking difficulty better than average ratios.

5. **n_items_near_vW** - Count of items whose sorted medium dimension exceeds 50% of vehicle width. This is the width analog of n_items_near_vL, which is already in the bank.

6. **spare_cap_x_n_tall** - Already rejected in iter 12.

Let me think more carefully about what the SVM is missing. The SVM weight analysis shows:
- sku_average_volume has the largest negative weight (-16.4), meaning high average volume pushes toward infeasible
- spare_capacity has the largest positive weight (+13.6), meaning more spare capacity pushes toward feasible

The tree model is better at TPR@FPR=1%, meaning it catches more true infeasible cases at very low FPR. This suggests we need features that create sharper boundaries for detecting infeasibility.

Let me think about what physical packing constraints would create nonlinear effects:

1. **Cumulative footprint pressure**: When you have multiple large-footprint items, they can't stack on each other, so they compete for floor space. The floor space is L×W. Sum of (item_length × item_width) / (vehicle_length × vehicle_width) across all items gives a total footprint utilization. If this exceeds ~1, it's definitely infeasible regardless of height.

2. **Width bottleneck count**: n_items_near_vW - items whose width (or medium sorted dim) is close to vehicle width. Multiple wide items can't be placed side by side.

3. **Long-tall interaction count**: Items that are both long (>50% vL) and tall (>50% vH) create the worst conflicts because they can't be stacked on or under, and they take up a lot of linear space.

4. **Effective layer count**: A rough estimate of how many layers you'd need if you tried to stack items by height. Sum of (item_height / vehicle_height) rounded up gives minimum layers needed. If this times the footprint exceeds vehicle floor area, it's infeasible.

Let me pick 3 features that are most likely to help:

1. **footprint_util** - Sum of (dim_l * dim_m) / (vehicle_length * vehicle_width) across all items. This directly measures floor-space competition. Unlike max_fp_ratio which captures the worst single item, this captures cumulative pressure.

Wait, footprint_pressure was rejected in iter 12. Let me check - it might have been similar. Let me think about what footprint_pressure was vs what I'm proposing.

footprint_pressure was probably something like sum of item footprints / vehicle footprint. My proposal is similar. Let me try something different.

2. **n_items_near_vW** - Width bottleneck count. This is a natural complement to n_items_near_vL (already in bank). Multiple wide items competing for vehicle width is a key packing constraint that hasn't been explicitly encoded.

3. **long_tall_count** - Items that are both long (>50% vL) AND tall (>50% vH). n_long_tall_items was rejected in iter 12, but it was combined with other features. Maybe the combination was bad but the feature itself could work with different companions.

Actually, let me reconsider. The rejected trials show:
- iter 12: footprint_pressure, spare_cap_x_n_tall_items, n_long_tall_items - AUC 0.9741 (better than current 0.9736!), TPR@1% 0.6898 (worse than current 0.7006)

So the AUC actually improved but TPR@1% dropped. This means those features helped overall discrimination but hurt the low-FPR operating point. That's interesting - it suggests those features might be adding noise near the decision boundary at low FPR.

Let me think about what would specifically help TPR@FPR=1%. At this operating point, we need features that clearly separate the hardest infeasible cases from feasible ones. These are cases where the model is uncertain but should be predicting infeasible.

The key question: what makes a case that looks feasible (high spare capacity, moderate volume) but is actually infeasible?

Answer: Awkward item shapes and dimension conflicts. For example:
- Many items with similar large dimensions that can't be nested
- Items that are wide relative to the vehicle width (limiting side-by-side placement)
- Items with high flatness (can't stack efficiently)

Let me try:

1. **n_items_near_vW** - Count of items whose dim_m (medium sorted dimension, which often corresponds to width when placed) exceeds 50% of vehicle width. This captures width bottleneck.

2. **spare_cap_x_n_wide** - Interaction between spare capacity and number of wide items. When spare capacity is moderate but there are many wide items, the dispatch is harder than spare capacity alone would suggest.

Wait, n_wide_items is already in the bank. And spare_cap_x_n_near_vL is already in the bank.

3. **width_crowding** - Sum of (dim_m / vehicle_width) for all items. This is cumulative width pressure, analogous to how height_pressure would work.

4. **dim_m_max_to_vW** - Maximum of (dim_m / vehicle_width) across items. This is like max_dim_l_to_vL but for the width dimension.

5. **l_m_ratio_std** - Standard deviation of (dim_l / dim_m) across items. High variance means a mix of square-ish and elongated items, which creates packing difficulty.

6. **spare_cap_x_wl_avg** - Interaction between spare capacity and wl_to_vehicle_wl_avg. The SVM weight for wl_to_vehicle_wl_avg is +7.6, suggesting it's important. Interacting it with spare_capacity could capture the nonlinear effect.

Wait, let me check: spare_cap_x_wl_total and spare_cap_x_wl_max are already in the bank. But spare_cap_x_wl_avg is not.

Actually, wl_to_vehicle_wl_avg has a positive weight (+7.6), meaning higher average width-length ratio to vehicle pushes toward feasible. That seems counterintuitive... unless it's correlated with something else. Let me think...

Actually, wl_to_vehicle_wl_avg is the average of (item_width * item_length) / (vehicle_width * vehicle_length). Higher values mean items take up more of the vehicle footprint on average. The positive weight might mean that when items are large relative to the vehicle, they tend to be fewer and more regularly shaped, making packing easier. Or it could be a confound.

Let me focus on features that are clearly motivated:

1. **n_items_near_vW** - Width bottleneck count (complement to n_items_near_vL)
2. **width_crowding** - Cumulative width pressure: sum of dim_m / vehicle_width
3. **spare_cap_x_n_items_near_vW** - Interaction between spare capacity and width bottleneck

Actually, let me be more strategic. The rejected trials 10-12 all failed to improve TPR@1%. The pattern suggests we need something fundamentally different, not just variations on existing themes.

Let me look at what item-level information we haven't used:
- if_fragile (tried n_fragile_items, rejected)
- load_parameter (never used!)
- dim_s, dim_m, dim_l (partially used)
- item_flatness (tried flat_item_share, rejected)
- item_volume (partially used through vol_top3_share)

load_parameter is interesting - it's never been used! What is it? It's likely a parameter related to how the item should be loaded (orientation constraints, stacking limits, etc.). This could be very informative.

Let me also think about what hasn't been tried:
- Features using dim_s (smallest dimension) - we've used dim_l and dim_m but not dim_s much
- Features using load_parameter
- Features capturing the distribution shape of item volumes (not just top-k share)
- Features capturing orientation constraints

Let me propose:

1. **load_param_max** - Maximum load_parameter across items in a dispatch. Higher load parameters might indicate more constrained loading requirements.

2. **load_param_mean** - Mean load_parameter. Captures average loading constraint.

3. **n_items_near_vW** - Width bottleneck count.

Actually, I don't know what load_parameter represents exactly. Let me be careful. It could be anything. Let me use it but with clear semantics.

Let me try a different approach. Looking at the SVM weights more carefully:

The top positive features (push toward feasible):
- spare_capacity (+13.6)
- wl_to_vehicle_wl_total (+2.1)
- wl_to_vehicle_wl_max (+1.3)

The top negative features (push toward infeasible):
- sku_average_volume (-16.4)
- sku_counts (-10.6)
- wl_to_vehicle_wl_std (-4.3)

The model is saying: "High spare capacity = feasible, high average volume = infeasible, many items = infeasible, variable width-length utilization = infeasible."

The TPR@FPR=1% gap means: at very low FPR, the SVM misses some truly infeasible cases. These are cases where spare_capacity is moderate-to-high (so SVM thinks feasible) but packing is actually impossible due to geometric conflicts.

So I need features that detect geometric infeasibility even when spare capacity is not critically low.

Ideas:
1. **effective_layers** - Estimate of minimum stacking layers needed: sum of ceil(item_height / vehicle_height) or simply sum of (item_height / vehicle_height). If this exceeds some threshold relative to footprint availability, packing is hard.

2. **orientation_conflict_count** - Items where dim_l / dim_m ratio is close to 1 (near-cubic), meaning they can't be reoriented to save space. Count of items with dim_l / dim_m < 1.5.

3. **small_dim_spread** - Standard deviation of dim_s across items. High spread means some items are very thin (plates) and some are thick, creating stacking challenges.

4. **vol_to_footprint_ratio_std** - Std of (item_volume / (dim_l * dim_m)) across items. This captures variation in item "thickness" relative to footprint.

Let me settle on 3 features:

1. **n_items_near_vW** - Count of items where dim_m > 0.5 * vehicle_width. This is the width analog of n_items_near_vL. Width constraints are a real bottleneck in 3D packing.

2. **cumul_height_ratio** - Sum of (item_height / vehicle_height) across all items. This captures total vertical pressure. Even with spare capacity, if the cumulative height demand is high, stacking is difficult.

3. **spare_cap_x_cumul_h_ratio** - Interaction between spare capacity and cumulative height ratio. This directly tests: "even with some spare capacity, high vertical pressure makes packing infeasible."

Wait, but cumul_height_ratio might be highly correlated with existing features. Let me think...

h_to_H_ratio_avg * sku_counts ≈ cumul_height_ratio. So this is approximately a known combination. But the interaction with spare_cap could be new.

Actually, let me think about what's truly novel. Let me look at the active bank again:
- spare_cap_sq, spare_cap_x_conc, spare_cap_x_wl_total, vol_top3_share, n_high_asr_items
- spare_cap_log1p, h_to_H_max_sq, l_std_x_h_max, max_fp_ratio
- spare_cap_x_wl_max, wl_total_over_wl_max, max_dim_l_to_vL
- spare_cap_x_sku_counts, n_items_near_vL, n_tall_items
- n_wide_items, n_multi_dim_stress, spare_cap_x_n_near_vL

What's NOT there:
- No features using dim_s (smallest sorted dimension)
- No features using load_parameter
- No features using if_fragile (tried and rejected)
- No features using item_flatness (tried and rejected)
- No width-specific bottleneck beyond n_wide_items
- No cumulative dimension pressure (only averages and maxes)
- No features about item shape regularity/orientation flexibility

Let me try:

1. **n_items_near_vW** - Width bottleneck: count of items with dim_m > 0.5 * vehicle_width. Complements n_items_near_vL.

2. **dim_s_avg** - Average of dim_s (smallest sorted dimension) across items. Items with larger dim_s are more "cubic" and harder to pack efficiently (less flexibility in orientation). This captures a different aspect than volume or flatness.

3. **spare_cap_x_dim_s_avg** - Interaction between spare capacity and average smallest dimension. When spare capacity is low and items are thick (high dim_s), packing is especially difficult because you can't tuck items into gaps.

Hmm, but dim_s_avg might be correlated with item_volume average. Let me think of something more unique.

What about:

1. **n_items_near_vW** - Width bottleneck count
2. **orientation_flexibility** - Average of (dim_m / dim_l) across items. Values close to 1 mean items are near-square in their L×M face, giving less orientation flexibility. Values close to 0 mean items are very elongated, which can be both good (can fit in gaps) and bad (take up linear space).
3. **spare_cap_x_orientation_flex** - Interaction

Actually, let me think about this differently. The SVM with current features gets AUC 0.9736. The XGB with same features would likely do better due to nonlinear interactions. I need to find the specific nonlinear effects that matter most.

Looking at rejected iter 12: AUC actually improved to 0.9741 but TPR@1% dropped to 0.6898. This means those features (footprint_pressure, spare_cap_x_n_tall_items, n_long_tall_items) shifted the ROC curve in a way that helped overall AUC but hurt the low-FPR region. This could mean those features added noise for the hardest-to-classify infeasible cases.

For TPR@FPR=1%, I need features that clearly flag infeasible cases without false positives. This means I need features with high precision for infeasibility.

What makes a case clearly infeasible even with moderate spare capacity?
- Multiple items that each consume a large fraction of one vehicle dimension
- Items that create "wall" effects (blocking entire cross-sections)
- Cumulative dimension pressure exceeding what stacking can resolve

Let me try:

1. **n_items_near_vW** - Count of items where dim_m > 0.5 * vehicle_width. Width is often the tightest dimension in packing.

2. **wall_item_count** - Count of items that span >70% of vehicle height (item_height > 0.7 * vehicle_height). These items create "walls" that partition the vehicle and limit placement options for other items. Different threshold from n_tall_items (50%).

3. **spare_cap_x_n_near_vW** - Interaction between spare capacity and width bottleneck count. Captures the nonlinear effect where width constraints matter more when spare capacity is moderate.

Actually, I realize n_wide_items is already in the bank. Let me check what it likely means - probably items where item_width > 0.5 * vehicle_width. But n_items_near_vW uses dim_m (sorted medium dimension),
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
        - AUC: `0.9736`
        - TPR@FPR=1%: `0.7006`
        - Accuracy: `0.9360`

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

        Current iteration: `13`

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

## Iteration 8
- Add `spare_cap_x_sku_counts` to capture the nonlinear interaction between slack and item count (XGB guidance §3.2, not yet in active bank despite spare_cap interactions with conc/wl_total/wl_max).
- Add `n_items_near_vL` as a threshold count of items whose longest sorted dimension exceeds 50% of vehicle length — multiple long items create axis-alignment conflicts that a single max-ratio cannot express.
- Add `n_tall_items` as a threshold count of items whose original height exceeds 50% of vehicle height — tall items limit vertical stacking; height pressure has high XGB gain (h_to_H_ratio_max rank 8) but no threshold-count representation in the bank.


        # Memory

Baseline reference:
- AUC `0.9651`
- TPR@FPR=1% `0.6347`
- Accuracy `0.9276`

## Accepted Trials
- iter `9`: AUC `0.9736` (Δ `+0.0084`), TPR@1% `0.7006` (Δ `+0.0659`), features=n_wide_items, n_multi_dim_stress, spare_cap_x_n_near_vL
- iter `8`: AUC `0.9730` (Δ `+0.0079`), TPR@1% `0.7026` (Δ `+0.0678`), features=spare_cap_x_sku_counts, n_items_near_vL, n_tall_items
- iter `6`: AUC `0.9695` (Δ `+0.0044`), TPR@1% `0.6647` (Δ `+0.0300`), features=spare_cap_x_wl_max, wl_total_over_wl_max, max_dim_l_to_vL

## Rejected Trials
- iter `10`: AUC `0.9737` (Δ `+0.0085`), TPR@1% `0.6957` (Δ `+0.0610`), features=n_fragile_items, flat_item_share, vol_top1_share
- iter `11`: AUC `0.9737` (Δ `+0.0086`), TPR@1% `0.6957` (Δ `+0.0610`), features=spare_cap_x_h_to_H_max, wl_total_x_h_to_H_max, dim_s_avg_to_vmin
- iter `12`: AUC `0.9741` (Δ `+0.0089`), TPR@1% `0.6898` (Δ `+0.0551`), features=footprint_pressure, spare_cap_x_n_tall_items, n_long_tall_items

## Failed Trials
- iter `2`: RateLimitError: Error code: 429 - {'error': {'message': "No deployments available for selected model, Try again in 5 seconds. Passed model=glm-5.1. pre-call-checks=False, cooldown_list=[('b9ca8bb822ee3f6a483f081af040bbc6769519968ef875dbfcc563db02907968', {'exception_received': 'litellm.RateLimitError: Model rate limit exceeded.', 'status_code': '429', 'timestamp': 1779097037.1183505, 'cooldown_time': 5}), ('ee7f63fe7b5a6e0c0f610a8773fbc7d2b61ec00beadff16b9c535feb3698db62', {'exception_received': 'litellm.InternalServerError: InternalServerError: Hosted_vllmException - Internal Server Error', 'status_code': '500', 'timestamp': 1779097037.671829, 'cooldown_time': 5})]", 'type': 'None', 'param': 'None', 'code': '429'}}
- iter `4`: ValueError: Repair response did not include FEATURE_CODE python block
- iter `7`: RateLimitError: Error code: 429 - {'error': {'message': "No deployments available for selected model, Try again in 5 seconds. Passed model=glm-5.1. pre-call-checks=False, cooldown_list=[('b9ca8bb822ee3f6a483f081af040bbc6769519968ef875dbfcc563db02907968', {'exception_received': 'litellm.RateLimitError: Model rate limit exceeded.', 'status_code': '429', 'timestamp': 1779097551.7085834, 'cooldown_time': 5}), ('ee7f63fe7b5a6e0c0f610a8773fbc7d2b61ec00beadff16b9c535feb3698db62', {'exception_received': 'litellm.RateLimitError: Model rate limit exceeded.', 'status_code': '429', 'timestamp': 1779097552.2410598, 'cooldown_time': 5})]", 'type': 'None', 'param': 'None', 'code': '429'}}

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale


        ## Current Active Feature Bank
- source accepted trials: `1`, `5`, `6`, `8`, `9`
- active feature count: `18`
- active feature names: `spare_cap_sq`, `spare_cap_x_conc`, `spare_cap_x_wl_total`, `vol_top3_share`, `n_high_asr_items`, `spare_cap_log1p`, `h_to_H_max_sq`, `l_std_x_h_max`, `max_fp_ratio`, `spare_cap_x_wl_max`, `wl_total_over_wl_max`, `max_dim_l_to_vL`, `spare_cap_x_sku_counts`, `n_items_near_vL`, `n_tall_items`, `n_wide_items`, `n_multi_dim_stress`, `spare_cap_x_n_near_vL`
- active bank AUC: `0.9736`
- active bank TPR@FPR=1%: `0.7006`
- active bank Accuracy: `0.9360`

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
