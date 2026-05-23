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
        - error: RateLimitError: Error code: 429 - {'error': {'message': "No deployments available for selected model, Try again in 5 seconds. Passed model=glm-5.1. pre-call-checks=False, cooldown_list=[('b9ca8bb822ee3f6a483f081af040bbc6769519968ef875dbfcc563db02907968', {'exception_received': 'litellm.RateLimitError: Model rate limit exceeded.', 'status_code': '429', 'timestamp': 1779097823.8184807, 'cooldown_time': 5}), ('ee7f63fe7b5a6e0c0f610a8773fbc7d2b61ec00beadff16b9c535feb3698db62', {'exception_received': 'litellm.InternalServerError: InternalServerError: Hosted_vllmException - Internal Server Error', 'status_code': '500', 'timestamp': 1779097823.8030097, 'cooldown_time': 5})]", 'type': 'None', 'param': 'None', 'code': '429'}}

        Original response:
        ```text
        Let me analyze the current situation carefully.

Current active feature bank (15 features):
- spare_cap_sq, spare_cap_x_conc, spare_cap_x_wl_total, vol_top3_share, n_high_asr_items
- spare_cap_log1p, h_to_H_max_sq, l_std_x_h_max, max_fp_ratio
- spare_cap_x_wl_max, wl_total_over_wl_max, max_dim_l_to_vL
- n_fragile_items, avg_item_flatness, spare_cap_x_sku_counts

Current best metrics:
- AUC: 0.9697
- TPR@FPR=1%: 0.6829
- Accuracy: 0.9296

To beat, I need:
- AUC improvement > 0.0005 (so AUC > 0.9702), OR
- TPR@FPR=1% improvement > 0.005 (so TPR@1% > 0.6879), OR
- Accuracy improvement > 0.0005 (so Accuracy > 0.9301)

The XGB guidance emphasizes:
1. spare_capacity nonlinear transforms - already have spare_cap_sq, spare_cap_log1p
2. spare_capacity interactions - already have spare_cap_x_conc, spare_cap_x_wl_total, spare_cap_x_wl_max, spare_cap_x_sku_counts
3. Utilization-ratio interactions - already have wl_total_over_wl_max
4. Dimensional-ratio dispersion features - already have l_std_x_h_max, h_to_H_max_sq
5. SKU-width distribution features - NOT yet explored much
6. Concentration-interaction cluster - already have spare_cap_x_conc

What's missing from the active bank that could help:

Looking at the XGB guidance more carefully:
- Section 3.3: `wl_max_residual` (wl_to_vehicle_wl_max - wl_to_vehicle_wl_total) - was in rejected trial iter 3
- Section 3.4: `l_to_L_ratio_std_is_zero` - binary for homogeneous loads
- Section 3.5: `sku_width_range` (sku_max_width - sku_min_width)
- Section 3.6: `sku_concentration_x_sku_counts`, `sku_concentration_x_sku_average_volume`

From the item-level data, I can compute:
- `dispatch_id`, `item_length`, `item_width`, `item_height`, `if_fragile`, `load_parameter`, `vehicle_capacity`, `dim_s`, `dim_m`, `dim_l`, `item_volume`, `item_flatness`

The item-level table has sorted dimensions (dim_s <= dim_m <= dim_l) which I haven't fully exploited.

Let me think about what features might help improve TPR@FPR=1% specifically. The gap between SVM and XGB is largest at the low-FPR operating point. This means I need features that help correctly identify infeasible cases with high confidence.

Key ideas for new features:

1. **Height pressure / wall pressure**: How many items have dim_m (medium dimension) close to vehicle height? This creates stacking difficulty. Items with dim_m close to vehicle height are hard to stack.

2. **Large-piece bottleneck count**: Count of items where dim_l > some threshold relative to vehicle length. Already have max_dim_l_to_vL but not a count.

3. **Tail ratio of item volumes**: The ratio of the largest item volume to total volume - this captures whether one dominant item creates a bottleneck. Already have vol_top3_share.

4. **Item dimension heterogeneity**: How varied are the sorted dimensions across items? If dim_s values are very spread, packing is harder.

5. **Footprint coverage pressure**: Sum of (dim_l * dim_m) / (vehicle_length * vehicle_width) - how much of the floor area items need.

6. **Height-layer utilization**: How many distinct height layers are needed?

Let me think about what's most likely to help based on the tree guidance:

The tree model uses `spare_capacity` as dominant splitter. The active bank already has many spare_capacity interactions. Let me focus on other angles:

1. **Width-pressure bottleneck**: The XGB guidance mentions `sku_max_width`, `sku_min_width`, `w_to_W_ratio_avg` as important. The active bank doesn't have width-specific features. A feature like "count of items with dim_m close to vehicle width" could capture width bottleneck.

2. **Height-wall pressure**: Count of items where dim_l (longest dimension) is close to vehicle height - these items must go against the wall and limit stacking.

3. **Load parameter signals**: The items_df has `load_parameter` which hasn't been used at all in the active bank.

4. **Sorted-dimension tail features**: Using dim_s, dim_m, dim_l from items_df to compute per-dispatch statistics like the max dim_l / vehicle_length ratio (already have max_dim_l_to_vL), but what about the second-largest? Or the count of items with dim_l > 0.5 * vehicle_length?

Let me propose 3 focused features:

1. **n_wide_items**: Count of items where dim_m > 0.5 * vehicle_width. These items create width bottlenecks.
   - Actually, I need vehicle dimensions. I can get them from agg_df which has vehicle_width, vehicle_height, vehicle_length.

Wait, but the function receives both agg_df and items_df. I can merge vehicle dimensions into items_df from agg_df.

2. **height_wall_pressure**: Sum of (dim_l / vehicle_height) for items where dim_l > 0.7 * vehicle_height. These items press against the height wall and limit vertical stacking.

3. **load_param_max**: Maximum load_parameter per dispatch. Higher load parameters may indicate harder-to-pack items.

Actually, let me reconsider. The `load_parameter` might be very informative and hasn't been used at all.

Let me think about what's most physically meaningful for 3D bin packing feasibility:

1. **n_tall_items**: Count of items where dim_l > 0.6 * vehicle_height. Items that are tall relative to the vehicle height create vertical bottlenecks.

2. **width_pressure_sum**: Sum of (dim_m / vehicle_width) for items with dim_m > 0.5 * vehicle_width. Captures cumulative width wall pressure.

3. **load_param_avg**: Average load_parameter per dispatch. Different load parameters may indicate orientation constraints that make packing harder.

Actually, looking more carefully at the problem structure:

- `dim_s <= dim_m <= dim_l` are sorted dimensions
- For 3D packing, the critical question is: can items be arranged so their dimensions align with vehicle dimensions?
- Items with dim_l close to vehicle_length must span the full length
- Items with dim_m close to vehicle_width are hard to place side-by-side
- Items with dim_l close to vehicle_height must be oriented vertically

Let me focus on features that capture bottleneck/pressure patterns that the linear SVM can't see from existing aggregates:

1. **n_length_dominant_items**: Count of items where dim_l > 0.7 * vehicle_length. These items force a specific orientation and limit placement flexibility.

2. **width_tail_pressure**: Sum of max(0, dim_m/vehicle_width - 0.5) for all items. Captures how much items press against the width constraint beyond a threshold.

3. **height_tail_pressure**: Sum of max(0, dim_l/vehicle_height - 0.5) for all items. Captures height wall pressure.

Wait, but I should be careful. Let me look at what's already in the active bank more carefully:

- max_dim_l_to_vL: This is max(dim_l) / vehicle_length - captures the single worst case
- n_high_asr_items: Count of high aspect ratio items
- vol_top3_share: Volume share of top 3 items
- max_fp_ratio: Maximum footprint ratio

So I already have some extreme-item features. What I don't have:

1. Cumulative pressure beyond thresholds (not just max, but sum of excess)
2. Features using `load_parameter`
3. Width-specific tail features
4. Features about the second-largest dimension alignment

Let me propose:

1. **width_tail_sum**: Sum of max(0, dim_m - 0.5 * vehicle_width) per dispatch. This captures cumulative width pressure beyond the 50% threshold. Items with medium dimension > half vehicle width can't be placed side by side easily.

2. **height_tail_sum**: Sum of max(0, dim_l - 0.6 * vehicle_height) per dispatch. Cumulative height pressure beyond 60% threshold.

3. **load_param_mean**: Average load_parameter per dispatch. This captures orientation flexibility constraints.

Actually wait - I need to get vehicle dimensions into the items_df. Let me check: agg_df has vehicle_length, vehicle_width, vehicle_height. I can merge these.

Let me write the code:

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Merge vehicle dimensions into items_df
    veh = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    items = items_df.merge(veh, on='dispatch_id', how='left')

    # Feature 1: width tail pressure - cumulative excess of medium dim over half vehicle width
    items['width_excess'] = (items['dim_m'] - 0.5 * items['vehicle_width']).clip(lower=0)
    width_tail = items.groupby('dispatch_id')['width_excess'].sum().reset_index()
    width_tail.columns = ['dispatch_id', 'width_tail_sum']

    # Feature 2: height tail pressure - cumulative excess of longest dim over 60% vehicle height
    items['height_excess'] = (items['dim_l'] - 0.6 * items['vehicle_height']).clip(lower=0)
    height_tail = items.groupby('dispatch_id')['height_excess'].sum().reset_index()
    height_tail.columns = ['dispatch_id', 'height_tail_sum']

    # Feature 3: average load parameter per dispatch
    load_avg = items.groupby('dispatch_id')['load_parameter'].mean().reset_index()
    load_avg.columns = ['dispatch_id', 'load_param_mean']

    # Merge all features
    result = agg_df[['dispatch_id']].copy()
    result = result.merge(width_tail, on='dispatch_id', how='left')
    result = result.merge(height_tail, on='dispatch_id', how='left')
    result = result.merge(load_avg, on='dispatch_id', how='left')

    return result
```

Hmm, but I should think more carefully about what will actually help. The main gap is TPR@FPR=1%. This means we need to better identify infeasible cases at high confidence.

Infeasible cases typically have:
- Low spare capacity (already captured well)
- Large items that don't fit well together
- Dimensional mismatches with the vehicle
- Orientation constraints

The `load_parameter` is interesting - it likely indicates how items can be oriented. Higher values might mean more restricted orientations.

Let me also think about whether I should try interaction features. The active bank already has many spare_capacity interactions. What about interactions NOT involving spare_capacity?

For example:
- `sku_concentration_x_sku_counts`: Already mentioned in XGB guidance section 3.6, not in active bank
- `wl_to_vehicle_wl_total_x_sku_counts`: Not in active bank

But the guidance says to prioritize §3.1 and §3.2 first (spare_capacity transforms and interactions), which are already well-covered in the active bank.

Let me focus on the item-level features that capture something genuinely new:

1. **width_tail_sum**: Cumulative width pressure beyond threshold
2. **height_tail_sum**: Cumulative height pressure beyond threshold  
3. **load_param_mean**: Average load parameter

Actually, let me reconsider. I want features that are most likely to improve TPR@FPR=1%. The key insight from the tree model is that it captures nonlinear thresholds. 

Let me think about what threshold effects might exist:

- If more than N items have dim_m > 0.5 * vehicle_width, packing becomes much harder (step function)
- If the total "width excess" exceeds some threshold, infeasibility jumps

These are exactly the kinds of features that would help. But I should also consider:

- **n_wide_items**: Simple count of items with dim_m > 0.5 * vehicle_width. This is a threshold count, which the tree guidance specifically recommends.
- **dim_l_p90_ratio**: The 90th percentile of dim_l / vehicle_length. Captures tail of length distribution without being as extreme as the max.

Let me revise my proposal:

1. **n_wide_items**: Count of items where dim_m > 0.5 * vehicle_width. Threshold count for width bottleneck.
2. **height_tail_sum**: Sum of max(0, dim_l - 0.6 * vehicle_height). Cumulative height pressure.
3. **load_param_mean**: Average load_parameter per dispatch.

Actually, I realize I should be more careful about what `load_parameter` means. Looking at the schema, it's a per-item field. It might represent weight or some other constraint. Without knowing exactly, using the mean seems reasonable as a first attempt.

Let me also consider: the XGB guidance mentions `sku_concentration_x_sku_counts` and `sku_concentration_x_sku_average_volume` as hypotheses. These use base40 features only and might be worth trying. But they're interactions of existing aggregates, not item-level features.

Actually, I can compute them from agg_df directly:
- `sku_concentration_x_sku_counts` = agg_df['sku_concentration'] * agg_df['sku_counts']
- `sku_concentration_x_sku_average_volume` = agg_df['sku_concentration'] * agg_df['sku_average_volume']

These are simple and interpretable. Let me include one of these.

Final proposal - 3 features:

1. **n_wide_items**: Count of items where dim_m > 0.5 * vehicle_width. Width bottleneck threshold count.
2. **height_tail_sum**: Sum of max(0, dim_l - 0.6 * vehicle_height). Cumulative height wall pressure.
3. **conc_x_avg_vol**: sku_concentration * sku_average_volume from agg_df. Captures whether concentrated loads are also large-volume.

Wait, but `conc_x_avg_vol` - is this similar to `spare_cap_x_conc` or `vol_top3_share`? Let me check:
- `spare_cap_x_conc` = spare_capacity * sku_concentration
- `vol_top3_share` = volume share of top 3 items

`conc_x_avg_vol` = sku_concentration * sku_average_volume captures something different: whether the load is both concentrated AND has large average volume. This could be a strong infeasibility signal.

Actually, let me think about this differently. The XGB guidance specifically says in §3.6:
- `sku_concentration_x_sku_average_volume` — concentration × average volume; captures whether concentrated loads are also large.

This is different from spare_cap_x_conc because it doesn't involve spare_capacity. It directly measures: "are the items both concentrated and large?"

Let me finalize with 3 features:

1. **n_wide_items**: Threshold count of items with medium dimension > 50% of vehicle width
2. **height_tail_sum**: Cumulative excess of longest item dimension beyond 60% of vehicle height  
3. **conc_x_avg_vol**: Interaction of concentration and average volume

Let me write clean code:

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # Merge vehicle dimensions into items for threshold computations
    veh = agg_df[['dispatch_id', 'vehicle_width', 'vehicle_height']].copy()
    items = items_df.merge(veh, on='dispatch_id', how='left')

    # Feature 1: n_wide_items - count of items with dim_m > 50% of vehicle width
    items['is_wide'] = (items['dim_m'] > 0.5 * items['vehicle_width']).astype(int)
    n_wide = items.groupby('dispatch_id')['is_wide'].sum().reset_index()
    n_wide.columns = ['dispatch_id', 'n_wide_items']
    result = result.merge(n_wide, on='dispatch_id', how='left')

    # Feature 2: height_tail_sum - cumulative excess of dim_l beyond 60% vehicle height
    items['h_excess'] = (items['dim_l'] - 0.6 * items['vehicle_height']).clip(lower=0)
    h_tail = items.groupby('dispatch_id')['h_excess'].sum().reset_index()
    h_tail.columns = ['dispatch_id', 'height_tail_sum']
    result = result.merge(h_tail, on='dispatch_id', how='left')

    # Feature 3: conc_x_avg_vol - concentration × average volume interaction
    result['conc_x_avg_vol'] = agg_df['sku_concentration'].values * agg_df['sku_average_volume'].values

    # Fill any NaN from left joins
    result = result.fillna(0)

    return result
```

Wait, the rules say "Keep feature computations in pandas objects; do not call `.values` unless you immediately wrap the result back into a `pd.Series` or `pd.DataFrame`." and "Do not call `.rename(...)` on numpy arrays."

Let me fix the .values usage:

```python
result['conc_x_avg_vol'] = agg_df['sku_concentration'] * agg_df['sku_average_volume']
```

But wait, I need to make sure the alignment is correct. Since result is built from agg_df[['dispatch_id']], and I'm multiplying columns from agg_df, the index alignment might be off after merges. Let me be more careful:

Actually, the result starts as a copy of dispatch_id from agg_df, then I merge features onto it. The conc_x_avg_vol should be computed before any merges that might reorder rows, or I should merge it properly.

Let me restructure:

```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    # Start with dispatch_id from agg_df
    result = agg_df[['dispatch_id']].copy()

    # Feature 3: conc_x_avg_vol - can compute directly from agg_df
    conc_vol = agg_df[['dispatch_id']].copy()
    conc_vol['conc_x_avg_vol'] = agg_df['sku_concentration'] * agg_df['sku_average_volume']
    result = result.merge(conc_vol, on='dispatch_id', how='left')

    # Merge vehicle dimensions into items for threshold computations
    veh = agg_df[['dispatch_id', 'vehicle_width', 'vehicle_height']].copy()
    items = items_df.merge(veh, on='dispatch_id', how='left')

    # Feature 1: n_wide_items
    items['is_wide'] = (items['dim_m'] > 0.5 * items['vehicle_width']).astype(int)
    n_wide = items.groupby('dispatch_id')['is
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
        - AUC: `0.9697`
        - TPR@FPR=1%: `0.6829`
        - Accuracy: `0.9296`

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

        Current iteration: `10`

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
- iter `8`: AUC `0.9697` (Δ `+0.0045`), TPR@1% `0.6829` (Δ `+0.0482`), features=n_fragile_items, avg_item_flatness, spare_cap_x_sku_counts
- iter `6`: AUC `0.9695` (Δ `+0.0044`), TPR@1% `0.6647` (Δ `+0.0300`), features=spare_cap_x_wl_max, wl_total_over_wl_max, max_dim_l_to_vL
- iter `5`: AUC `0.9689` (Δ `+0.0037`), TPR@1% `0.6593` (Δ `+0.0246`), features=spare_cap_log1p, h_to_H_max_sq, l_std_x_h_max, max_fp_ratio

## Rejected Trials
- iter `3`: AUC `0.9682` (Δ `+0.0030`), TPR@1% `0.6608` (Δ `+0.0261`), features=spare_cap_x_avg_vol, wl_max_residual, n_long_items
- iter `9`: AUC `0.9694` (Δ `+0.0042`), TPR@1% `0.6657` (Δ `+0.0310`), features=width_tail_sum, height_tail_sum, load_param_mean

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
- source accepted trials: `1`, `5`, `6`, `8`
- active feature count: `15`
- active feature names: `spare_cap_sq`, `spare_cap_x_conc`, `spare_cap_x_wl_total`, `vol_top3_share`, `n_high_asr_items`, `spare_cap_log1p`, `h_to_H_max_sq`, `l_std_x_h_max`, `max_fp_ratio`, `spare_cap_x_wl_max`, `wl_total_over_wl_max`, `max_dim_l_to_vL`, `n_fragile_items`, `avg_item_flatness`, `spare_cap_x_sku_counts`
- active bank AUC: `0.9697`
- active bank TPR@FPR=1%: `0.6829`
- active bank Accuracy: `0.9296`

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
