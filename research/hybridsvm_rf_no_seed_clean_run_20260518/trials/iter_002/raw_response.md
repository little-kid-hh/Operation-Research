## POLICY_UPDATE
- Continue targeting the TPR@FPR=1% gap by linearizing RF's conditional splits on the top features.
- Iteration 1 covered spare-capacity interactions with `wl_total` and `sku_counts`, a quadratic slack term, cross-dimension tightness, and a volume-based big-piece share.
- Iteration 2 priorities: (a) the remaining H2 interaction—`spare_capacity × sku_average_volume`—since `sku_average_volume` carries the largest negative SVM weight (-16.4) and RF importance 0.064; (b) an H3 wl-family shape signal capturing single-item footprint dominance; (c) an item-level elongation-threshold share that trees create via binary splits but SVM cannot express.

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # 1. spare_capacity x sku_average_volume interaction
    #    sku_average_volume has the largest |w_j| in the SVM; interacting it
    #    with spare_capacity linearizes the conditional split trees make
    #    when spare_capacity is low AND items are large.
    result['spare_cap_x_sku_avg_vol'] = (
        agg_df['spare_capacity'] * agg_df['sku_average_volume']
    )

    # 2. wl max residual — gap between the largest item wl-ratio and the mean.
    #    Captures whether a single item dominates the weight-length footprint
    #    (high residual) versus a uniform distribution (low residual).
    #    RF can split on max vs avg separately; SVM needs their difference.
    result['wl_max_residual'] = (
        agg_df['wl_to_vehicle_wl_max'] - agg_df['wl_to_vehicle_wl_avg']
    )

    # 3. awkward_shape_share — fraction of items whose longest sorted dimension
    #    exceeds 3× the middle sorted dimension (dim_l / dim_m > 3).
    #    Very elongated items are hard to place and create local bottlenecks
    #    that trees capture via threshold splits on individual item shapes.
    is_awkward = (items_df['dim_l'] > 3.0 * items_df['dim_m']).astype(int)
    awkward_share = is_awkward.groupby(items_df['dispatch_id']).mean()
    awkward_df = awkward_share.to_frame('awkward_shape_share').reset_index()
    result = result.merge(awkward_df, on='dispatch_id', how='left')

    return result
```

## RATIONALE
- **`spare_cap_x_sku_avg_vol`**: `sku_average_volume` has the single largest SVM weight (−16.4) and RF importance 0.064. Iteration 1 interacted spare capacity with `wl_total` and `sku_counts`; the remaining top interaction with average volume directly addresses the conditional regime where low slack coincides with bulky items—exactly where trees switch branches but SVM sees only additive effects.
- **`wl_max_residual`**: The `wl_to_vehicle_wl_*` family accounts for ~22% of RF importance. The existing bank has no feature capturing the *shape* of this distribution—whether one item dominates the footprint (high max–avg gap) or utilization is spread evenly. This residual is a clean, interpretable dispersion signal distinct from `wl_to_vehicle_wl_std`.
- **`awkward_shape_share`**: A threshold-based item-level count (dim_l/dim_m > 3) that trees naturally form via binary splits. Elongated items are structurally hard to pack because they constrain placement along one axis while wasting space on others. This is orthogonal to `big_piece_share` (volume-based) because an item can be small in volume yet extremely elongated.