## POLICY_UPDATE
- Add dimensional pressure features capturing stacking difficulty and floor competition
- Focus on sum-based pressure signals rather than percentile/average ratios already in the bank
- height_layer_pressure captures vertical stacking demand via smallest dimensions
- footprint_sum_ratio captures floor-area bottleneck using sorted dims (best-case orientation)
- length_sum_pressure captures total length competition as a count×avg interaction

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    vL = agg_df.set_index('dispatch_id')['vehicle_length']
    vW = agg_df.set_index('dispatch_id')['vehicle_width']
    vH = agg_df.set_index('dispatch_id')['vehicle_height']

    # height_layer_pressure: sum of smallest dims / vehicle height
    # estimates how many vertical layers are needed
    dim_s_sum = items_df.groupby('dispatch_id')['dim_s'].sum()
    height_layer_pressure = (dim_s_sum / vH).reindex(agg_df['dispatch_id']).fillna(0)
    result['height_layer_pressure'] = height_layer_pressure.values

    # footprint_sum_ratio: sum(dim_l * dim_m) / (vL * vW)
    # minimum floor area needed under best-case orientation
    footprint_per_item = items_df['dim_l'] * items_df['dim_m']
    footprint_sum = items_df.groupby('dispatch_id').apply(
        lambda g: (g['dim_l'] * g['dim_m']).sum()
    )
    floor_area = vL * vW
    footprint_sum_ratio = (footprint_sum / floor_area).reindex(agg_df['dispatch_id']).fillna(0)
    result['footprint_sum_ratio'] = footprint_sum_ratio.values

    # length_sum_pressure: sum(dim_l) / vehicle_length
    # total length demand — an implicit count × avg interaction
    dim_l_sum = items_df.groupby('dispatch_id')['dim_l'].sum()
    length_sum_pressure = (dim_l_sum / vL).reindex(agg_df['dispatch_id']).fillna(0)
    result['length_sum_pressure'] = length_sum_pressure.values

    return result
```

## RATIONALE
- height_layer_pressure: Sum of dim_s over vehicle height estimates vertical stacking layers; high values mean items must stack tall, making packing harder. Unlike dim_s_p90_to_vh (rejected iter 4), this captures cumulative pressure from all items, not just the tail.
- footprint_sum_ratio: Sum of optimal-orientation footprints over vehicle floor area. When >1, items must stack vertically regardless of arrangement. Uses sorted dims (dim_l×dim_m) which gives the minimum possible footprint per item.
- length_sum_pressure: Total length demand as sum(dim_l)/vL. This is effectively sku_counts × avg(dim_l/vL), creating an interaction the linear SVM cannot build on its own. Captures how many items compete for the length dimension.