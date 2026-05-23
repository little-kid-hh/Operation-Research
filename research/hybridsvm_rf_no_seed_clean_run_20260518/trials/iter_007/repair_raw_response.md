## POLICY_UPDATE
- Add **sku_counts × h_to_H** (H5, untried): isolates "many items competing for vertical space" regardless of overall slack — distinct from the rejected `spare_cap × hH` which captured slack-conditional vertical tightness.
- Introduce **mean item flatness** from `items_df`: flat items stack more easily, easing packing in a way no existing aggregate captures.
- Add **near_limit_share**: fraction of items whose longest sorted dimension exceeds 75% of vehicle length — a threshold-based length-bottleneck signal that linearizes the kind of depth-1 split trees make on long-item prevalence.

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # 1. Vertical pressure: many items competing for height space (H5)
    result['sku_counts_x_hH'] = (
        agg_df['sku_counts'] * agg_df['h_to_H_ratio_avg']
    )

    # 2. Mean item flatness: flat items stack more easily, easing packing
    flat_mean = items_df.groupby('dispatch_id')['item_flatness'].mean()
    result['flat_item_mean'] = result['dispatch_id'].map(flat_mean).fillna(0)

    # 3. Share of items whose longest sorted dim exceeds 75% of vehicle length
    veh_length_map = agg_df.set_index('dispatch_id')['vehicle_length']
    items_ext = items_df.copy()
    items_ext['vehicle_length'] = items_ext['dispatch_id'].map(veh_length_map)
    items_ext['near_limit'] = (
        items_ext['dim_l'] > 0.75 * items_ext['vehicle_length']
    ).astype(int)
    near_limit_share = items_ext.groupby('dispatch_id')['near_limit'].mean()
    result['near_limit_share'] = result['dispatch_id'].map(near_limit_share).fillna(0)

    return result
```

## RATIONALE
- `sku_counts_x_hH` targets the H5 hypothesis (sku_counts × height ratio interaction) not yet in the active bank; it captures vertical crowding pressure that trees model via sequential splits on count then height ratio.
- `flat_item_mean` is a genuinely new item-level signal — flatness enables stable stacking and reduces effective height consumption, a packing-ease factor absent from all existing aggregates.
- `near_limit_share` linearizes the threshold effect trees exploit when splitting on long-item prevalence: a high share of items approaching the vehicle length limit creates a longitudinal bottleneck that smooth averages of `l_to_L_ratio` underweight.