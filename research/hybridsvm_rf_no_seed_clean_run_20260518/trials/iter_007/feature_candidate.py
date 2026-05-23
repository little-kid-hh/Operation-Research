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
