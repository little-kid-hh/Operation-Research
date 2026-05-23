def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # 1. spare_cap_x_sku_counts: spare capacity behaves differently at high vs low item counts
    result['spare_cap_x_sku_counts'] = (
        agg_df['spare_capacity'] * agg_df['sku_counts']
    )

    # 2. n_items_near_vL: count of items whose longest sorted dim > 50% of vehicle length
    #    Multiple such items compete for the same axis, creating packing conflicts
    merged_l = items_df[['dispatch_id', 'dim_l']].merge(
        agg_df[['dispatch_id', 'vehicle_length']], on='dispatch_id'
    )
    merged_l['is_near_vL'] = (merged_l['dim_l'] > 0.5 * merged_l['vehicle_length']).astype(int)
    near_vl_counts = merged_l.groupby('dispatch_id')['is_near_vL'].sum()
    result['n_items_near_vL'] = result['dispatch_id'].map(near_vl_counts).fillna(0)

    # 3. n_tall_items: count of items whose height > 50% of vehicle height
    #    Tall items block vertical stacking; a threshold count captures cumulative pressure
    merged_h = items_df[['dispatch_id', 'item_height']].merge(
        agg_df[['dispatch_id', 'vehicle_height']], on='dispatch_id'
    )
    merged_h['is_tall'] = (merged_h['item_height'] > 0.5 * merged_h['vehicle_height']).astype(int)
    tall_counts = merged_h.groupby('dispatch_id')['is_tall'].sum()
    result['n_tall_items'] = result['dispatch_id'].map(tall_counts).fillna(0)

    return result
