def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Get vehicle dims from agg_df
    vdims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    
    items = items_df.merge(vdims, on='dispatch_id', how='left')
    
    # Feature 1: three_dim_near_limit_share
    # Items large in all three sorted dimensions relative to vehicle
    items['three_dim_flag'] = (
        (items['dim_l'] > 0.5 * items['vehicle_length']) &
        (items['dim_m'] > 0.5 * items['vehicle_width']) &
        (items['dim_s'] > 0.5 * items['vehicle_height'])
    ).astype(int)
    
    three_dim_share = items.groupby('dispatch_id')['three_dim_flag'].mean().rename('three_dim_near_limit_share')
    
    # Feature 2: vol_top3_concentration
    # Volume share of top-3 largest items
    items_sorted = items.sort_values(['dispatch_id', 'item_volume'], ascending=[True, False])
    top3_vol = items_sorted.groupby('dispatch_id').head(3).groupby('dispatch_id')['item_volume'].sum()
    total_vol = items.groupby('dispatch_id')['item_volume'].sum()
    vol_top3_ratio = (top3_vol / total_vol).rename('vol_top3_concentration')
    
    # Feature 3: spare_x_wlstd
    # Interaction of spare_capacity and wl_to_vehicle_wl_std
    result = agg_df[['dispatch_id']].copy()
    result = result.merge(three_dim_share, on='dispatch_id', how='left')
    result = result.merge(vol_top3_ratio, on='dispatch_id', how='left')
    
    result['spare_x_wlstd'] = agg_df['spare_capacity'] * agg_df['wl_to_vehicle_wl_std']
    
    result = result.fillna(0)
    
    return result
