def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Get vehicle dimensions per dispatch
    v_height = agg_df.set_index('dispatch_id')['vehicle_height']
    
    # Height pressure share
    items_df = items_df.copy()
    items_df['height_ratio'] = items_df['dim_s'] / items_df['dispatch_id'].map(v_height)
    items_df['is_height_pressure'] = (items_df['height_ratio'] > 0.5).astype(int)
    height_pressure = items_df.groupby('dispatch_id')['is_height_pressure'].mean()
    
    # Long and wide item share
    v_length = agg_df.set_index('dispatch_id')['vehicle_length']
    v_width = agg_df.set_index('dispatch_id')['vehicle_width']
    items_df['length_ratio'] = items_df['dim_l'] / items_df['dispatch_id'].map(v_length)
    items_df['width_ratio'] = items_df['dim_m'] / items_df['dispatch_id'].map(v_width)
    items_df['is_long_wide'] = ((items_df['length_ratio'] > 0.5) & (items_df['width_ratio'] > 0.4)).astype(int)
    long_wide_share = items_df.groupby('dispatch_id')['is_long_wide'].mean()
    
    # Spare x vol_cv interaction
    vol_stats = items_df.groupby('dispatch_id')['item_volume'].agg(['mean', 'std'])
    vol_cv = vol_stats['std'] / vol_stats['mean']
    spare_cap = agg_df.set_index('dispatch_id')['spare_capacity']
    spare_x_volcv = (1 - spare_cap) * vol_cv
    
    # Combine
    result = pd.DataFrame({
        'height_pressure_share': height_pressure,
        'long_wide_item_share': long_wide_share,
        'spare_x_volcv': spare_x_volcv
    })
    result.index.name = 'dispatch_id'
    result = result.reset_index()
    
    return result
