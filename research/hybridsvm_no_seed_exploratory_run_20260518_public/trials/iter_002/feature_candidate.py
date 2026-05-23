def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()
    
    # Feature 1: spare_capacity × wl_to_vehicle_wl_max
    result['spare_cap_x_wl_max'] = agg_df['spare_capacity'] * agg_df['wl_to_vehicle_wl_max']
    
    # Feature 2: wl_max concentration ratio
    result['wl_max_over_total'] = agg_df['wl_to_vehicle_wl_max'] / (agg_df['wl_to_vehicle_wl_total'] + 1e-8)
    
    # Feature 3: h_to_H_ratio_max squared (threshold effect)
    result['h_ratio_max_sq'] = agg_df['h_to_H_ratio_max'] ** 2
    
    # Feature 4: count of items with dim_l near vehicle length limit
    merged = items_df.merge(agg_df[['dispatch_id', 'vehicle_length']], on='dispatch_id', how='left')
    near_len_mask = merged['dim_l'] > 0.7 * merged['vehicle_length']
    n_near_len = near_len_mask.groupby(merged['dispatch_id']).sum().astype(float)
    result = result.merge(near_len.rename('n_near_len_items'), left_on='dispatch_id', right_index=True, how='left')
    result['n_near_len_items'] = result['n_near_len_items'].fillna(0)
    
    return result
