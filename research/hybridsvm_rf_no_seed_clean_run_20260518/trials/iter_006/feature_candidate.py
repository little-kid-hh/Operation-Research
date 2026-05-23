def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()
    
    # H3: wl_max_to_avg - single-item wl dominance
    result['wl_max_to_avg'] = agg_df['wl_to_vehicle_wl_max'] / (agg_df['wl_to_vehicle_wl_avg'] + 1e-6)
    
    # H6: tight_x_high_wl - regime interaction
    is_tight = (agg_df['spare_capacity'] < 0.15).astype(int)
    is_high_wl = (agg_df['wl_to_vehicle_wl_total'] > 0.7).astype(int)
    result['tight_x_high_wl'] = is_tight * is_high_wl
    
    # H5: sku_counts_x_lL - many items in tight longitudinal space
    result['sku_counts_x_lL'] = agg_df['sku_counts'] * agg_df['l_to_L_ratio_avg']
    
    return result
