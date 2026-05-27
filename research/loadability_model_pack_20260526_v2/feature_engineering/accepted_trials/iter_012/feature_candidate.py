def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Vehicle dimensions from agg_df (using sorted dims to align with item dim_s, dim_m, dim_l)
    vL = agg_df.set_index('dispatch_id')['vehicle_length']
    vW = agg_df.set_index('dispatch_id')['vehicle_width']
    vH = agg_df.set_index('dispatch_id')['vehicle_height']
    
    # Compute sorted vehicle dimensions per dispatch
    v_dims = pd.DataFrame({
        'v_s': np.minimum(np.minimum(vL, vW), vH),
        'v_m': np.median(np.array([vL, vW, vH]), axis=0),
        'v_l': np.maximum(np.maximum(vL, vW), vH)
    })
    
    # Merge vehicle sorted dims to items
    items = items_df.merge(v_dims, left_on='dispatch_id', right_index=True, how='left')
    
    # 1. m_ratio_p90: 90th percentile of dim_m / v_m ratio per dispatch
    items['m_ratio'] = items['dim_m'] / items['v_m']
    m_ratio_p90 = items.groupby('dispatch_id')['m_ratio'].quantile(0.9).rename('m_ratio_p90')
    
    # 2. l_ratio_iqr: Interquartile range of dim_l / v_l ratio per dispatch
    items['l_ratio'] = items['dim_l'] / items['v_l']
    l_ratio_q75 = items.groupby('dispatch_id')['l_ratio'].quantile(0.75)
    l_ratio_q25 = items.groupby('dispatch_id')['l_ratio'].quantile(0.25)
    l_ratio_iqr = (l_ratio_q75 - l_ratio_q25).rename('l_ratio_iqr')
    
    # 3. awkward_vol_x_slack_inv: Share of items with volume > 50% of mean item volume, 
    # multiplied by 1 / (1 + spare_capacity)
    mean_vol = items.groupby('dispatch_id')['item_volume'].transform('mean')
    items['is_large_vol'] = (items['item_volume'] > 0.5 * mean_vol).astype(float)
    large_vol_share = items.groupby('dispatch_id')['is_large_vol'].mean().rename('large_vol_share')
    spare = agg_df.set_index('dispatch_id')['spare_capacity']
    awkward_vol_x_slack_inv = (large_vol_share / (1.0 + spare)).rename('awkward_vol_x_slack_inv')
    
    # Assemble output
    out = pd.concat([m_ratio_p90, l_ratio_iqr, awkward_vol_x_slack_inv], axis=1).reset_index()
    return out
