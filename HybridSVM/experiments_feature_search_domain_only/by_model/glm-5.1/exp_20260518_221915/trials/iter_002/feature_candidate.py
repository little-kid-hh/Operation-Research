def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Merge vehicle dimensions for per-item computations
    veh = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    items = items_df.merge(veh, on='dispatch_id', how='left')
    
    # Feature 1: dim_l_occupancy_sum - total length demand
    items['dim_l_vlen_ratio'] = items['dim_l'] / items['vehicle_length']
    f1 = items.groupby('dispatch_id')['dim_l_vlen_ratio'].sum().rename('dim_l_occupancy_sum')
    
    # Feature 2: fragile_vol_share - volume fraction that is fragile
    items['frag_vol'] = items['item_volume'] * items['if_fragile']
    frag_vol_sum = items.groupby('dispatch_id')['frag_vol'].sum()
    total_vol_sum = items.groupby('dispatch_id')['item_volume'].sum()
    f2 = (frag_vol_sum / total_vol_sum).rename('fragile_vol_share')
    
    # Feature 3: needle_frac - fraction of very elongated items
    items['aspect_ratio'] = items['dim_l'] / items['dim_s'].clip(lower=1)
    items['is_needle'] = (items['aspect_ratio'] > 5).astype(int)
    f3 = items.groupby('dispatch_id')['is_needle'].mean().rename('needle_frac')
    
    # Feature 4: dim_m_tail_ratio - p90/median of dim_m
    dim_m_p90 = items.groupby('dispatch_id')['dim_m'].quantile(0.9)
    dim_m_med = items.groupby('dispatch_id')['dim_m'].median()
    f4 = (dim_m_p90 / dim_m_med.clip(lower=1)).rename('dim_m_tail_ratio')
    
    out = pd.concat([f1, f2, f3, f4], axis=1).reset_index()
    return out
