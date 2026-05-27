def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    out = items_df[['dispatch_id']].copy()
    
    # 1. Chubby item share: items with dim_m / vehicle_height > 0.4
    # Captures "chubby" items that create localized packing gaps invisible to length/height extremes
    vh = items_df['vehicle_capacity']  # proxy for vehicle dims availability
    # Since vehicle_height is not in items_df, use dim_l as reference for vehicle scale
    # We use the sorted dimensions to identify items where the medium dimension is unusually large
    dim_m_ratio = items_df['dim_m'] / (items_df['dim_l'] + 1e-9)
    chubby = (dim_m_ratio > 0.85).astype(float)
    out['chubby_item_share'] = chubby.groupby(items_df['dispatch_id']).transform('mean')
    
    # 2. Tall x low slack interaction: tall_item_share * (1 / (spare_capacity + 0.01))
    # Translates XGB's conditional split where tall items make packing infeasible at low slack
    tall_share = (items_df['dim_l'] / (items_df['vehicle_capacity'] + 1e-9) > 0.7).astype(float)
    tall_share_disp = tall_share.groupby(items_df['dispatch_id']).transform('mean')
    spare_disp = items_df.groupby('dispatch_id')['vehicle_capacity'].transform('first')
    # Use spare_capacity from agg_df via merge
    merge_df = agg_df[['dispatch_id', 'spare_capacity']].copy()
    out = out.merge(merge_df, on='dispatch_id', how='left')
    out['tall_x_slack_inv'] = tall_share_disp * (1.0 / (out['spare_capacity'] + 0.01))
    out = out.drop(columns=['spare_capacity'])
    
    # 3. Extreme height ratio share: h_to_H > 0.85
    # Isolates the most severe vertical bottlenecks, linearizing XGB's h_to_H_ratio_max splits
    extreme_h = (items_df['item_height'] / (items_df['vehicle_capacity'] + 1e-9) > 0.85).astype(float)
    out['extreme_height_share'] = extreme_h.groupby(items_df['dispatch_id']).transform('mean')
    
    # 4. Long x low slack interaction: long_item_share * (1 / (spare_capacity + 0.01))
    # Captures conditional boundary where long items drastically reduce feasibility at low slack
    long_share = (items_df['dim_l'] / (items_df['vehicle_capacity'] + 1e-9) > 0.8).astype(float)
    long_share_disp = long_share.groupby(items_df['dispatch_id']).transform('mean')
    out_merge = agg_df[['dispatch_id', 'spare_capacity']].copy()
    out = out.merge(out_merge, on='dispatch_id', how='left', suffixes=('', '_r'))
    out['long_x_slack_inv'] = long_share_disp * (1.0 / (out['spare_capacity'] + 0.01))
    out = out.drop(columns=['spare_capacity'])
    
    # Deduplicate to one row per dispatch
    out = out.drop_duplicates(subset=['dispatch_id'])
    
    # Select only the new feature columns plus dispatch_id
    new_features = ['chubby_item_share', 'tall_x_slack_inv', 'extreme_height_share', 'long_x_slack_inv']
    result = out[['dispatch_id'] + new_features].copy()
    
    return result
