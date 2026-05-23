def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Get vehicle dimensions per dispatch
    vdims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    
    # Merge vehicle dims to items
    items = items_df.merge(vdims, on='dispatch_id', how='left')
    
    # Feature 1: max dim_l relative to vehicle length
    feat1 = items.groupby('dispatch_id').apply(
        lambda g: g['dim_l'].max() / g['vehicle_length'].iloc[0]
    ).rename('dim_l_max_to_vl')
    
    # Feature 2: share of items where dim_m > 0.5 * vehicle_width
    items['width_dominant'] = (items['dim_m'] > 0.5 * items['vehicle_width']).astype(int)
    feat2 = items.groupby('dispatch_id')['width_dominant'].mean().rename('width_dominant_share')
    
    # Feature 3: share of items large in both floor dims
    items['dual_dominant'] = (
        (items['dim_l'] > 0.5 * items['vehicle_length']) & 
        (items['dim_m'] > 0.5 * items['vehicle_width'])
    ).astype(int)
    feat3 = items.groupby('dispatch_id')['dual_dominant'].mean().rename('dual_dominant_share')
    
    result = pd.concat([feat1, feat2, feat3], axis=1).reset_index()
    return result
