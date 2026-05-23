def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Merge vehicle dimensions into items_df
    vdims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    items = items_df.merge(vdims, on='dispatch_id', how='left')
    
    # Feature 1: Cross-section pressure
    items['face_area_ratio'] = (items['dim_m'] * items['dim_s']) / (items['vehicle_width'] * items['vehicle_height'])
    f1 = items.groupby('dispatch_id')['face_area_ratio'].sum().rename('cross_sect_pressure')
    
    # Feature 2: Count of items consuming >50% of vehicle width
    items['is_half_width'] = (items['dim_m'] > 0.5 * items['vehicle_width']).astype(int)
    f2 = items.groupby('dispatch_id')['is_half_width'].sum().rename('n_half_width')
    
    # Feature 3: Max face area ratio (single item bottleneck)
    f3 = items.groupby('dispatch_id')['face_area_ratio'].max().rename('max_face_vratio')
    
    result = pd.concat([f1, f2, f3], axis=1).reset_index()
    return result
