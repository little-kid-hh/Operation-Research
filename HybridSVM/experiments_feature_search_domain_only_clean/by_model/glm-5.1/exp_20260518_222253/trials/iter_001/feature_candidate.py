def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Get vehicle dimensions per dispatch
    vdims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    
    # Merge vehicle dims into items
    items = items_df.merge(vdims, on='dispatch_id', how='left')
    
    # Feature 1: tall_item_share - fraction of items where longest dim > 70% of vehicle height
    items['is_tall'] = (items['dim_l'] > 0.7 * items['vehicle_height']).astype(int)
    tall_share = items.groupby('dispatch_id')['is_tall'].mean()
    
    # Feature 2: volume_tail_ratio - P90 / P50 of item_volume
    vol_p90 = items.groupby('dispatch_id')['item_volume'].quantile(0.9)
    vol_p50 = items.groupby('dispatch_id')['item_volume'].quantile(0.5)
    vol_tail = vol_p90 / vol_p50.clip(lower=1e-9)
    
    # Feature 3: fragile_share
    fragile_share = items.groupby('dispatch_id')['if_fragile'].mean()
    
    # Feature 4: flat_item_share - fraction where item_flatness > 5
    items['is_flat'] = (items['item_flatness'] > 5.0).astype(int)
    flat_share = items.groupby('dispatch_id')['is_flat'].mean()
    
    # Feature 5: dim_l_p90_to_vh - 90th percentile of dim_l / vehicle_height
    items['dim_l_to_vh'] = items['dim_l'] / items['vehicle_height'].clip(lower=1e-9)
    dim_l_p90_vh = items.groupby('dispatch_id')['dim_l_to_vh'].quantile(0.9)
    
    # Assemble
    result = pd.DataFrame({
        'dispatch_id': agg_df['dispatch_id'],
        'tall_item_share': tall_share.reindex(agg_df['dispatch_id']).values,
        'volume_tail_ratio': vol_tail.reindex(agg_df['dispatch_id']).values,
        'fragile_share': fragile_share.reindex(agg_df['dispatch_id']).values,
        'flat_item_share': flat_share.reindex(agg_df['dispatch_id']).values,
        'dim_l_p90_to_vh': dim_l_p90_vh.reindex(agg_df['dispatch_id']).values,
    })
    
    # Fill NaN from single-item dispatches
    result = result.fillna(0.0)
    
    return result
