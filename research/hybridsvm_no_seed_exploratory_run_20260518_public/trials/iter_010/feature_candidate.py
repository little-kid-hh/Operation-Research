def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # 1. n_fragile_items: count of fragile items per dispatch
    frag_counts = items_df.groupby('dispatch_id')['if_fragile'].sum().rename('n_fragile_items')
    
    # 2. flat_item_share: fraction of items with item_flatness > 3
    items_flat = items_df.copy()
    items_flat['is_flat'] = (items_flat['item_flatness'] > 3).astype(int)
    flat_share = items_flat.groupby('dispatch_id')['is_flat'].mean().rename('flat_item_share')
    
    # 3. vol_top1_share: largest item's volume as fraction of total volume
    vol_sorted = items_df.groupby('dispatch_id')['item_volume'].apply(
        lambda s: s.nlargest(1).sum() / s.sum() if s.sum() > 0 else 0
    ).rename('vol_top1_share')
    
    result = agg_df[['dispatch_id']].copy()
    result = result.set_index('dispatch_id')
    result['n_fragile_items'] = frag_counts
    result['flat_item_share'] = flat_share
    result['vol_top1_share'] = vol_sorted
    result = result.reset_index()
    
    return result[['dispatch_id', 'n_fragile_items', 'flat_item_share', 'vol_top1_share']]
