def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Merge vehicle dimensions from agg_df to items_df for per-item computations
    vehicle_dims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    items_merged = items_df.merge(vehicle_dims, on='dispatch_id', how='left')
    
    # Aspect ratio polarity: dim_l / dim_s (already sorted: dim_s <= dim_m <= dim_l)
    aspect_polarity = items_merged['dim_l'] / (items_merged['dim_s'] + 1e-9)
    
    # 1. Dimension polarity tail ratio: 90th percentile of aspect ratio per dispatch
    polarity_p90 = aspect_polarity.groupby(items_merged['dispatch_id']).quantile(0.9)
    polarity_p90.name = 'polarity_p90'
    
    # 2. Rod-item share: fraction of items with aspect ratio > 5.0 (needle/rod shapes)
    is_rod = (aspect_polarity > 5.0).astype(float)
    rod_item_share = is_rod.groupby(items_merged['dispatch_id']).mean()
    rod_item_share.name = 'rod_item_share'
    
    # 3. Low-slack rod bottleneck interaction: rod_item_share * (1 / (spare_capacity + 1e-9))
    # spare_capacity is already in agg_df, so compute interaction at dispatch level
    rod_x_slack_inv = rod_item_share * (1.0 / (agg_df.set_index('dispatch_id')['spare_capacity'] + 1e-9))
    rod_x_slack_inv.name = 'rod_x_slack_inv'
    
    # Assemble output DataFrame with dispatch_id and new feature columns
    out_df = pd.DataFrame({
        'dispatch_id': polarity_p90.index,
        'polarity_p90': polarity_p90.values,
        'rod_item_share': rod_item_share.values,
        'rod_x_slack_inv': rod_x_slack_inv.values
    }).reset_index(drop=True)
    
    return out_df
