def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    # Index by dispatch_id for clean alignment
    agg_idx = agg_df.set_index('dispatch_id')
    sc = agg_idx['spare_capacity']
    vl = agg_idx['vehicle_length']
    wl_total = agg_idx['wl_to_vehicle_wl_total']
    sku_ct = agg_idx['sku_counts']

    # Merge vehicle length into items for per-item threshold computation
    items_ext = items_df.merge(
        agg_df[['dispatch_id', 'vehicle_length']],
        on='dispatch_id', how='left'
    )

    # 1. spare_cap_x_long_item: spare_capacity × share of items with dim_l > 50% vehicle_length
    items_ext['is_long'] = (items_ext['dim_l'] > 0.5 * items_ext['vehicle_length']).astype(int)
    long_share = items_ext.groupby('dispatch_id')['is_long'].mean()
    f_spare_x_long = sc * long_share

    # 2. dim_l_max_to_veh_l: max sorted-longest dimension / vehicle length
    dim_l_max = items_ext.groupby('dispatch_id')['dim_l'].max()
    f_dim_l_max_ratio = dim_l_max / vl

    # 3. wl_total_x_sku_counts: wl_to_vehicle_wl_total × sku_counts
    f_wl_x_ct = wl_total * sku_ct

    # Assemble into DataFrame indexed by dispatch_id
    feats = pd.DataFrame({
        'spare_cap_x_long_item': f_spare_x_long,
        'dim_l_max_to_veh_l': f_dim_l_max_ratio,
        'wl_total_x_sku_counts': f_wl_x_ct,
    })

    # Reset index to get dispatch_id as column
    feats = feats.reset_index()

    # Ensure all dispatch_ids from agg_df are present
    result = agg_df[['dispatch_id']].merge(feats, on='dispatch_id', how='left')

    return result
