def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # Feature 1: spare_cap_x_wl_max
    result['spare_cap_x_wl_max'] = (
        agg_df['spare_capacity'].values * agg_df['wl_to_vehicle_wl_max'].values
    )
    
    # Feature 2: item_vol_cv - coefficient of variation of item volumes
    vol_stats = items_df.groupby('dispatch_id')['item_volume'].agg(['std', 'mean'])
    vol_stats['item_vol_cv'] = vol_stats['std'] / (vol_stats['mean'] + 1e-9)
    result = result.merge(
        vol_stats[['item_vol_cv']].reset_index(),
        on='dispatch_id',
        how='left'
    )
    
    # Feature 3: long_item_share - fraction of items where dim_l > 0.5 * vehicle_length
    items_with_vl = items_df.merge(
        agg_df[['dispatch_id', 'vehicle_length']],
        on='dispatch_id',
        how='left'
    )
    items_with_vl['is_long'] = (items_with_vl['dim_l'] > 0.5 * items_with_vl['vehicle_length']).astype(int)
    long_share = items_with_vl.groupby('dispatch_id')['is_long'].mean().reset_index()
    long_share.columns = ['dispatch_id', 'long_item_share']
    result = result.merge(long_share, on='dispatch_id', how='left')
    
    return result
