def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # Get vehicle dimensions from agg_df
    veh_dims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    
    # Merge vehicle dims to items
    items = items_df.merge(veh_dims, on='dispatch_id', how='left')
    
    # Feature 1: q90_w_to_W - 90th percentile of width-to-vehicle-width ratio
    items['w_to_W'] = items['item_width'] / items['vehicle_width']
    q90_w = items.groupby('dispatch_id')['w_to_W'].quantile(0.9).reset_index()
    q90_w.columns = ['dispatch_id', 'q90_w_to_W']
    result = result.merge(q90_w, on='dispatch_id', how='left')
    
    # Feature 2: spare_cap_x_height_tail_share
    # height_tail_share is already in the bank, so I need to compute it again here
    # Actually, I should compute it from items_df since I can't use the active bank features
    items['h_to_H'] = items['item_height'] / items['vehicle_height']
    ht_share = items.groupby('dispatch_id').apply(
        lambda g: (g['h_to_H'] > 0.5).mean()
    ).reset_index()
    ht_share.columns = ['dispatch_id', 'height_tail_share_local']
    
    # Merge with spare_capacity from agg_df
    temp = agg_df[['dispatch_id', 'spare_capacity']].merge(ht_share, on='dispatch_id', how='left')
    temp['spare_cap_x_height_tail'] = temp['spare_capacity'] * temp['height_tail_share_local']
    result = result.merge(temp[['dispatch_id', 'spare_cap_x_height_tail']], on='dispatch_id', how='left')
    
    # Feature 3: n_long_and_wide_items
    items['is_long'] = items['dim_l'] > 0.5 * items['vehicle_length']
    items['is_wide'] = items['dim_m'] > 0.5 * items['vehicle_width']
    items['is_long_and_wide'] = items['is_long'] & items['is_wide']
    n_lw = items.groupby('dispatch_id')['is_long_and_wide'].sum().reset_index()
    n_lw.columns = ['dispatch_id', 'n_long_and_wide']
    result = result.merge(n_lw, on='dispatch_id', how='left')
    
    result = result.fillna(0)
    return result
