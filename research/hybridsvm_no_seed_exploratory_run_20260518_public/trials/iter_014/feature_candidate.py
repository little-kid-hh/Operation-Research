def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # Get vehicle dimensions from agg_df
    v_dims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    
    # Merge vehicle dimensions to items
    items = items_df.merge(v_dims, on='dispatch_id', how='left')
    
    # Feature 1: q90_l_to_L - 90th percentile of dim_l/vehicle_length
    items['l_to_vL'] = items['dim_l'] / items['vehicle_length']
    q90_l = items.groupby('dispatch_id')['l_to_vL'].quantile(0.9).reset_index()
    q90_l.columns = ['dispatch_id', 'q90_l_to_L']
    result = result.merge(q90_l, on='dispatch_id', how='left')
    
    # Feature 2: spare_cap_x_n_wide_items
    # n_wide_items is already in active bank, but the interaction with spare_cap is not
    # Need to recompute n_wide_items to create the interaction
    # Actually, I should use the spare_capacity from agg_df and compute n_wide_items from items_df
    items['is_wide'] = (items['item_width'] > 0.5 * items['vehicle_width']).astype(int)
    n_wide = items.groupby('dispatch_id')['is_wide'].sum().reset_index()
    n_wide.columns = ['dispatch_id', 'n_wide_temp']
    result = result.merge(n_wide, on='dispatch_id', how='left')
    result['spare_cap_x_n_wide'] = agg_df['spare_capacity'].values * result['n_wide_temp'].values
    result = result.drop(columns=['n_wide_temp'])
    
    # Feature 3: height_tail_share - share of items with height > 60% of vehicle height
    items['is_very_tall'] = (items['item_height'] > 0.6 * items['vehicle_height']).astype(int)
    ht_share = items.groupby('dispatch_id')['is_very_tall'].mean().reset_index()
    ht_share.columns = ['dispatch_id', 'height_tail_share']
    result = result.merge(ht_share, on='dispatch_id', how='left')
    
    return result[['dispatch_id', 'q90_l_to_L', 'spare_cap_x_n_wide', 'height_tail_share']]
