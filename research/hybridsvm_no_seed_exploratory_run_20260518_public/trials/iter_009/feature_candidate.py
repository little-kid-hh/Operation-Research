def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # Merge vehicle dimensions into items_df
    items = items_df.merge(
        agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']],
        on='dispatch_id',
        how='left'
    )
    
    # Feature 1: n_wide_items - count of items with width > 50% of vehicle width
    items['is_wide'] = (items['item_width'] > 0.5 * items['vehicle_width']).astype(int)
    n_wide = items.groupby('dispatch_id')['is_wide'].sum().rename('n_wide_items')
    result = result.merge(n_wide, on='dispatch_id', how='left')
    
    # Feature 2: n_multi_dim_stress - items large in >=2 of 3 dimensions (>40% of vehicle)
    items['is_long'] = (items['dim_l'] > 0.4 * items['vehicle_length']).astype(int)
    items['is_wide_dim'] = (items['dim_m'] > 0.4 * items['vehicle_width']).astype(int)
    items['is_tall_dim'] = (items['dim_s'] > 0.4 * items['vehicle_height']).astype(int)
    items['dim_stress_count'] = items['is_long'] + items['is_wide_dim'] + items['is_tall_dim']
    items['is_multi_stress'] = (items['dim_stress_count'] >= 2).astype(int)
    n_multi = items.groupby('dispatch_id')['is_multi_stress'].sum().rename('n_multi_dim_stress')
    result = result.merge(n_multi, on='dispatch_id', how='left')
    
    # Feature 3: spare_cap_x_n_near_vL - spare_capacity * n_items_near_vL
    # n_items_near_vL is already in active bank, but the interaction with spare_cap is not
    # We need to recompute n_items_near_vL to create the interaction
    items['near_vL'] = (items['dim_l'] > 0.5 * items['vehicle_length']).astype(int)
    n_near_vL = items.groupby('dispatch_id')['near_vL'].sum()
    result = result.merge(
        n_near_vL.rename('n_near_vL_temp'), on='dispatch_id', how='left'
    )
    result['spare_cap_x_n_near_vL'] = agg_df['spare_capacity'] * result['n_near_vL_temp']
    result = result.drop(columns=['n_near_vL_temp'])
    
    # Fill NAs
    result = result.fillna(0)
    
    return result
