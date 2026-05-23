def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # Feature 1: spare_capacity × sku_average_volume interaction
    result['spare_cap_x_avg_vol'] = (
        agg_df['spare_capacity'] * agg_df['sku_average_volume']
    )
    
    # Feature 2: wl_to_vehicle_wl_max - wl_to_vehicle_wl_total (footprint concentration gap)
    result['wl_max_residual'] = (
        agg_df['wl_to_vehicle_wl_max'] - agg_df['wl_to_vehicle_wl_total']
    )
    
    # Feature 3: count of long items (dim_l > 60% of vehicle_length)
    veh_dims = agg_df[['dispatch_id', 'vehicle_length']].copy()
    items_with_veh = items_df.merge(veh_dims, on='dispatch_id', how='left')
    items_with_veh['is_long'] = (items_with_veh['dim_l'] > 0.6 * items_with_veh['vehicle_length']).astype(int)
    n_long = items_with_veh.groupby('dispatch_id')['is_long'].sum().reset_index()
    n_long.columns = ['dispatch_id', 'n_long_items']
    result = result.merge(n_long, on='dispatch_id', how='left')
    result['n_long_items'] = result['n_long_items'].fillna(0)
    
    return result
