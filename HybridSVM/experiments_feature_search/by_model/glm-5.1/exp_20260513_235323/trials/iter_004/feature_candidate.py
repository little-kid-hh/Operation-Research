def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # Feature 1: spare_x_wlmax - interaction of spare capacity with max piece footprint ratio
    result['spare_x_wlmax'] = (
        agg_df['spare_capacity'] * agg_df['wl_to_vehicle_wl_max']
    )
    
    # Feature 2: total_footprint_ratio - sum of item footprints / vehicle floor area
    items_copy = items_df.copy()
    items_copy['footprint'] = items_copy['dim_l'] * items_copy['dim_m']
    footprint_sum = items_copy.groupby('dispatch_id')['footprint'].sum().reset_index()
    footprint_sum.columns = ['dispatch_id', 'total_footprint']
    result = result.merge(footprint_sum, on='dispatch_id', how='left')
    result['total_footprint_ratio'] = (
        result['total_footprint'] / (agg_df['vehicle_length'] * agg_df['vehicle_width'])
    )
    result = result.drop(columns=['total_footprint'])
    
    # Feature 3: cross_dim_pressure_count - items large in two dimensions
    items_copy2 = items_df.copy()
    vl = agg_df.set_index('dispatch_id')['vehicle_length']
    vw = agg_df.set_index('dispatch_id')['vehicle_width']
    items_copy2['vl'] = items_copy2['dispatch_id'].map(vl)
    items_copy2['vw'] = items_copy2['dispatch_id'].map(vw)
    items_copy2['cross_pressure'] = (
        (items_copy2['dim_l'] > 0.6 * items_copy2['vl']) & 
        (items_copy2['dim_m'] > 0.5 * items_copy2['vw'])
    ).astype(int)
    cross_count = items_copy2.groupby('dispatch_id')['cross_pressure'].sum().reset_index()
    cross_count.columns = ['dispatch_id', 'cross_dim_pressure_count']
    result = result.merge(cross_count, on='dispatch_id', how='left')
    
    result = result.drop(columns=['dispatch_id'])
    result.insert(0, 'dispatch_id', agg_df['dispatch_id'].values)
    
    return result
