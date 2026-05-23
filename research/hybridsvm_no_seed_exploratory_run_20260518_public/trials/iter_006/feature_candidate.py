def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # Feature 1: spare_capacity × wl_to_vehicle_wl_max interaction
    result['spare_cap_x_wl_max'] = (
        agg_df['spare_capacity'] * agg_df['wl_to_vehicle_wl_max']
    )
    
    # Feature 2: footprint concentration ratio
    result['wl_total_over_wl_max'] = (
        agg_df['wl_to_vehicle_wl_total'] / (agg_df['wl_to_vehicle_wl_max'] + 1e-9)
    )
    
    # Feature 3: max item longest-dim ratio to vehicle length
    max_dim_l = items_df.groupby('dispatch_id')['dim_l'].max().reset_index()
    max_dim_l.columns = ['dispatch_id', 'max_dim_l']
    merged = result.merge(max_dim_l, on='dispatch_id', how='left')
    result['max_dim_l_to_vL'] = merged['max_dim_l'] / (agg_df['vehicle_length'] + 1e-9)
    
    return result
