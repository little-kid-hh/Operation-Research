def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # Feature 1: spare_cap_x_h_to_H_max
    # spare_capacity from agg_df, h_to_H_ratio_max from agg_df
    result['spare_cap_x_h_to_H_max'] = (
        agg_df['spare_capacity'] * agg_df['h_to_H_ratio_max']
    )
    
    # Feature 2: wl_total_x_h_to_H_max  
    # wl_to_vehicle_wl_total and h_to_H_ratio_max from agg_df
    result['wl_total_x_h_to_H_max'] = (
        agg_df['wl_to_vehicle_wl_total'] * agg_df['h_to_H_ratio_max']
    )
    
    # Feature 3: dim_s_avg_to_vmin
    # Average ratio of each item's smallest sorted dimension to the smallest vehicle dimension
    v_min = agg_df.set_index('dispatch_id')[['vehicle_length', 'vehicle_width', 'vehicle_height']].min(axis=1)
    
    items_with_vmin = items_df.merge(
        v_min.rename('v_min'),
        left_on='dispatch_id',
        right_index=True,
        how='left'
    )
    items_with_vmin['dim_s_ratio'] = items_with_vmin['dim_s'] / items_with_vmin['v_min']
    dim_s_avg = items_with_vmin.groupby('dispatch_id')['dim_s_ratio'].mean()
    result['dim_s_avg_to_vmin'] = result['dispatch_id'].map(dim_s_avg).fillna(0)
    
    return result
