def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # Feature 1: volume_top3_share - fraction of total volume in top 3 items
    items_sorted = items_df.sort_values(['dispatch_id', 'item_volume'], ascending=[True, False])
    items_sorted['vol_rank'] = items_sorted.groupby('dispatch_id').cumcount()
    top3_vol = items_sorted[items_sorted['vol_rank'] < 3].groupby('dispatch_id')['item_volume'].sum()
    total_vol = items_df.groupby('dispatch_id')['item_volume'].sum()
    result['volume_top3_share'] = (top3_vol / total_vol).reindex(result['dispatch_id'].values).fillna(0.0)
    
    # Feature 2: multi_dim_near_limit_count - items large in 2+ sorted dims relative to vehicle
    vl = items_df.merge(agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']], on='dispatch_id')
    vl['long_near'] = (vl['dim_l'] > 0.5 * vl['vehicle_length']).astype(int)
    vl['mid_near'] = (vl['dim_m'] > 0.5 * vl['vehicle_width']).astype(int)
    vl['multi_near'] = (vl['long_near'] + vl['mid_near'] >= 2).astype(int)
    multi_near_cnt = vl.groupby('dispatch_id')['multi_near'].sum()
    item_counts = items_df.groupby('dispatch_id').size()
    result['multi_dim_near_limit_share'] = (multi_near_cnt / item_counts).reindex(result['dispatch_id'].values).fillna(0.0)
    
    # Feature 3: side_wall_area_load - sum of (dim_l * dim_s) / (vehicle_length * vehicle_height)
    # This measures how much item face area presses against the side wall
    vl['side_face'] = vl['dim_l'] * vl['dim_s']
    side_load = vl.groupby('dispatch_id')['side_face'].sum()
    side_cap = vl.groupby('dispatch_id').apply(lambda g: g['vehicle_length'].iloc[0] * g['vehicle_height'].iloc[0])
    result['side_wall_area_load'] = (side_load / side_cap).reindex(result['dispatch_id'].values).fillna(0.0)
    
    return result
