def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    # Get vehicle dimensions per dispatch
    veh = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    
    # Merge vehicle dims into items
    items = items_df.merge(veh, on='dispatch_id', how='left')
    
    # Feature 1: load_param_max - max load parameter per dispatch
    load_max = items.groupby('dispatch_id')['load_parameter'].max().rename('load_param_max')
    
    # Feature 2: dim_m_sum_to_vw - sum of medium dims / vehicle width
    items['dim_m_ratio_vw'] = items['dim_m'] / items['vehicle_width']
    dim_m_pressure = items.groupby('dispatch_id')['dim_m_ratio_vw'].sum().rename('dim_m_sum_to_vw')
    
    # Feature 3: awkward_floor_share - share of items where dim_m > 0.5 * min(VL, VW)
    items['min_floor_dim'] = items[['vehicle_length', 'vehicle_width']].min(axis=1)
    items['is_awkward_floor'] = (items['dim_m'] > 0.5 * items['min_floor_dim']).astype(int)
    awkward_share = items.groupby('dispatch_id')['is_awkward_floor'].mean().rename('awkward_floor_share')
    
    # Combine
    result = pd.concat([load_max, dim_m_pressure, awkward_share], axis=1).reset_index()
    
    return result
