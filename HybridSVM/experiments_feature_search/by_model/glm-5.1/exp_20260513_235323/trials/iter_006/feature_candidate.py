def build_candidate_features(agg_df, items_df):
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # Merge vehicle dimensions onto items
    items = items_df.merge(agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height', 'spare_capacity', 'sku_average_volume']], on='dispatch_id', how='left')
    
    # Feature 1: near_limit_mid_share
    # Share of items where dim_m is near the vehicle's second dimension
    # Use vehicle_width as proxy for second-largest vehicle dimension
    items['mid_ratio'] = items['dim_m'] / items['vehicle_width']
    items['near_limit_mid'] = (items['mid_ratio'] > 0.6).astype(int)
    feat1 = items.groupby('dispatch_id')['near_limit_mid'].mean().rename('near_limit_mid_share')
    result = result.merge(feat1, on='dispatch_id', how='left')
    
    # Feature 2: spare_x_avgvol
    result['spare_x_avgvol'] = agg_df['spare_capacity'] * agg_df['sku_average_volume']
    
    # Feature 3: long_thin_item_share
    items['aspect_ratio'] = items['dim_l'] / items['dim_s'].clip(lower=1)
    items['long_thin'] = (items['aspect_ratio'] > 5).astype(int)
    feat3 = items.groupby('dispatch_id')['long_thin'].mean().rename('long_thin_item_share')
    result = result.merge(feat3, on='dispatch_id', how='left')
    
    return result
