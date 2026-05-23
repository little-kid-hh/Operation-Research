def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # Feature 1: spare_vol_per_item - pressure-slack interaction
    result['spare_vol_per_item'] = agg_df['spare_capacity'] / agg_df['sku_counts']
    
    # Feature 2: bulky_item_share - items large in smallest sorted dimension
    # relative to smallest vehicle dimension
    items = items_df.copy()
    items['min_vh_dim'] = items.groupby('dispatch_id')['vehicle_capacity'].transform('first')
    # Actually, vehicle_capacity might not be the dimension. Let me use agg_df to get vehicle dims.
    
    # Need to merge vehicle dimensions from agg_df
    vh_dims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    vh_dims['min_vh_dim'] = vh_dims[['vehicle_length', 'vehicle_width', 'vehicle_height']].min(axis=1)
    vh_dims['vh_width'] = vh_dims['vehicle_width']
    
    items = items_df.merge(vh_dims[['dispatch_id', 'min_vh_dim', 'vh_width']], on='dispatch_id', how='left')
    
    # Bulky items: dim_s > 0.5 * min vehicle dimension
    items['is_bulky'] = (items['dim_s'] > 0.5 * items['min_vh_dim']).astype(int)
    bulky_share = items.groupby('dispatch_id')['is_bulky'].mean()
    result = result.merge(bulky_share.rename('bulky_item_share'), on='dispatch_id', how='left')
    
    # Feature 3: dim_m_p90_to_vw - 90th percentile of medium sorted dim / vehicle width
    dim_m_p90 = items.groupby('dispatch_id')['dim_m'].quantile(0.9)
    result = result.merge(dim_m_p90.rename('dim_m_p90'), on='dispatch_id', how='left')
    result['dim_m_p90_to_vw'] = result['dim_m_p90'] / result.merge(
        agg_df[['dispatch_id', 'vehicle_width']], on='dispatch_id', how='left'
    )['vehicle_width']
    result = result.drop(columns=['dim_m_p90'])
    
    return result
