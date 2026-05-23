def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # Get vehicle_length per dispatch
    vl = agg_df.set_index('dispatch_id')['vehicle_length']
    
    # 1. n_large_dim_l: count of items where dim_l > 0.65 * vehicle_length
    items_with_vl = items_df.merge(agg_df[['dispatch_id', 'vehicle_length']], on='dispatch_id', how='left')
    items_with_vl['is_large'] = (items_with_vl['dim_l'] > 0.65 * items_with_vl['vehicle_length']).astype(int)
    n_large = items_with_vl.groupby('dispatch_id')['is_large'].sum().rename('n_large_dim_l')
    result = result.merge(n_large.reset_index(), on='dispatch_id', how='left')
    
    # 2. vol_top2_ratio: fraction of total volume in the 2 largest items
    items_sorted = items_df.sort_values(['dispatch_id', 'item_volume'], ascending=[True, False])
    top2 = items_sorted.groupby('dispatch_id').head(2)
    top2_vol = top2.groupby('dispatch_id')['item_volume'].sum()
    total_vol = items_df.groupby('dispatch_id')['item_volume'].sum()
    vol_ratio = (top2_vol / total_vol).fillna(0).rename('vol_top2_ratio')
    result = result.merge(vol_ratio.reset_index(), on='dispatch_id', how='left')
    
    # 3. flat_frac: fraction of items that are flat
    items_flat = (items_df['item_flatness'] > 5).astype(int)
    flat_frac = items_flat.groupby(items_df['dispatch_id']).mean().rename('flat_frac')
    result = result.merge(flat_frac.reset_index(), on='dispatch_id', how='left')
    
    # 4. max_dim_l_vratio: max dim_l / vehicle_length
    max_diml = items_df.groupby('dispatch_id')['dim_l'].max()
    max_diml_vratio = (max_diml / vl).rename('max_dim_l_vratio')
    result = result.merge(max_diml_vratio.reset_index(), on='dispatch_id', how='left')
    
    # 5. spare_cap_x_nolarge: spare_capacity * (1 - n_large_dim_l / sku_counts)
    result = result.set_index('dispatch_id')
    spare = agg_df.set_index('dispatch_id')['spare_capacity']
    counts = agg_df.set_index('dispatch_id')['sku_counts']
    n_large_idx = n_large  # already indexed by dispatch_id
    result['spare_cap_x_nolarge'] = spare * (1 - n_large_idx / counts)
    result = result.reset_index()
    
    # Fill NaN
    result = result.fillna(0)
    
    return result
