def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    
    result = agg_df[['dispatch_id']].copy()
    
    # 1. Quadratic spare capacity
    result['spare_cap_sq'] = agg_df['spare_capacity'] ** 2
    
    # 2. Interaction: spare capacity x concentration
    result['spare_cap_x_conc'] = agg_df['spare_capacity'] * agg_df['sku_concentration']
    
    # 3. Interaction: spare capacity x total wl utilization
    result['spare_cap_x_wl_total'] = agg_df['spare_capacity'] * agg_df['wl_to_vehicle_wl_total']
    
    # 4. Volume concentration: share of total volume in top 3 largest items
    items_sorted = items_df.sort_values(['dispatch_id', 'item_volume'], ascending=[True, False])
    items_sorted['rank_within_dispatch'] = items_sorted.groupby('dispatch_id').cumcount()
    top3_vol = items_sorted[items_sorted['rank_within_dispatch'] < 3].groupby('dispatch_id')['item_volume'].sum()
    total_vol = items_df.groupby('dispatch_id')['item_volume'].sum()
    result['vol_top3_share'] = result['dispatch_id'].map(top3_vol).fillna(0) / result['dispatch_id'].map(total_vol).replace(0, np.nan)
    result['vol_top3_share'] = result['vol_top3_share'].fillna(0)
    
    # 5. Count of high aspect-ratio items (awkward to pack)
    items_df_copy = items_df.copy()
    items_df_copy['aspect_ratio'] = items_df_copy['dim_l'] / items_df_copy['dim_s'].clip(lower=1)
    items_df_copy['is_high_asr'] = (items_df_copy['aspect_ratio'] > 5).astype(int)
    n_high_asr = items_df_copy.groupby('dispatch_id')['is_high_asr'].sum()
    result['n_high_asr_items'] = result['dispatch_id'].map(n_high_asr).fillna(0)
    
    return result
