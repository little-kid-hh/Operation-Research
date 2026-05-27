import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # 1. Dominant item footprint share (max item footprint / sum of all item footprints)
    items_df['item_footprint'] = items_df['dim_s'] * items_df['dim_m']
    footprint_sums = items_df.groupby('dispatch_id')['item_footprint'].sum()
    footprint_maxs = items_df.groupby('dispatch_id')['item_footprint'].max()
    dominant_footprint_share = (footprint_maxs / footprint_sums).fillna(0.0)

    # 2. Tower-block count: items taking >50% of vehicle length AND height simultaneously
    # Use agg_df for vehicle dimensions to avoid missing keys in items_df
    merged_items = items_df.merge(agg_df[['dispatch_id', 'vehicle_length', 'vehicle_height']], on='dispatch_id', how='left')
    merged_items['is_tower_block'] = ((merged_items['dim_l'] > 0.5 * merged_items['vehicle_length']) & 
                                      (merged_items['dim_l'] > 0.5 * merged_items['vehicle_height'])).astype(int)
    tower_block_count = merged_items.groupby('dispatch_id')['is_tower_block'].sum()

    # 3. Length ratio tail pressure: 90th percentile of l_to_L_ratio
    # Compute l_to_L_ratio from item dimensions and the joined vehicle_length
    merged_items['l_to_L_ratio'] = merged_items['dim_l'] / merged_items['vehicle_length']
    l_ratio_p90 = merged_items.groupby('dispatch_id')['l_to_L_ratio'].quantile(0.9).fillna(0.0)

    # Assemble the new features
    result = pd.DataFrame({
        'dispatch_id': agg_df['dispatch_id'],
        'dominant_footprint_share': dominant_footprint_share.reindex(agg_df['dispatch_id'], fill_value=0.0).values,
        'tower_block_count': tower_block_count.reindex(agg_df['dispatch_id'], fill_value=0).values,
        'l_ratio_p90': l_ratio_p90.reindex(agg_df['dispatch_id'], fill_value=0.0).values
    })

    return result
