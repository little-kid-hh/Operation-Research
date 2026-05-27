import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Vehicle dimensions from sorted dims
    veh_l = items_df.groupby('dispatch_id')['dim_l'].transform('max')
    veh_w = items_df.groupby('dispatch_id')['dim_m'].transform('max')
    veh_h = items_df.groupby('dispatch_id')['dim_s'].transform('max')
    
    # 1. Cross-dimensional extreme item share: items long AND wide
    is_long = items_df['dim_l'] > 0.7 * veh_l
    is_wide = items_df['dim_m'] > 0.7 * veh_w
    long_wide_share = (is_long & is_wide).groupby(items_df['dispatch_id']).mean().rename('long_wide_share')
    
    # 2. Cross-dimensional extreme item share: items long AND tall
    is_tall = items_df['dim_s'] > 0.7 * veh_h
    long_tall_share = (is_long & is_tall).groupby(items_df['dispatch_id']).mean().rename('long_tall_share')
    
    # 3. Footprint slack saturation: ratio of total item footprint to vehicle footprint
    item_footprint = items_df['dim_l'] * items_df['dim_m']
    total_footprint = item_footprint.groupby(items_df['dispatch_id']).sum()
    vehicle_footprint = veh_l.groupby(items_df['dispatch_id']).first() * veh_w.groupby(items_df['dispatch_id']).first()
    footprint_slack_sat = (total_footprint / vehicle_footprint.clip(lower=1e-6)).rename('footprint_slack_sat')
    
    # 4. Volume IQR: spread of core item volumes
    vol_q75 = items_df.groupby('dispatch_id')['item_volume'].quantile(0.75)
    vol_q25 = items_df.groupby('dispatch_id')['item_volume'].quantile(0.25)
    volume_iqr = (vol_q75 - vol_q25).rename('volume_iqr')
    
    # Assemble features
    features = pd.concat([long_wide_share, long_tall_share, footprint_slack_sat, volume_iqr], axis=1).reset_index()
    return features
