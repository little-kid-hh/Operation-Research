import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # Feature 1: spare_capacity × sku_counts interaction
    # Low spare AND many items → very constrained packing (nonlinear pressure)
    result['spare_x_count'] = (
        agg_df['spare_capacity'].values * agg_df['sku_counts'].values
    )

    # Feature 2: near_limit_long_share
    # Share of items whose long dim exceeds 75% of vehicle length
    veh_dims = agg_df[['dispatch_id', 'vehicle_length']].copy()
    items_with_veh = items_df.merge(veh_dims, on='dispatch_id', how='left')
    items_with_veh['is_near_limit'] = (
        items_with_veh['dim_l'] > 0.75 * items_with_veh['vehicle_length']
    ).astype(int)
    near_limit_share = items_with_veh.groupby('dispatch_id')['is_near_limit'].mean()
    result['near_limit_long_share'] = result['dispatch_id'].map(near_limit_share).fillna(0)

    # Feature 3: vol_cv - coefficient of variation of item volumes
    # High CV → mix of very large and very small items → harder to pack efficiently
    vol_stats = items_df.groupby('dispatch_id')['item_volume'].agg(
        vol_mean='mean', vol_std='std'
    )
    vol_stats['vol_cv'] = vol_stats['vol_std'] / vol_stats['vol_mean'].clip(lower=1e-9)
    result['vol_cv'] = result['dispatch_id'].map(vol_stats['vol_cv']).fillna(0)

    return result
