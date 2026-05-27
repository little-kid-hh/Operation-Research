import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # 1. Tall item share: fraction of items with height-to-vehicle-height ratio > 0.5
    #    Linearizes XGB's sharp splits on h_to_H_ratio_max by isolating hard-to-stack vertical outliers
    tall_ratio_threshold = 0.5
    items_df['h_to_H'] = items_df['item_height'] / agg_df.set_index('dispatch_id').loc[items_df['dispatch_id'], 'vehicle_height'].values
    tall_share = (
        items_df.groupby('dispatch_id')['h_to_H']
        .apply(lambda s: (s > tall_ratio_threshold).mean())
        .reset_index(name='tall_item_share')
    )
    result = result.merge(tall_share, on='dispatch_id', how='left')

    # 2. Footprint pressure interaction: wl_to_vehicle_wl_total * (1 / (spare_capacity + 0.01))
    #    Captures the conditional boundary where XGB splits on total footprint utilization
    #    primarily when spare capacity is low (high packing stress)
    result['footprint_x_slack_inv'] = (
        agg_df['wl_to_vehicle_wl_total'] / (agg_df['spare_capacity'] + 0.01)
    )

    # 3. Vertical stacking pressure: sum of h_to_H ratios for items exceeding 0.5
    #    Proxies the cumulative vertical bottleneck; complements h_to_H_max_sq by capturing
    #    multiple moderately-tall items rather than just the single tallest outlier
    vert_pressure = (
        items_df[items_df['h_to_H'] > tall_ratio_threshold]
        .groupby('dispatch_id')['h_to_H']
        .sum()
        .reset_index(name='vert_stacking_pressure')
    )
    result = result.merge(vert_pressure, on='dispatch_id', how='left')
    result['vert_stacking_pressure'] = result['vert_stacking_pressure'].fillna(0.0)

    return result
