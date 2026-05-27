import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    out = agg_df[['dispatch_id']].copy()

    # Logarithmic compression of spare capacity to capture XGB's concentrated splits at low values
    out['spare_capacity_log1p'] = np.log1p(agg_df['spare_capacity'].clip(lower=0))

    # Interaction of top-1 and top-2 XGB gain features (spare_capacity and sku_average_volume)
    out['spare_x_avg_vol'] = agg_df['spare_capacity'] * agg_df['sku_average_volume']

    # Tail-ratio piece pressure: fraction of items with extreme length ratios (dim_l / vehicle_length > 0.8)
    # Use vehicle_length from agg_df to avoid KeyError on items_df
    items_with_vl = items_df.merge(agg_df[['dispatch_id', 'vehicle_length']], on='dispatch_id', how='left')
    items_with_vl['l_ratio'] = items_with_vl['dim_l'] / items_with_vl['vehicle_length']
    awkward_share = (
        items_with_vl.groupby('dispatch_id')['l_ratio']
        .apply(lambda s: (s > 0.8).mean())
        .reset_index(name='awkward_length_share')
    )
    out = out.merge(awkward_share, on='dispatch_id', how='left')

    return out
