def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # Map vehicle dimensions
    vl = agg_df.set_index('dispatch_id')['vehicle_length']
    vw = agg_df.set_index('dispatch_id')['vehicle_width']
    vh = agg_df.set_index('dispatch_id')['vehicle_height']

    # Feature 1: three_dim_min_fill - bottleneck dimension fill rate
    dim_l_sum = items_df.groupby('dispatch_id')['dim_l'].sum()
    dim_m_sum = items_df.groupby('dispatch_id')['dim_m'].sum()
    dim_s_sum = items_df.groupby('dispatch_id')['dim_s'].sum()

    fill_l = dim_l_sum / vl
    fill_w = dim_m_sum / vw
    fill_h = dim_s_sum / vh

    three_dim_min_fill = pd.concat([fill_l, fill_w, fill_h], axis=1).min(axis=1)
    result = result.merge(
        three_dim_min_fill.rename('three_dim_min_fill').reset_index(),
        on='dispatch_id', how='left'
    )

    # Feature 2: fragile_floor_demand - floor area needed by fragile items
    fragile = items_df[items_df['if_fragile'] == 1].copy()
    fragile_floor = fragile.groupby('dispatch_id').apply(
        lambda g: (g['dim_l'] * g['dim_m']).sum()
    )
    vehicle_floor = vl * vw
    fragile_floor_demand = (fragile_floor / vehicle_floor).fillna(0)
    result = result.merge(
        fragile_floor_demand.rename('fragile_floor_demand').reset_index(),
        on='dispatch_id', how='left'
    )

    # Feature 3: dim_s_cv - coefficient of variation of smallest dimension
    dim_s_mean = items_df.groupby('dispatch_id')['dim_s'].mean()
    dim_s_std = items_df.groupby('dispatch_id')['dim_s'].std().fillna(0)
    dim_s_cv = (dim_s_std / dim_s_mean).fillna(0)
    result = result.merge(
        dim_s_cv.rename('dim_s_cv').reset_index(),
        on='dispatch_id', how='left'
    )

    return result
