def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # Feature 1: spare_cap_x_wl_cv
    # Interaction: low spare capacity × uneven wl utilization = acute packing stress.
    # Trees approximate this via sequential splits; the product linearizes it.
    wl_cv = agg_df['wl_to_vehicle_wl_std'] / (agg_df['wl_to_vehicle_wl_avg'] + 1e-8)
    result['spare_cap_x_wl_cv'] = agg_df['spare_capacity'] * wl_cv

    # Feature 2: near_h_limit_share
    # Fraction of items whose largest sorted dimension > 70% of vehicle height.
    # A thresholded vertical-bottleneck count that trees create naturally
    # but a linear model cannot express from smooth averages alone.
    veh_h_map = agg_df.set_index('dispatch_id')['vehicle_height']
    items_tmp = items_df[['dispatch_id', 'dim_l']].copy()
    items_tmp['vehicle_height'] = items_tmp['dispatch_id'].map(veh_h_map)
    items_tmp['near_h_limit'] = (
        items_tmp['dim_l'] > 0.7 * items_tmp['vehicle_height']
    ).astype(np.float64)
    near_h = items_tmp.groupby('dispatch_id')['near_h_limit'].mean()
    result['near_h_limit_share'] = result['dispatch_id'].map(near_h).fillna(0.0)

    # Feature 3: wl_total_sq
    # Quadratic term on wl_to_vehicle_wl_total (RF importance #2).
    # Feasibility degrades super-linearly as total wl utilization approaches 1;
    # this term captures that curvature, complementing spare_cap_sq.
    result['wl_total_sq'] = agg_df['wl_to_vehicle_wl_total'] ** 2

    return result
