import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Flatness tail interaction: share of items with extreme flatness AND extreme length ratio
    items_df['l_ratio'] = items_df['dim_l'] / agg_df.set_index('dispatch_id')['vehicle_length']
    flat_extreme_mask = (items_df['item_flatness'] > 5.0) & (items_df['l_ratio'] > 0.5)
    flat_extreme_share = flat_extreme_mask.groupby(items_df['dispatch_id']).mean().rename('flat_extreme_share')

    # Slack-saturation quadratic: nonlinear steepening of infeasibility at high footprint utilization
    wl_total = agg_df.set_index('dispatch_id')['wl_to_vehicle_wl_total']
    slack_sat_quad = (wl_total ** 2).rename('slack_sat_quad')

    # Cross-axis peak interaction: simultaneous peak footprint and peak vertical stress
    wl_max = agg_df.set_index('dispatch_id')['wl_to_vehicle_wl_max']
    h_max = agg_df.set_index('dispatch_id')['h_to_H_ratio_max']
    wl_max_x_h_max = (wl_max * h_max).rename('wl_max_x_h_max')

    result = pd.concat([flat_extreme_share, slack_sat_quad, wl_max_x_h_max], axis=1).reset_index()
    result.columns = ['dispatch_id', 'flat_extreme_share', 'slack_sat_quad', 'wl_max_x_h_max']
    return result
