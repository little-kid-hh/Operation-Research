import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()
    
    # 1. Quadratic expansion of spare_capacity
    #    Linearizes the dominant curvature XGB exploits via 1860 splits.
    result['spare_capacity_sq'] = agg_df['spare_capacity'] ** 2
    
    # 2. Threshold indicator for critical low spare capacity
    #    Captures the hard boundary effect near p25 that trees partition on.
    p25_spare = agg_df['spare_capacity'].quantile(0.25)
    result['spare_capacity_is_low'] = (agg_df['spare_capacity'] < p25_spare).astype(np.int32)
    
    # 3. Interaction: spare capacity × concentration
    #    Approximates tree paths where high concentration shifts the spare capacity threshold.
    result['spare_x_concentration'] = agg_df['spare_capacity'] * agg_df['sku_concentration']
    
    # 4. Interaction: spare capacity × total footprint utilization
    #    Captures the joint regime where total floor pressure amplifies remaining slack necessity.
    result['spare_x_wl_total'] = agg_df['spare_capacity'] * agg_df['wl_to_vehicle_wl_total']
    
    # 5. Footprint stress residual (peak minus total utilization)
    #    Measures whether the load is dominated by a single large item's footprint vs. distributed stress.
    result['wl_max_residual'] = agg_df['wl_to_vehicle_wl_max'] - agg_df['wl_to_vehicle_wl_total']
    
    # 6. Height ceiling pressure squared
    #    Amplifies the sharp threshold effect of items approaching the vehicle height limit.
    result['h_to_H_max_sq'] = agg_df['h_to_H_ratio_max'] ** 2
    
    return result
