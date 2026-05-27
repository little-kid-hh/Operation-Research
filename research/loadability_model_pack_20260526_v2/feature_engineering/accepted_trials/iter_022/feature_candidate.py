import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    items_df = items_df.copy()
    
    # Compute item-level dimension-to-vehicle ratios using agg_df vehicle dimensions
    vehicle_dims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].set_index('dispatch_id')
    items_df = items_df.set_index('dispatch_id')
    
    items_df['l_ratio'] = items_df['dim_l'] / vehicle_dims['vehicle_length']
    items_df['h_ratio'] = items_df['dim_s'] / vehicle_dims['vehicle_height']
    items_df['w_ratio'] = items_df['dim_m'] / vehicle_dims['vehicle_width']
    
    # Reset index to make dispatch_id a column again for groupby
    items_df = items_df.reset_index()

    # 1. Cross-axis tail interaction: 90th percentile length ratio * 90th percentile height ratio
    p90_l = items_df.groupby('dispatch_id')['l_ratio'].quantile(0.9)
    p90_h = items_df.groupby('dispatch_id')['h_ratio'].quantile(0.9)
    p90_l_x_p90_h = p90_l * p90_h

    # 2. Volume tail pressure: 90th percentile volume / average volume
    vol_p90 = items_df.groupby('dispatch_id')['item_volume'].quantile(0.9)
    vol_avg = items_df.groupby('dispatch_id')['item_volume'].mean()
    vol_tail_pressure = vol_p90 / vol_avg.clip(lower=1e-9)

    # 3. Low-slack multi-axis bottleneck: share of items extreme in >=2 dims * (1 / spare_capacity)
    extreme_l = items_df['l_ratio'] > 0.75
    extreme_h = items_df['h_ratio'] > 0.75
    extreme_w = items_df['w_ratio'] > 0.75
    extreme_count = (extreme_l.astype(int) + extreme_h.astype(int) + extreme_w.astype(int))
    
    multi_extreme = items_df.groupby('dispatch_id').apply(
        lambda g: (extreme_count.loc[g.index] >= 2).mean()
    )
    
    spare_inv = 1.0 / agg_df.set_index('dispatch_id')['spare_capacity'].clip(lower=1e-6)
    multi_axis_x_slack_inv = multi_extreme * spare_inv

    result = pd.DataFrame({
        'dispatch_id': agg_df['dispatch_id'],
        'p90_l_x_p90_h': p90_l_x_p90_h.reindex(agg_df['dispatch_id']).values,
        'vol_tail_pressure': vol_tail_pressure.reindex(agg_df['dispatch_id']).values,
        'multi_axis_x_slack_inv': multi_axis_x_slack_inv.reindex(agg_df['dispatch_id']).values
    })

    return result
