import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Vehicle dimensions from agg_df
    v_width = agg_df.set_index('dispatch_id')['vehicle_width']
    
    # Sorted medium dimension (dim_m) represents the item's second-longest physical extent
    # When dim_m is large relative to vehicle width, items create lateral stacking bottlenecks
    m_dim = items_df['dim_m']
    
    # 1. Width-sorting stress index: ratio of average sorted dim_m to vehicle width
    # Captures the physical mismatch between items' intermediate dimensions and lateral capacity
    m_dim_avg = items_df.groupby('dispatch_id')['dim_m'].mean()
    width_sort_stress = (m_dim_avg / v_width).fillna(0)
    width_sort_stress = width_sort_stress.replace([np.inf, -np.inf], 0)
    
    # 2. Sorted-width tail pressure: 90th percentile of dim_m / vehicle_width
    # Isolates extreme lateral stress from "chubby" items that resist width-wise rotation
    m_dim_p90 = items_df.groupby('dispatch_id')['dim_m'].quantile(0.9)
    width_tail_p90 = (m_dim_p90 / v_width).fillna(0)
    width_tail_p90 = width_tail_p90.replace([np.inf, -np.inf], 0)
    
    # 3. Lateral bottleneck × slack interaction
    # Translates tree-conditional boundaries where extreme sorted widths cause infeasibility
    # specifically when slack is constrained
    slack_inv = 1.0 / (agg_df.set_index('dispatch_id')['spare_capacity'] + 1e-9)
    slack_inv = slack_inv.replace([np.inf, -np.inf], 0)
    
    lateral_x_slack_inv = (width_tail_p90 * slack_inv).fillna(0)
    lateral_x_slack_inv = lateral_x_slack_inv.replace([np.inf, -np.inf], 0)
    
    # Assemble output
    result = pd.DataFrame({
        'width_sort_stress': width_sort_stress,
        'width_tail_p90': width_tail_p90,
        'lateral_x_slack_inv': lateral_x_slack_inv
    })
    
    # Ensure dispatch_id is a column, not index
    result = result.reset_index()
    
    # Final NaN/inf cleanup
    result = result.fillna(0)
    result = result.replace([np.inf, -np.inf], 0)
    
    return result
