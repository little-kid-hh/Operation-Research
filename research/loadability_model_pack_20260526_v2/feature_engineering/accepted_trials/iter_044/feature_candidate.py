import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # 1. Footprint pressure skew: (top-1 footprint - mean footprint) / mean footprint
    #    Detects asymmetric floor monopolization by a single large base item
    fp = items_df['dim_s'] * items_df['dim_m']
    fp_mean = fp.groupby(items_df['dispatch_id']).mean()
    fp_max = fp.groupby(items_df['dispatch_id']).max()
    fp_pressure_skew = (fp_max - fp_mean) / (fp_mean + 1e-9)
    fp_pressure_skew = fp_pressure_skew.rename('fp_pressure_skew')
    
    # 2. Height heterogeneity index: std(item_height) / mean(item_height)
    #    Captures "staircase" stacking conflicts from mixed-height loads
    h_mean = items_df.groupby('dispatch_id')['item_height'].mean()
    h_std = items_df.groupby('dispatch_id')['item_height'].std()
    height_heterogeneity = h_std / (h_mean + 1e-9)
    height_heterogeneity = height_heterogeneity.rename('height_heterogeneity')
    
    # 3. Low-slack floor skew interaction: fp_pressure_skew * (1 / (spare_capacity + 1e-9))
    #    Translates tree-conditional splits where asymmetric floor monopolization
    #    causes infeasibility specifically when slack is constrained
    spare_cap = agg_df.set_index('dispatch_id')['spare_capacity']
    fp_skew_inv_idx = fp_pressure_skew.reindex(agg_df['dispatch_id'])
    spare_inv = 1.0 / (spare_cap + 1e-9)
    skew_x_slack_inv = fp_skew_inv_idx * spare_inv
    skew_x_slack_inv = skew_x_slack_inv.rename('skew_x_slack_inv')
    
    result = pd.concat([fp_pressure_skew, height_heterogeneity, skew_x_slack_inv], axis=1).reset_index()
    return result
