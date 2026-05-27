import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Join vehicle dimensions to items for physical stacking computation
    vehicle_dims = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height', 'spare_capacity']].copy()
    items_merged = items_df.merge(vehicle_dims, on='dispatch_id', how='left')
    
    # 1. Medium-dimension IQR: spread of the intermediate sorted dimension (dim_m)
    m_dim_iqr = items_merged.groupby('dispatch_id')['dim_m'].agg(
        lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).reset_index(name='m_dim_iqr')
    
    # 2. Vertical stacking efficiency: total footprint area / total volume
    # High values = flat pancake loads spreading horizontally; low values = tall/blocky loads stacking efficiently
    stacking_eff = items_merged.groupby('dispatch_id').apply(
        lambda g: pd.Series({
            'stacking_eff': g['item_volume'].sum() / (g['dim_s'] * g['dim_m']).sum()
        })
    ).reset_index()
    
    # 3. Low-slack stacking inefficiency interaction
    # Inverse stacking efficiency * inverse spare capacity
    stacking_eff_merged = stacking_eff.merge(agg_df[['dispatch_id', 'spare_capacity']], on='dispatch_id', how='left')
    stacking_eff_merged['stacking_ineff_x_slack_inv'] = (
        (1.0 / (stacking_eff_merged['stacking_eff'] + 1e-9)) * 
        (1.0 / (stacking_eff_merged['spare_capacity'] + 1e-9))
    )
    
    # Assemble final DataFrame
    result = m_dim_iqr.merge(stacking_eff_merged[['dispatch_id', 'stacking_eff', 'stacking_ineff_x_slack_inv']], on='dispatch_id', how='left')
    
    # Robust fill for NaN/inf
    result = result.fillna(0.0)
    for col in result.columns:
        if col != 'dispatch_id':
            result[col] = result[col].replace([np.inf, -np.inf], 0.0)
    
    return result
