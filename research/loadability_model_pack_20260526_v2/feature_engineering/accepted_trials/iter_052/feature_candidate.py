import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Compute item footprint (smallest two dimensions)
    items_df = items_df.copy()
    items_df['item_footprint'] = items_df['dim_s'] * items_df['dim_m']
    
    dispatch_ids = agg_df['dispatch_id']

    # 1. 3D Volume Packing Density: ratio of total item volume to vehicle volume
    total_item_vol = items_df.groupby('dispatch_id')['item_volume'].sum()
    vehicle_vol = agg_df.set_index('dispatch_id')['vehicle_capacity']
    vol_packing_density = total_item_vol / (vehicle_vol + 1e-9)
    vol_packing_density = vol_packing_density.reindex(dispatch_ids).fillna(0)

    # 2. Item Footprint Floor Coverage: ratio of total item footprint area to vehicle floor area
    total_item_fp = items_df.groupby('dispatch_id')['item_footprint'].sum()
    vehicle_floor = (agg_df.set_index('dispatch_id')['vehicle_length'] * 
                     agg_df.set_index('dispatch_id')['vehicle_width'])
    floor_coverage = total_item_fp / (vehicle_floor + 1e-9)
    floor_coverage = floor_coverage.reindex(dispatch_ids).fillna(0)

    # 3. Low-slack floor saturation interaction: floor_coverage × (1 / spare_capacity)
    spare_cap = agg_df.set_index('dispatch_id')['spare_capacity']
    floor_sat_x_slack_inv = floor_coverage / (spare_cap.reindex(dispatch_ids) + 1e-9)
    
    # Replace any inf or NaN values to prevent IntCastingNaNError downstream
    floor_sat_x_slack_inv = floor_sat_x_slack_inv.replace([np.inf, -np.inf], 0).fillna(0)

    result = pd.DataFrame({
        'dispatch_id': dispatch_ids,
        'vol_packing_density': vol_packing_density.values,
        'floor_coverage': floor_coverage.values,
        'floor_sat_x_slack_inv': floor_sat_x_slack_inv.values
    })

    return result
