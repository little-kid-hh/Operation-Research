def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    items = items_df.copy()

    # --- Feature 1: floor_overshoot ---
    # 2D floor area pressure, thresholded: only positive when footprints exceed floor
    items['footprint'] = items['dim_l'] * items['dim_m']
    fp_sum = items.groupby('dispatch_id')['footprint'].sum().rename('fp_sum')

    # --- Feature 2: near_limit_long_x_mid ---
    # Interaction of near-limit shares for long and mid dimensions
    items['is_near_limit_long'] = (items['dim_l'] > 0.9 * items['vehicle_length']).astype(int)
    items['is_near_limit_mid'] = (items['dim_m'] > 0.9 * items['vehicle_width']).astype(int)
    nl_shares = items.groupby('dispatch_id').agg(
        nl_long_share=('is_near_limit_long', 'mean'),
        nl_mid_share=('is_near_limit_mid', 'mean'),
    )

    # --- Feature 3: max_item_vol_ratio ---
    # Largest item volume / vehicle volume — single-item bottleneck
    max_vol = items.groupby('dispatch_id')['item_volume'].max().rename('max_vol')

    # Combine item-level aggregates
    item_agg = pd.concat([fp_sum, nl_shares, max_vol], axis=1)

    # Merge with dispatch-level vehicle dimensions
    result = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].merge(
        item_agg, on='dispatch_id', how='left'
    ).fillna(0)

    # Compute final features
    vehicle_floor = result['vehicle_length'] * result['vehicle_width']
    vehicle_floor = vehicle_floor.replace(0, 1e-9)
    floor_ratio = result['fp_sum'] / vehicle_floor
    result['floor_overshoot'] = np.maximum(0.0, floor_ratio - 1.0)

    result['near_limit_long_x_mid'] = result['nl_long_share'] * result['nl_mid_share']

    vehicle_vol = result['vehicle_length'] * result['vehicle_width'] * result['vehicle_height']
    vehicle_vol = vehicle_vol.replace(0, 1e-9)
    result['max_item_vol_ratio'] = result['max_vol'] / vehicle_vol

    return result[['dispatch_id', 'floor_overshoot', 'near_limit_long_x_mid', 'max_item_vol_ratio']]
