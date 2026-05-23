## POLICY_UPDATE
- Target the **cross-dimension near-limit interaction**: when many items are near the vehicle long limit AND many are near the mid limit (possibly different items), the overall packing is doubly constrained — a nonlinear regime trees capture through successive splits.
- Add a **thresholded floor-overshoot** signal: sum of item footprints (dim_l × dim_m) relative to vehicle floor area, clipped at zero excess. This is distinct from the rejected `total_footprint_ratio` because the thresholding creates a piecewise-linear feature that only activates when stacking is forced.
- Add a **max-item volumetric bottleneck**: the largest item's volume share of the vehicle. Different from `sku_average_volume` (which is an average), this captures the extreme single-item constraint on placement.

## FEATURE_CODE
```python
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
```

## RATIONALE
- **floor_overshoot** linearizes the tree's implicit split on "is total footprint > vehicle floor?"; the ReLU-style threshold at zero means the feature is silent for easy dispatches and grows with stacking pressure for hard ones — a regime the SVM cannot reach through smooth averages alone.
- **near_limit_long_x_mid** captures a cross-dimension interaction: when a dispatch has many items near the vehicle length limit AND many near the width limit (even if they are different items), the 2D placement freedom collapses nonlinearly. Trees model this via sequential splits; the product term makes it accessible to the linear SVM.
- **max_item_vol_ratio** targets the single-item bottleneck that `sku_average_volume` (rank-1 negative weight) smooths over; an extremely large item relative to vehicle volume forces placement first and constrains all subsequent packing, a threshold effect trees exploit but the linear model misses without an explicit extreme-value feature.