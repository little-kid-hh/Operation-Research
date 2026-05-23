## POLICY_UPDATE
- Target the stubborn TPR@FPR=1% gap by adding features that linearize the conditional structure trees exploit via depth-2 splits on wl-family dispersion conditional on spare capacity.
- Introduce a **spare_capacity × wl_cv** interaction: trees split first on spare_capacity then on wl spread; this product lets the linear SVM approximate that conditional regime.
- Add a **thresholded vertical bottleneck** signal (near_h_limit_share) from items_df: fraction of items whose largest sorted dimension exceeds 70% of vehicle height. This is a count-based signal distinct from the smooth h_to_H_ratio_avg already in the base.
- Add **wl_total_sq**: a quadratic term on the #2 RF-importance feature (wl_to_vehicle_wl_total), mirroring how spare_cap_sq already captured curvature on the #1 feature. Feasibility drops sharply as wl_total approaches 1.0.

## FEATURE_CODE
```python
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
```

## RATIONALE
- **spare_cap_x_wl_cv** targets the conditional interaction where trees first split on spare_capacity then on wl dispersion: when slack is scarce AND utilization is uneven (high cv), some SKUs dominate the length-width footprint while others are small, creating an irregular packing landscape that averages miss. This product lets the linear SVM weight that specific regime.
- **near_h_limit_share** is a genuinely new item-level threshold signal. Unlike h_to_H_ratio_avg (a smooth mean), it counts how many items nearly saturate the vertical dimension—each such item blocks a full-height column, a discrete constraint that only a thresholded count can express. It complements long_item_share (which targets the length axis) by capturing the height-axis bottleneck.
- **wl_total_sq** mirrors the successful spare_cap_sq pattern: the #2 RF feature (wl_to_vehicle_wl_total) likely has a nonlinear feasibility boundary where infeasibility accelerates sharply past ~0.8 utilization. A quadratic term lets the linear SVM approximate that curvature without needing tree depth.