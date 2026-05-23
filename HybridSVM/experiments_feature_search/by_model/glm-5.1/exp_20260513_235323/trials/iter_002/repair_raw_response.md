## POLICY_UPDATE
- Focus on slack-pressure interactions: the two strongest SVM features (spare_capacity +13.6, sku_counts -10.6) suggest a nonlinear interaction when both are in adverse regimes simultaneously.
- Add a thresholded near-limit count share for the long dimension, complementing the existing p90 quantile with a hard-threshold perspective.
- Add volume heterogeneity (CV) to capture the diversity vs. repetition axis that trees exploit but the linear SVM cannot without an explicit feature.
- These three features are compact, interpretable, and target the three main tree-advantage directions: interaction structure, threshold counts, and distribution shape.

## FEATURE_CODE
```python
import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # Feature 1: spare_capacity × sku_counts interaction
    # Low spare AND many items → very constrained packing (nonlinear pressure)
    result['spare_x_count'] = (
        agg_df['spare_capacity'].values * agg_df['sku_counts'].values
    )

    # Feature 2: near_limit_long_share
    # Share of items whose long dim exceeds 75% of vehicle length
    veh_dims = agg_df[['dispatch_id', 'vehicle_length']].copy()
    items_with_veh = items_df.merge(veh_dims, on='dispatch_id', how='left')
    items_with_veh['is_near_limit'] = (
        items_with_veh['dim_l'] > 0.75 * items_with_veh['vehicle_length']
    ).astype(int)
    near_limit_share = items_with_veh.groupby('dispatch_id')['is_near_limit'].mean()
    result['near_limit_long_share'] = result['dispatch_id'].map(near_limit_share).fillna(0)

    # Feature 3: vol_cv - coefficient of variation of item volumes
    # High CV → mix of very large and very small items → harder to pack efficiently
    vol_stats = items_df.groupby('dispatch_id')['item_volume'].agg(
        vol_mean='mean', vol_std='std'
    )
    vol_stats['vol_cv'] = vol_stats['vol_std'] / vol_stats['vol_mean'].clip(lower=1e-9)
    result['vol_cv'] = result['dispatch_id'].map(vol_stats['vol_cv']).fillna(0)

    return result
```

## RATIONALE
- **spare_x_count**: The SVM's two strongest features (spare_capacity +13.6, sku_counts -10.6) have opposing directions. Trees can split on their combination (low spare + high count = very infeasible), but a linear SVM needs the explicit interaction term to capture this nonlinear pressure regime.
- **near_limit_long_share**: Complements the existing p90_long_over_bin_long (a quantile of the ratio) with a hard-threshold count-based measure. Trees benefit from threshold splits on "how many items are near the vehicle limit," which a single quantile cannot express.
- **vol_cv**: Captures volume heterogeneity — a high CV means a mix of very large and very small items, creating awkward packing situations that uniform-item aggregates miss. This addresses the "heterogeneity vs repetition" tree-guidance direction not yet covered by the active bank.