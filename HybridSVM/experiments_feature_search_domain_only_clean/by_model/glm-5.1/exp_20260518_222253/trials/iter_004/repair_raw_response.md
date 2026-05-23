## POLICY_UPDATE
- Add the missing smallest-dimension percentile signal (`dim_s_p90_to_vh`) to complete the sorted-dimension coverage alongside existing `dim_l_p90_to_vh` and `dim_m_p90_to_vw`.
- Introduce a cross-dimension pressure feature (`two_dim_large_share`) that counts items large in **both** sorted dim_l and dim_m simultaneously — these are the true packing bottlenecks a linear SVM cannot infer from marginal percentiles alone.
- Add a shape-heterogeneity signal (`aspect_cv`) measuring the coefficient of variation of item aspect ratios (dim_l/dim_s), capturing whether a dispatch mixes rod-like and cubic items, which makes stacking harder.

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    # Merge vehicle dims onto items for per-item computations
    veh = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    it = items_df.merge(veh, on='dispatch_id', how='left')

    # 1. dim_s_p90_to_vh: 90th percentile of smallest sorted dim / vehicle height
    it['dim_s_ratio'] = it['dim_s'] / it['vehicle_height']
    dim_s_p90 = it.groupby('dispatch_id')['dim_s_ratio'].quantile(0.90).reset_index()
    dim_s_p90.columns = ['dispatch_id', 'dim_s_p90_to_vh']

    # 2. two_dim_large_share: fraction of items where dim_l > 0.5*VL AND dim_m > 0.35*VW
    it['l_large'] = (it['dim_l'] > 0.5 * it['vehicle_length']).astype(int)
    it['m_large'] = (it['dim_m'] > 0.35 * it['vehicle_width']).astype(int)
    it['both_large'] = it['l_large'] * it['m_large']
    two_dim = it.groupby('dispatch_id')['both_large'].mean().reset_index()
    two_dim.columns = ['dispatch_id', 'two_dim_large_share']

    # 3. aspect_cv: coefficient of variation of (dim_l / dim_s) across items
    it['aspect'] = it['dim_l'] / it['dim_s'].clip(lower=1e-6)
    aspect_stats = it.groupby('dispatch_id')['aspect'].agg(['mean', 'std']).reset_index()
    aspect_stats.columns = ['dispatch_id', 'aspect_mean', 'aspect_std']
    aspect_stats['aspect_cv'] = aspect_stats['aspect_std'] / aspect_stats['aspect_mean'].clip(lower=1e-6)
    aspect_cv = aspect_stats[['dispatch_id', 'aspect_cv']].copy()

    # Merge all features
    out = dim_s_p90.merge(two_dim, on='dispatch_id').merge(aspect_cv, on='dispatch_id')
    return out
```

## RATIONALE
- `dim_s_p90_to_vh` completes the sorted-dimension percentile triad (dim_s, dim_m, dim_l → vh, vw, vh). Items with thick minimum dimensions resist nesting and vertical stacking; this signal is orthogonal to the existing dim_l and dim_m percentiles because it captures the *floor* of the dimension distribution rather than the ceiling.
- `two_dim_large_share` captures cross-dimension bottleneck items that are large in two sorted dimensions simultaneously. A linear SVM cannot infer this interaction from marginal percentile features alone; these items create "wall" constraints that block efficient 3D arrangement.
- `aspect_cv` measures shape heterogeneity: when a dispatch mixes very flat/rod-like items (high aspect ratio) with near-cubic items (low aspect ratio), no single stacking strategy works well, increasing infeasibility risk beyond what mean/variance of individual dimensions convey.