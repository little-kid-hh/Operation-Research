## POLICY_UPDATE
- Add non-linear interaction features that the linear SVM cannot capture on its own
- Focus on the multiplicative pressure interaction between vertical stacking and floor coverage
- Introduce a worst-case single-item bottleneck against the tighter vehicle floor dimension
- Encode the compounding difficulty of high item count combined with high volume fill

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    # Vehicle dimensions from agg_df (not items_df)
    if 'dispatch_id' in agg_df.columns:
        vdims = agg_df.set_index('dispatch_id')[['vehicle_length', 'vehicle_width', 'vehicle_height']]
    else:
        vdims = agg_df[['vehicle_length', 'vehicle_width', 'vehicle_height']]

    # Item-level computations
    items = items_df.copy()
    items['footprint'] = items['dim_m'] * items['dim_l']

    grp = items.groupby('dispatch_id')
    stats = grp.agg(
        n_items=('dim_s', 'count'),
        total_volume=('item_volume', 'sum'),
        max_dim_l=('dim_l', 'max'),
        sum_dim_s=('dim_s', 'sum'),
        sum_footprint=('footprint', 'sum'),
    )

    # Merge vehicle dims
    stats = stats.join(vdims, how='left')

    # Derived vehicle quantities
    stats['vehicle_vol'] = stats['vehicle_length'] * stats['vehicle_width'] * stats['vehicle_height']
    stats['vehicle_floor'] = stats['vehicle_length'] * stats['vehicle_width']
    stats['vehicle_min_floor_dim'] = stats[['vehicle_length', 'vehicle_width']].min(axis=1)

    # Feature 1: pressure_product — multiplicative interaction of vertical and floor pressure
    # The linear SVM cannot combine height_layer_pressure and footprint_sum_ratio
    # non-linearly; this explicit product captures compounding packing difficulty.
    stats['height_press'] = stats['sum_dim_s'] / stats['vehicle_height']
    stats['footprint_press'] = stats['sum_footprint'] / stats['vehicle_floor']
    stats['pressure_product'] = stats['height_press'] * stats['footprint_press']

    # Feature 2: max_dim_l_to_min_floor — worst-case single-item length vs tighter
    # floor dimension. Captures orientation-constrained items that can only fit one way.
    stats['max_dim_l_to_min_floor'] = stats['max_dim_l'] / stats['vehicle_min_floor_dim']

    # Feature 3: count_vol_fill_interaction — item count × volume fill rate.
    # Many items at high fill is exponentially harder; the linear model misses this.
    stats['count_vol_fill_interaction'] = stats['n_items'] * (stats['total_volume'] / stats['vehicle_vol'])

    result = stats[['pressure_product', 'max_dim_l_to_min_floor', 'count_vol_fill_interaction']].copy()
    result.index.name = 'dispatch_id'
    result = result.reset_index()
    return result
```

## RATIONALE
- **pressure_product**: The active bank already has `height_layer_pressure` and `footprint_sum_ratio` as separate linear terms, but packing difficulty is multiplicative—when both vertical and floor pressure are high simultaneously, feasibility drops far more than either alone would predict. The explicit product lets the linear SVM capture this non-linear interaction.
- **max_dim_l_to_min_floor**: Unlike `dim_l_p90_to_vh` (p90 of longest dim vs height), this feature targets the single largest item's longest dimension against the tighter vehicle floor dimension. An item whose longest side exceeds the shorter floor dimension can only be placed in one orientation, severely constraining the packing search.
- **count_vol_fill_interaction**: The SVM gives large negative weights to both `sku_counts` and `sku_average_volume` independently, but their interaction is missing. High item count × high volume fill creates combinatorial explosion in arrangement difficulty; this product encodes that compounding effect directly.