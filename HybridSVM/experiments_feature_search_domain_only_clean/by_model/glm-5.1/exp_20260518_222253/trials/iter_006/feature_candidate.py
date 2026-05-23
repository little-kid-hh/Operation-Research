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
