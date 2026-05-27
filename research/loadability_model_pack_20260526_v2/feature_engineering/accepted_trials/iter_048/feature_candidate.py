def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    # Compute footprint Gini per dispatch from items_df
    def gini(series):
        vals = series.sort_values()
        n = len(vals)
        if n == 0:
            return 0.0
        cumvals = vals.cumsum()
        return (2 * cumvals.sum() / (vals.sum() * n) - (n + 1) / n)

    fp_gini = items_df.groupby('dispatch_id')['item_volume'].apply(
        lambda v: gini(v / (items_df.loc[v.index, 'dim_l'] * items_df.loc[v.index, 'dim_m'] + 1e-9))
    )
    fp_gini = fp_gini.rename('footprint_gini_calc')

    # Get vehicle dimensions and existing aggregates per dispatch
    dispatch_info = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height', 
                            'sku_counts', 'spare_capacity', 'wl_to_vehicle_wl_max']].set_index('dispatch_id')

    # Compute footprint per item residual: 
    # (vehicle floor area / sku_counts) - peak footprint monopolization proxy (wl_to_vehicle_wl_max * vehicle floor area)
    vehicle_floor_area = dispatch_info['vehicle_length'] * dispatch_info['vehicle_width']
    peak_fp_monopoly = dispatch_info['wl_to_vehicle_wl_max'] * vehicle_floor_area
    avg_fp_per_item = vehicle_floor_area / dispatch_info['sku_counts']
    fp_per_item_residual = avg_fp_per_item - peak_fp_monopoly

    # Align footprint Gini to dispatch index
    fp_gini_aligned = fp_gini.reindex(dispatch_info.index).fillna(0.0)

    # Compute interactions
    count_x_gini = dispatch_info['sku_counts'] * fp_gini_aligned
    count_x_slack_inv = dispatch_info['sku_counts'] / (dispatch_info['spare_capacity'] + 1e-6)

    # Assemble output DataFrame
    result = pd.DataFrame({
        'dispatch_id': dispatch_info.index,
        'fp_per_item_residual': fp_per_item_residual.values,
        'count_x_gini': count_x_gini.values,
        'count_x_slack_inv': count_x_slack_inv.values
    }).reset_index(drop=True)

    return result
