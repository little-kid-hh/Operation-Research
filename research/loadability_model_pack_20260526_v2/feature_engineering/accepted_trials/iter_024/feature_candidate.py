import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Vehicle dimensions for ratio computations
    vl = agg_df.set_index('dispatch_id')['vehicle_length']
    vw = agg_df.set_index('dispatch_id')['vehicle_width']
    vh = agg_df.set_index('dispatch_id')['vehicle_height']
    sc = agg_df.set_index('dispatch_id')['spare_capacity']

    # Item footprint area (using unsorted dims for physical floor projection)
    items_df['item_footprint'] = items_df['item_length'] * items_df['item_width']

    # --- Feature 1: footprint_gini ---
    # Gini coefficient of item footprints per dispatch
    # Measures floor-space monopolization; high Gini = few items dominate floor
    def gini(series):
        if len(series) == 0:
            return 0.0
        sorted_vals = series.sort_values()
        n = len(sorted_vals)
        cumsum = sorted_vals.cumsum()
        return (2.0 * cumsum.sum() / (n * sorted_vals.sum() + 1e-9)) - (n + 1.0) / n

    footprint_gini = items_df.groupby('dispatch_id')['item_footprint'].agg(gini)
    footprint_gini = footprint_gini.reindex(agg_df['dispatch_id']).fillna(0.0)

    # --- Feature 2: cross_axis_peak_ratio ---
    # Ratio of wl_max * h_max to wl_total, detecting "spiky" loads
    # where a single item stresses both floor and ceiling disproportionately
    wl_max = agg_df.set_index('dispatch_id')['wl_to_vehicle_wl_max']
    h_max = items_df.groupby('dispatch_id')['item_height'].max() / vh
    h_max = h_max.reindex(agg_df['dispatch_id']).fillna(0.0)
    wl_total = agg_df.set_index('dispatch_id')['wl_to_vehicle_wl_total']

    cross_axis_peak_ratio = (wl_max * h_max) / (wl_total + 1e-9)
    cross_axis_peak_ratio = cross_axis_peak_ratio.reindex(agg_df['dispatch_id']).fillna(0.0)

    # --- Feature 3: gini_x_slack_inv ---
    # Interaction: footprint fragmentation * inverse spare capacity
    # Translates tree-conditional splits where floor fragmentation
    # causes infeasibility specifically when slack is constrained
    gini_x_slack_inv = footprint_gini / (sc + 1e-9)
    gini_x_slack_inv = gini_x_slack_inv.reindex(agg_df['dispatch_id']).fillna(0.0)

    # Assemble output
    result = pd.DataFrame({
        'dispatch_id': agg_df['dispatch_id'],
        'footprint_gini': footprint_gini.values,
        'cross_axis_peak_ratio': cross_axis_peak_ratio.values,
        'gini_x_slack_inv': gini_x_slack_inv.values,
    })

    return result
