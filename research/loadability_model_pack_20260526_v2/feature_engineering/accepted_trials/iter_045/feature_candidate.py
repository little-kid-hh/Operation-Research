import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Join vehicle dimensions to items_df for threshold computations
    vehicle_dims = agg_df[['dispatch_id', 'vehicle_height', 'vehicle_width']].copy()
    items_merged = items_df.merge(vehicle_dims, on='dispatch_id', how='left')

    # 1. pillar_item_share: share of items exceeding BOTH 50% of vehicle height AND 50% of vehicle width
    # Targets "pillar" items that create localized floor AND vertical stacking conflicts
    is_pillar = (
        (items_merged['item_height'] > 0.5 * items_merged['vehicle_height']) &
        (items_merged['item_width'] > 0.5 * items_merged['vehicle_width'])
    )
    pillar_share = is_pillar.groupby(items_merged['dispatch_id']).mean().rename('pillar_item_share')

    # 2. hw_tail_interaction: p90(h_to_H) * p90(w_to_W)
    # Captures cross-dimensional tail stress where extreme height AND width items co-occur
    h_ratio = items_merged['item_height'] / items_merged['vehicle_height']
    w_ratio = items_merged['item_width'] / items_merged['vehicle_width']
    p90_h = h_ratio.groupby(items_merged['dispatch_id']).quantile(0.9)
    p90_w = w_ratio.groupby(items_merged['dispatch_id']).quantile(0.9)
    hw_tail_interaction = (p90_h * p90_w).rename('hw_tail_interaction')

    # 3. pillar_x_slack_inv: pillar_item_share * (1 / (spare_capacity + 1e-6))
    # Translates tree-conditional boundary: pillar items cause infeasibility specifically when slack is constrained
    pillar_x_slack_inv = (
        pillar_share / (agg_df.set_index('dispatch_id')['spare_capacity'] + 1e-6)
    ).rename('pillar_x_slack_inv')

    # Assemble and align to dispatch_id
    result = pd.concat([pillar_share, hw_tail_interaction, pillar_x_slack_inv], axis=1).reset_index()
    result = result.merge(agg_df[['dispatch_id']], on='dispatch_id', how='right')
    result = result.fillna(0.0)

    return result
