import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Merge vehicle dimensions into items_df for threshold computations
    items = items_df.merge(
        agg_df[['dispatch_id', 'vehicle_height', 'spare_capacity']],
        on='dispatch_id', how='left'
    )

    # 1. Volume dominance ratio: largest single item volume vs total dispatch volume
    top1_vol = items.groupby('dispatch_id')['item_volume'].max().rename('top1_vol')
    total_vol = items.groupby('dispatch_id')['item_volume'].sum().rename('total_vol')
    vol_dominance = (top1_vol / (total_vol + 1e-9)).rename('vol_dominance')

    # 2. Ceiling-crunch share: fraction of items with max(h_to_H) > 0.90
    items['h_ratio'] = items['dim_s'] / (items['vehicle_height'] + 1e-9)
    items['h_ratio_m'] = items['dim_m'] / (items['vehicle_height'] + 1e-9)
    items['h_ratio_l'] = items['dim_l'] / (items['vehicle_height'] + 1e-9)
    items['max_h_ratio'] = items[['h_ratio', 'h_ratio_m', 'h_ratio_l']].max(axis=1)
    items['is_ceiling_crunch'] = (items['max_h_ratio'] > 0.90).astype(int)
    ceiling_crunch_share = items.groupby('dispatch_id')['is_ceiling_crunch'].mean().rename('ceiling_crunch_share')

    # 3. Low-slack volume anchor interaction
    spare_inv = (1.0 / (agg_df['spare_capacity'] + 1e-9)).rename('spare_inv')
    vol_anchor_x_slack_inv = (vol_dominance * spare_inv).rename('vol_anchor_x_slack_inv')

    # Assemble output
    result = pd.DataFrame({
        'vol_dominance': vol_dominance,
        'ceiling_crunch_share': ceiling_crunch_share,
        'vol_anchor_x_slack_inv': vol_anchor_x_slack_inv
    })

    # Ensure dispatch_id is a column (it comes from the groupby index)
    result = result.reset_index()

    return result
