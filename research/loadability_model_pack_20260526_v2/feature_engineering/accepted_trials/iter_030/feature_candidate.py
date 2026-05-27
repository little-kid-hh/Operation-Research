import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Extract vehicle dimensions from agg_df
    v_w = agg_df.set_index('dispatch_id')['vehicle_width']
    v_h = agg_df.set_index('dispatch_id')['vehicle_height']
    v_l = agg_df.set_index('dispatch_id')['vehicle_length']
    
    # Map vehicle dimensions to items_df
    items_w = items_df['dispatch_id'].map(v_w)
    items_h = items_df['dispatch_id'].map(v_h)
    
    # 1. orientation_inflexibility: share of items where BOTH dim_m and dim_l exceed vehicle_width,
    # meaning the item cannot rotate to fit the width constraint (only fits length-wise).
    # Higher values -> fewer rotation options -> harder packing.
    inflexible_mask = (items_df['dim_m'] > items_w) & (items_df['dim_l'] > items_w)
    orientation_inflexibility = inflexible_mask.groupby(items_df['dispatch_id']).mean()
    
    # 2. stacking_conflict_score: measures the probability of vertical stacking collision.
    # Computed as the probability that two randomly sampled items both exceed 40% of vehicle height.
    # This proxies the physical constraint that tall items cannot stack on top of each other.
    tall_mask = items_df['dim_l'] > (0.4 * items_h)
    tall_prob = tall_mask.groupby(items_df['dispatch_id']).mean()
    stacking_conflict_score = tall_prob ** 2
    
    # 3. rotation_x_slack_inv: interaction of orientation inflexibility with inverse spare capacity.
    # Captures the conditional tree boundary where rigid items cause infeasibility specifically 
    # when vehicle slack is constrained.
    spare_cap = agg_df.set_index('dispatch_id')['spare_capacity']
    rotation_x_slack_inv = orientation_inflexibility / (spare_cap + 1e-9)
    
    # Assemble the final DataFrame
    dispatch_ids = agg_df['dispatch_id']
    features_df = pd.DataFrame({
        'dispatch_id': dispatch_ids,
        'orientation_inflexibility': orientation_inflexibility.reindex(dispatch_ids).fillna(0.0).values,
        'stacking_conflict_score': stacking_conflict_score.reindex(dispatch_ids).fillna(0.0).values,
        'rotation_x_slack_inv': rotation_x_slack_inv.reindex(dispatch_ids).fillna(0.0).values
    })
    
    return features_df
