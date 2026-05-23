def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # --- Feature 1: near_limit_short_share ---
    # Share of items whose shortest sorted dimension exceeds 70% of vehicle height.
    # Fills the gap: near_limit_long_share and near_limit_mid_share already exist.
    items = items_df.copy()
    items['short_ratio'] = items['dim_s'] / items['vehicle_capacity']
    near_limit_short = (
        items.groupby('dispatch_id')['short_ratio']
        .apply(lambda x: (x > 0.7).mean())
        .reset_index()
    )
    near_limit_short.columns = ['dispatch_id', 'near_limit_short_share']
    result = result.merge(near_limit_short, on='dispatch_id', how='left')

    # --- Feature 2: fragile_share ---
    # Mean of if_fragile per dispatch. Fragile items cannot be stacked upon,
    # creating vertical dead space — a novel signal from an unused column.
    fragile_share = (
        items.groupby('dispatch_id')['if_fragile']
        .mean()
        .reset_index()
    )
    fragile_share.columns = ['dispatch_id', 'fragile_share']
    result = result.merge(fragile_share, on='dispatch_id', how='left')

    # --- Feature 3: spare_x_fragile ---
    # Interaction: spare_capacity * fragile_share.
    # When spare is low AND fragile share is high, packing is especially hard
    # because vertical stacking is constrained. Linearizes a tree-like threshold interaction.
    spare = agg_df[['dispatch_id', 'spare_capacity']].copy()
    result = result.merge(spare, on='dispatch_id', how='left')
    result['spare_x_fragile'] = result['spare_capacity'] * result['fragile_share']
    result = result.drop(columns=['spare_capacity'])

    # Fill any NaN from merges
    result = result.fillna(0.0)

    return result
