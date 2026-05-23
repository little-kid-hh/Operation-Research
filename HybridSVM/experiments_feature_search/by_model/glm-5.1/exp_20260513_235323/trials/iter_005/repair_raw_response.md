## POLICY_UPDATE
- Target the two-dimensional coupling gap: current bank has single-dimension pressure (near_limit_long_share, p90_long) but no feature capturing items large in **both** long and mid dimensions simultaneously.
- Add a spare × p90_long interaction to linearize the nonlinear regime where low slack coincides with extreme long-dimension tail pressure — trees exploit this threshold interaction.
- Add a cube-item share signal: items with dim_s/dim_l > 0.5 resist nesting and create voids, a shape class not covered by thin_item_share or long_wide_item_share.

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    # Merge vehicle dimensions into items_df for threshold computations
    items = items_df.merge(
        agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height', 'spare_capacity']],
        on='dispatch_id',
        how='left'
    )

    # --- Feature 1: two_dim_large_share ---
    # Share of items where dim_l > 0.5*vehicle_length AND dim_m > 0.5*vehicle_width
    items['is_two_dim_large'] = (
        (items['dim_l'] > 0.5 * items['vehicle_length']) &
        (items['dim_m'] > 0.5 * items['vehicle_width'])
    ).astype(int)

    two_dim_large = (
        items.groupby('dispatch_id')['is_two_dim_large']
        .mean()
        .reset_index()
        .rename(columns={'is_two_dim_large': 'two_dim_large_share'})
    )

    # --- Feature 2: spare_x_p90long ---
    # spare_capacity * p90(dim_l / vehicle_length)
    p90_long_ratio = (
        items.groupby('dispatch_id')
        .apply(lambda g: g['dim_l'].quantile(0.9) / g['vehicle_length'].iloc[0])
        .reset_index()
        .rename(columns={0: 'p90_long_ratio'})
    )

    spare_cap = agg_df[['dispatch_id', 'spare_capacity']].copy()
    spare_x_p90long = p90_long_ratio.merge(spare_cap, on='dispatch_id')
    spare_x_p90long['spare_x_p90long'] = (
        spare_x_p90long['spare_capacity'] * spare_x_p90long['p90_long_ratio']
    )
    spare_x_p90long = spare_x_p90long[['dispatch_id', 'spare_x_p90long']]

    # --- Feature 3: cube_item_share ---
    # Share of items where dim_s / dim_l > 0.5 (cube-like items resisting nesting)
    items['is_cube'] = (
        items['dim_s'] / items['dim_l'].clip(lower=1e-9) > 0.5
    ).astype(int)

    cube_share = (
        items.groupby('dispatch_id')['is_cube']
        .mean()
        .reset_index()
        .rename(columns={'is_cube': 'cube_item_share'})
    )

    # --- Merge all features ---
    result = (
        two_dim_large
        .merge(spare_x_p90long, on='dispatch_id')
        .merge(cube_share, on='dispatch_id')
    )

    return result
```

## RATIONALE
- **two_dim_large_share** captures dimensional coupling at the item level: items large in both long and mid dimensions simultaneously consume floor space in two directions and cannot be nested around each other. This is distinct from near_limit_long_share (single-dimension) and long_wide_item_share (different threshold definition).
- **spare_x_p90long** linearizes the nonlinear interaction trees exploit: when spare_capacity is low AND the 90th-percentile item is near the vehicle length limit, feasibility drops sharply. The SVM cannot capture this product interaction from the two features alone.
- **cube_item_share** identifies items with low aspect ratio (dim_s/dim_l > 0.5) that resist nesting and create voids — a shape class orthogonal to thin_item_share (high aspect ratio) and long_wide_item_share (large in specific dimensions).