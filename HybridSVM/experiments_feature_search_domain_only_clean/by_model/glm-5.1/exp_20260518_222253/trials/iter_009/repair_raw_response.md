## POLICY_UPDATE
- Add features that capture **average stacking constraint** (complementing the existing max), **volume-weighted fragility** (large fragile items are doubly problematic), and **worst-case vertical orientation bottleneck** (items whose medium dimension approaches vehicle height force a single orientation).
- These target non-linear interactions and extreme-case signals that the linear SVM cannot derive from existing bank features alone.

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    grp = items_df.groupby('dispatch_id')

    # 1. Average load parameter — complements load_param_max in the active bank.
    #    High mean means stacking is constrained across most items, not just the worst.
    load_param_mean = grp['load_parameter'].mean().rename('load_param_mean')

    # 2. Volume fraction from fragile items — fragile_share is count-based;
    #    this weights by volume so a few large fragile items register strongly.
    total_vol = grp['item_volume'].sum()
    fragile_vol = (
        items_df.loc[items_df['if_fragile'] == 1]
        .groupby('dispatch_id')['item_volume']
        .sum()
    )
    fragile_volume_share = (fragile_vol / total_vol).fillna(0.0).rename('fragile_volume_share')

    # 3. Max medium dimension / vehicle height — if any item's dim_m is close
    #    to vehicle height it must be laid with dim_s vertical, forcing a
    #    single orientation and consuming vertical space.  Distinct from
    #    dim_m_p90_to_vw and dim_m_sum_to_vw already in the bank.
    dim_m_max = grp['dim_m'].max().rename('dim_m_max')

    # Assemble item-level aggregations
    feats = pd.concat([load_param_mean, fragile_volume_share, dim_m_max], axis=1).reset_index()

    # Merge vehicle height from agg_df to compute the ratio
    feats = feats.merge(agg_df[['dispatch_id', 'vehicle_height']], on='dispatch_id', how='left')

    feats['dim_m_max_to_vh'] = feats['dim_m_max'] / feats['vehicle_height']
    feats['dim_m_max_to_vh'] = feats['dim_m_max_to_vh'].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out = feats[['dispatch_id', 'load_param_mean', 'fragile_volume_share', 'dim_m_max_to_vh']]
    return out
```

## RATIONALE
- **load_param_mean**: The active bank has `load_param_max` (worst-case stacking constraint), but the average load parameter captures the pervasive stacking difficulty across all items. A dispatch where most items have moderate load limits is harder to pack than one where only a single item is constrained.
- **fragile_volume_share**: The existing `fragile_share` is count-based; two small fragile items and two large fragile items get the same value. Volume-weighting ensures that large fragile items—which consume top-layer space without allowing stacking above—register as a stronger infeasibility signal.
- **dim_m_max_to_vh**: When an item's medium sorted dimension approaches vehicle height, that item can only be oriented with its smallest dimension vertical, forcing a single packing orientation and consuming vertical clearance. This is a distinct bottleneck from the p90-to-width and sum-to-width signals already in the bank.