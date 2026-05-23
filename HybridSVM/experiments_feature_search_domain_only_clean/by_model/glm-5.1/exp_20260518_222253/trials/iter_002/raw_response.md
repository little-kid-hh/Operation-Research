## POLICY_UPDATE
- Add geometric bottleneck signals that address packing dimensions not yet covered by the active bank: floor-space pressure, length-direction constraints, and item shape difficulty.
- The active bank covers tall items, volume tails, fragile/flat shares, and height-direction p90. Missing are: 2D floor-area demand vs. vehicle floor, length-direction long-item bottlenecks, and cubic (hard-to-nest) item prevalence.
- Target the TPR@FPR=1% gap by surfacing dispatches where volume fits but geometry fails.

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Merge vehicle dimensions to items for per-item threshold computations
    veh = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']]
    items = items_df.merge(veh, on='dispatch_id', how='left')

    # Feature 1: floor_area_pressure
    # Sum of minimal item footprints (dim_s * dim_m) divided by vehicle floor area.
    # Captures 2D floor-space demand; high values mean stacking is required,
    # which may be impossible if items are fragile or load-constrained.
    items['footprint'] = items['dim_s'] * items['dim_m']
    items['veh_floor'] = items['vehicle_length'] * items['vehicle_width']
    grp = items.groupby('dispatch_id')
    floor_area_pressure = grp['footprint'].sum() / grp['veh_floor'].first()

    # Feature 2: long_item_share
    # Fraction of items whose longest sorted dimension exceeds 60% of vehicle length.
    # Long items constrain placement orientation and block vehicle length,
    # complementing the existing tall_item_share (height direction).
    items['is_long'] = (items['dim_l'] > 0.6 * items['vehicle_length']).astype(int)
