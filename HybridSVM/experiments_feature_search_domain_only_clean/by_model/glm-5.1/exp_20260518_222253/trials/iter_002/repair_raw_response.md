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
    long_item_share = grp['is_long'].mean()

    # Feature 3: cubic_item_share
    # Fraction of items whose aspect ratio (dim_l / dim_s) is below 2.5,
    # meaning they are near-cubic and hard to nest efficiently.
    # Cubic items waste space because they resist interleaving.
    items['aspect_ratio'] = items['dim_l'] / items['dim_s'].clip(lower=1e-6)
    items['is_cubic'] = (items['aspect_ratio'] < 2.5).astype(int)
    cubic_item_share = grp['is_cubic'].mean()

    # Assemble output
    out = pd.DataFrame({
        'dispatch_id': floor_area_pressure.index,
        'floor_area_pressure': floor_area_pressure.values,
        'long_item_share': long_item_share.values,
        'cubic_item_share': cubic_item_share.values,
    }).reset_index(drop=True)

    return out
```

## RATIONALE
- **floor_area_pressure** captures 2D floor-space competition: when the sum of minimal footprints exceeds the vehicle floor, stacking is forced, which is risky for fragile or load-constrained items. This is orthogonal to the existing `spare_capacity` (3D volume) and `tall_item_share` (height direction).
- **long_item_share** complements the existing `tall_item_share` by targeting the length axis: items spanning >60% of vehicle length constrain orientation and create dead space, a common failure mode when volume alone looks feasible.
- **cubic_item_share** identifies dispatches dominated by near-cubic items that resist efficient nesting; these items waste interstitial space that more elongated or flat items would fill, directly addressing geometry-vs-volume mismatches at low FPR.