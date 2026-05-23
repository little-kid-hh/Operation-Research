## POLICY_UPDATE
- Shift from spare_capacity interactions (already well-covered in active bank) to **dispersion and constraint signals** that trees exploit via sequential splits on wl-family spread and item-level orientation restrictions.
- Add **wl_cv** (H3) to capture whether weight-length utilization is dominated by a single SKU or uniformly spread — a pattern trees pick up via depth-2 splits on wl_std conditional on wl_avg.
- Introduce **fragile_share** from items_df as a genuinely new direction: fragile items cannot be reoriented, tightening the effective packing search space in a way not captured by any existing aggregate.
- Extend the spare_capacity polynomial family with **spare_cap_cubed** (H1): the accepted quadratic term helped, but the TPR@1% gap suggests the feasibility boundary is steeper than quadratic near zero slack.

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # wl_cv: coefficient of variation of wl-to-vehicle ratios
    # captures whether a single SKU dominates wl utilization vs. uniform spread
    wl_cv = agg_df['wl_to_vehicle_wl_std'] / (agg_df['wl_to_vehicle_wl_avg'] + 1e-6)
    result['wl_cv'] = wl_cv

    # fragile_share: fraction of items that are fragile (cannot be reoriented)
    # constrains the effective packing search space; a new signal from items_df
    fragile_share = items_df.groupby('dispatch_id')['if_fragile'].mean()
    fragile_df = fragile_share.reset_index()
    fragile_df.columns = ['dispatch_id', 'fragile_share']
    result = result.merge(fragile_df, on='dispatch_id', how='left')
    result['fragile_share'] = result['fragile_share'].fillna(0.0)

    # spare_cap_cubed: cubic spare capacity for sharper nonlinear response near zero
    # extends the accepted spare_cap_sq with a steeper curvature in the tight-fit regime
    result['spare_cap_cubed'] = agg_df['spare_capacity'] ** 3

    return result
```

## RATIONALE
- **wl_cv** linearizes the tree pattern where RF splits on wl_std conditional on wl_avg: a high CV means one or few SKUs dominate the wl footprint (potential bottleneck), while low CV means uniform spread (easier packing). This is a pure ratio of existing base features, not duplicating any active-bank entry.
- **fragile_share** introduces an entirely new item-level constraint signal: fragile items cannot be rotated, reducing the effective orientable packing degrees of freedom. No existing feature captures orientation constraints, and trees can exploit this via splits on if_fragile prevalence within a dispatch.
- **spare_cap_cubed** extends the accepted spare_cap_sq with a steeper polynomial term. The remaining TPR@1% gap (0.707 vs RF's 0.806) suggests the feasibility boundary is sharper than quadratic near zero slack; the cubic term gives the linear SVM a lever to assign disproportionately higher infeasibility scores to extremely tight dispatches.