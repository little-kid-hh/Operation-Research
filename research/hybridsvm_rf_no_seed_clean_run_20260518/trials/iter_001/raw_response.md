## POLICY_UPDATE
- Focus on the TPR@FPR=1% gap (0.17 below RF): the SVM misses hard positives because it cannot express nonlinear interactions and threshold effects on the top features.
- Priority 1: explicit interactions between `spare_capacity` and the next-highest RF features (`wl_to_vehicle_wl_total`, `sku_counts`) to linearize the conditional splits trees make.
- Priority 2: a threshold-based item-level signal (`big_piece_share`) that trees create naturally via binary splits but SVM cannot.
- Priority 3: cross-dimension tightness product (`hH × lL`) and a quadratic spare-capacity term to capture curvature near the decision boundary.

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # H2: spare_capacity interactions with top RF features
    result['spare_cap_x_wl_total'] = (
        agg_df['spare_capacity'] * agg_df['wl_to_vehicle_wl_total']
    )
    result['spare_cap_x_sku_counts'] = (
        agg_df['spare_capacity'] * agg_df['sku_counts']
    )

    # H1: nonlinear curvature on the single most important feature
    result['spare_cap_sq'] = agg_df['spare_capacity'] ** 2

    # H4: multi-axis tightness — high only when BOTH vertical and longitudinal are tight
    result['hH_x_lL'] = (
        agg_df['h_to_H_ratio_avg'] * agg_df['l_to_L_ratio_avg']
    )

    # Threshold count from item-level data: share of items whose longest
    # sorted dimension exceeds half the vehicle length (bottleneck pieces)
    vl_map = agg_df[['dispatch_id', 'vehicle_length']]
    items_merged = items_df.merge(vl_map, on='dispatch_id', how='left')
    items_merged['is_big'] = (
        items_merged['dim_l'] > 0.5 * items_merged['vehicle_length']
    ).astype(int)
    big_share = (
        items_merged.groupby('dispatch_id')['is_big']
        .mean()
        .reset_index()
    )
    big_share.columns = ['dispatch_id', 'big_piece_share']
    result = result.merge(big_share, on='dispatch_id', how='left')
    result['big_piece_share'] = result['big_piece_share'].fillna(0.0)

    return result
```

## RATIONALE
- **spare_cap_x_wl_total**: The two highest-importance RF features (31% + 14%) interact conditionally in tree splits; the product lets the linear SVM assign a separate weight to the joint high-pressure regime.
- **spare_cap_x_sku_counts**: Many SKUs with low spare capacity is nonlinearly harder to pack than either factor alone; this interaction directly encodes that tree-observed pattern.
- **spare_cap_sq**: RF can partition spare_capacity into step-wise regions; the quadratic term approximates the curvature near the feasibility boundary where the marginal effect of additional slack changes.
- **hH_x_lL**: A dispatch is hardest to pack when *both* vertical and longitudinal ratios are high simultaneously; the product is large only in that joint-tightness corner, which trees capture via sequential axis splits.
- **big_piece_share**: Trees naturally threshold individual item dimensions (e.g., "is any item > 50% of vehicle length?"); this per-dispatch share linearizes that binary-split signal for the SVM, directly flagging footprint bottlenecks that averages and variances miss.