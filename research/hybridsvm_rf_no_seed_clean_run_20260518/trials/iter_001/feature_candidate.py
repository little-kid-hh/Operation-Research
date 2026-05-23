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
