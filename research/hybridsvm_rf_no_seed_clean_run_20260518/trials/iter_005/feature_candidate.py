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
