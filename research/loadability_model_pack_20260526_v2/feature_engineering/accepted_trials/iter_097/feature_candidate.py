def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Join vehicle dimensions to items_df for safe per-item ratio computation
    veh_dims = agg_df[['dispatch_id', 'vehicle_length']].copy()
    items_merged = items_df.merge(veh_dims, on='dispatch_id', how='left')

    # 1. Long-sorting stress index: average mismatch between sorted longest dim (dim_l) and vehicle length
    # Captures longitudinal packing pressure from the physically longest item dimension
    items_merged['l_sort_ratio'] = items_merged['dim_l'] / (items_merged['vehicle_length'] + 1e-9)
    long_sort_stress = items_merged.groupby('dispatch_id')['l_sort_ratio'].mean().reset_index()
    long_sort_stress.columns = ['dispatch_id', 'long_sort_stress']

    # 2. Sorted-length tail pressure: p90 of dim_l / vehicle_length
    # Isolates "beam" items that monopolize longitudinal space, orthogonal to l_ratio_p90 (unsorted)
    l_sort_p90 = items_merged.groupby('dispatch_id')['l_sort_ratio'].quantile(0.90).reset_index()
    l_sort_p90.columns = ['dispatch_id', 'l_sort_tail_p90']

    # 3. Longitudinal bottleneck × slack interaction
    # Translates tree-conditional splits where extreme sorted lengths cause infeasibility at constrained slack
    base_df = agg_df[['dispatch_id', 'spare_capacity']].copy()
    merged = base_df.merge(long_sort_stress, on='dispatch_id', how='left')
    merged = merged.merge(l_sort_p90, on='dispatch_id', how='left')

    # Compute interaction: long_sort_stress * (1 / (spare_capacity + 1e-9))
    # High longitudinal pressure at low slack = strong infeasibility signal
    merged['longitudinal_x_slack_inv'] = merged['long_sort_stress'] / (merged['spare_capacity'] + 1e-9)

    # Clean up: fill any NaN/inf with 0, drop helper columns
    result = merged[['dispatch_id', 'long_sort_stress', 'l_sort_tail_p90', 'longitudinal_x_slack_inv']].copy()
    result = result.fillna(0.0)
    for col in ['long_sort_stress', 'l_sort_tail_p90', 'longitudinal_x_slack_inv']:
        result[col] = result[col].replace([np.inf, -np.inf], 0.0)

    return result
