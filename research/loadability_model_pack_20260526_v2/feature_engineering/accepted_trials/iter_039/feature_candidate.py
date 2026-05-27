def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Join vehicle dimensions to items for per-item calculations
    veh = agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']].copy()
    items = items_df.merge(veh, on='dispatch_id', how='left')

    # 1. Orientation Flexibility: fraction of items where smallest dim < 50% of medium dim
    #    (can rotate to fit narrow widths, enabling better packing)
    items['can_rotate'] = (items['dim_s'] < 0.5 * items['dim_m']).astype(float)
    orientation_flex = items.groupby('dispatch_id')['can_rotate'].mean().rename('orientation_flexibility')

    # 2. Footprint Aspect Ratio Diversity: IQR of dim_l / dim_m
    #    Captures heterogeneous footprint shapes that fragment floor layout
    items['fp_aspect'] = items['dim_l'] / (items['dim_m'] + 1e-9)
    fp_q75 = items.groupby('dispatch_id')['fp_aspect'].quantile(0.75)
    fp_q25 = items.groupby('dispatch_id')['fp_aspect'].quantile(0.25)
    fp_aspect_iqr = (fp_q75 - fp_q25).rename('fp_aspect_iqr')

    # 3. Low-slack shape diversity interaction
    #    Diverse shapes cause infeasibility specifically when slack is constrained
    spare = agg_df.set_index('dispatch_id')['spare_capacity']
    shape_div_x_slack_inv = (fp_aspect_iqr / (spare + 0.01)).rename('shape_div_x_slack_inv')

    # Assemble output — use the exact Series variable names in concat
    result = pd.concat([orientation_flex, fp_aspect_iqr, shape_div_x_slack_inv], axis=1).reset_index()

    # Ensure all dispatches present
    result = agg_df[['dispatch_id']].merge(result, on='dispatch_id', how='left').fillna(0)

    return result[['dispatch_id', 'orientation_flexibility', 'fp_aspect_iqr', 'shape_div_x_slack_inv']]
