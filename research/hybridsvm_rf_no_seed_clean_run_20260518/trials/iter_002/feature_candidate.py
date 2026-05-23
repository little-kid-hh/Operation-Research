def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # 1. spare_capacity x sku_average_volume interaction
    #    sku_average_volume has the largest |w_j| in the SVM; interacting it
    #    with spare_capacity linearizes the conditional split trees make
    #    when spare_capacity is low AND items are large.
    result['spare_cap_x_sku_avg_vol'] = (
        agg_df['spare_capacity'] * agg_df['sku_average_volume']
    )

    # 2. wl max residual — gap between the largest item wl-ratio and the mean.
    #    Captures whether a single item dominates the weight-length footprint
    #    (high residual) versus a uniform distribution (low residual).
    #    RF can split on max vs avg separately; SVM needs their difference.
    result['wl_max_residual'] = (
        agg_df['wl_to_vehicle_wl_max'] - agg_df['wl_to_vehicle_wl_avg']
    )

    # 3. awkward_shape_share — fraction of items whose longest sorted dimension
    #    exceeds 3× the middle sorted dimension (dim_l / dim_m > 3).
    #    Very elongated items are hard to place and create local bottlenecks
    #    that trees capture via threshold splits on individual item shapes.
    is_awkward = (items_df['dim_l'] > 3.0 * items_df['dim_m']).astype(int)
    awkward_share = is_awkward.groupby(items_df['dispatch_id']).mean()
    awkward_df = awkward_share.to_frame('awkward_shape_share').reset_index()
    result = result.merge(awkward_df, on='dispatch_id', how='left')

    return result
