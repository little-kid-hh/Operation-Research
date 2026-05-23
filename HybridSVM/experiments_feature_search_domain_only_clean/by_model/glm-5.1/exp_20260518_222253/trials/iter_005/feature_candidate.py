def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    vL = agg_df.set_index('dispatch_id')['vehicle_length']
    vW = agg_df.set_index('dispatch_id')['vehicle_width']
    vH = agg_df.set_index('dispatch_id')['vehicle_height']

    # height_layer_pressure: sum of smallest dims / vehicle height
    # estimates how many vertical layers are needed
    dim_s_sum = items_df.groupby('dispatch_id')['dim_s'].sum()
    height_layer_pressure = (dim_s_sum / vH).reindex(agg_df['dispatch_id']).fillna(0)
    result['height_layer_pressure'] = height_layer_pressure.values

    # footprint_sum_ratio: sum(dim_l * dim_m) / (vL * vW)
    # minimum floor area needed under best-case orientation
    footprint_per_item = items_df['dim_l'] * items_df['dim_m']
    footprint_sum = items_df.groupby('dispatch_id').apply(
        lambda g: (g['dim_l'] * g['dim_m']).sum()
    )
    floor_area = vL * vW
    footprint_sum_ratio = (footprint_sum / floor_area).reindex(agg_df['dispatch_id']).fillna(0)
    result['footprint_sum_ratio'] = footprint_sum_ratio.values

    # length_sum_pressure: sum(dim_l) / vehicle_length
    # total length demand — an implicit count × avg interaction
    dim_l_sum = items_df.groupby('dispatch_id')['dim_l'].sum()
    length_sum_pressure = (dim_l_sum / vL).reindex(agg_df['dispatch_id']).fillna(0)
    result['length_sum_pressure'] = length_sum_pressure.values

    return result
