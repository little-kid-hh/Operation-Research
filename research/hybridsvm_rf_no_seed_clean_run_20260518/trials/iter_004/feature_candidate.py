def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # 1. Height-pressure interaction: spare_capacity × h_to_H_ratio_avg
    # RF uses h_to_H_ratio_avg (importance 0.030); no spare_cap interaction yet
    result['spare_cap_x_hH'] = (
        agg_df['spare_capacity'] * agg_df['h_to_H_ratio_avg']
    )

    # Merge vehicle dimensions for item-level computations
    item_veh = items_df.merge(
        agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']],
        on='dispatch_id',
        how='left'
    )

    # 2. Floor area pressure: sum of minimum item footprints / vehicle floor area
    # dim_s × dim_m = smallest face (item on its largest face)
    # Captures horizontal competition distinct from volume-based spare_capacity
    item_veh['min_footprint'] = item_veh['dim_s'] * item_veh['dim_m']
    item_veh['veh_floor'] = item_veh['vehicle_length'] * item_veh['vehicle_width']
    item_veh['fp_ratio'] = item_veh['min_footprint'] / item_veh['veh_floor']

    fp = item_veh.groupby('dispatch_id')['fp_ratio'].sum().reset_index()
    fp.columns = ['dispatch_id', 'floor_area_pressure']
    result = result.merge(fp, on='dispatch_id', how='left')

    # 3. Tall item share: fraction of items with dim_l > 50% of vehicle height
    # Tall items create ceiling dead-zones blocking vertical stacking above them
    item_veh['is_tall'] = (item_veh['dim_l'] > 0.5 * item_veh['vehicle_height']).astype(int)
    ts = item_veh.groupby('dispatch_id')['is_tall'].mean().reset_index()
    ts.columns = ['dispatch_id', 'tall_item_share']
    result = result.merge(ts, on='dispatch_id', how='left')

    return result
