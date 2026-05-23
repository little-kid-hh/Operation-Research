def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Merge vehicle dimensions into items_df
    items = items_df.merge(
        agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height', 'spare_capacity']],
        on='dispatch_id',
        how='left'
    )
    
    # 1. footprint_pressure: sum of item footprints / vehicle floor area
    items['item_footprint'] = items['dim_s'] * items['dim_m']
    items['vehicle_floor'] = items['vehicle_length'] * items['vehicle_width']
    fp = items.groupby('dispatch_id').apply(
        lambda g: g['item_footprint'].sum() / g['vehicle_floor'].iloc[0]
    ).reset_index(name='footprint_pressure')
    
    # 2. spare_cap_x_n_tall_items
    # n_tall_items is already in active bank, so we compute it here for the interaction
    items['is_tall'] = (items['item_height'] > 0.5 * items['vehicle_height']).astype(int)
    tall_counts = items.groupby('dispatch_id')['is_tall'].sum().reset_index(name='_n_tall')
    sc = agg_df[['dispatch_id', 'spare_capacity']]
    sc_x_tall = sc.merge(tall_counts, on='dispatch_id')
    sc_x_tall['spare_cap_x_n_tall_items'] = sc_x_tall['spare_capacity'] * sc_x_tall['_n_tall']
    
    # 3. n_long_tall_items: items with dim_l > 0.5*vL AND item_height > 0.5*vH
    items['is_long_tall'] = (
        (items['dim_l'] > 0.5 * items['vehicle_length']) & 
        (items['item_height'] > 0.5 * items['vehicle_height'])
    ).astype(int)
    lt = items.groupby('dispatch_id')['is_long_tall'].sum().reset_index(name='n_long_tall_items')
    
    # Merge all
    result = agg_df[['dispatch_id']].copy()
    result = result.merge(fp, on='dispatch_id', how='left')
    result = result.merge(sc_x_tall[['dispatch_id', 'spare_cap_x_n_tall_items']], on='dispatch_id', how='left')
    result = result.merge(lt, on='dispatch_id', how='left')
    
    return result
