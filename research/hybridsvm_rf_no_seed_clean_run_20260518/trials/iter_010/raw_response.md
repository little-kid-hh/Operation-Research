## POLICY_UPDATE
- Pivot from pure spatial-constraint features to **weight-capacity pressure**, an entirely orthogonal constraint axis not yet represented in the active bank.
- Add a **per-item multi-dimensional tightness** threshold count: trees naturally isolate cases where a single item stresses 2+ vehicle axes simultaneously, which no existing feature captures.
- Introduce a **dim_l upper-tail ratio** (p90/p50) to capture long-item heterogeneity within a dispatch — distinct from the smooth `sku_std_length` and the binary `long_item_share` already in the bank.

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    result = agg_df[['dispatch_id']].copy()

    # Feature 1: load_util_ratio — total item load vs vehicle weight capacity
    weight_agg = items_df.groupby('dispatch_id').agg(
        total_load=('load_parameter', 'sum'),
        veh_cap=('vehicle_capacity', 'max')
    ).reset_index()
    weight_agg['load_util_ratio'] = weight_agg['total_load'] / (weight_agg['veh_cap'] + 1e-9)
    result = result.merge(weight_agg[['dispatch_id', 'load_util_ratio']], on='dispatch_id', how='left')

    # Feature 2: multi_dim_tight_share — fraction of items large in 2+ sorted dims vs vehicle
    items_merged = items_df[['dispatch_id', 'dim_s', 'dim_m', 'dim_l']].merge(
        agg_df[['dispatch_id', 'vehicle_length', 'vehicle_width', 'vehicle_height']],
        on='dispatch_id', how='left'
    )
    l_tight = (items_merged['dim_l'] > 0.5 * items_merged['vehicle_length']).astype(int)
    m_tight = (items_merged['dim_m'] > 0.5 * items_merged['vehicle_width']).astype(int)
    s_tight = (items_merged['dim_s'] > 0.5 * items_merged['vehicle_height']).astype(int)
    multi_tight = ((l_tight + m_tight + s_tight) >= 2).astype(int)
    multi_df = items_merged[['dispatch_id']].copy()
    multi_df['multi_tight'] = multi_tight
    multi_share = multi_df.groupby('dispatch_id')['multi_tight'].mean().reset_index()
    multi_share.columns = ['dispatch_id', 'multi_dim_tight_share']
    result = result.merge(multi_share, on='dispatch_id', how='left')

    # Feature 3: dim_l_tail_ratio — p90/p50 of longest sorted dimension per dispatch
    dim_l_p50 = items_df.groupby('dispatch_id')['dim_l'].quantile(0.5).reset_index()
    dim_l_p50.columns = ['dispatch_id', 'p50']
    dim_l_p90 = items_df.groupby('dispatch_id')['dim_l'].quantile(0.9).reset_index()
    dim_l_p90.columns = ['dispatch_id', 'p90']
    dim_l_stats = dim_l_p50.merge(dim_l_p90, on='dispatch_id')
    dim_l_stats['dim_l_tail_ratio'] = dim_l_stats['p90'] / (dim_l_stats['p50'] + 1e-9)
    result = result.merge(dim_l_stats[['dispatch_id', 'dim_l_tail_ratio']], on='dispatch_id', how='left')

    return result
```

## RATIONALE
- **load_util_ratio** opens a genuinely new constraint axis (weight vs. spatial). All 17 active-bank features are purely dimensional; weight pressure is orthogonal and directly affects feasibility, especially for dispatches that appear spatially feasible but exceed weight limits — a pattern trees can isolate via splits on `vehicle_capacity` that the linear SVM currently cannot express.
- **multi_dim_tight_share** captures per-item multi-axis stress: items whose sorted dimensions each exceed 50% of the corresponding vehicle dimension in 2+ axes simultaneously create packing conflicts that single-dimension ratios (h_to_H, l_to_L) miss. Trees recover this via depth-2 conditional splits on different ratio features; this thresholded count linearizes that structure.
- **dim_l_tail_ratio** (p90/p50) measures upper-tail heaviness of the longest-item-dimension distribution within a dispatch, complementing the smooth `sku_std_length` and binary `long_item_share`. A high ratio means a few outlier-long items coexist with shorter ones, creating arrangement difficulty that averages and binary flags smooth over — exactly the kind of local threshold pattern trees exploit.