import pandas as pd
import numpy as np

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    # Merge vehicle dimensions into items_df for ratio calculations
    items_merged = items_df.merge(
        agg_df[['dispatch_id', 'vehicle_width', 'vehicle_height', 'spare_capacity']],
        on='dispatch_id',
        how='left'
    )
    
    # 1. wide_item_share: Fraction of items with dim_l exceeding 60% of vehicle width
    # Thresholded count linearizing XGB splits on sku_max_width / w_to_W_ratio
    items_merged['wide_flag'] = (items_merged['dim_l'] > 0.6 * items_merged['vehicle_width']).astype(int)
    wide_share = items_merged.groupby('dispatch_id')['wide_flag'].mean()
    
    # 2. footprint_cv: Coefficient of variation of item footprints (dim_s * dim_m)
    # Captures footprint heterogeneity; high CV means mixed large/small bases creating stacking gaps
    items_merged['item_footprint'] = items_merged['dim_s'] * items_merged['dim_m']
    fp_mean = items_merged.groupby('dispatch_id')['item_footprint'].mean()
    fp_std = items_merged.groupby('dispatch_id')['item_footprint'].std().fillna(0.0)
    footprint_cv = (fp_std / fp_mean.replace(0, np.nan)).fillna(0.0)
    
    # 3. width_pressure_x_low_slack: Interaction of wide_item_share with constrained slack
    # Translates XGB conditional split: wide items are critical bottlenecks only when slack is low
    # spare_capacity is already in agg_df, so we compute directly on dispatch level
    width_pressure_x_low_slack = wide_share * (1.0 / (1.0 + agg_df.set_index('dispatch_id')['spare_capacity']))
    
    # Assemble the new features into a per-dispatch DataFrame
    feat_df = pd.DataFrame({
        'wide_item_share': wide_share,
        'footprint_cv': footprint_cv,
        'width_pressure_x_low_slack': width_pressure_x_low_slack
    }).reset_index(drop=False)
    
    # Ensure dispatch_id is properly aligned and all dispatches are present
    feat_df = agg_df[['dispatch_id']].merge(feat_df, on='dispatch_id', how='left')
    feat_df = feat_df.fillna(0.0)
    
    return feat_df
