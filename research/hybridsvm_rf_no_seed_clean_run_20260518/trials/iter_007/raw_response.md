## POLICY_UPDATE
- Shift toward **genuinely new item-level signals** not yet represented in the active bank: `item_flatness` and `load_parameter` from `items_df` carry packing-ease and stacking-constraint information that no existing aggregate or interaction captures.
- Add the **sku_counts × h_to_H** interaction (H5, untried) — distinct from the rejected `spare_cap × hH` because it isolates "many items competing for vertical space" regardless of overall slack.
- Introduce a **threshold-based length-bottleneck share** (`near_limit_share`) using dim_l vs. vehicle_length at 75%, linearizing the kind of depth-1 split trees make on long-item prevalence.

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    result = agg_df[['dispatch_id']].copy()

    # 1. Vertical pressure: many items competing for height space
    result['sku_counts_x_hH'] = (
        agg_df['sku_counts'] * agg_df['h_to_H_ratio_avg']
    )

    # 2. Mean item flatness: flat items can be stacked, easing packing
    flat_mean = items_df.groupby('dispatch_id')['item_flatness'].mean()
    result['flat_item_mean'] = result['dispatch_id'].map(flat_mean).fillna(0)

    # 3. Share of items whose longest sorted dim exceeds 75% of vehicle length
    merged = items_df