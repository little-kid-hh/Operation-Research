# Memory

Baseline reference:
- AUC `0.9651`
- TPR@FPR=1% `0.6347`
- Accuracy `0.9276`

## Accepted Trials
- iter `8`: AUC `0.9731` (Δ `+0.0079`), TPR@1% `0.6726` (Δ `+0.0379`), features=dim_l_max_to_vl, width_dominant_share, dual_dominant_share
- iter `7`: AUC `0.9726` (Δ `+0.0074`), TPR@1% `0.6662` (Δ `+0.0315`), features=load_param_max, dim_m_sum_to_vw, awkward_floor_share
- iter `6`: AUC `0.9719` (Δ `+0.0068`), TPR@1% `0.6534` (Δ `+0.0187`), features=pressure_product, max_dim_l_to_min_floor, count_vol_fill_interaction

## Rejected Trials
- iter `2`: AUC `0.9691` (Δ `+0.0040`), TPR@1% `0.6264` (Δ `-0.0084`), features=floor_area_pressure, long_item_share, cubic_item_share
- iter `4`: AUC `0.9708` (Δ `+0.0057`), TPR@1% `0.6308` (Δ `-0.0039`), features=dim_s_p90_to_vh, two_dim_large_share, aspect_cv
- iter `9`: AUC `0.9730` (Δ `+0.0079`), TPR@1% `0.6740` (Δ `+0.0393`), features=load_param_mean, fragile_volume_share, dim_m_max_to_vh

## Failed Trials
- iter `10`: TypeError: Index(...) must be called with a collection of some kind, 'fragile_floor_demand' was passed

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale
