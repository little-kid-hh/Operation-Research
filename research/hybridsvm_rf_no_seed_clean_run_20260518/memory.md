# Memory

Baseline reference:
- AUC `0.9651`
- TPR@FPR=1% `0.6347`
- Accuracy `0.9276`

## Accepted Trials
- iter `10`: AUC `0.9733` (Δ `+0.0082`), TPR@1% `0.6903` (Δ `+0.0556`), features=load_util_ratio, multi_dim_tight_share, dim_l_tail_ratio
- iter `9`: AUC `0.9723` (Δ `+0.0072`), TPR@1% `0.6853` (Δ `+0.0506`), features=spare_cap_x_wl_cv, near_h_limit_share, wl_total_sq
- iter `6`: AUC `0.9714` (Δ `+0.0062`), TPR@1% `0.7040` (Δ `+0.0693`), features=wl_max_to_avg, tight_x_high_wl, sku_counts_x_lL

## Rejected Trials
- iter `4`: AUC `0.9704` (Δ `+0.0053`), TPR@1% `0.7060` (Δ `+0.0713`), features=spare_cap_x_hH, floor_area_pressure, tall_item_share
- iter `7`: AUC `0.9716` (Δ `+0.0064`), TPR@1% `0.7050` (Δ `+0.0703`), features=sku_counts_x_hH, flat_item_mean, near_limit_share
- iter `8`: AUC `0.9718` (Δ `+0.0067`), TPR@1% `0.6927` (Δ `+0.0580`), features=spare_cap_x_long_item, dim_l_max_to_veh_l, wl_total_x_sku_counts

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale
