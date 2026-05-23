# Memory

Baseline reference:
- AUC `0.9651`
- TPR@FPR=1% `0.6347`
- Accuracy `0.9276`

## Accepted Trials
- iter `10`: AUC `0.9765` (Δ `+0.0113`), TPR@1% `0.7458` (Δ `+0.1111`), features=three_dim_near_limit_share, vol_top3_concentration, spare_x_wlstd
- iter `6`: AUC `0.9758` (Δ `+0.0106`), TPR@1% `0.7665` (Δ `+0.1318`), features=near_limit_mid_share, spare_x_avgvol, long_thin_item_share
- iter `5`: AUC `0.9758` (Δ `+0.0106`), TPR@1% `0.7463` (Δ `+0.1116`), features=two_dim_large_share, spare_x_p90long, cube_item_share

## Rejected Trials
- iter `1`: AUC `0.9719` (Δ `+0.0067`), TPR@1% `0.7070` (Δ `+0.0723`), features=volume_top3_share, multi_dim_near_limit_share, side_wall_area_load
- iter `4`: AUC `0.9749` (Δ `+0.0097`), TPR@1% `0.7119` (Δ `+0.0772`), features=spare_x_wlmax, total_footprint_ratio, cross_dim_pressure_count
- iter `8`: AUC `0.9758` (Δ `+0.0106`), TPR@1% `0.7665` (Δ `+0.1318`), features=near_limit_short_share, fragile_share, spare_x_fragile

## Failed Trials
- iter `7`: ValueError: Candidate code must define build_candidate_features
- iter `9`: KeyError: 'vehicle_length'

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale
