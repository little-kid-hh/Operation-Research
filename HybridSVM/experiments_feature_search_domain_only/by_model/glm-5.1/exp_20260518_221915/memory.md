# Memory

Baseline reference:
- AUC `0.9651`
- TPR@FPR=1% `0.6347`
- Accuracy `0.9276`

## Accepted Trials
- iter `1`: AUC `0.9688` (Δ `+0.0037`), TPR@1% `0.6853` (Δ `+0.0506`), features=n_large_dim_l, vol_top2_ratio, flat_frac, max_dim_l_vratio, spare_cap_x_nolarge
- iter `3`: AUC `0.9688` (Δ `+0.0037`), TPR@1% `0.6917` (Δ `+0.0570`), features=cross_sect_pressure, n_half_width, max_face_vratio

## Rejected Trials
- iter `2`: AUC `0.9680` (Δ `+0.0029`), TPR@1% `0.6608` (Δ `+0.0261`), features=dim_l_occupancy_sum, fragile_vol_share, needle_frac, dim_m_tail_ratio

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale
