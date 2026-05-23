# Memory

Baseline reference:
- AUC `0.9651`
- TPR@FPR=1% `0.6347`
- Accuracy `0.9276`

## Accepted Trials
- iter `14`: AUC `0.9745` (Δ `+0.0094`), TPR@1% `0.7207` (Δ `+0.0860`), features=q90_l_to_L, spare_cap_x_n_wide, height_tail_share
- iter `9`: AUC `0.9736` (Δ `+0.0084`), TPR@1% `0.7006` (Δ `+0.0659`), features=n_wide_items, n_multi_dim_stress, spare_cap_x_n_near_vL
- iter `8`: AUC `0.9730` (Δ `+0.0079`), TPR@1% `0.7026` (Δ `+0.0678`), features=spare_cap_x_sku_counts, n_items_near_vL, n_tall_items

## Rejected Trials
- iter `11`: AUC `0.9737` (Δ `+0.0086`), TPR@1% `0.6957` (Δ `+0.0610`), features=spare_cap_x_h_to_H_max, wl_total_x_h_to_H_max, dim_s_avg_to_vmin
- iter `12`: AUC `0.9741` (Δ `+0.0089`), TPR@1% `0.6898` (Δ `+0.0551`), features=footprint_pressure, spare_cap_x_n_tall_items, n_long_tall_items
- iter `15`: AUC `0.9745` (Δ `+0.0094`), TPR@1% `0.7193` (Δ `+0.0846`), features=q90_w_to_W, spare_cap_x_height_tail, n_long_and_wide

## Failed Trials
- iter `4`: ValueError: Repair response did not include FEATURE_CODE python block
- iter `7`: RateLimitError: Error code: 429 - {'error': {'message': "No deployments available for selected model, Try again in 5 seconds. Passed model=glm-5.1. pre-call-checks=False, cooldown_list=[('b9ca8bb822ee3f6a483f081af040bbc6769519968ef875dbfcc563db02907968', {'exception_received': 'litellm.RateLimitError: Model rate limit exceeded.', 'status_code': '429', 'timestamp': 1779097551.7085834, 'cooldown_time': 5}), ('ee7f63fe7b5a6e0c0f610a8773fbc7d2b61ec00beadff16b9c535feb3698db62', {'exception_received': 'litellm.RateLimitError: Model rate limit exceeded.', 'status_code': '429', 'timestamp': 1779097552.2410598, 'cooldown_time': 5})]", 'type': 'None', 'param': 'None', 'code': '429'}}
- iter `13`: KeyError: "Label(s) ['vehicle_height', 'vehicle_length', 'vehicle_width'] do not exist"

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale
