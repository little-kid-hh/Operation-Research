# Base40-Only XGBoost Guidance Package

This package is the formal teacher-analysis source for the HybridSVM
feature-search route. It intentionally uses only the original base40
aggregate feature set and does not use any later LLM-derived features.

## Scope
- feature_set: `base40`
- active_bank_used: `false`
- data: `repro_bundle/data/training_2orientations.csv`
- xgb_cfg_source: `repro_bundle/reference/Ensemble_baseline/experiments/xgb_search_20260511_202834_base40/summary.json`

## Metrics
- svm (base40): AUC `0.9651`, TPR@1% `0.6347`, ACC `0.9276`
- xgb (base40): AUC `0.9868`, TPR@1% `0.8289`, ACC `0.9532`

## Top base40 XGB features
- `spare_capacity`
- `sku_average_volume`
- `wl_to_vehicle_wl_total`
- `wl_to_vehicle_wl_max`
- `sku_counts`
- `l_to_L_ratio_std`
- `sku_max_width`
- `h_to_H_ratio_max`
- `sku_concentration`
- `w_to_W_ratio_avg`
- `w_to_W_ratio_min`
- `sku_min_width`
- `sku_height_avg`
- `sku_length_avg`
- `sku_min_height`

## High-contrast features on rows recovered by XGB over SVM
- `spare_capacity`
- `sku_average_volume`
- `wl_to_vehicle_wl_total`
- `wl_to_vehicle_wl_max`
- `wl_to_vehicle_wl_std`
- `wl_to_vehicle_wl_avg`
- `sku_length_var`
- `wl_to_vehicle_wl_min`
- `sku_counts`
- `sku_height_var`
- `sku_max_length`
- `sku_width_var`
- `sku_height_avg`
- `sku_length_avg`
- `sku_max_width`

Files:
- `summary.json`
- `base40_gain_importance.csv`
- `xgb_vs_svm_recovery_contrast_base40.csv`
