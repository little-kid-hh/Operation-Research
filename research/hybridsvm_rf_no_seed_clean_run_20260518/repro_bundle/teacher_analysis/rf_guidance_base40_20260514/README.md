# Base40-Only RandomForest Guidance Package

This package is the formal teacher-analysis source for the HybridSVM
feature-search route. It intentionally uses only the original base40
aggregate feature set and does not use any later LLM-derived features.

## Scope
- feature_set: `base40`
- active_bank_used: `false`
- data: `/Users/zhongxiaochuan/Operation-Research/FunSearch_test/training_2orientations.csv`
- rf_cfg_source: `/Users/zhongxiaochuan/Operation-Research/Ensemble_baseline/run_ensemble_ablation.py`

## Metrics
- svm (base40): AUC `0.9651`, TPR@1% `0.6347`, ACC `0.9276`
- rf (base40): AUC `0.9823`, TPR@1% `0.8058`, ACC `0.9436`

## Top base40 RF features
- `spare_capacity`
- `wl_to_vehicle_wl_total`
- `sku_counts`
- `sku_average_volume`
- `wl_to_vehicle_wl_max`
- `h_to_H_ratio_avg`
- `sku_height_avg`
- `wl_to_vehicle_wl_std`
- `wl_to_vehicle_wl_avg`
- `sku_length_avg`
- `l_to_L_ratio_avg`
- `sku_concentration`
- `w_to_W_ratio_avg`
- `sku_width_avg`
- `sku_height_var`

## High-contrast features on rows recovered by RF over SVM
- `spare_capacity`
- `sku_average_volume`
- `wl_to_vehicle_wl_total`
- `wl_to_vehicle_wl_max`
- `wl_to_vehicle_wl_avg`
- `wl_to_vehicle_wl_std`
- `sku_counts`
- `sku_length_var`
- `sku_max_length`
- `sku_height_avg`
- `sku_length_avg`
- `sku_max_height`
- `sku_width_var`
- `sku_max_width`
- `sku_height_var`

Files:
- `summary.json`
- `base40_rf_importance.csv`
- `rf_vs_svm_recovery_contrast_base40.csv`
