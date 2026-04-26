## Hybrid after iteration 10

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 103 / 2034 = 5.06%
- Avg SVM P(y=1) (reference): 0.5298
- Std probability: 0.1798
- Most variable features:
  - `spare_capacity`: mean=11230.5146, var=14021856.7352
  - `sku_average_volume`: mean=3326.4432, var=168607.4092
  - `wl_to_vehicle_wl_total`: mean=1103.1675, var=29524.8022
  - `wl_to_vehicle_wl_max`: mean=186.1853, var=318.9852
  - `wl_to_vehicle_wl_min`: mean=46.8042, var=175.2171
- Typical FN cases:
  - dispatch=195565, prob=0.8475, sku_counts=7.0, fill_ratio=18901.0
  - dispatch=267667, prob=0.8437, sku_counts=10.0, fill_ratio=14911.0
  - dispatch=196274, prob=0.8369, sku_counts=8.0, fill_ratio=17909.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 97 / 466 = 20.82%
- Avg SVM P(y=1) (reference): 0.6739
- Std probability: 0.1461
- Most variable features:
  - `spare_capacity`: mean=11496.7835, var=14351456.4377
  - `sku_average_volume`: mean=3066.5113, var=157016.0240
  - `wl_to_vehicle_wl_total`: mean=1143.3162, var=39561.8589
  - `wl_to_vehicle_wl_max`: mean=180.3608, var=542.4600
  - `sku_length_var`: mean=45.5454, var=196.0447
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=20288.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).