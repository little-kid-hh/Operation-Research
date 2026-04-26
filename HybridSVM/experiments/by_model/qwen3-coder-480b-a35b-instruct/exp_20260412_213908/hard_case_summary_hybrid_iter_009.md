## Hybrid after iteration 9

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 86 / 2034 = 4.23%
- Avg SVM P(y=1) (reference): 0.4862
- Std probability: 0.1670
- Most variable features:
  - `spare_capacity`: mean=9178.4651, var=7319144.1092
  - `sku_average_volume`: mean=3005.7684, var=385999.0375
  - `wl_to_vehicle_wl_total`: mean=1229.3023, var=33624.6384
  - `wl_to_vehicle_wl_max`: mean=175.1211, var=493.0994
  - `wl_to_vehicle_wl_avg`: mean=101.0150, var=186.2783
- Typical FN cases:
  - dispatch=302575, prob=0.9156, sku_counts=16.0, fill_ratio=12427.0
  - dispatch=301804, prob=0.8932, sku_counts=16.0, fill_ratio=10784.0
  - dispatch=300658, prob=0.8806, sku_counts=16.0, fill_ratio=10339.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 110 / 466 = 23.61%
- Avg SVM P(y=1) (reference): 0.6952
- Std probability: 0.1356
- Most variable features:
  - `spare_capacity`: mean=12563.7182, var=15000443.7842
  - `sku_average_volume`: mean=3175.8056, var=121823.1149
  - `wl_to_vehicle_wl_total`: mean=1087.7197, var=38159.9296
  - `wl_to_vehicle_wl_max`: mean=184.4697, var=421.6001
  - `sku_length_var`: mean=47.5675, var=208.4218
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=20288.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).