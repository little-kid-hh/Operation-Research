## Hybrid after iteration 4

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 79 / 2034 = 3.88%
- Avg SVM P(y=1) (reference): 0.4928
- Std probability: 0.1726
- Most variable features:
  - `spare_capacity`: mean=9162.9747, var=7844474.3538
  - `sku_average_volume`: mean=2976.5531, var=407871.5535
  - `wl_to_vehicle_wl_total`: mean=1235.1793, var=35965.8961
  - `wl_to_vehicle_wl_max`: mean=175.4958, var=467.2340
  - `wl_to_vehicle_wl_avg`: mean=100.3188, var=193.3231
- Typical FN cases:
  - dispatch=302575, prob=0.9156, sku_counts=16.0, fill_ratio=12427.0
  - dispatch=301804, prob=0.8932, sku_counts=16.0, fill_ratio=10784.0
  - dispatch=300658, prob=0.8806, sku_counts=16.0, fill_ratio=10339.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 124 / 466 = 26.61%
- Avg SVM P(y=1) (reference): 0.6578
- Std probability: 0.1676
- Most variable features:
  - `spare_capacity`: mean=12159.3145, var=14792469.4414
  - `sku_average_volume`: mean=3188.5237, var=112904.5401
  - `wl_to_vehicle_wl_total`: mean=1098.4879, var=35265.5339
  - `wl_to_vehicle_wl_max`: mean=183.9852, var=418.1122
  - `sku_length_var`: mean=47.2555, var=189.0393
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=20288.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).