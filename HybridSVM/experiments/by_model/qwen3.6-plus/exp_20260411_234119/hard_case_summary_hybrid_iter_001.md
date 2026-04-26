## Hybrid after iteration 1

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 334 / 2034 = 16.42%
- Avg SVM P(y=1) (reference): 0.8239
- Std probability: 0.2220
- Most variable features:
  - `spare_capacity`: mean=15511.7575, var=27420808.3274
  - `sku_average_volume`: mean=2766.6735, var=200064.1450
  - `wl_to_vehicle_wl_total`: mean=1040.0836, var=42383.5462
  - `wl_to_vehicle_wl_max`: mean=166.1178, var=528.4599
  - `wl_to_vehicle_wl_min`: mean=42.3940, var=122.0587
- Typical FN cases:
  - dispatch=84579, prob=0.9999, sku_counts=7.0, fill_ratio=28239.0
  - dispatch=341828, prob=0.9999, sku_counts=8.0, fill_ratio=29004.0
  - dispatch=300262, prob=0.9996, sku_counts=9.0, fill_ratio=24184.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 98 / 466 = 21.03%
- Avg SVM P(y=1) (reference): 0.6802
- Std probability: 0.1392
- Most variable features:
  - `spare_capacity`: mean=12419.9592, var=16030865.7943
  - `sku_average_volume`: mean=3192.9224, var=125703.2461
  - `wl_to_vehicle_wl_total`: mean=1095.2466, var=40927.3449
  - `wl_to_vehicle_wl_max`: mean=184.5918, var=435.2040
  - `sku_length_var`: mean=45.5886, var=199.9223
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=20288.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).