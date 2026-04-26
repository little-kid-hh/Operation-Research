## Hybrid after iteration 2

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 767 / 2034 = 37.71%
- Avg SVM P(y=1) (reference): 0.9037
- Std probability: 0.1702
- Most variable features:
  - `spare_capacity`: mean=20526.8422, var=50211771.7365
  - `sku_average_volume`: mean=2967.2083, var=247346.0261
  - `wl_to_vehicle_wl_total`: mean=850.1766, var=65774.7364
  - `wl_to_vehicle_wl_max`: mean=169.9995, var=787.3010
  - `wl_to_vehicle_wl_min`: mean=49.5458, var=307.5648
- Typical FN cases:
  - dispatch=84625, prob=1.0000, sku_counts=2.0, fill_ratio=42204.0
  - dispatch=301226, prob=1.0000, sku_counts=1.0, fill_ratio=42600.0
  - dispatch=42071, prob=1.0000, sku_counts=4.0, fill_ratio=37147.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 86 / 466 = 18.45%
- Avg SVM P(y=1) (reference): 0.6124
- Std probability: 0.1850
- Most variable features:
  - `spare_capacity`: mean=10491.0930, var=14036516.1076
  - `sku_average_volume`: mean=3114.2433, var=114443.2227
  - `wl_to_vehicle_wl_total`: mean=1170.3391, var=34356.4418
  - `wl_to_vehicle_wl_max`: mean=181.2161, var=455.9642
  - `sku_length_var`: mean=43.7943, var=175.1179
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0
  - dispatch=228399, prob=0.0685, sku_counts=11.0, fill_ratio=4122.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).