## Hybrid after iteration 4

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 884 / 2034 = 43.46%
- Avg SVM P(y=1) (reference): 0.9053
- Std probability: 0.1615
- Most variable features:
  - `spare_capacity`: mean=19929.7941, var=46320650.6273
  - `sku_average_volume`: mean=2951.7088, var=227696.9826
  - `wl_to_vehicle_wl_total`: mean=877.6541, var=62204.2908
  - `wl_to_vehicle_wl_max`: mean=171.2481, var=759.9555
  - `wl_to_vehicle_wl_min`: mean=49.2147, var=282.8495
- Typical FN cases:
  - dispatch=84625, prob=1.0000, sku_counts=2.0, fill_ratio=42204.0
  - dispatch=301226, prob=1.0000, sku_counts=1.0, fill_ratio=42600.0
  - dispatch=42071, prob=1.0000, sku_counts=4.0, fill_ratio=37147.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 82 / 466 = 17.60%
- Avg SVM P(y=1) (reference): 0.5794
- Std probability: 0.1846
- Most variable features:
  - `spare_capacity`: mean=9584.2805, var=14021242.9823
  - `sku_average_volume`: mean=3037.6992, var=162243.7075
  - `wl_to_vehicle_wl_total`: mean=1207.5711, var=34558.5569
  - `wl_to_vehicle_wl_max`: mean=175.3913, var=510.3716
  - `sku_length_var`: mean=42.5713, var=158.5446
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0
  - dispatch=228399, prob=0.0685, sku_counts=11.0, fill_ratio=4122.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).