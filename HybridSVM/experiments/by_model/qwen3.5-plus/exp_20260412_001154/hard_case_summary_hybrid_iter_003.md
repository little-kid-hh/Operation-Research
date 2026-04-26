## Hybrid after iteration 3

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 747 / 2034 = 36.73%
- Avg SVM P(y=1) (reference): 0.9084
- Std probability: 0.1692
- Most variable features:
  - `spare_capacity`: mean=20840.3628, var=47745950.9380
  - `sku_average_volume`: mean=2987.7073, var=237720.9852
  - `wl_to_vehicle_wl_total`: mean=835.3670, var=58660.1676
  - `wl_to_vehicle_wl_max`: mean=170.4786, var=792.6624
  - `wl_to_vehicle_wl_min`: mean=49.9024, var=310.0137
- Typical FN cases:
  - dispatch=84625, prob=1.0000, sku_counts=2.0, fill_ratio=42204.0
  - dispatch=301226, prob=1.0000, sku_counts=1.0, fill_ratio=42600.0
  - dispatch=42071, prob=1.0000, sku_counts=4.0, fill_ratio=37147.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 92 / 466 = 19.74%
- Avg SVM P(y=1) (reference): 0.6092
- Std probability: 0.1795
- Most variable features:
  - `spare_capacity`: mean=10265.1739, var=13894920.9263
  - `sku_average_volume`: mean=3057.0482, var=154289.1880
  - `wl_to_vehicle_wl_total`: mean=1185.7201, var=36594.4033
  - `wl_to_vehicle_wl_max`: mean=178.8134, var=521.0256
  - `sku_length_var`: mean=44.8373, var=180.7481
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0
  - dispatch=228399, prob=0.0685, sku_counts=11.0, fill_ratio=4122.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).