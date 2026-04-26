## Hybrid after iteration 13

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 58 / 2034 = 2.85%
- Avg SVM P(y=1) (reference): 0.5368
- Std probability: 0.1764
- Most variable features:
  - `spare_capacity`: mean=8729.8966, var=4645883.6445
  - `sku_average_volume`: mean=2792.3282, var=323637.0726
  - `wl_to_vehicle_wl_total`: mean=1277.8951, var=29681.5519
  - `wl_to_vehicle_wl_max`: mean=172.0833, var=477.7658
  - `sku_length_var`: mean=50.8703, var=164.6684
- Typical FN cases:
  - dispatch=302575, prob=0.9156, sku_counts=16.0, fill_ratio=12427.0
  - dispatch=301804, prob=0.8932, sku_counts=16.0, fill_ratio=10784.0
  - dispatch=300658, prob=0.8806, sku_counts=16.0, fill_ratio=10339.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 318 / 466 = 68.24%
- Avg SVM P(y=1) (reference): 0.3704
- Std probability: 0.2712
- Most variable features:
  - `spare_capacity`: mean=7341.4245, var=31240583.2191
  - `sku_average_volume`: mean=3223.3260, var=148923.5253
  - `wl_to_vehicle_wl_total`: mean=1257.6114, var=53710.0259
  - `wl_to_vehicle_wl_max`: mean=182.0100, var=470.4633
  - `sku_length_var`: mean=45.3849, var=154.8974
- Typical FP cases:
  - dispatch=341853, prob=0.0010, sku_counts=21.0, fill_ratio=-9604.0
  - dispatch=226675, prob=0.0014, sku_counts=12.0, fill_ratio=-5329.0
  - dispatch=2417, prob=0.0047, sku_counts=18.0, fill_ratio=-7459.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).