## Hybrid after iteration 2

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 45 / 2034 = 2.21%
- Avg SVM P(y=1) (reference): 0.8018
- Std probability: 0.1063
- Most variable features:
  - `spare_capacity`: mean=12537.3778, var=11748065.9240
  - `sku_average_volume`: mean=2716.9981, var=223874.1309
  - `wl_to_vehicle_wl_total`: mean=1186.7778, var=36260.0339
  - `wl_to_vehicle_wl_max`: mean=169.0833, var=505.8457
  - `sku_length_var`: mean=53.5989, var=196.0048
- Typical FN cases:
  - dispatch=341590, prob=0.9961, sku_counts=10.0, fill_ratio=21974.0
  - dispatch=228536, prob=0.9881, sku_counts=10.0, fill_ratio=18093.0
  - dispatch=228652, prob=0.9859, sku_counts=10.0, fill_ratio=19730.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 463 / 466 = 99.36%
- Avg SVM P(y=1) (reference): 0.3215
- Std probability: 0.2519
- Most variable features:
  - `spare_capacity`: mean=6144.7084, var=29313797.0446
  - `sku_average_volume`: mean=3155.5024, var=194184.1992
  - `wl_to_vehicle_wl_total`: mean=1303.1524, var=52192.3828
  - `wl_to_vehicle_wl_max`: mean=180.4491, var=467.2738
  - `sku_length_var`: mean=46.7874, var=146.8468
- Typical FP cases:
  - dispatch=156193, prob=0.0008, sku_counts=15.0, fill_ratio=-7768.0
  - dispatch=341853, prob=0.0010, sku_counts=21.0, fill_ratio=-9604.0
  - dispatch=226675, prob=0.0014, sku_counts=12.0, fill_ratio=-5329.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).