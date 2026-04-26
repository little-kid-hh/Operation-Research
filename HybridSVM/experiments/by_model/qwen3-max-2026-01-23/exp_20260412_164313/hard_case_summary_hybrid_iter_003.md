## Hybrid after iteration 3

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 27 / 2034 = 1.33%
- Avg SVM P(y=1) (reference): 0.8205
- Std probability: 0.1182
- Most variable features:
  - `spare_capacity`: mean=14730.2593, var=6305263.4513
  - `sku_average_volume`: mean=3063.5893, var=69489.6666
  - `wl_to_vehicle_wl_total`: mean=1051.8364, var=1675.5306
  - `wl_to_vehicle_wl_max`: mean=180.8642, var=311.3031
  - `sku_length_var`: mean=55.0383, var=186.2725
- Typical FN cases:
  - dispatch=341590, prob=0.9961, sku_counts=10.0, fill_ratio=21974.0
  - dispatch=228536, prob=0.9881, sku_counts=10.0, fill_ratio=18093.0
  - dispatch=228652, prob=0.9859, sku_counts=10.0, fill_ratio=19730.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 464 / 466 = 99.57%
- Avg SVM P(y=1) (reference): 0.3222
- Std probability: 0.2520
- Most variable features:
  - `spare_capacity`: mean=6148.7069, var=29258023.1037
  - `sku_average_volume`: mean=3153.6856, var=195293.9777
  - `wl_to_vehicle_wl_total`: mean=1302.9580, var=52097.4098
  - `wl_to_vehicle_wl_max`: mean=180.3655, var=469.5013
  - `sku_length_var`: mean=46.7963, var=146.5667
- Typical FP cases:
  - dispatch=156193, prob=0.0008, sku_counts=15.0, fill_ratio=-7768.0
  - dispatch=341853, prob=0.0010, sku_counts=21.0, fill_ratio=-9604.0
  - dispatch=226675, prob=0.0014, sku_counts=12.0, fill_ratio=-5329.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).