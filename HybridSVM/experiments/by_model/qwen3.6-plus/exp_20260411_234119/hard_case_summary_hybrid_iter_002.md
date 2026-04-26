## Hybrid after iteration 2

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 468 / 2034 = 23.01%
- Avg SVM P(y=1) (reference): 0.8613
- Std probability: 0.1996
- Most variable features:
  - `spare_capacity`: mean=17341.8205, var=38495110.0618
  - `sku_average_volume`: mean=2838.7910, var=212848.0592
  - `wl_to_vehicle_wl_total`: mean=976.4913, var=57621.3527
  - `wl_to_vehicle_wl_max`: mean=172.8641, var=627.6723
  - `wl_to_vehicle_wl_min`: mean=43.5310, var=151.4678
- Typical FN cases:
  - dispatch=300197, prob=1.0000, sku_counts=6.0, fill_ratio=31727.0
  - dispatch=267516, prob=0.9999, sku_counts=4.0, fill_ratio=34424.0
  - dispatch=267576, prob=0.9999, sku_counts=4.0, fill_ratio=34564.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 88 / 466 = 18.88%
- Avg SVM P(y=1) (reference): 0.6714
- Std probability: 0.1420
- Most variable features:
  - `spare_capacity`: mean=11887.6705, var=14232918.0391
  - `sku_average_volume`: mean=3156.6985, var=121380.3276
  - `wl_to_vehicle_wl_total`: mean=1122.8030, var=37259.8435
  - `wl_to_vehicle_wl_max`: mean=183.3286, var=460.0793
  - `sku_length_var`: mean=43.5442, var=167.7196
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=20288.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).