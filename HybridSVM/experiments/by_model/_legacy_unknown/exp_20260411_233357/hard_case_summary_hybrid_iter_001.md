## Hybrid after iteration 1

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 300 / 2034 = 14.75%
- Avg SVM P(y=1) (reference): 0.8248
- Std probability: 0.2331
- Most variable features:
  - `spare_capacity`: mean=19298.2000, var=63659610.8933
  - `sku_average_volume`: mean=3073.6244, var=241669.2584
  - `wl_to_vehicle_wl_total`: mean=882.8236, var=81791.6273
  - `wl_to_vehicle_wl_max`: mean=175.8611, var=709.7272
  - `wl_to_vehicle_wl_min`: mean=48.5903, var=227.0943
- Typical FN cases:
  - dispatch=301226, prob=1.0000, sku_counts=1.0, fill_ratio=42600.0
  - dispatch=41329, prob=1.0000, sku_counts=5.0, fill_ratio=33946.0
  - dispatch=303933, prob=1.0000, sku_counts=4.0, fill_ratio=35030.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 100 / 466 = 21.46%
- Avg SVM P(y=1) (reference): 0.6753
- Std probability: 0.1509
- Most variable features:
  - `spare_capacity`: mean=11969.8700, var=13087692.9531
  - `sku_average_volume`: mean=3158.0355, var=148053.1846
  - `wl_to_vehicle_wl_total`: mean=1108.8625, var=35948.7287
  - `wl_to_vehicle_wl_max`: mean=182.7750, var=426.2542
  - `sku_length_var`: mean=46.7565, var=203.3645
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=20288.0
  - dispatch=226571, prob=0.0299, sku_counts=7.0, fill_ratio=10405.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).