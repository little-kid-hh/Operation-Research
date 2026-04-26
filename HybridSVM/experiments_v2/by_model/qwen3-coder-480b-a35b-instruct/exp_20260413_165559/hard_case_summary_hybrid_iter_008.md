## Hybrid after iteration 8

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 510 / 2034 = 25.07%
- Avg SVM P(y=1) (reference): 0.8500
- Std probability: 0.1379
- Most variable features:
  - `spare_capacity`: mean=14994.0157, var=50268817.4821
  - `sku_average_volume`: mean=2763.4307, var=178863.6418
  - `wl_to_vehicle_wl_total`: mean=1107.4845, var=79874.0837
  - `wl_to_vehicle_wl_max`: mean=168.2655, var=619.5975
  - `wl_to_vehicle_wl_min`: mean=51.4698, var=214.0057
- Typical FN cases:
  - dispatch=41673, prob=1.0000, sku_counts=3.0, fill_ratio=N/A
  - dispatch=303390, prob=1.0000, sku_counts=3.0, fill_ratio=N/A
  - dispatch=122168, prob=1.0000, sku_counts=2.0, fill_ratio=N/A

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 360 / 466 = 77.25%
- Avg SVM P(y=1) (reference): 0.3293
- Std probability: 0.2365
- Most variable features:
  - `spare_capacity`: mean=7170.1583, var=21431831.9055
  - `sku_average_volume`: mean=3195.0497, var=184024.9451
  - `wl_to_vehicle_wl_total`: mean=1254.9340, var=41500.6443
  - `wl_to_vehicle_wl_max`: mean=181.1840, var=441.7400
  - `sku_length_var`: mean=46.7290, var=145.5090
- Typical FP cases:
  - dispatch=344476, prob=0.0136, sku_counts=12.0, fill_ratio=N/A
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=N/A
  - dispatch=226368, prob=0.0151, sku_counts=11.0, fill_ratio=N/A

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).