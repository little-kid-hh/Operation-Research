## Hybrid after iteration 1

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 292 / 2034 = 14.36%
- Avg SVM P(y=1) (reference): 0.8327
- Std probability: 0.1403
- Most variable features:
  - `spare_capacity`: mean=13920.5240, var=52938426.9960
  - `sku_average_volume`: mean=2682.4184, var=169673.0918
  - `wl_to_vehicle_wl_total`: mean=1177.0006, var=86284.5881
  - `wl_to_vehicle_wl_max`: mean=168.2120, var=534.3109
  - `wl_to_vehicle_wl_min`: mean=50.3382, var=266.3079
- Typical FN cases:
  - dispatch=156180, prob=1.0000, sku_counts=4.0, fill_ratio=N/A
  - dispatch=119445, prob=1.0000, sku_counts=3.0, fill_ratio=N/A
  - dispatch=266280, prob=1.0000, sku_counts=2.0, fill_ratio=N/A

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 373 / 466 = 80.04%
- Avg SVM P(y=1) (reference): 0.3413
- Std probability: 0.2420
- Most variable features:
  - `spare_capacity`: mean=7320.7534, var=21515502.3091
  - `sku_average_volume`: mean=3192.5674, var=179576.7388
  - `wl_to_vehicle_wl_total`: mean=1251.4768, var=40573.2809
  - `wl_to_vehicle_wl_max`: mean=181.3517, var=444.7176
  - `sku_length_var`: mean=46.6974, var=143.2944
- Typical FP cases:
  - dispatch=344476, prob=0.0136, sku_counts=12.0, fill_ratio=N/A
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=N/A
  - dispatch=226368, prob=0.0151, sku_counts=11.0, fill_ratio=N/A

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).