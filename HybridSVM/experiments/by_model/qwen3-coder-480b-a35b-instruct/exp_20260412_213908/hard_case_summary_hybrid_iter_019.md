## Hybrid after iteration 19

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 29 / 2034 = 1.43%
- Avg SVM P(y=1) (reference): 0.4303
- Std probability: 0.0601
- Most variable features:
  - `spare_capacity`: mean=8140.6552, var=2456227.1914
  - `sku_average_volume`: mean=2987.3311, var=87498.2807
  - `wl_to_vehicle_wl_total`: mean=1246.7241, var=17907.0609
  - `wl_to_vehicle_wl_max`: mean=182.6437, var=224.5962
  - `sku_length_var`: mean=51.9169, var=153.9620
- Typical FN cases:
  - dispatch=229081, prob=0.2966, sku_counts=14.0, fill_ratio=5606.0
  - dispatch=342320, prob=0.3084, sku_counts=14.0, fill_ratio=7085.0
  - dispatch=196911, prob=0.3242, sku_counts=13.0, fill_ratio=5995.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 357 / 466 = 76.61%
- Avg SVM P(y=1) (reference): 0.3517
- Std probability: 0.2695
- Most variable features:
  - `spare_capacity`: mean=6905.6275, var=30931377.8696
  - `sku_average_volume`: mean=3242.0805, var=174628.1400
  - `wl_to_vehicle_wl_total`: mean=1264.8693, var=49827.4358
  - `wl_to_vehicle_wl_max`: mean=181.9760, var=458.3004
  - `sku_length_var`: mean=45.3437, var=146.4171
- Typical FP cases:
  - dispatch=341853, prob=0.0010, sku_counts=21.0, fill_ratio=-9604.0
  - dispatch=226675, prob=0.0014, sku_counts=12.0, fill_ratio=-5329.0
  - dispatch=229229, prob=0.0029, sku_counts=12.0, fill_ratio=-4832.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).