## Hybrid after iteration 3

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 466 / 2034 = 22.91%
- Avg SVM P(y=1) (reference): 0.8616
- Std probability: 0.1996
- Most variable features:
  - `spare_capacity`: mean=17587.7146, var=36013980.4228
  - `sku_average_volume`: mean=2888.3727, var=212087.6576
  - `wl_to_vehicle_wl_total`: mean=959.2766, var=49622.3856
  - `wl_to_vehicle_wl_max`: mean=174.4447, var=626.5608
  - `wl_to_vehicle_wl_min`: mean=44.0817, var=155.7171
- Typical FN cases:
  - dispatch=300197, prob=1.0000, sku_counts=6.0, fill_ratio=31727.0
  - dispatch=267516, prob=0.9999, sku_counts=4.0, fill_ratio=34424.0
  - dispatch=267576, prob=0.9999, sku_counts=4.0, fill_ratio=34564.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 87 / 466 = 18.67%
- Avg SVM P(y=1) (reference): 0.6580
- Std probability: 0.1441
- Most variable features:
  - `spare_capacity`: mean=11241.7471, var=14575390.9016
  - `sku_average_volume`: mean=3075.9170, var=166870.3690
  - `wl_to_vehicle_wl_total`: mean=1157.9262, var=40152.3067
  - `wl_to_vehicle_wl_max`: mean=179.6360, var=557.5862
  - `sku_length_var`: mean=43.7910, var=181.5263
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=20288.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).