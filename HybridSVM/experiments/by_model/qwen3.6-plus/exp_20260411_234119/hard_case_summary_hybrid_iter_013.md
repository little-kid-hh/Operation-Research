## Hybrid after iteration 13

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 101 / 2034 = 4.97%
- Avg SVM P(y=1) (reference): 0.5542
- Std probability: 0.1935
- Most variable features:
  - `spare_capacity`: mean=11805.2079, var=15575747.8479
  - `sku_average_volume`: mean=3353.2833, var=169095.3757
  - `wl_to_vehicle_wl_total`: mean=1081.4563, var=32287.9834
  - `wl_to_vehicle_wl_max`: mean=186.6337, var=305.5923
  - `wl_to_vehicle_wl_min`: mean=47.2071, var=172.3444
- Typical FN cases:
  - dispatch=226712, prob=0.9128, sku_counts=8.0, fill_ratio=16906.0
  - dispatch=227396, prob=0.8823, sku_counts=8.0, fill_ratio=16853.0
  - dispatch=197307, prob=0.8733, sku_counts=6.0, fill_ratio=21900.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 105 / 466 = 22.53%
- Avg SVM P(y=1) (reference): 0.6493
- Std probability: 0.1594
- Most variable features:
  - `spare_capacity`: mean=11077.7524, var=13789854.9482
  - `sku_average_volume`: mean=3072.3192, var=146643.3031
  - `wl_to_vehicle_wl_total`: mean=1155.4921, var=36324.8075
  - `wl_to_vehicle_wl_max`: mean=178.5397, var=557.2921
  - `sku_length_var`: mean=45.4805, var=181.1578
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=20288.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).