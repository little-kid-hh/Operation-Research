## Hybrid after iteration 5

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 337 / 2034 = 16.57%
- Avg SVM P(y=1) (reference): 0.8313
- Std probability: 0.2243
- Most variable features:
  - `spare_capacity`: mean=17596.9050, var=42200934.8812
  - `sku_average_volume`: mean=2983.7143, var=223153.8176
  - `wl_to_vehicle_wl_total`: mean=940.6763, var=54512.0864
  - `wl_to_vehicle_wl_max`: mean=179.2594, var=628.7073
  - `wl_to_vehicle_wl_min`: mean=44.4881, var=183.4896
- Typical FN cases:
  - dispatch=300197, prob=1.0000, sku_counts=6.0, fill_ratio=31727.0
  - dispatch=267516, prob=0.9999, sku_counts=4.0, fill_ratio=34424.0
  - dispatch=267576, prob=0.9999, sku_counts=4.0, fill_ratio=34564.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 93 / 466 = 19.96%
- Avg SVM P(y=1) (reference): 0.6653
- Std probability: 0.1429
- Most variable features:
  - `spare_capacity`: mean=11281.8280, var=13672006.8091
  - `sku_average_volume`: mean=3063.7398, var=159463.9979
  - `wl_to_vehicle_wl_total`: mean=1153.3109, var=38172.9740
  - `wl_to_vehicle_wl_max`: mean=179.7805, var=554.3332
  - `sku_length_var`: mean=44.7259, var=185.0063
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=20288.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).