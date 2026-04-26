## Hybrid after iteration 12

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 60 / 2034 = 2.95%
- Avg SVM P(y=1) (reference): 0.5320
- Std probability: 0.1754
- Most variable features:
  - `spare_capacity`: mean=8694.6167, var=4719676.7031
  - `sku_average_volume`: mean=2808.3974, var=321923.0459
  - `wl_to_vehicle_wl_total`: mean=1273.4375, var=29293.6147
  - `wl_to_vehicle_wl_max`: mean=171.6667, var=469.9363
  - `sku_length_var`: mean=50.6337, var=161.0343
- Typical FN cases:
  - dispatch=302575, prob=0.9156, sku_counts=16.0, fill_ratio=12427.0
  - dispatch=301804, prob=0.8932, sku_counts=16.0, fill_ratio=10784.0
  - dispatch=300658, prob=0.8806, sku_counts=16.0, fill_ratio=10339.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 294 / 466 = 63.09%
- Avg SVM P(y=1) (reference): 0.3823
- Std probability: 0.2775
- Most variable features:
  - `spare_capacity`: mean=7479.8707, var=32904322.7792
  - `sku_average_volume`: mean=3220.5557, var=154608.7784
  - `wl_to_vehicle_wl_total`: mean=1256.1409, var=57008.0834
  - `wl_to_vehicle_wl_max`: mean=183.7075, var=453.7267
  - `sku_length_var`: mean=45.1708, var=161.0259
- Typical FP cases:
  - dispatch=341853, prob=0.0010, sku_counts=21.0, fill_ratio=-9604.0
  - dispatch=226675, prob=0.0014, sku_counts=12.0, fill_ratio=-5329.0
  - dispatch=2417, prob=0.0047, sku_counts=18.0, fill_ratio=-7459.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).