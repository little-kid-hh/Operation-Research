## Hybrid after iteration 16

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 33 / 2034 = 1.62%
- Avg SVM P(y=1) (reference): 0.4159
- Std probability: 0.0746
- Most variable features:
  - `spare_capacity`: mean=7916.1515, var=2614877.6437
  - `sku_average_volume`: mean=3026.3875, var=94164.0929
  - `wl_to_vehicle_wl_total`: mean=1245.6944, var=16064.7587
  - `wl_to_vehicle_wl_max`: mean=183.5606, var=204.9883
  - `sku_length_var`: mean=50.2672, var=157.7872
- Typical FN cases:
  - dispatch=226709, prob=0.1750, sku_counts=11.0, fill_ratio=5282.0
  - dispatch=229081, prob=0.2966, sku_counts=14.0, fill_ratio=5606.0
  - dispatch=196215, prob=0.3077, sku_counts=13.0, fill_ratio=5579.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 324 / 466 = 69.53%
- Avg SVM P(y=1) (reference): 0.3740
- Std probability: 0.2700
- Most variable features:
  - `spare_capacity`: mean=7335.6019, var=30675851.7952
  - `sku_average_volume`: mean=3205.0654, var=163958.2095
  - `wl_to_vehicle_wl_total`: mean=1260.3627, var=53425.4365
  - `wl_to_vehicle_wl_max`: mean=181.3130, var=490.9173
  - `sku_length_var`: mean=45.6516, var=156.2129
- Typical FP cases:
  - dispatch=341853, prob=0.0010, sku_counts=21.0, fill_ratio=-9604.0
  - dispatch=226675, prob=0.0014, sku_counts=12.0, fill_ratio=-5329.0
  - dispatch=2417, prob=0.0047, sku_counts=18.0, fill_ratio=-7459.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).