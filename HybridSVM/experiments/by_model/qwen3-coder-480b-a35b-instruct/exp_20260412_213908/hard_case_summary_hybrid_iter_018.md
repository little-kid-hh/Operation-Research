## Hybrid after iteration 18

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 30 / 2034 = 1.47%
- Avg SVM P(y=1) (reference): 0.4262
- Std probability: 0.0631
- Most variable features:
  - `spare_capacity`: mean=8055.2667, var=2585797.6622
  - `sku_average_volume`: mean=2988.8329, var=84647.0767
  - `wl_to_vehicle_wl_total`: mean=1248.0556, var=17361.5664
  - `wl_to_vehicle_wl_max`: mean=183.0139, var=221.0843
  - `sku_length_var`: mean=51.2041, var=163.5649
- Typical FN cases:
  - dispatch=229081, prob=0.2966, sku_counts=14.0, fill_ratio=5606.0
  - dispatch=196215, prob=0.3077, sku_counts=13.0, fill_ratio=5579.0
  - dispatch=342320, prob=0.3084, sku_counts=14.0, fill_ratio=7085.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 339 / 466 = 72.75%
- Avg SVM P(y=1) (reference): 0.3624
- Std probability: 0.2707
- Most variable features:
  - `spare_capacity`: mean=7088.6696, var=31321148.4867
  - `sku_average_volume`: mean=3228.6824, var=173044.4118
  - `wl_to_vehicle_wl_total`: mean=1260.7633, var=51572.3090
  - `wl_to_vehicle_wl_max`: mean=181.8240, var=475.0633
  - `sku_length_var`: mean=45.5206, var=150.4173
- Typical FP cases:
  - dispatch=341853, prob=0.0010, sku_counts=21.0, fill_ratio=-9604.0
  - dispatch=226675, prob=0.0014, sku_counts=12.0, fill_ratio=-5329.0
  - dispatch=229229, prob=0.0029, sku_counts=12.0, fill_ratio=-4832.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).