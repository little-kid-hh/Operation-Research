## Hybrid after iteration 1

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 285 / 2034 = 14.01%
- Avg SVM P(y=1) (reference): 0.8635
- Std probability: 0.2626
- Most variable features:
  - `spare_capacity`: mean=13310.6070, var=12154751.3894
  - `sku_average_volume`: mean=2799.8271, var=159626.7035
  - `wl_to_vehicle_wl_total`: mean=1143.6594, var=16731.9618
  - `wl_to_vehicle_wl_max`: mean=170.6082, var=525.1308
  - `sku_length_var`: mean=46.2389, var=159.7421
- Typical FN cases:
  - dispatch=226709, prob=0.0140, sku_counts=11.0, fill_ratio=N/A
  - dispatch=156435, prob=0.9826, sku_counts=12.0, fill_ratio=N/A
  - dispatch=341618, prob=0.9826, sku_counts=11.0, fill_ratio=N/A

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 95 / 466 = 20.39%
- Avg SVM P(y=1) (reference): 0.7333
- Std probability: 0.2538
- Most variable features:
  - `spare_capacity`: mean=12578.9053, var=13828868.5279
  - `sku_average_volume`: mean=3233.5613, var=109760.6242
  - `wl_to_vehicle_wl_total`: mean=1072.7895, var=29100.7276
  - `wl_to_vehicle_wl_max`: mean=182.6096, var=446.0973
  - `wl_to_vehicle_wl_min`: mean=47.4167, var=176.5592
- Typical FP cases:
  - dispatch=197884, prob=0.0227, sku_counts=10.0, fill_ratio=N/A
  - dispatch=195221, prob=0.0234, sku_counts=10.0, fill_ratio=N/A
  - dispatch=300240, prob=0.9727, sku_counts=9.0, fill_ratio=N/A

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).