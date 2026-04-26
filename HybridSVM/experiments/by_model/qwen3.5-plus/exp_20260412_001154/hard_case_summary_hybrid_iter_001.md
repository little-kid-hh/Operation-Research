## Hybrid after iteration 1

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 514 / 2034 = 25.27%
- Avg SVM P(y=1) (reference): 0.8913
- Std probability: 0.1987
- Most variable features:
  - `spare_capacity`: mean=22352.1887, var=60108225.7601
  - `sku_average_volume`: mean=3087.1430, var=269893.7022
  - `wl_to_vehicle_wl_total`: mean=772.0185, var=71552.5108
  - `wl_to_vehicle_wl_max`: mean=170.1200, var=814.5892
  - `wl_to_vehicle_wl_min`: mean=53.2255, var=364.8585
- Typical FN cases:
  - dispatch=84625, prob=1.0000, sku_counts=2.0, fill_ratio=42204.0
  - dispatch=301226, prob=1.0000, sku_counts=1.0, fill_ratio=42600.0
  - dispatch=42071, prob=1.0000, sku_counts=4.0, fill_ratio=37147.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 104 / 466 = 22.32%
- Avg SVM P(y=1) (reference): 0.6433
- Std probability: 0.1845
- Most variable features:
  - `spare_capacity`: mean=11116.0288, var=13847054.3165
  - `sku_average_volume`: mean=3112.8125, var=103254.3159
  - `wl_to_vehicle_wl_total`: mean=1143.9663, var=32683.1663
  - `wl_to_vehicle_wl_max`: mean=182.2796, var=435.4265
  - `sku_length_var`: mean=46.3435, var=191.0840
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0
  - dispatch=120724, prob=0.9493, sku_counts=9.0, fill_ratio=18470.0

### 3. Note

Section titles refer to the **predictor** used for hard labels. Probability columns still show **SVM P(y=1)** on each row (reference score).