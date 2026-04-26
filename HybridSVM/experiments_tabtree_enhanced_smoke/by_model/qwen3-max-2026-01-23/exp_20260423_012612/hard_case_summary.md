## 3D-BPP Hard Case Analysis

### 1. FN Cases (TabTreeFormer(RF) predicts NO, GT = YES — under-confident)
- Count: 151 / 2034 = 7.42%
- Avg Ensemble baseline probability: 0.0335
- Std probability: 0.0931
- Most variable features:
  - `spare_capacity`: mean=12707.1060, var=31199744.5848
  - `sku_average_volume`: mean=3165.2768, var=149961.7318
  - `wl_to_vehicle_wl_total`: mean=1079.5833, var=48265.4824
  - `wl_to_vehicle_wl_max`: mean=179.6523, var=390.7548
  - `sku_length_var`: mean=46.6419, var=193.5874
- Typical FN cases:
  - dispatch=119035, prob=0.0000, sku_counts=12.0, fill_ratio=N/A
  - dispatch=341372, prob=0.0000, sku_counts=7.0, fill_ratio=N/A
  - dispatch=303367, prob=0.0000, sku_counts=10.0, fill_ratio=N/A

### 2. FP Cases (TabTreeFormer(RF) predicts YES, GT = NO — over-confident)
- Count: 69 / 466 = 14.81%
- Avg Ensemble baseline probability: 0.9659
- Std probability: 0.0976
- Most variable features:
  - `spare_capacity`: mean=11794.4493, var=19648424.6242
  - `sku_average_volume`: mean=3231.9345, var=185817.5018
  - `wl_to_vehicle_wl_total`: mean=1101.1111, var=40779.0895
  - `wl_to_vehicle_wl_max`: mean=183.4964, var=429.3137
  - `wl_to_vehicle_wl_min`: mean=46.4915, var=168.0324
- Typical FP cases:
  - dispatch=197978, prob=1.0000, sku_counts=11.0, fill_ratio=N/A
  - dispatch=195220, prob=1.0000, sku_counts=9.0, fill_ratio=N/A
  - dispatch=380300, prob=1.0000, sku_counts=10.0, fill_ratio=N/A

### 3. Easy TN (SVM predicts NO, GT = NO — correct negatives)
- Count: 397 / 466 = 85.19% of GT=NO
- Avg Ensemble baseline probability: 0.0058
- Std probability: 0.0413
- Most variable features:
  - `spare_capacity`: mean=5211.4509, var=24756170.0360
  - `sku_average_volume`: mean=3139.3711, var=194884.4736
  - `wl_to_vehicle_wl_total`: mean=1336.7832, var=45892.2107
  - `wl_to_vehicle_wl_max`: mean=179.8510, var=472.6102
  - `sku_length_var`: mean=46.4067, var=142.0075
- Borderline-correct TN (closest to prob=0.5 — do not harm these patterns):
  - dispatch=228509, prob=0.4383, sku_counts=9.0, fill_ratio=N/A
  - dispatch=84532, prob=0.3984, sku_counts=12.0, fill_ratio=N/A
  - dispatch=344525, prob=0.3604, sku_counts=15.0, fill_ratio=N/A

### 4. Easy TP (SVM predicts YES, GT = YES — correct positives)
- Count: 1883 / 2034 = 92.58% of GT=YES
- Avg Ensemble baseline probability: 0.9978
- Std probability: 0.0232
- Most variable features:
  - `spare_capacity`: mean=19106.6516, var=50587077.6694
  - `sku_average_volume`: mean=2826.0934, var=204317.5311
  - `wl_to_vehicle_wl_total`: mean=907.2327, var=71232.4651
  - `wl_to_vehicle_wl_max`: mean=161.1723, var=736.6897
  - `wl_to_vehicle_wl_min`: mean=48.8310, var=250.4699
- Borderline-correct TP (closest to prob=0.5 — do not harm these patterns):
  - dispatch=342371, prob=0.5432, sku_counts=8.0, fill_ratio=N/A
  - dispatch=157150, prob=0.6907, sku_counts=8.0, fill_ratio=N/A
  - dispatch=2065, prob=0.7005, sku_counts=7.0, fill_ratio=N/A

### 5a. Easy TP rows (feature-similar to FN cloud — SVM still correct)
These are **positive** examples that sit near FN cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=119335, prob=1.0000, sku_counts=10.0, fill_ratio=N/A
  - dispatch=43771, prob=0.9947, sku_counts=10.0, fill_ratio=N/A
  - dispatch=121108, prob=1.0000, sku_counts=10.0, fill_ratio=N/A

### 5b. Easy TN rows (feature-similar to FP cloud — SVM still correct)
These are **negative** examples that sit near FP cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=156372, prob=0.0000, sku_counts=10.0, fill_ratio=N/A
  - dispatch=156977, prob=0.0001, sku_counts=10.0, fill_ratio=N/A
  - dispatch=195685, prob=0.0004, sku_counts=11.0, fill_ratio=N/A

### 6. Debug Instruction
请分析以上FN/FP错题与Easy TN/TP对照样本，找出SVM线性模型在错题上的失效规律，设计**窄**的条件：只在确有必要时覆盖SVM。对 Easy TN/TP 及 §5 中的对照行，除非逻辑上必须触发，否则应保持 `return -1`。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。