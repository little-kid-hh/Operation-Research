## 3D-BPP Hard Case Analysis

### 1. FN Cases (TabTreeFormer(RF) predicts NO, GT = YES — under-confident)
- Count: 109 / 2034 = 5.36%
- Avg Ensemble baseline probability: 0.0599
- Std probability: 0.1187
- Most variable features:
  - `spare_capacity`: mean=11799.9450, var=26165559.6667
  - `sku_average_volume`: mean=3167.5529, var=186316.2784
  - `wl_to_vehicle_wl_total`: mean=1113.1919, var=49303.1225
  - `wl_to_vehicle_wl_max`: mean=180.9901, var=373.9417
  - `sku_length_var`: mean=47.4187, var=223.8595
- Typical FN cases:
  - dispatch=227608, prob=0.0003, sku_counts=14.0, fill_ratio=N/A
  - dispatch=226709, prob=0.0003, sku_counts=11.0, fill_ratio=N/A
  - dispatch=196215, prob=0.0003, sku_counts=13.0, fill_ratio=N/A

### 2. FP Cases (TabTreeFormer(RF) predicts YES, GT = NO — over-confident)
- Count: 82 / 466 = 17.60%
- Avg Ensemble baseline probability: 0.9644
- Std probability: 0.0976
- Most variable features:
  - `spare_capacity`: mean=10907.6463, var=17827737.1554
  - `sku_average_volume`: mean=3165.1771, var=154393.8786
  - `wl_to_vehicle_wl_total`: mean=1137.1545, var=31560.8224
  - `wl_to_vehicle_wl_max`: mean=181.7276, var=388.2796
  - `sku_length_var`: mean=46.9511, var=188.8484
- Typical FP cases:
  - dispatch=300240, prob=0.9999, sku_counts=9.0, fill_ratio=N/A
  - dispatch=380300, prob=0.9999, sku_counts=10.0, fill_ratio=N/A
  - dispatch=341161, prob=0.9999, sku_counts=12.0, fill_ratio=N/A

### 3. Easy TN (SVM predicts NO, GT = NO — correct negatives)
- Count: 384 / 466 = 82.40% of GT=NO
- Avg Ensemble baseline probability: 0.0095
- Std probability: 0.0500
- Most variable features:
  - `spare_capacity`: mean=5177.9583, var=26174962.9774
  - `sku_average_volume`: mean=3150.4929, var=203175.3379
  - `wl_to_vehicle_wl_total`: mean=1337.0649, var=49503.8159
  - `wl_to_vehicle_wl_max`: mean=180.1053, var=484.4096
  - `sku_length_var`: mean=46.7958, var=137.5761
- Borderline-correct TN (closest to prob=0.5 — do not harm these patterns):
  - dispatch=85886, prob=0.4833, sku_counts=13.0, fill_ratio=N/A
  - dispatch=343035, prob=0.4586, sku_counts=15.0, fill_ratio=N/A
  - dispatch=84872, prob=0.4191, sku_counts=11.0, fill_ratio=N/A

### 4. Easy TP (SVM predicts YES, GT = YES — correct positives)
- Count: 1925 / 2034 = 94.64% of GT=YES
- Avg Ensemble baseline probability: 0.9964
- Std probability: 0.0318
- Most variable features:
  - `spare_capacity`: mean=19018.3917, var=50630847.4165
  - `sku_average_volume`: mean=2833.3649, var=203442.5519
  - `wl_to_vehicle_wl_total`: mean=909.0900, var=70597.3313
  - `wl_to_vehicle_wl_max`: mean=161.4998, var=734.5370
  - `wl_to_vehicle_wl_min`: mean=48.8019, var=251.4930
- Borderline-correct TP (closest to prob=0.5 — do not harm these patterns):
  - dispatch=196344, prob=0.5115, sku_counts=11.0, fill_ratio=N/A
  - dispatch=226519, prob=0.5467, sku_counts=11.0, fill_ratio=N/A
  - dispatch=342039, prob=0.6023, sku_counts=8.0, fill_ratio=N/A

### 5a. Easy TP rows (feature-similar to FN cloud — SVM still correct)
These are **positive** examples that sit near FN cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=41140, prob=0.9999, sku_counts=10.0, fill_ratio=N/A
  - dispatch=757, prob=0.9962, sku_counts=11.0, fill_ratio=N/A
  - dispatch=195919, prob=0.9999, sku_counts=11.0, fill_ratio=N/A

### 5b. Easy TN rows (feature-similar to FP cloud — SVM still correct)
These are **negative** examples that sit near FP cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=303204, prob=0.3175, sku_counts=11.0, fill_ratio=N/A
  - dispatch=341177, prob=0.0006, sku_counts=10.0, fill_ratio=N/A
  - dispatch=195934, prob=0.0004, sku_counts=10.0, fill_ratio=N/A

### 6. Debug Instruction
请分析以上FN/FP错题与Easy TN/TP对照样本，找出SVM线性模型在错题上的失效规律，设计**窄**的条件：只在确有必要时覆盖SVM。对 Easy TN/TP 及 §5 中的对照行，除非逻辑上必须触发，否则应保持 `return -1`。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。