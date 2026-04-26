## 3D-BPP Hard Case Analysis

### 1. FN Cases (TabTreeFormer(RF) predicts NO, GT = YES — under-confident)
- Count: 170 / 2034 = 8.36%
- Avg Ensemble baseline probability: 0.0624
- Std probability: 0.1217
- Most variable features:
  - `spare_capacity`: mean=12874.9529, var=32129899.3743
  - `sku_average_volume`: mean=3235.1025, var=199253.6936
  - `wl_to_vehicle_wl_total`: mean=1067.8848, var=55728.5070
  - `wl_to_vehicle_wl_max`: mean=179.6078, var=347.3319
  - `sku_length_var`: mean=45.5832, var=229.4177
- Typical FN cases:
  - dispatch=227608, prob=0.0002, sku_counts=14.0, fill_ratio=N/A
  - dispatch=341960, prob=0.0003, sku_counts=14.0, fill_ratio=N/A
  - dispatch=196215, prob=0.0003, sku_counts=13.0, fill_ratio=N/A

### 2. FP Cases (TabTreeFormer(RF) predicts YES, GT = NO — over-confident)
- Count: 51 / 466 = 10.94%
- Avg Ensemble baseline probability: 0.9110
- Std probability: 0.1360
- Most variable features:
  - `spare_capacity`: mean=11741.9608, var=16614421.7240
  - `sku_average_volume`: mean=3145.7000, var=152870.0101
  - `wl_to_vehicle_wl_total`: mean=1123.3333, var=36820.1593
  - `wl_to_vehicle_wl_max`: mean=180.9150, var=425.5244
  - `sku_length_var`: mean=48.5735, var=203.2922
- Typical FP cases:
  - dispatch=300240, prob=0.9999, sku_counts=9.0, fill_ratio=N/A
  - dispatch=341673, prob=0.9999, sku_counts=10.0, fill_ratio=N/A
  - dispatch=226909, prob=0.9999, sku_counts=7.0, fill_ratio=N/A

### 3. Easy TN (SVM predicts NO, GT = NO — correct negatives)
- Count: 415 / 466 = 89.06% of GT=NO
- Avg Ensemble baseline probability: 0.0088
- Std probability: 0.0469
- Most variable features:
  - `spare_capacity`: mean=5503.4289, var=26786446.7510
  - `sku_average_volume`: mean=3153.9834, var=199746.2910
  - `wl_to_vehicle_wl_total`: mean=1323.8303, var=49624.7183
  - `wl_to_vehicle_wl_max`: mean=180.3263, var=473.0424
  - `sku_length_var`: mean=46.6080, var=139.2122
- Borderline-correct TN (closest to prob=0.5 — do not harm these patterns):
  - dispatch=119312, prob=0.4289, sku_counts=14.0, fill_ratio=N/A
  - dispatch=342159, prob=0.4065, sku_counts=10.0, fill_ratio=N/A
  - dispatch=156854, prob=0.3944, sku_counts=11.0, fill_ratio=N/A

### 4. Easy TP (SVM predicts YES, GT = YES — correct positives)
- Count: 1864 / 2034 = 91.64% of GT=YES
- Avg Ensemble baseline probability: 0.9946
- Std probability: 0.0348
- Most variable features:
  - `spare_capacity`: mean=19156.5751, var=50473278.0534
  - `sku_average_volume`: mean=2816.2678, var=194342.2002
  - `wl_to_vehicle_wl_total`: mean=906.5428, var=70837.9589
  - `wl_to_vehicle_wl_max`: mean=160.9880, var=740.8108
  - `wl_to_vehicle_wl_min`: mean=48.6671, var=246.9248
- Borderline-correct TP (closest to prob=0.5 — do not harm these patterns):
  - dispatch=343110, prob=0.5111, sku_counts=8.0, fill_ratio=N/A
  - dispatch=119133, prob=0.5391, sku_counts=14.0, fill_ratio=N/A
  - dispatch=341916, prob=0.5817, sku_counts=12.0, fill_ratio=N/A

### 5a. Easy TP rows (feature-similar to FN cloud — SVM still correct)
These are **positive** examples that sit near FN cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=228380, prob=0.9999, sku_counts=10.0, fill_ratio=N/A
  - dispatch=300771, prob=0.9999, sku_counts=10.0, fill_ratio=N/A
  - dispatch=121108, prob=0.9999, sku_counts=10.0, fill_ratio=N/A

### 5b. Easy TN rows (feature-similar to FP cloud — SVM still correct)
These are **negative** examples that sit near FP cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=195685, prob=0.0004, sku_counts=11.0, fill_ratio=N/A
  - dispatch=156372, prob=0.0005, sku_counts=10.0, fill_ratio=N/A
  - dispatch=156977, prob=0.0016, sku_counts=10.0, fill_ratio=N/A

### 6. Debug Instruction
请分析以上FN/FP错题与Easy TN/TP对照样本，找出SVM线性模型在错题上的失效规律，设计**窄**的条件：只在确有必要时覆盖SVM。对 Easy TN/TP 及 §5 中的对照行，除非逻辑上必须触发，否则应保持 `return -1`。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。