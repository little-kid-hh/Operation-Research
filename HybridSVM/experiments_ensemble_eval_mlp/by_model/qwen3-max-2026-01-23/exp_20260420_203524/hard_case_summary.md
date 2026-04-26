## 3D-BPP Hard Case Analysis

### 1. FN Cases (Ensemble predicts NO, GT = YES — under-confident)
- Count: 37 / 2034 = 1.82%
- Avg Ensemble baseline probability: 0.2205
- Std probability: 0.1340
- Most variable features:
  - `spare_capacity`: mean=8431.7027, var=7593554.1008
  - `sku_average_volume`: mean=3198.7724, var=229058.3610
  - `wl_to_vehicle_wl_total`: mean=1224.0203, var=27565.3267
  - `wl_to_vehicle_wl_max`: mean=180.6306, var=376.3478
  - `sku_length_var`: mean=48.1945, var=117.3758
- Typical FN cases:
  - dispatch=226709, prob=0.0226, sku_counts=11.0, fill_ratio=N/A
  - dispatch=301553, prob=0.0343, sku_counts=11.0, fill_ratio=N/A
  - dispatch=119035, prob=0.0462, sku_counts=12.0, fill_ratio=N/A

### 2. FP Cases (Ensemble predicts YES, GT = NO — over-confident)
- Count: 90 / 466 = 19.31%
- Avg Ensemble baseline probability: 0.8451
- Std probability: 0.1221
- Most variable features:
  - `spare_capacity`: mean=12692.6222, var=13928010.4795
  - `sku_average_volume`: mean=3150.4993, var=142913.4254
  - `wl_to_vehicle_wl_total`: mean=1081.2407, var=35913.2792
  - `wl_to_vehicle_wl_max`: mean=182.0602, var=436.4173
  - `sku_length_var`: mean=49.7542, var=169.2555
- Typical FP cases:
  - dispatch=226909, prob=0.9806, sku_counts=7.0, fill_ratio=N/A
  - dispatch=300240, prob=0.9776, sku_counts=9.0, fill_ratio=N/A
  - dispatch=341161, prob=0.9772, sku_counts=12.0, fill_ratio=N/A

### 3. Easy TN (SVM predicts NO, GT = NO — correct negatives)
- Count: 376 / 466 = 80.69% of GT=NO
- Avg Ensemble baseline probability: 0.0782
- Std probability: 0.1055
- Most variable features:
  - `spare_capacity`: mean=4628.7952, var=20627218.5565
  - `sku_average_volume`: mean=3153.6938, var=206998.0010
  - `wl_to_vehicle_wl_total`: mean=1354.7019, var=41583.0325
  - `wl_to_vehicle_wl_max`: mean=179.9911, var=474.5788
  - `sku_length_var`: mean=46.1215, var=138.6307
- Borderline-correct TN (closest to prob=0.5 — do not harm these patterns):
  - dispatch=530, prob=0.4889, sku_counts=11.0, fill_ratio=N/A
  - dispatch=196962, prob=0.4775, sku_counts=12.0, fill_ratio=N/A
  - dispatch=301936, prob=0.4773, sku_counts=12.0, fill_ratio=N/A

### 4. Easy TP (SVM predicts YES, GT = YES — correct positives)
- Count: 1997 / 2034 = 98.18% of GT=YES
- Avg Ensemble baseline probability: 0.9702
- Std probability: 0.0495
- Most variable features:
  - `spare_capacity`: mean=18820.5433, var=50821212.2892
  - `sku_average_volume`: mean=2844.8353, var=205523.5069
  - `wl_to_vehicle_wl_total`: mean=914.3953, var=70640.3375
  - `wl_to_vehicle_wl_max`: mean=162.2091, var=734.9414
  - `wl_to_vehicle_wl_min`: mean=48.7131, var=247.8298
- Borderline-correct TP (closest to prob=0.5 — do not harm these patterns):
  - dispatch=119520, prob=0.5283, sku_counts=6.0, fill_ratio=N/A
  - dispatch=122247, prob=0.5286, sku_counts=12.0, fill_ratio=N/A
  - dispatch=122066, prob=0.5370, sku_counts=9.0, fill_ratio=N/A

### 5a. Easy TP rows (feature-similar to FN cloud — SVM still correct)
These are **positive** examples that sit near FN cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=122164, prob=0.9413, sku_counts=12.0, fill_ratio=N/A
  - dispatch=158166, prob=0.8281, sku_counts=12.0, fill_ratio=N/A
  - dispatch=300358, prob=0.9470, sku_counts=12.0, fill_ratio=N/A

### 5b. Easy TN rows (feature-similar to FP cloud — SVM still correct)
These are **negative** examples that sit near FP cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=344708, prob=0.0676, sku_counts=9.0, fill_ratio=N/A
  - dispatch=1765, prob=0.1664, sku_counts=10.0, fill_ratio=N/A
  - dispatch=195860, prob=0.2331, sku_counts=9.0, fill_ratio=N/A

### 6. Debug Instruction
请分析以上FN/FP错题与Easy TN/TP对照样本，找出SVM线性模型在错题上的失效规律，设计**窄**的条件：只在确有必要时覆盖SVM。对 Easy TN/TP 及 §5 中的对照行，除非逻辑上必须触发，否则应保持 `return -1`。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。