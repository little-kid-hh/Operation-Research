## 3D-BPP Hard Case Analysis

### 1. FN Cases (TabTreeFormer(RF) predicts NO, GT = YES — under-confident)
- Count: 77 / 2034 = 3.79%
- Avg Ensemble baseline probability: 0.0868
- Std probability: 0.1161
- Most variable features:
  - `spare_capacity`: mean=10513.8571, var=17231492.2783
  - `sku_average_volume`: mean=3220.1929, var=251834.1857
  - `wl_to_vehicle_wl_total`: mean=1159.4318, var=42308.6666
  - `wl_to_vehicle_wl_max`: mean=182.2186, var=366.8896
  - `sku_length_var`: mean=47.6636, var=184.5185
- Typical FN cases:
  - dispatch=196215, prob=0.0004, sku_counts=13.0, fill_ratio=N/A
  - dispatch=226709, prob=0.0004, sku_counts=11.0, fill_ratio=N/A
  - dispatch=227608, prob=0.0005, sku_counts=14.0, fill_ratio=N/A

### 2. FP Cases (TabTreeFormer(RF) predicts YES, GT = NO — over-confident)
- Count: 76 / 466 = 16.31%
- Avg Ensemble baseline probability: 0.9369
- Std probability: 0.0968
- Most variable features:
  - `spare_capacity`: mean=12312.1184, var=13615295.0518
  - `sku_average_volume`: mean=3137.9474, var=107653.5160
  - `wl_to_vehicle_wl_total`: mean=1096.8257, var=30117.2330
  - `wl_to_vehicle_wl_max`: mean=180.6908, var=419.6297
  - `sku_length_var`: mean=49.4698, var=167.4384
- Typical FP cases:
  - dispatch=156927, prob=0.9996, sku_counts=12.0, fill_ratio=N/A
  - dispatch=156245, prob=0.9996, sku_counts=12.0, fill_ratio=N/A
  - dispatch=300240, prob=0.9995, sku_counts=9.0, fill_ratio=N/A

### 3. Easy TN (SVM predicts NO, GT = NO — correct negatives)
- Count: 390 / 466 = 83.69% of GT=NO
- Avg Ensemble baseline probability: 0.0183
- Std probability: 0.0633
- Most variable features:
  - `spare_capacity`: mean=4992.4154, var=23817338.3710
  - `sku_average_volume`: mean=3156.0251, var=211517.2921
  - `wl_to_vehicle_wl_total`: mean=1341.8483, var=46641.9239
  - `wl_to_vehicle_wl_max`: mean=180.3323, var=477.2565
  - `sku_length_var`: mean=46.3074, var=140.9102
- Borderline-correct TN (closest to prob=0.5 — do not harm these patterns):
  - dispatch=342205, prob=0.4634, sku_counts=14.0, fill_ratio=N/A
  - dispatch=226511, prob=0.4545, sku_counts=9.0, fill_ratio=N/A
  - dispatch=119225, prob=0.4388, sku_counts=10.0, fill_ratio=N/A

### 4. Easy TP (SVM predicts YES, GT = YES — correct positives)
- Count: 1957 / 2034 = 96.21% of GT=YES
- Avg Ensemble baseline probability: 0.9930
- Std probability: 0.0369
- Most variable features:
  - `spare_capacity`: mean=18950.9612, var=50634168.5851
  - `sku_average_volume`: mean=2836.7582, var=200905.9558
  - `wl_to_vehicle_wl_total`: mean=910.6081, var=70376.4095
  - `wl_to_vehicle_wl_max`: mean=161.7701, var=733.1129
  - `wl_to_vehicle_wl_min`: mean=48.7568, var=247.8723
- Borderline-correct TP (closest to prob=0.5 — do not harm these patterns):
  - dispatch=342371, prob=0.5124, sku_counts=8.0, fill_ratio=N/A
  - dispatch=85868, prob=0.5355, sku_counts=14.0, fill_ratio=N/A
  - dispatch=301765, prob=0.5780, sku_counts=11.0, fill_ratio=N/A

### 5a. Easy TP rows (feature-similar to FN cloud — SVM still correct)
These are **positive** examples that sit near FN cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=156264, prob=0.8857, sku_counts=11.0, fill_ratio=N/A
  - dispatch=84793, prob=0.9985, sku_counts=11.0, fill_ratio=N/A
  - dispatch=195239, prob=0.9086, sku_counts=11.0, fill_ratio=N/A

### 5b. Easy TN rows (feature-similar to FP cloud — SVM still correct)
These are **negative** examples that sit near FP cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=84872, prob=0.0551, sku_counts=11.0, fill_ratio=N/A
  - dispatch=119045, prob=0.0275, sku_counts=9.0, fill_ratio=N/A
  - dispatch=226511, prob=0.4545, sku_counts=9.0, fill_ratio=N/A

### 6. Debug Instruction
请分析以上FN/FP错题与Easy TN/TP对照样本，找出SVM线性模型在错题上的失效规律，设计**窄**的条件：只在确有必要时覆盖SVM。对 Easy TN/TP 及 §5 中的对照行，除非逻辑上必须触发，否则应保持 `return -1`。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。