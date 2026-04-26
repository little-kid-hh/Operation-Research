## 3D-BPP Hard Case Analysis

### 1. FN Cases (Ensemble predicts NO, GT = YES — under-confident)
- Count: 44 / 2034 = 2.16%
- Avg Ensemble baseline probability: 0.2681
- Std probability: 0.1544
- Most variable features:
  - `spare_capacity`: mean=8649.8864, var=9515996.9644
  - `sku_average_volume`: mean=3206.9960, var=259530.3848
  - `wl_to_vehicle_wl_total`: mean=1221.7330, var=33156.8572
  - `wl_to_vehicle_wl_max`: mean=180.1136, var=366.4644
  - `wl_to_vehicle_wl_avg`: mean=106.1935, var=124.7461
- Typical FN cases:
  - dispatch=226709, prob=0.0140, sku_counts=11.0, fill_ratio=N/A
  - dispatch=301553, prob=0.0244, sku_counts=11.0, fill_ratio=N/A
  - dispatch=196215, prob=0.0289, sku_counts=13.0, fill_ratio=N/A

### 2. FP Cases (Ensemble predicts YES, GT = NO — over-confident)
- Count: 91 / 466 = 19.53%
- Avg Ensemble baseline probability: 0.8087
- Std probability: 0.1331
- Most variable features:
  - `spare_capacity`: mean=12761.4286, var=14061739.2559
  - `sku_average_volume`: mean=3166.8322, var=134895.1718
  - `wl_to_vehicle_wl_total`: mean=1074.5467, var=32754.8445
  - `wl_to_vehicle_wl_max`: mean=182.2253, var=460.4404
  - `wl_to_vehicle_wl_min`: mean=47.6877, var=181.6555
- Typical FP cases:
  - dispatch=341161, prob=0.9745, sku_counts=12.0, fill_ratio=N/A
  - dispatch=300240, prob=0.9727, sku_counts=9.0, fill_ratio=N/A
  - dispatch=226909, prob=0.9714, sku_counts=7.0, fill_ratio=N/A

### 3. Easy TN (SVM predicts NO, GT = NO — correct negatives)
- Count: 375 / 466 = 80.47% of GT=NO
- Avg Ensemble baseline probability: 0.0667
- Std probability: 0.1008
- Most variable features:
  - `spare_capacity`: mean=4590.5947, var=20167339.0784
  - `sku_average_volume`: mean=3149.7389, var=209059.5754
  - `wl_to_vehicle_wl_total`: mean=1357.0556, var=41260.3377
  - `wl_to_vehicle_wl_max`: mean=179.9456, var=468.6651
  - `sku_length_var`: mean=46.3151, var=138.0235
- Borderline-correct TN (closest to prob=0.5 — do not harm these patterns):
  - dispatch=195228, prob=0.4751, sku_counts=13.0, fill_ratio=N/A
  - dispatch=341212, prob=0.4738, sku_counts=11.0, fill_ratio=N/A
  - dispatch=302799, prob=0.4677, sku_counts=17.0, fill_ratio=N/A

### 4. Easy TP (SVM predicts YES, GT = YES — correct positives)
- Count: 1990 / 2034 = 97.84% of GT=YES
- Avg Ensemble baseline probability: 0.9646
- Std probability: 0.0527
- Most variable features:
  - `spare_capacity`: mean=18852.2628, var=50649293.0470
  - `sku_average_volume`: mean=2843.4085, var=204194.0712
  - `wl_to_vehicle_wl_total`: mean=913.3568, var=70361.1266
  - `wl_to_vehicle_wl_max`: mean=162.1558, var=735.6400
  - `wl_to_vehicle_wl_min`: mean=48.7274, var=248.4003
- Borderline-correct TP (closest to prob=0.5 — do not harm these patterns):
  - dispatch=342217, prob=0.5008, sku_counts=9.0, fill_ratio=N/A
  - dispatch=226578, prob=0.5068, sku_counts=9.0, fill_ratio=N/A
  - dispatch=2439, prob=0.5106, sku_counts=11.0, fill_ratio=N/A

### 5a. Easy TP rows (feature-similar to FN cloud — SVM still correct)
These are **positive** examples that sit near FN cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=300318, prob=0.9120, sku_counts=11.0, fill_ratio=N/A
  - dispatch=158189, prob=0.5392, sku_counts=12.0, fill_ratio=N/A
  - dispatch=300767, prob=0.9641, sku_counts=12.0, fill_ratio=N/A

### 5b. Easy TN rows (feature-similar to FP cloud — SVM still correct)
These are **negative** examples that sit near FP cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=1765, prob=0.1322, sku_counts=10.0, fill_ratio=N/A
  - dispatch=344708, prob=0.0592, sku_counts=9.0, fill_ratio=N/A
  - dispatch=195860, prob=0.1811, sku_counts=9.0, fill_ratio=N/A

### 6. Debug Instruction
请分析以上FN/FP错题与Easy TN/TP对照样本，找出SVM线性模型在错题上的失效规律，设计**窄**的条件：只在确有必要时覆盖SVM。对 Easy TN/TP 及 §5 中的对照行，除非逻辑上必须触发，否则应保持 `return -1`。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。