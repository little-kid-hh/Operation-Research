## 3D-BPP Hard Case Analysis

### 1. FN Cases (SVM predicts NO, GT = YES — under-confident)
- Count: 65 / 2034 = 3.20%
- Avg SVM probability: 0.4043
- Std probability: 0.0744
- Most variable features:
  - `spare_capacity`: mean=9239.7538, var=9011310.8009
  - `sku_average_volume`: mean=3268.1364, var=226878.5756
  - `wl_to_vehicle_wl_total`: mean=1175.9231, var=26980.8146
  - `wl_to_vehicle_wl_max`: mean=182.9359, var=313.9820
  - `wl_to_vehicle_wl_min`: mean=47.2051, var=174.6620
- Typical FN cases:
  - dispatch=301553, prob=0.1719, sku_counts=11.0, fill_ratio=N/A
  - dispatch=226709, prob=0.1750, sku_counts=11.0, fill_ratio=N/A
  - dispatch=300791, prob=0.2482, sku_counts=11.0, fill_ratio=N/A

### 2. FP Cases (SVM predicts YES, GT = NO — over-confident)
- Count: 116 / 466 = 24.89%
- Avg SVM probability: 0.6884
- Std probability: 0.1354
- Most variable features:
  - `spare_capacity`: mean=12277.3362, var=15761680.4646
  - `sku_average_volume`: mean=3127.2596, var=159062.2927
  - `wl_to_vehicle_wl_total`: mean=1104.1918, var=42023.1959
  - `wl_to_vehicle_wl_max`: mean=182.3958, var=488.2030
  - `sku_length_var`: mean=48.1996, var=206.1213
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=N/A
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=N/A
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=N/A

### 3. Easy TN (SVM predicts NO, GT = NO — correct negatives)
- Count: 350 / 466 = 75.11% of GT=NO
- Avg SVM probability: 0.2039
- Std probability: 0.1462
- Most variable features:
  - `spare_capacity`: mean=4167.4057, var=17636425.2354
  - `sku_average_volume`: mean=3161.6334, var=206114.3693
  - `wl_to_vehicle_wl_total`: mean=1367.4095, var=38248.2299
  - `wl_to_vehicle_wl_max`: mean=179.7262, var=459.3645
  - `sku_length_var`: mean=46.3669, var=126.0392
- Borderline-correct TN (closest to prob=0.5 — do not harm these patterns):
  - dispatch=197978, prob=0.4960, sku_counts=11.0, fill_ratio=N/A
  - dispatch=302738, prob=0.4913, sku_counts=16.0, fill_ratio=N/A
  - dispatch=343035, prob=0.4841, sku_counts=15.0, fill_ratio=N/A

### 4. Easy TP (SVM predicts YES, GT = YES — correct positives)
- Count: 1969 / 2034 = 96.80% of GT=YES
- Avg SVM probability: 0.9204
- Std probability: 0.1132
- Most variable features:
  - `spare_capacity`: mean=18941.6018, var=50372381.0675
  - `sku_average_volume`: mean=2837.5123, var=201646.0134
  - `wl_to_vehicle_wl_total`: mean=911.5801, var=70807.8338
  - `wl_to_vehicle_wl_max`: mean=161.8711, var=734.1803
  - `wl_to_vehicle_wl_min`: mean=48.6857, var=247.5397
- Borderline-correct TP (closest to prob=0.5 — do not harm these patterns):
  - dispatch=156469, prob=0.5045, sku_counts=14.0, fill_ratio=N/A
  - dispatch=303512, prob=0.5063, sku_counts=11.0, fill_ratio=N/A
  - dispatch=226456, prob=0.5087, sku_counts=13.0, fill_ratio=N/A

### 5a. Easy TP rows (feature-similar to FN cloud — SVM still correct)
These are **positive** examples that sit near FN cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=226446, prob=0.5447, sku_counts=11.0, fill_ratio=N/A
  - dispatch=196177, prob=0.7799, sku_counts=11.0, fill_ratio=N/A
  - dispatch=156241, prob=0.7672, sku_counts=12.0, fill_ratio=N/A

### 5b. Easy TN rows (feature-similar to FP cloud — SVM still correct)
These are **negative** examples that sit near FP cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=156372, prob=0.3964, sku_counts=10.0, fill_ratio=N/A
  - dispatch=344708, prob=0.3389, sku_counts=9.0, fill_ratio=N/A
  - dispatch=84589, prob=0.3901, sku_counts=9.0, fill_ratio=N/A

### 6. Debug Instruction
请分析以上FN/FP错题与Easy TN/TP对照样本，找出SVM线性模型在错题上的失效规律，设计**窄**的条件：只在确有必要时覆盖SVM。对 Easy TN/TP 及 §5 中的对照行，除非逻辑上必须触发，否则应保持 `return -1`。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。