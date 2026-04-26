## 3D-BPP Hard Case Analysis

### 1. FN Cases (Ensemble predicts NO, GT = YES — under-confident)
- Count: 50 / 2034 = 2.46%
- Avg Ensemble baseline probability: 0.2792
- Std probability: 0.1438
- Most variable features:
  - `spare_capacity`: mean=8374.6200, var=7287988.7556
  - `sku_average_volume`: mean=3130.1312, var=211269.8626
  - `wl_to_vehicle_wl_total`: mean=1245.9917, var=29305.8326
  - `wl_to_vehicle_wl_max`: mean=179.2667, var=393.3997
  - `sku_length_var`: mean=46.7758, var=114.4897
- Typical FN cases:
  - dispatch=226709, prob=0.0124, sku_counts=11.0, fill_ratio=N/A
  - dispatch=227608, prob=0.0257, sku_counts=14.0, fill_ratio=N/A
  - dispatch=301553, prob=0.0383, sku_counts=11.0, fill_ratio=N/A

### 2. FP Cases (Ensemble predicts YES, GT = NO — over-confident)
- Count: 94 / 466 = 20.17%
- Avg Ensemble baseline probability: 0.7891
- Std probability: 0.1336
- Most variable features:
  - `spare_capacity`: mean=12624.7447, var=12752879.2540
  - `sku_average_volume`: mean=3154.3492, var=125953.4649
  - `wl_to_vehicle_wl_total`: mean=1083.9761, var=30487.6170
  - `wl_to_vehicle_wl_max`: mean=182.9832, var=437.2536
  - `sku_length_var`: mean=49.4514, var=179.5934
- Typical FP cases:
  - dispatch=341161, prob=0.9709, sku_counts=12.0, fill_ratio=N/A
  - dispatch=380300, prob=0.9678, sku_counts=10.0, fill_ratio=N/A
  - dispatch=226909, prob=0.9663, sku_counts=7.0, fill_ratio=N/A

### 3. Easy TN (SVM predicts NO, GT = NO — correct negatives)
- Count: 372 / 466 = 79.83% of GT=NO
- Avg Ensemble baseline probability: 0.0689
- Std probability: 0.1050
- Most variable features:
  - `spare_capacity`: mean=4559.2392, var=20567614.8487
  - `sku_average_volume`: mean=3152.7553, var=211974.1434
  - `wl_to_vehicle_wl_total`: mean=1356.9512, var=42581.9755
  - `wl_to_vehicle_wl_max`: mean=179.7357, var=473.4862
  - `sku_length_var`: mean=46.1590, var=136.0785
- Borderline-correct TN (closest to prob=0.5 — do not harm these patterns):
  - dispatch=300165, prob=0.4970, sku_counts=10.0, fill_ratio=N/A
  - dispatch=195940, prob=0.4882, sku_counts=7.0, fill_ratio=N/A
  - dispatch=195309, prob=0.4745, sku_counts=11.0, fill_ratio=N/A

### 4. Easy TP (SVM predicts YES, GT = YES — correct positives)
- Count: 1984 / 2034 = 97.54% of GT=YES
- Avg Ensemble baseline probability: 0.9610
- Std probability: 0.0568
- Most variable features:
  - `spare_capacity`: mean=18890.0539, var=50370165.0752
  - `sku_average_volume`: mean=2844.2460, var=206102.2069
  - `wl_to_vehicle_wl_total`: mean=911.8128, var=69888.8318
  - `wl_to_vehicle_wl_max`: mean=162.1228, var=735.8498
  - `wl_to_vehicle_wl_min`: mean=48.7418, var=248.6346
- Borderline-correct TP (closest to prob=0.5 — do not harm these patterns):
  - dispatch=228452, prob=0.5034, sku_counts=8.0, fill_ratio=N/A
  - dispatch=341824, prob=0.5057, sku_counts=13.0, fill_ratio=N/A
  - dispatch=300214, prob=0.5274, sku_counts=11.0, fill_ratio=N/A

### 5a. Easy TP rows (feature-similar to FN cloud — SVM still correct)
These are **positive** examples that sit near FN cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=158166, prob=0.6693, sku_counts=12.0, fill_ratio=N/A
  - dispatch=122164, prob=0.8484, sku_counts=12.0, fill_ratio=N/A
  - dispatch=300358, prob=0.9505, sku_counts=12.0, fill_ratio=N/A

### 5b. Easy TN rows (feature-similar to FP cloud — SVM still correct)
These are **negative** examples that sit near FP cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=344708, prob=0.0964, sku_counts=9.0, fill_ratio=N/A
  - dispatch=195860, prob=0.1911, sku_counts=9.0, fill_ratio=N/A
  - dispatch=1765, prob=0.3900, sku_counts=10.0, fill_ratio=N/A

### 6. Debug Instruction
请分析以上FN/FP错题与Easy TN/TP对照样本，找出SVM线性模型在错题上的失效规律，设计**窄**的条件：只在确有必要时覆盖SVM。对 Easy TN/TP 及 §5 中的对照行，除非逻辑上必须触发，否则应保持 `return -1`。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。