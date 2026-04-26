## 3D-BPP Hard Case Analysis

### 1. FN Cases (Ensemble predicts NO, GT = YES — under-confident)
- Count: 43 / 2034 = 2.11%
- Avg Ensemble baseline probability: 0.2390
- Std probability: 0.1486
- Most variable features:
  - `spare_capacity`: mean=8468.3721, var=7486840.7918
  - `sku_average_volume`: mean=3184.8596, var=222055.9241
  - `wl_to_vehicle_wl_total`: mean=1229.6802, var=29402.4777
  - `wl_to_vehicle_wl_max`: mean=179.6899, var=385.3124
  - `sku_length_var`: mean=48.7711, var=108.3691
- Typical FN cases:
  - dispatch=226709, prob=0.0139, sku_counts=11.0, fill_ratio=N/A
  - dispatch=301553, prob=0.0276, sku_counts=11.0, fill_ratio=N/A
  - dispatch=119035, prob=0.0382, sku_counts=12.0, fill_ratio=N/A

### 2. FP Cases (Ensemble predicts YES, GT = NO — over-confident)
- Count: 88 / 466 = 18.88%
- Avg Ensemble baseline probability: 0.8213
- Std probability: 0.1189
- Most variable features:
  - `spare_capacity`: mean=12620.0909, var=13843354.0372
  - `sku_average_volume`: mean=3146.0092, var=147333.9945
  - `wl_to_vehicle_wl_total`: mean=1082.6089, var=35949.0510
  - `wl_to_vehicle_wl_max`: mean=181.2547, var=457.0569
  - `sku_length_var`: mean=49.5128, var=171.5543
- Typical FP cases:
  - dispatch=226909, prob=0.9742, sku_counts=7.0, fill_ratio=N/A
  - dispatch=341161, prob=0.9705, sku_counts=12.0, fill_ratio=N/A
  - dispatch=300240, prob=0.9692, sku_counts=9.0, fill_ratio=N/A

### 3. Easy TN (SVM predicts NO, GT = NO — correct negatives)
- Count: 378 / 466 = 81.12% of GT=NO
- Avg Ensemble baseline probability: 0.0657
- Std probability: 0.0993
- Most variable features:
  - `spare_capacity`: mean=4688.3466, var=21223084.5915
  - `sku_average_volume`: mean=3154.7222, var=205617.4260
  - `wl_to_vehicle_wl_total`: mean=1352.9365, var=42111.0225
  - `wl_to_vehicle_wl_max`: mean=180.1896, var=470.1801
  - `sku_length_var`: mean=46.1969, var=138.7164
- Borderline-correct TN (closest to prob=0.5 — do not harm these patterns):
  - dispatch=197529, prob=0.4858, sku_counts=8.0, fill_ratio=N/A
  - dispatch=228234, prob=0.4707, sku_counts=13.0, fill_ratio=N/A
  - dispatch=342159, prob=0.4688, sku_counts=10.0, fill_ratio=N/A

### 4. Easy TP (SVM predicts YES, GT = YES — correct positives)
- Count: 1991 / 2034 = 97.89% of GT=YES
- Avg Ensemble baseline probability: 0.9646
- Std probability: 0.0508
- Most variable features:
  - `spare_capacity`: mean=18851.0588, var=50644033.3100
  - `sku_average_volume`: mean=2844.0692, var=205434.2360
  - `wl_to_vehicle_wl_total`: mean=913.3400, var=70364.0643
  - `wl_to_vehicle_wl_max`: mean=162.1739, var=735.5339
  - `wl_to_vehicle_wl_min`: mean=48.7243, var=248.1787
- Borderline-correct TP (closest to prob=0.5 — do not harm these patterns):
  - dispatch=119520, prob=0.5099, sku_counts=6.0, fill_ratio=N/A
  - dispatch=122066, prob=0.5172, sku_counts=9.0, fill_ratio=N/A
  - dispatch=227315, prob=0.5178, sku_counts=9.0, fill_ratio=N/A

### 5a. Easy TP rows (feature-similar to FN cloud — SVM still correct)
These are **positive** examples that sit near FN cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=122164, prob=0.9117, sku_counts=12.0, fill_ratio=N/A
  - dispatch=158166, prob=0.7829, sku_counts=12.0, fill_ratio=N/A
  - dispatch=343146, prob=0.9073, sku_counts=12.0, fill_ratio=N/A

### 5b. Easy TN rows (feature-similar to FP cloud — SVM still correct)
These are **negative** examples that sit near FP cases in feature space; your rule must **not** override SVM on rows like these unless necessary.
  - dispatch=344708, prob=0.0561, sku_counts=9.0, fill_ratio=N/A
  - dispatch=195860, prob=0.1623, sku_counts=9.0, fill_ratio=N/A
  - dispatch=1765, prob=0.1589, sku_counts=10.0, fill_ratio=N/A

### 6. Debug Instruction
请分析以上FN/FP错题与Easy TN/TP对照样本，找出SVM线性模型在错题上的失效规律，设计**窄**的条件：只在确有必要时覆盖SVM。对 Easy TN/TP 及 §5 中的对照行，除非逻辑上必须触发，否则应保持 `return -1`。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。