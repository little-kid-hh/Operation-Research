## Hybrid hard cases (SVM + rules, test split)

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 35 / 2034 = 1.72%
- Avg SVM P(y=1) (reference): 0.4299
- Std probability: 0.0555
- Most variable features:
  - `spare_capacity`: mean=8146.0857, var=2351902.0784
  - `sku_average_volume`: mean=2956.0543, var=94830.6190
  - `wl_to_vehicle_wl_total`: mean=1255.2500, var=18567.0258
  - `wl_to_vehicle_wl_max`: mean=181.6310, var=306.6108
  - `sku_length_var`: mean=50.3517, var=157.5104
- Typical FN cases:
  - dispatch=229081, prob=0.2966, sku_counts=14.0, fill_ratio=5606.0
  - dispatch=342320, prob=0.3084, sku_counts=14.0, fill_ratio=7085.0
  - dispatch=196911, prob=0.3242, sku_counts=13.0, fill_ratio=5995.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 305 / 466 = 65.45%
- Avg SVM P(y=1) (reference): 0.3854
- Std probability: 0.2711
- Most variable features:
  - `spare_capacity`: mean=7731.8951, var=27548928.0087
  - `sku_average_volume`: mean=3290.0295, var=165157.6815
  - `wl_to_vehicle_wl_total`: mean=1223.4508, var=37071.5128
  - `wl_to_vehicle_wl_max`: mean=180.9426, var=424.0841
  - `sku_length_var`: mean=45.3111, var=148.0763
- Typical FP cases:
  - dispatch=226675, prob=0.0014, sku_counts=12.0, fill_ratio=-5329.0
  - dispatch=229229, prob=0.0029, sku_counts=12.0, fill_ratio=-4832.0
  - dispatch=226681, prob=0.0046, sku_counts=11.0, fill_ratio=-3012.0

### 3. Debug Instruction
请分析以上FN和FP错题数据，找出SVM线性模型在这些case上失效的规律，设计针对性规则对SVM预测进行后处理修正。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。