## Hybrid hard cases (SVM + rules, test split)

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 774 / 2034 = 38.05%
- Avg SVM P(y=1) (reference): 0.8967
- Std probability: 0.1584
- Most variable features:
  - `spare_capacity`: mean=17235.4225, var=25063939.2724
  - `sku_average_volume`: mean=2826.8031, var=185369.1851
  - `wl_to_vehicle_wl_total`: mean=975.2837, var=42969.2372
  - `wl_to_vehicle_wl_max`: mean=169.9074, var=569.1945
  - `wl_to_vehicle_wl_min`: mean=44.1591, var=143.0456
- Typical FN cases:
  - dispatch=85325, prob=0.9999, sku_counts=9.0, fill_ratio=26247.0
  - dispatch=341828, prob=0.9999, sku_counts=8.0, fill_ratio=29004.0
  - dispatch=196542, prob=0.9999, sku_counts=6.0, fill_ratio=29448.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 100 / 466 = 21.46%
- Avg SVM P(y=1) (reference): 0.6041
- Std probability: 0.1523
- Most variable features:
  - `spare_capacity`: mean=10857.6800, var=12026037.7576
  - `sku_average_volume`: mean=3136.1949, var=155589.3475
  - `wl_to_vehicle_wl_total`: mean=1157.9917, var=35016.5117
  - `wl_to_vehicle_wl_max`: mean=179.7708, var=501.1749
  - `sku_length_var`: mean=45.3488, var=186.5129
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0
  - dispatch=196266, prob=0.9344, sku_counts=7.0, fill_ratio=19610.0

### 3. Debug Instruction
请分析以上FN和FP错题数据，找出SVM线性模型在这些case上失效的规律，设计针对性规则对SVM预测进行后处理修正。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。