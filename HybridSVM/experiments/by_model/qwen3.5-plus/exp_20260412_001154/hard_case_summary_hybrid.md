## Hybrid hard cases (SVM + rules, test split)

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 773 / 2034 = 38.00%
- Avg SVM P(y=1) (reference): 0.8894
- Std probability: 0.1700
- Most variable features:
  - `spare_capacity`: mean=18658.8305, var=38889353.7009
  - `sku_average_volume`: mean=2934.8102, var=208551.7223
  - `wl_to_vehicle_wl_total`: mean=925.5703, var=52504.8710
  - `wl_to_vehicle_wl_max`: mean=175.1445, var=698.9653
  - `wl_to_vehicle_wl_min`: mean=47.2186, var=212.8691
- Typical FN cases:
  - dispatch=84625, prob=1.0000, sku_counts=2.0, fill_ratio=42204.0
  - dispatch=84677, prob=1.0000, sku_counts=6.0, fill_ratio=34067.0
  - dispatch=119445, prob=1.0000, sku_counts=3.0, fill_ratio=37200.0

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 78 / 466 = 16.74%
- Avg SVM P(y=1) (reference): 0.6175
- Std probability: 0.1603
- Most variable features:
  - `spare_capacity`: mean=10428.9615, var=14482640.4729
  - `sku_average_volume`: mean=3074.3454, var=186950.5765
  - `wl_to_vehicle_wl_total`: mean=1174.5513, var=41259.2760
  - `wl_to_vehicle_wl_max`: mean=176.4957, var=533.2195
  - `wl_to_vehicle_wl_min`: mean=48.7874, var=174.3242
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=20288.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0

### 3. Debug Instruction
请分析以上FN和FP错题数据，找出SVM线性模型在这些case上失效的规律，设计针对性规则对SVM预测进行后处理修正。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。