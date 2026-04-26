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
  - dispatch=301553, prob=0.1719, sku_counts=11.0, fill_ratio=7126.0
  - dispatch=226709, prob=0.1750, sku_counts=11.0, fill_ratio=5282.0
  - dispatch=300791, prob=0.2482, sku_counts=11.0, fill_ratio=6228.0

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
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=21007.0
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=20288.0
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=18165.0

### 3. Debug Instruction
请分析以上FN和FP错题数据，找出SVM线性模型在这些case上失效的规律，设计针对性规则对SVM预测进行后处理修正。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。