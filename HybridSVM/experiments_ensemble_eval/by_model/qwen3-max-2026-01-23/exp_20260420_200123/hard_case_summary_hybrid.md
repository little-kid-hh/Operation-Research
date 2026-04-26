## Hybrid hard cases (SVM + rules, test split)

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 68 / 2034 = 3.34%
- Avg SVM P(y=1) (reference): 0.5024
- Std probability: 0.1826
- Most variable features:
  - `spare_capacity`: mean=8863.8382, var=7909089.8415
  - `sku_average_volume`: mean=2912.2822, var=400724.5019
  - `wl_to_vehicle_wl_total`: mean=1265.3002, var=33671.7261
  - `wl_to_vehicle_wl_max`: mean=172.2488, var=483.6608
  - `wl_to_vehicle_wl_avg`: mean=99.8815, var=209.8757
- Typical FN cases:
  - dispatch=302575, prob=0.9156, sku_counts=16.0, fill_ratio=N/A
  - dispatch=301804, prob=0.8932, sku_counts=16.0, fill_ratio=N/A
  - dispatch=300658, prob=0.8806, sku_counts=16.0, fill_ratio=N/A

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 139 / 466 = 29.83%
- Avg SVM P(y=1) (reference): 0.6163
- Std probability: 0.2015
- Most variable features:
  - `spare_capacity`: mean=12061.4820, var=13361813.1849
  - `sku_average_volume`: mean=3248.7694, var=126832.1222
  - `wl_to_vehicle_wl_total`: mean=1091.0402, var=31017.7542
  - `wl_to_vehicle_wl_max`: mean=184.7152, var=412.6849
  - `sku_length_var`: mean=48.4953, var=181.5834
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=N/A
  - dispatch=195395, prob=0.9735, sku_counts=7.0, fill_ratio=N/A
  - dispatch=85889, prob=0.9697, sku_counts=8.0, fill_ratio=N/A

### 3. Debug Instruction
请分析以上FN/FP错题与Easy TN/TP对照样本，找出SVM线性模型在错题上的失效规律，设计**窄**的条件：只在确有必要时覆盖SVM。对 Easy TN/TP 及 §5 中的对照行，除非逻辑上必须触发，否则应保持 `return -1`。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。