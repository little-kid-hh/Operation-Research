## Hybrid hard cases (SVM + rules, test split)

### 1. FN Cases (Hybrid (SVM + rules) predicts NO, GT = YES — under-confident)
- Count: 510 / 2034 = 25.07%
- Avg SVM P(y=1) (reference): 0.8500
- Std probability: 0.1379
- Most variable features:
  - `spare_capacity`: mean=14994.0157, var=50268817.4821
  - `sku_average_volume`: mean=2763.4307, var=178863.6418
  - `wl_to_vehicle_wl_total`: mean=1107.4845, var=79874.0837
  - `wl_to_vehicle_wl_max`: mean=168.2655, var=619.5975
  - `wl_to_vehicle_wl_min`: mean=51.4698, var=214.0057
- Typical FN cases:
  - dispatch=41673, prob=1.0000, sku_counts=3.0, fill_ratio=N/A
  - dispatch=303390, prob=1.0000, sku_counts=3.0, fill_ratio=N/A
  - dispatch=122168, prob=1.0000, sku_counts=2.0, fill_ratio=N/A

### 2. FP Cases (Hybrid (SVM + rules) predicts YES, GT = NO — over-confident)
- Count: 324 / 466 = 69.53%
- Avg SVM P(y=1) (reference): 0.3590
- Std probability: 0.2306
- Most variable features:
  - `spare_capacity`: mean=7867.3056, var=18919293.7986
  - `sku_average_volume`: mean=3203.8282, var=180113.6697
  - `wl_to_vehicle_wl_total`: mean=1230.9298, var=38362.0229
  - `wl_to_vehicle_wl_max`: mean=181.3683, var=423.3003
  - `sku_length_var`: mean=46.8845, var=149.0774
- Typical FP cases:
  - dispatch=226909, prob=0.9860, sku_counts=7.0, fill_ratio=N/A
  - dispatch=196613, prob=0.0178, sku_counts=11.0, fill_ratio=N/A
  - dispatch=226935, prob=0.0250, sku_counts=10.0, fill_ratio=N/A

### 3. Debug Instruction
请分析以上FN/FP错题与Easy TN/TP对照样本，找出SVM线性模型在错题上的失效规律，设计**窄**的条件：只在确有必要时覆盖SVM。对 Easy TN/TP 及 §5 中的对照行，除非逻辑上必须触发，否则应保持 `return -1`。输出一个Python函数 `apply_rule_patch(svm_prob, features) -> int`，当规则判断应覆盖SVM时返回被修正的标签(0或1)，否则返回-1表示不修改。