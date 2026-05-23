# HybridSVM XGB -> GLM 无 Seed Exploratory Run 报告（2026-05-18）

## 0. 报告范围与状态说明

本报告只记录一条无 seed 的特征搜索实验：

- 路线：`XGB -> GLM -> linear SVM`
- 实验目录：`trials_archive/exp_20260518_173257/`

这条实验应当被视为 **exploratory run**，而不是最终的正式 10 轮实验，原因如下：

- 它最初是以无 seed 的新实验启动的；
- 之后又在同一个 `exp_dir` 上执行过 resume；
- 在 `code/HybridSVM/scripts/run_feature_search_agent.py` 中，resume 模式下的
  `--n-iters` 含义是“追加多少轮”，而不是“总共跑到多少轮”。

因此，这个目录最终累计了 `15` 条 trial，而不是最初预期的 `10` 条。

尽管如此，这条实验仍然具有明确的方法学价值，因为它回答了一个关键问题：

- `XGB -> GLM -> linear SVM` 这条路线，是否能够在完全没有人工 seed 特征的情况下自行启动？

这条实验给出的答案是 **可以**：第 `001` 轮已经被接受。

---

## 1. 端到端 Pipeline

```text
raw dispatch table + raw item table
    |
    v
数据加载与字段规范化
    |
    v
形成两种内部视图
  - agg_df: 每个 dispatch 一行
  - items_df: 每个 item 一行
    |
    v
固定划分上的 base40 SVM baseline
    |
    v
base40 XGB teacher analysis
    |
    v
GLM 生成冻结版 tree-guidance markdown
    |
    v
组装 feature-search prompt:
context + policy + memory + active bank
    |
    v
GLM 返回 POLICY_UPDATE + FEATURE_CODE + RATIONALE
    |
    v
在本地对全量 agg_df 与全量 items_df 执行 FEATURE_CODE
    |
    v
特征落表 + SVM 评估 + validator 判定
    |
    v
accept / reject / fail
    |
    v
更新 trials、memory、active bank 与 summary
```

---

## 2. Raw 数据在 Pipeline 中的作用

### 2.1 `dispatch` 的含义

在本仓库中，`dispatch` 是对原始字段 `发车号` 的规范化命名。

从业务含义上看，一个 `dispatch` 就是一条具体的发车 / 装车任务。模型的预测目标定义在这个粒度上。因此：

- 一条样本 = 一个 `dispatch`；
- 一个标签 = 该 `dispatch` 是否能够装载成功（`if_loaded`）；
- 同一个 `dispatch` 下可以对应多条 item 明细。

### 2.2 原始输入

这条 pipeline 从两张原始表开始：

- dispatch 级表：`repro_bundle/data/training_2orientations.csv`
- item 级表：`repro_bundle/data/物品信息和dblf信息.csv`

这两张表在后续流程中的职责不同，不能混为一谈。

### 2.3 raw dispatch 表的作用

dispatch 表承担了四类核心职责：

1. **定义样本**
   - 每一行对应一个 `dispatch`；
   - train/test 划分也是在这个粒度上完成的。

2. **提供监督标签**
   - 目标变量 `if_loaded` 来自这张表。

3. **提供 base40 聚合特征**
   - 去重并删除不需要的列之后，保留下来的聚合列构成固定的
     baseline 特征集合，供 SVM 与 teacher 模型使用。

4. **提供车辆级上下文**
   - 如 `vehicle_length`、`vehicle_width`、`vehicle_height`、
     `spare_capacity` 等字段，后续会在候选特征中作为分母、阈值或交互锚点使用。

相关实现位置：

- `repro_bundle/code/HybridSVM/src/svm_train.py`
- `repro_bundle/code/HybridSVM/src/feature_search.py:374-394`

### 2.4 raw item 表的作用

item 表并不是另一张独立的预测输入表。它的主要作用是为特征工程提供细粒度的几何信息。

具体而言，这张表被用来构造：

- 原始 item 尺寸：`item_length`, `item_width`, `item_height`
- 排序后的尺寸：`dim_s`, `dim_m`, `dim_l`
- 单件体积：`item_volume`
- 形状代理量：`item_flatness`
- 单件属性：例如 `if_fragile`

在候选特征生成阶段，这些 item 级记录会按照 `dispatch_id` 分组，以便计算例如：

- top-k 体积集中度；
- 长件 / 宽件 / 高件数量；
- 尾部 quantile；
- footprint pressure；
- item 级压力与 dispatch 级 slack 的交互项。

相关实现位置：

- `repro_bundle/code/HybridSVM/src/feature_search.py:337-371`

### 2.5 raw 数据在搜索循环中最终变成什么

完成加载与规范化后，原始数据会变成两种内部视图：

1. **`agg_df`**
   - 每个 `dispatch` 一行；
   - 包含聚合特征与车辆维度；
   - 既是 baseline 特征表，也是候选代码的 dispatch 级输入。

2. **`items_df`**
   - 每个 item 一行；
   - 保留细粒度几何信息；
   - 是候选代码的 item 级输入。

这一区分是整个 pipeline 的关键：

- LLM **不会**在 prompt 中直接看到完整 raw 表；
- LLM 只是在 prompt 中被告知：后续本地会向它写出的函数传入 `agg_df` 和 `items_df`；
- LLM 返回代码后，本地再用 **全量** `agg_df` 与 **全量** `items_df` 执行该代码。

因此，raw 数据在这条 pipeline 中同时承担了两种角色：

- 作为固定 baseline 表征的来源；
- 作为每一轮候选特征计算的完整执行底座。

---

## 3. 实际喂给 GLM 的内容

当前实现 **不会**把完整 dispatch 数值表或完整 item 明细表直接送入 GLM。

GLM 实际收到的是由以下四部分文本拼装而成的 prompt：

1. `context.md`
2. `policy.md`
3. `memory.md`
4. `active_feature_bank.md`

### 3.1 数据摘要

`context.md` 中与数据相关的部分包括：

- 去重后的 dispatch 数量；
- train/test 划分规模；
- 正例比例；
- 匹配到的 item 总行数；
- 每个 dispatch 的 item 数统计；
- base40 列名；
- item 表 schema。

这意味着，GLM 能看到的是：

- 表结构；
- 数据规模；
- 可用字段集合。

GLM 看不到的是：

- 全量 `10000` 行 dispatch 数值表；
- 全量 `100317` 行 item 数值表；
- 每个样本的原始数值明细。

### 3.2 学生模型摘要

prompt 还包含当前 baseline 线性 SVM 的压缩描述：

- 截距；
- 按绝对值排序的主要系数；
- 系数正负方向的解释。

这部分的作用是告诉 GLM：当前线性模型已经在哪些方向上较为敏感，哪些地方可能仍然需要通过显式非线性特征来补足。

### 3.3 teacher 模型摘要

prompt 中会插入冻结版 XGB guidance 文件：

- `repro_bundle/guidance/XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

这部分为 GLM 提供的是 teacher 侧证据，包括：

- 固定划分下 SVM 与 XGB 的性能差距；
- recovery count（`SVM 错、XGB 对` 等）；
- XGB 的高 gain 特征；
- 推荐的特征方向；
- 不建议投入的方向。

### 3.4 迭代记忆

prompt 还包含当前搜索状态：

- `policy.md`：硬约束与偏好的特征家族；
- `memory.md`：accepted / rejected / failed trial 的摘要；
- `active_feature_bank.md`：当前已经接受的增量特征及其指标。

因此，GLM 在每一轮真正依据的是：

- 数据可用性摘要；
- 当前线性模型的行为摘要；
- teacher 模型证据；
- 累积的搜索历史。

---

## 4. Teacher Analysis

### 4.1 Teacher package

本实验使用的 teacher package 为：

- `repro_bundle/teacher_analysis/xgb_guidance_base40_20260514/`

核心文件包括：

- `summary.json`
- `base40_gain_importance.csv`
- `xgb_vs_svm_recovery_contrast_base40.csv`

### 4.2 固定协议

teacher analysis 采用以下固定协议：

- 数据：`repro_bundle/data/training_2orientations.csv`
- 划分：`test_size=0.25`, `random_state=42`
- 学生模型：线性 SVM，`C=10.0`
- teacher scope：仅限 `base40`

XGB 配置来自：

- `Ensemble_baseline/experiments/xgb_search_20260511_202834_base40/summary.json`

最优配置为：

- `n_estimators=700`
- `max_depth=8`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.9`
- `reg_lambda=0.5`
- `min_child_weight=2`
- `gamma=0.0`

### 4.3 Teacher 侧发现

固定划分下，base40 的性能为：

| 模型 | Accuracy | AUC | TPR@FPR=1% |
|---|---:|---:|---:|
| linear SVM | `0.9276` | `0.9651` | `0.6347` |
| XGB | `0.9532` | `0.9868` | `0.8289` |

recovery 统计：

- `SVM wrong, XGB right = 92`
- `SVM right, XGB wrong = 28`

XGB 的主要信号集中在：

- `spare_capacity`
- `sku_average_volume`
- `wl_to_vehicle_wl_total`
- `wl_to_vehicle_wl_max`
- `sku_counts`
- `l_to_L_ratio_std`
- `sku_concentration`
- `h_to_H_ratio_max`

这意味着，在特征搜索阶段，应优先尝试将树模型依赖的非线性结构显式编码为数值特征，尤其是：

- slack 相关非线性；
- slack-pressure 交互；
- tail / threshold count；
- 局部 bottleneck 结构；
- concentration 与 footprint pressure。

---

## 5. Validator 逻辑

### 5.1 返回格式要求

每轮 GLM 返回必须包含三部分：

- `POLICY_UPDATE`
- `FEATURE_CODE`
- `RATIONALE`

其中 `FEATURE_CODE` 必须定义：

```python
build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame
```

### 5.2 本地执行流程

候选代码是在本地执行的，而不是由模型直接运行。执行流程为：

1. 校验返回代码；
2. 在受限命名空间中执行代码；
3. 调用 `build_candidate_features(agg_df.copy(), items_df.copy())`；
4. 将返回结果对齐为“每个 `dispatch` 一行”；
5. 拼接：
   - base40 特征；
   - active feature bank；
   - 新候选特征；
6. 在固定划分上重新训练 / 评估线性 SVM。

### 5.3 `accept / reject / fail`

trial 会被判定为：

- **fail**
  - 没有合法代码块；
  - 代码不可执行；
  - 返回对象不合法；
  - `dispatch_id` 缺失或重复；
  - 特征名冲突或非 ASCII。

- **reject**
  - 代码执行成功，但在接受规则下未能超过当前 best accepted。

- **accept**
  - 代码执行成功，且在接受规则下超过当前 best accepted。

接受规则为：

- 如果 AUC 提升超过 `0.0005`，则接受；
- 否则要求 `TPR@FPR=1%` 提升超过 `0.005`；
- 若仍持平，则要求 Accuracy 提升超过 `0.0005`。

### 5.4 Repair 机制

若首次验证失败，pipeline 会再发一次 repair prompt，其中包含：

- 原始 prompt；
- 原始 response；
- 错误信息。

若 repair 后仍失败，则该轮保持 `fail`。

---

## 6. 实验结果总览

### 6.1 Baseline

- Accuracy `0.9276`
- AUC `0.9651`
- TPR@FPR=1% `0.6347`

### 6.2 总体结果

总 trial 数：

- `15` total
- `6` accepted
- `5` rejected
- `4` failed

accepted 轮次：

- `1`, `5`, `6`, `8`, `9`, `14`

best accepted trial：

- iteration `14`
- features:
  - `q90_l_to_L`
  - `spare_cap_x_n_wide`
  - `height_tail_share`

best accepted metrics：

- Accuracy `0.9372`
- AUC `0.9745`
- TPR@FPR=1% `0.7207`

相对 baseline 的提升：

- Accuracy `+0.0096`
- AUC `+0.0094`
- TPR@FPR=1% `+0.0860`

还需要额外说明两点：

- raw Accuracy 最高的是 iteration `12`（`0.9388`），但由于低 FPR 区域召回没有超过当前 accepted best，因此被拒绝；
- raw AUC 最高的是 iteration `15`，但同样因为低 FPR 区域召回未超过当前 accepted best 而被拒绝。

这表明当前 validator 的目标并不是孤立地追求 Accuracy 或 AUC，而是明确保护低 FPR 区域的召回表现。

---

## 7. 每轮结果

| 轮次 | 状态 | 新特征 | 结果 |
|---|---|---|---|
| `1` | accept | `spare_cap_sq=spare_capacity^2`; `spare_cap_x_conc=spare_capacity*sku_concentration`; `spare_cap_x_wl_total=spare_capacity*wl_to_vehicle_wl_total`; `vol_top3_share=sum(top3 item_volume)/sum(item_volume)`; `n_high_asr_items=count(dim_l/dim_s>5)` | AUC `0.9681`, TPR@1% `0.6672`, ACC `0.9292` |
| `2` | fail | backend `429`，未返回可执行候选 | fail |
| `3` | reject | `spare_cap_x_avg_vol=spare_capacity*sku_average_volume`; `wl_max_residual=wl_to_vehicle_wl_max-wl_to_vehicle_wl_total`; `n_long_items=count(dim_l>0.6*vehicle_length)` | AUC `0.9682`, TPR@1% `0.6608`, ACC `0.9292` |
| `4` | fail | repair response 未返回合法 `FEATURE_CODE` 代码块 | fail |
| `5` | accept | `spare_cap_log1p=log1p(max(spare_capacity,0))`; `h_to_H_max_sq=(h_to_H_ratio_max)^2`; `l_std_x_h_max=l_to_L_ratio_std*h_to_H_ratio_max`; `max_fp_ratio=max(dim_s*dim_m)/(vehicle_length*vehicle_width)` | AUC `0.9689`, TPR@1% `0.6593`, ACC `0.9292` |
| `6` | accept | `spare_cap_x_wl_max=spare_capacity*wl_to_vehicle_wl_max`; `wl_total_over_wl_max=wl_to_vehicle_wl_total/wl_to_vehicle_wl_max`; `max_dim_l_to_vL=max(dim_l)/vehicle_length` | AUC `0.9695`, TPR@1% `0.6647`, ACC `0.9292` |
| `7` | fail | backend `429`，未返回可执行候选 | fail |
| `8` | accept | `spare_cap_x_sku_counts=spare_capacity*sku_counts`; `n_items_near_vL=count(dim_l>0.5*vehicle_length)`; `n_tall_items=count(item_height>0.5*vehicle_height)` | AUC `0.9730`, TPR@1% `0.7026`, ACC `0.9308` |
| `9` | accept | `n_wide_items=count(item_width>0.5*vehicle_width)`; `n_multi_dim_stress=count[(dim_l>0.4*L)+(dim_m>0.4*W)+(dim_s>0.4*H)>=2]`; `spare_cap_x_n_near_vL=spare_capacity*count(dim_l>0.5*vehicle_length)` | AUC `0.9736`, TPR@1% `0.7006`, ACC `0.9360` |
| `10` | reject | `n_fragile_items=sum(if_fragile)`; `flat_item_share=mean(item_flatness>3)`; `vol_top1_share=max(item_volume)/sum(item_volume)` | AUC `0.9737`, TPR@1% `0.6957`, ACC `0.9360` |
| `11` | reject | `spare_cap_x_h_to_H_max=spare_capacity*h_to_H_ratio_max`; `wl_total_x_h_to_H_max=wl_to_vehicle_wl_total*h_to_H_ratio_max`; `dim_s_avg_to_vmin=mean(dim_s/min(vehicle_length,vehicle_width,vehicle_height))` | AUC `0.9737`, TPR@1% `0.6957`, ACC `0.9348` |
| `12` | reject | `footprint_pressure=sum(dim_s*dim_m)/(vehicle_length*vehicle_width)`; `spare_cap_x_n_tall_items=spare_capacity*count(item_height>0.5*vehicle_height)`; `n_long_tall_items=count(dim_l>0.5*vehicle_length and item_height>0.5*vehicle_height)` | AUC `0.9741`, TPR@1% `0.6898`, ACC `0.9388` |
| `13` | fail | schema mismatch：候选代码引用了不存在的车辆列 | fail |
| `14` | accept | `q90_l_to_L=Q90(dim_l/vehicle_length)`; `spare_cap_x_n_wide=spare_capacity*count(item_width>0.5*vehicle_width)`; `height_tail_share=mean(item_height>0.6*vehicle_height)` | AUC `0.9745`, TPR@1% `0.7207`, ACC `0.9372` |
| `15` | reject | `q90_w_to_W=Q90(item_width/vehicle_width)`; `spare_cap_x_height_tail=spare_capacity*mean(item_height/vehicle_height>0.5)`; `n_long_and_wide=count(dim_l>0.5*vehicle_length and dim_m>0.5*vehicle_width)` | AUC `0.9745`, TPR@1% `0.7193`, ACC `0.9376` |

---

## 8. 结果解读

本实验支持以下四点结论。

### 8.1 无 seed 路线是可行的

第 `001` 轮已经被接受。这说明该路线并不依赖早期那套人工编写的 6 个 seed 特征才能启动。

### 8.2 raw item 表对性能提升具有实质贡献

被接受的特征并不是对 base40 聚合特征的简单重写，而是显著依赖了 item 级结构：

- top-k 体积集中度；
- 长件 / 高件 / 宽件数量；
- 尾部 quantile；
- 基于 footprint 的压力特征。

这说明细粒度 item 表并不是辅助性材料，而是 base40 之外新增几何信息的主要来源。

### 8.3 XGB guidance 具有明确方向性价值

最终被接受的特征，与 teacher 侧提出的方向高度一致：

- `spare_capacity` 交互；
- 局部阈值计数；
- tail pressure；
- bottleneck 型结构。

### 8.4 这条实验是证据性的，不是规范性的

由于该实验因 resume-on-same-directory 而超出了原始预算，它的价值主要是方法学上的，而不是规范实验口径上的：

- 它证明了 pure GLM bootstrapping 是可行的；
- 它证明了无 seed 路线可以对 SVM 带来实质提升；
- 但它 **不能** 取代一条 fixed-budget、clean run 的正式重跑结果。

---

## 9. 关键文件

主报告：

- `research/hybridsvm_no_seed_exploratory_run_20260518/hybridsvm_xgb_no_seed_exploratory_report_20260518.md`

英文版报告：

- `research/hybridsvm_no_seed_exploratory_run_20260518/hybridsvm_xgb_no_seed_exploratory_report_20260518_en.md`

配套摘要：

- `research/hybridsvm_no_seed_exploratory_run_20260518/README.md`

原始实验目录：

- `trials_archive/exp_20260518_173257/`

teacher analysis：

- `repro_bundle/teacher_analysis/xgb_guidance_base40_20260514/`

冻结版 teacher guidance：

- `repro_bundle/guidance/XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

核心实现：

- `code/HybridSVM/scripts/generate_guidance_with_llm.py`
- `code/HybridSVM/scripts/run_feature_search_agent.py`
- `repro_bundle/code/HybridSVM/src/feature_search.py`
- `repro_bundle/code/HybridSVM/src/svm_train.py`
- `repro_bundle/code/research/analyze_xgb_guidance_base40.py`

建议阅读顺序：

1. 本报告
2. `README.md`
3. `summary.json`
4. `trials.csv`
5. `active_feature_bank.md`
6. `trials/iter_014/feature_candidate.py`
