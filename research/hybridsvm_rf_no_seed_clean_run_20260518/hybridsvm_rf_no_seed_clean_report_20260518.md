# HybridSVM RF -> GLM 无 Seed Clean Run 报告（2026-05-18）

## 0. 报告范围与状态说明

本报告只记录一条正式的无 seed 特征搜索实验：

- 路线：`RF -> GLM -> linear SVM`
- 实验目录：`HybridSVM/experiments_feature_search/by_model/glm-5.1/exp_20260518_194131/`

这条实验应视为正式 clean run，原因是：

- 没有使用人工 seed 特征；
- 没有在同一 `exp_dir` 上 resume 追加轮次；
- 固定预算为 `10` 轮；
- 使用冻结版 `RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md` 作为唯一 teacher guidance。

因此，这条实验可以作为当前 `RF -> GLM -> SVM` 无 seed 路线的正式记录。

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
base40 RF teacher analysis
    |
    v
GLM 生成冻结版 RF guidance markdown
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
候选特征落表 + SVM 评估 + validator 判定
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

在本仓库中，`dispatch` 是原始字段 `发车号` 的规范化命名。

模型的预测粒度定义在 `dispatch` 上，因此：

- 一条样本 = 一个 `dispatch`；
- 一个标签 = 该 `dispatch` 是否装载成功（`if_loaded`）；
- 同一个 `dispatch` 下可对应多条 item 明细。

### 2.2 原始输入

本条 pipeline 从两张原始表开始：

- dispatch 级表：`FunSearch_test/training_2orientations.csv`
- item 级表：`FunSearch_test/物品信息和dblf信息.csv`

两张表的职责不同。

### 2.3 raw dispatch 表的作用

dispatch 表承担四类核心职责：

1. **定义样本**
   - 每一行对应一个 `dispatch`；
   - train/test 划分在这个粒度上进行。

2. **提供监督标签**
   - `if_loaded` 来自这张表。

3. **提供 base40 聚合特征**
   - 去重、删列后保留的聚合列，构成固定的 baseline 特征集合；
   - 这也是 SVM baseline 和 RF teacher 的输入。

4. **提供车辆级上下文**
   - 如 `vehicle_length`、`vehicle_width`、`vehicle_height`、
     `spare_capacity` 等字段；
   - 后续候选特征会将这些字段作为分母、阈值或交互锚点。

相关实现：

- `HybridSVM/src/svm_train.py`
- `HybridSVM/src/feature_search.py`

### 2.4 raw item 表的作用

item 表不是另一张独立的监督学习输入表。它的作用是为特征工程提供更细粒度的几何与约束信息。

它会被规范化为：

- 原始尺寸：`item_length`, `item_width`, `item_height`
- 排序尺寸：`dim_s`, `dim_m`, `dim_l`
- 单件体积：`item_volume`
- 形状代理量：`item_flatness`
- 单件属性：如 `if_fragile`
- 载重相关量：`load_parameter`, `vehicle_capacity`

在候选特征生成阶段，这些记录按 `dispatch_id` 分组，以便计算例如：

- 长件 / 高件 / 多轴紧张件占比；
- 维度 tail quantile 与 tail ratio；
- fragile 占比；
- 重量利用率；
- 车辆 slack 与 item-level 几何压力的交互项。

### 2.5 raw 数据在搜索循环中最终变成什么

完成加载与规范化后，raw 数据被转成两种内部视图：

1. **`agg_df`**
   - 每个 `dispatch` 一行；
   - 包含 base40 聚合特征与车辆维度；
   - 既是 baseline 特征表，也是候选代码的 dispatch 级输入。

2. **`items_df`**
   - 每个 item 一行；
   - 保留细粒度的尺寸与属性信息；
   - 是候选代码的 item 级输入。

因此，LLM 不会直接在 prompt 中看到全量 raw 表。它只会被告知：

- 后续本地会向它写出的函数传入 `agg_df`；
- 后续本地会向它写出的函数传入 `items_df`。

然后，本地再用 **全量** `agg_df` 与 **全量** `items_df` 执行候选代码。

---

## 3. 实际喂给 GLM 的内容

当前实现不会把完整 dispatch 数值表或完整 item 明细表直接送入 GLM。

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

因此，GLM 能看到的是：

- 表结构；
- 数据规模；
- 可用字段集合。

GLM 看不到的是：

- 全量 `10000` 行 dispatch 数值表；
- 全量 `100317` 行 item 数值表；
- 每条样本的原始数值明细。

### 3.2 学生模型摘要

prompt 还包含当前 baseline 线性 SVM 的压缩描述：

- 截距；
- 按绝对值排序的主要系数；
- 系数正负方向解释。

这部分的作用是告诉 GLM：

- 当前线性模型已经对哪些 base40 方向较敏感；
- 哪些方向可能还需要显式非线性特征来补足。

### 3.3 teacher 模型摘要

prompt 中插入冻结版 RF guidance 文件：

- `HybridSVM/RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

这部分为 GLM 提供 teacher 侧证据，包括：

- 固定划分下 SVM 与 RF 的性能差距；
- recovery count（`SVM 错、RF 对` 等）；
- RF 的高重要度特征；
- 推荐的特征方向；
- 不建议投入的方向。

### 3.4 迭代记忆

prompt 还包含当前搜索状态：

- `policy.md`：硬约束与偏好的特征家族；
- `memory.md`：accepted / rejected trial 的摘要；
- `active_feature_bank.md`：当前已经接受的增量特征及其指标。

因此，GLM 每一轮真正依据的是：

- 数据可用性摘要；
- 当前线性模型行为摘要；
- RF teacher evidence；
- 累积搜索历史。

---

## 4. RF Teacher Analysis

### 4.1 Teacher package

本实验使用的 teacher package 为：

- `research/rf_guidance_base40_20260514/`

核心文件包括：

- `summary.json`
- `base40_rf_importance.csv`
- `rf_vs_svm_recovery_contrast_base40.csv`

### 4.2 固定协议

teacher analysis 采用以下固定协议：

- 数据：`FunSearch_test/training_2orientations.csv`
- 划分：`test_size=0.25`, `random_state=42`
- 学生模型：线性 SVM，`C=10.0`
- teacher scope：仅限 `base40`

RF 配置来自：

- `Ensemble_baseline/run_ensemble_ablation.py`

使用配置为：

- `n_estimators=200`
- `max_depth=20`
- `min_samples_leaf=2`

### 4.3 Teacher 侧发现

固定划分下，base40 的性能为：

| 模型 | Accuracy | AUC | TPR@FPR=1% |
|---|---:|---:|---:|
| linear SVM | `0.9276` | `0.9651` | `0.6347` |
| RF | `0.9436` | `0.9823` | `0.8058` |

recovery 统计：

- `SVM wrong, RF right = 66`
- `SVM right, RF wrong = 26`

RF 的主要信号集中在：

- `spare_capacity`
- `wl_to_vehicle_wl_total`
- `sku_counts`
- `sku_average_volume`
- `wl_to_vehicle_wl_max`
- `h_to_H_ratio_avg`
- `sku_height_avg`
- `wl_to_vehicle_wl_std`
- `wl_to_vehicle_wl_avg`

这意味着，在特征搜索阶段，应优先尝试将树模型依赖的非线性结构显式编码为数值特征，尤其是：

- `spare_capacity` 的非线性与交互项；
- `wl` 家族的 dispersion / tail 结构；
- 基于 item 级阈值的 bottleneck count / share；
- 多轴紧张与局部局促结构；
- 与重量约束相关、而 base40 尚未显式表达的压力信号。

### 4.4 “怎么把 RF 的知识教给 GLM”

这个步骤完全通过程序化文本接口完成，不依赖人工手写摘要。

具体流程是：

1. 运行 `research/analyze_rf_guidance_base40.py`
   - 输出结构化 RF teacher package。

2. 运行 `HybridSVM/scripts/generate_guidance_with_llm.py`
   - 读取上一步的 `summary.json`；
   - 把以下内容送给 GLM：
     - teacher scope
     - split
     - data path
     - RF config
     - SVM / RF 指标差距
     - recovery 统计
     - top base40 RF features
   - 要求 GLM 生成冻结版 guidance markdown。

3. 保存完整轨迹：
   - prompt：`repro_bundle/guidance/guidance_gen_20260514_181154_glm-5.1/prompt.md`
   - raw response：`.../raw_response.md`
   - manifest：`.../manifest.json`
   - final guidance：`repro_bundle/guidance/RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

因此，“RF 如何指导 GLM 做特征工程”本身就是可复现的。

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
   - 当前 active feature bank；
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
  - 代码执行成功；
  - 但在接受规则下未能超过当前 best accepted。

- **accept**
  - 代码执行成功；
  - 且在接受规则下超过当前 best accepted。

接受规则为：

- 如果 AUC 提升超过 `0.0005`，则接受；
- 否则要求 `TPR@FPR=1%` 提升超过 `0.005`；
- 若仍持平，则要求 Accuracy 提升超过 `0.0005`。

这意味着 validator 的目标不是单独追求某个指标，而是优先保护当前定义下的主目标序。

### 5.4 Repair 机制

若首次响应不满足格式或代码要求，pipeline 会再发一次 repair prompt，其中包含：

- 原始 prompt；
- 原始 response；
- 错误信息。

本次 RF clean run 中没有最终 fail，但：

- `iter_007`
- `iter_008`

都记录了 repair 轨迹，说明系统确实执行过 repair 逻辑。

---

## 6. 实验结果总览

### 6.1 Baseline

- Accuracy `0.9276`
- AUC `0.9651`
- TPR@FPR=1% `0.6347`

### 6.2 总体结果

总 trial 数：

- `10` total
- `6` accepted
- `4` rejected
- `0` failed

accepted 轮次：

- `1`, `3`, `5`, `6`, `9`, `10`

final best accepted trial：

- iteration `10`
- features:
  - `load_util_ratio`
  - `multi_dim_tight_share`
  - `dim_l_tail_ratio`

best accepted metrics：

- Accuracy `0.9312`
- AUC `0.9733`
- TPR@FPR=1% `0.6903`

相对 baseline 的提升：

- Accuracy `+0.0036`
- AUC `+0.0082`
- TPR@FPR=1% `+0.0556`

还需要额外说明一点：

- 按最终 accepted 状态看，best TPR@FPR=1% 其实出现在 `iter_006`，其值为 `0.7040`；
- 但 `iter_010` 的 AUC 提升更大，且超过 acceptance margin，所以最终 active bank 更新到了 `iter_010`。

这说明当前 validator 的目标函数是**按 AUC 优先的字典序**，而不是单独锁死低 FPR 召回。

---

## 7. 每轮结果

| 轮次 | 状态 | 新特征 | 结果 |
|---|---|---|---|
| `1` | accept | `spare_cap_x_wl_total=spare_capacity*wl_to_vehicle_wl_total`; `spare_cap_x_sku_counts=spare_capacity*sku_counts`; `spare_cap_sq=spare_capacity^2`; `hH_x_lL=h_to_H_ratio_avg*l_to_L_ratio_avg`; `big_piece_share=mean(dim_l>0.5*vehicle_length)` | AUC `0.9686`, TPR@1% `0.6691`, ACC `0.9288` |
| `2` | reject | `spare_cap_x_sku_avg_vol=spare_capacity*sku_average_volume`; `wl_max_residual=wl_to_vehicle_wl_max-wl_to_vehicle_wl_avg`; `awkward_shape_share=mean(dim_l>3*dim_m)` | AUC `0.9687`, TPR@1% `0.6667`, ACC `0.9292` |
| `3` | accept | `spare_cap_x_wl_max=spare_capacity*wl_to_vehicle_wl_max`; `item_vol_cv=std(item_volume)/mean(item_volume)`; `long_item_share=mean(dim_l>0.5*vehicle_length)` | AUC `0.9704`, TPR@1% `0.7070`, ACC `0.9284` |
| `4` | reject | `spare_cap_x_hH=spare_capacity*h_to_H_ratio_avg`; `floor_area_pressure=sum(dim_s*dim_m/(vehicle_length*vehicle_width))`; `tall_item_share=mean(dim_l>0.5*vehicle_height)` | AUC `0.9704`, TPR@1% `0.7060`, ACC `0.9284` |
| `5` | accept | `wl_cv=wl_to_vehicle_wl_std/wl_to_vehicle_wl_avg`; `fragile_share=mean(if_fragile)`; `spare_cap_cubed=spare_capacity^3` | AUC `0.9711`, TPR@1% `0.6957`, ACC `0.9296` |
| `6` | accept | `wl_max_to_avg=wl_to_vehicle_wl_max/wl_to_vehicle_wl_avg`; `tight_x_high_wl=1[spare_capacity<0.15]*1[wl_to_vehicle_wl_total>0.7]`; `sku_counts_x_lL=sku_counts*l_to_L_ratio_avg` | AUC `0.9714`, TPR@1% `0.7040`, ACC `0.9288` |
| `7` | reject | `sku_counts_x_hH=sku_counts*h_to_H_ratio_avg`; `flat_item_mean=mean(item_flatness)`; `near_limit_share=mean(dim_l>0.75*vehicle_length)` | AUC `0.9716`, TPR@1% `0.7050`, ACC `0.9288` |
| `8` | reject | `spare_cap_x_long_item=spare_capacity*mean(dim_l>0.5*vehicle_length)`; `dim_l_max_to_veh_l=max(dim_l)/vehicle_length`; `wl_total_x_sku_counts=wl_to_vehicle_wl_total*sku_counts` | AUC `0.9718`, TPR@1% `0.6927`, ACC `0.9304` |
| `9` | accept | `spare_cap_x_wl_cv=spare_capacity*(wl_to_vehicle_wl_std/wl_to_vehicle_wl_avg)`; `near_h_limit_share=mean(dim_l>0.7*vehicle_height)`; `wl_total_sq=(wl_to_vehicle_wl_total)^2` | AUC `0.9723`, TPR@1% `0.6853`, ACC `0.9304` |
| `10` | accept | `load_util_ratio=sum(load_parameter)/max(vehicle_capacity)`; `multi_dim_tight_share=mean[(dim_l>0.5*L)+(dim_m>0.5*W)+(dim_s>0.5*H)>=2]`; `dim_l_tail_ratio=Q90(dim_l)/Q50(dim_l)` | AUC `0.9733`, TPR@1% `0.6903`, ACC `0.9312` |

---

## 8. 结果解读

本实验支持以下四点结论。

### 8.1 无 seed 路线是可行的

第 `001` 轮已经被接受。这说明 RF-guided 路线并不依赖人工 seed 特征才能启动。

### 8.2 item 级数据提供了实质新增信息

被接受的特征中，有相当一部分来自 item 级统计，而不是对 base40 聚合特征的简单变形。例如：

- `big_piece_share`
- `item_vol_cv`
- `fragile_share`
- `near_h_limit_share`
- `multi_dim_tight_share`
- `dim_l_tail_ratio`

这说明 item 表不是辅助材料，而是这条路线获得新增可解释几何信息的主要来源。

### 8.3 RF guidance 的作用是“给方向”，不是“直接给答案”

RF teacher 并没有直接给出候选特征公式。它提供的是：

- 哪些 base40 变量更重要；
- 哪些错误样本是 RF 能恢复而 SVM 不能恢复的；
- 哪些方向值得优先测试。

随后，GLM 再把这些方向转成：

- 交互项；
- threshold share；
- tail ratio；
- slack-pressure 型显式数值特征。

因此，RF 的作用更接近“结构化归纳偏置”，而不是 feature 模板库。

### 8.4 当前 acceptance rule 会偏向 AUC 更强的累计特征集

这一点在本次实验中非常明显：

- `iter_006` 的 TPR@FPR=1% 更高；
- `iter_010` 的 AUC 与 Accuracy 更高；
- 由于 rule 先比较 AUC，最终 active bank 走向了 `iter_010`。

因此，如果后续研究目标更强调低 FPR 区域召回，那么 validator 设计本身也可能成为一个单独需要讨论的因素。

---

## 9. 关键文件

主报告：

- `research/hybridsvm_rf_no_seed_clean_run_20260518/hybridsvm_rf_no_seed_clean_report_20260518.md`

英文版报告：

- `research/hybridsvm_rf_no_seed_clean_run_20260518/hybridsvm_rf_no_seed_clean_report_20260518_en.md`

配套摘要：

- `research/hybridsvm_rf_no_seed_clean_run_20260518/README.md`

原始实验目录：

- `HybridSVM/experiments_feature_search/by_model/glm-5.1/exp_20260518_194131/`

teacher analysis：

- `research/rf_guidance_base40_20260514/`

冻结版 teacher guidance：

- `HybridSVM/RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

guidance 生成轨迹：

- `research/hybridsvm_rf_no_seed_clean_run_20260518/repro_bundle/guidance/guidance_gen_20260514_181154_glm-5.1/`

核心实现：

- `HybridSVM/scripts/generate_guidance_with_llm.py`
- `HybridSVM/scripts/run_feature_search_agent.py`
- `HybridSVM/src/feature_search.py`
- `HybridSVM/src/svm_train.py`
- `research/analyze_rf_guidance_base40.py`
