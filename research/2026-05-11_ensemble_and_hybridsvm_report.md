# 3D-BPP 阶段性实验报告（2026-05-11）

本报告只整理当前两条主线：

1. `Ensemble_baseline/`：独立的 ensemble 路线
2. `HybridSVM/`：线性 SVM + LLM 特征工程路线

`TabTreeFormer/Former` 视为历史路线，本报告不展开。更早期的纯 `svm` 路线作为两条主线的共同参照基线。

## 1. 仓库内当前可追溯入口

### Ensemble 路线

- 训练/消融入口：`Ensemble_baseline/run_ensemble_ablation.py`
- 说明文档：`Ensemble_baseline/README.md`
- 主要实验目录：
  - `Ensemble_baseline/experiments/ablation_20260509_170335/`
  - `Ensemble_baseline/experiments/ablation_20260509_170103/`

每个实验目录当前都包含：

- `README.md`
- `single_model_ablation.csv`
- `stacking_ablation.csv`
- `summary.json`

### HybridSVM + LLM 特征工程路线

- 主循环入口：`HybridSVM/scripts/run_feature_search_agent.py`
- 特征搜索核心：`HybridSVM/src/feature_search.py`
- 树模型启发记录：`HybridSVM/TREE_INSPIRED_FEATURE_HYPOTHESES.md`
- 当前主实验目录：
  - `tmp_glm_probe_v4/by_model/glm-5.1/exp_20260510_181255/`

该目录当前包含：

- `context.md`
- `policy.md`
- `memory.md`
- `trials.csv`
- `summary.json`
- `active_feature_bank.csv`
- `active_feature_bank.md`
- `trials/iter_*`

### 手工特征实验（与 glm 历史分离）

- 入口：`HybridSVM/scripts/run_manual_feature_trial.py`
- 目录：
  - `HybridSVM/experiments_feature_search/manual_incremental/wall_pressure_v1_20260510_212617/`
  - `HybridSVM/experiments_feature_search/manual_incremental/heterogeneity_v1_20260510_212617/`

这些实验是单独落盘的，没有混入 glm 的 `trials.csv` 和 `summary.json`。

## 2. 统一实验背景与口径

### 任务

任务是做 3D bin packing feasibility classification。

### 主要数据

- 聚合 dispatch 级数据：`FunSearch_test/training_2orientations.csv`
- item 级数据：`FunSearch_test/物品信息和dblf信息.csv`

### 主要指标

- Accuracy
- ROC AUC
- `TPR@FPR=1%`

其中 `TPR@FPR=1%` 对“低误报条件下能找回多少正类”更敏感，适合判断强分类器是否真正改善了排序前端。

### 共同切分

两条主线都固定在：

- `test_size = 0.25`
- `random_state = 42`

### 一个需要明确的配置差异

两条主线虽然使用同一份数据和同一切分，但线性 SVM 的固定参数并不完全相同：

- `Ensemble_baseline/` 中的线性 SVM 使用 `C = 1.9124740037393821`
- `HybridSVM` 的 glm 特征搜索实验固定 `svm_c = 10.0`

因此：

- 同一路线内部的提升幅度可以直接比较
- 两条路线之间的绝对数值可以对照，但不应当过度解读到小数点后 3 位

## 3. Ensemble 路线结果

### 3.1 训练配置

来自 `Ensemble_baseline/experiments/ablation_20260509_170335/summary.json`：

- 基学习器请求：`svm, lr, rf, xgb`
- 元学习器：`logreg`
- stacking 使用 `cv_folds = 5`
- 最终本地配置：
  - linear SVM: `C = 1.9124740037393821`
  - logistic regression: `C = 4.736286181285866`, `max_iter = 3000`
  - random forest: `n_estimators = 200`, `max_depth = 20`, `min_samples_leaf = 2`
  - `xgb` 在当前环境回退为 `HistGradientBoostingClassifier`
  - GBDT fallback: `max_iter = 400`, `max_depth = 6`, `learning_rate = 0.1`
  - meta logreg: `C = 1.5688869685092588`, `max_iter = 2000`

训练轮次记录：

- SVM：`max_iter = -1`
- LR：`3000`
- RF：`200` 棵树
- GBDT fallback：`400` 轮
- full stack (`svm,lr,rf,xgb`) 的 `training_rounds_hint` 记录为：
  - `n_base_fits_total = 24`
  - `n_meta_fits_total = 1`

### 3.2 单模型结果

固定 40 个聚合特征，固定切分下：

| 模型 | Accuracy | AUC | TPR@1% |
| --- | ---: | ---: | ---: |
| `svm` | 0.9276 | 0.9644 | 0.6057 |
| `lr` | 0.9220 | 0.9649 | 0.6283 |
| `rf` | 0.9436 | 0.9823 | 0.8058 |
| `xgb -> gbdt_fallback` | 0.9468 | 0.9845 | 0.8132 |

结论很直接：树模型单体已经显著强于线性模型。

补充说明：

- 上面这组是早期 ablation 记录，当时环境里的 `xgb` 实际回退成了 `HistGradientBoostingClassifier`
- 因此这里更准确的表述应当是 “GBDT fallback 已经明显强于线性模型”

### 3.2b 真实 XGBoost 复核结果

为避免 `gbdt_fallback` 和真实 `xgb` 混淆，后续又单独补跑了真实 XGBoost：

- 结果目录：`research/real_xgb_with_hybrid_features_20260511/`
- 配置文件：`research/real_xgb_with_hybrid_features_20260511/summary.json`
- 使用配置：
  - `n_estimators = 500`
  - `max_depth = 6`
  - `learning_rate = 0.05`
  - `subsample = 0.9`
  - `colsample_bytree = 0.9`
  - `reg_lambda = 1.0`
  - `random_state = 42`

在相同固定切分下，真实 `xgb` 的单模型结果为：

| 模型 | 特征集 | Accuracy | AUC | TPR@1% |
| --- | --- | ---: | ---: | ---: |
| `xgb` | `base40` | 0.9524 | 0.9864 | 0.8550 |
| `xgb` | `base40_plus_h15` | 0.9516 | 0.9880 | 0.8633 |

这组结果说明两点：

1. 真实 `xgb` 比 `gbdt_fallback` 还要更强，尤其体现在 AUC 和 `TPR@1%`
2. 我们追加出来的 15 个 HybridSVM 特征，不只对线性 SVM 有用，对真实 `xgb` 也依然有增益

因此当前更准确的性能排序应理解为：

- 线性模型（`svm/lr`）明显落后
- `rf` 已经很强
- `gbdt_fallback` 比 `rf` 略强
- 真实 `xgb` 又比 `gbdt_fallback` 再强一截

### 3.3 非树模型能否自己“集成成强分类器”

这是单独做过的消融：`Ensemble_baseline/experiments/ablation_20260509_170103/`

| 组合 | Meta | Accuracy | AUC | TPR@1% |
| --- | --- | ---: | ---: | ---: |
| `svm + lr` | `logreg` | 0.9248 | 0.9648 | 0.6219 |
| `svm + lr` | `mlp` | 0.9276 | 0.9647 | 0.6150 |

结论：

- 仅靠非树模型互相堆叠，基本没有形成新的强分类器
- `svm` 和 `lr` 的信息高度同质，主要都在做线性边界重加权
- 因此“把 `rf/xgb` 全删掉，只靠非集成分类器集成出强分类器”这条路目前不成立

### 3.4 核心 stacking 结果

来自 `Ensemble_baseline/experiments/ablation_20260509_170335/summary.json`：

| 基学习器 | Meta | Accuracy | AUC | TPR@1% |
| --- | --- | ---: | ---: | ---: |
| `rf + xgb` | `logreg` | 0.9452 | 0.9844 | 0.8102 |
| `svm + rf + xgb` | `logreg` | 0.9472 | 0.9841 | 0.8274 |
| `svm + lr + rf + xgb` | `logreg` | 0.9476 | 0.9842 | 0.8171 |

这里的 `xgb` 实际解析为 `gbdt_fallback`。

补充一组真实 `xgb` 的 stacking 复核：

- 结果目录：`research/real_xgb_with_hybrid_features_20260511/`
- 汇总文件：`research/real_xgb_with_hybrid_features_20260511/stacking_comparison.csv`

| 基学习器 | 特征集 | Accuracy | AUC | TPR@1% |
| --- | --- | ---: | ---: | ---: |
| `svm + lr + rf + xgb` | `base40` | 0.9500 | 0.9851 | 0.8083 |
| `svm + lr + rf + xgb` | `base40_plus_h15` | 0.9548 | 0.9874 | 0.8736 |

这组结果说明：

1. 在真实 `xgb` 已经很强的情况下，full stack 不一定继续全面压过 `xgb` 单体
2. 但在加入 `base40_plus_h15` 后，stacking 的 `TPR@1%` 继续升到 `0.8736`，说明异质模型互补仍然存在
3. 对我们当前任务而言，更合理的理解不是 “stacking 必然超过最强单体”，而是：
   - 排序主体往往已经由 boosting 决定
   - stacking 有时会把极低误报段再往上推一点
   - 是否值得保留 stack，要看我们更重视单模型强度、可复现性，还是极低误报段召回

另外，这组真实 `xgb` full stack 的 meta logreg 权重为：

- `svm`: `-1.3098`
- `lr`: `+2.7137`
- `rf`: `+3.8442`
- `xgb`: `+3.4399`

### 3.5 为什么 ensemble 总会更好

当前证据支持下面这个解释：

1. 真正的提升主体来自树模型，尤其是 boosting，不是线性模型之间的互相投票。
2. `rf`、`gbdt_fallback`、真实 `xgb` 都已经能从 40 个聚合特征里抓到大量阈值效应、分段效应和交互效应。
3. `svm + lr` 的信息高度相关，所以二者单独互相堆叠时提升很小。
4. 从真实 `xgb` 结果看，boosting 已经能够单体吃到大部分排序优势；stacking 更多是在局部边界区域做补充。
5. 从指标看：
   - 真实 `xgb` 单体在 `base40` 上已经达到 `AUC = 0.9864`, `TPR@1% = 0.8550`
   - 加入 `base40_plus_h15` 后，真实 `xgb` 单体涨到 `AUC = 0.9880`, `TPR@1% = 0.8633`
   - full stack 在 `base40_plus_h15` 下到 `TPR@1% = 0.8736`，说明极低误报段仍有少量互补空间

因此更准确的说法不是“ensemble 本身神奇”，而是：

> ensemble 提升来自异质基学习器的互补，但主导者其实是 boosting；它先吃掉了大部分阈值/交互结构，stacking 再在少量边界样本上做补充。

### 3.6 真实 XGBoost 调参记录

为了确认真实 `xgb` 的上限有没有再高一些，补做了随机搜索：

- 脚本：`Ensemble_baseline/search_xgb_hparams.py`
- 选优规则：按 `AUC` -> `TPR@1%` -> `Accuracy` 依次排序
- 每个特征集随机搜索 `24` 个 trial

结果目录：

- `Ensemble_baseline/experiments/xgb_search_20260511_202834_base40/`
- `Ensemble_baseline/experiments/xgb_search_20260511_202925_base40_plus_h15/`

`base40` 的最优 trial：

- trial `10`
- 参数：
  - `n_estimators = 700`
  - `max_depth = 8`
  - `learning_rate = 0.05`
  - `subsample = 0.8`
  - `colsample_bytree = 0.9`
  - `reg_lambda = 0.5`
  - `min_child_weight = 2`
  - `gamma = 0.0`
- 指标：
  - Accuracy `0.9532`
  - AUC `0.9868`
  - `TPR@1% = 0.8289`

`base40_plus_h15` 的最优 trial：

- trial `14`
- 参数：
  - `n_estimators = 900`
  - `max_depth = 8`
  - `learning_rate = 0.02`
  - `subsample = 0.7`
  - `colsample_bytree = 0.7`
  - `reg_lambda = 1.0`
  - `min_child_weight = 2`
  - `gamma = 0.5`
- 指标：
  - Accuracy `0.9524`
  - AUC `0.9884`
  - `TPR@1% = 0.8628`

解释上要谨慎：

1. 这不是全局最优，只是固定切分上的一次局部随机搜索
2. `base40_plus_h15` 下，调参最优 `xgb` 与默认配置 `xgb` 的结论一致，说明 15 个新特征对 boosting 的增益比较稳
3. `base40` 下，调参后的 AUC 略高于默认配置，但 `TPR@1%` 不如默认配置，说明 boosting 的最优点会随目标指标不同而偏移

## 4. HybridSVM + LLM 特征工程路线结果

### 4.1 当前工作流

当前主循环由 `HybridSVM/scripts/run_feature_search_agent.py` 驱动，每轮执行：

1. 读取 `context.md`
2. 读取 `policy.md`
3. 读取 `memory.md`
4. 读取当前 active feature bank
5. 调用 LLM 生成新的 dispatch 级特征代码
6. 用固定切分、固定 `svm_c` 跑线性 SVM
7. 写回 trial 产物、指标、消融、summary
8. 进入下一轮

当前模式已经不是“整组候选特征替换旧特征”，而是：

- `feature_bank_mode = cumulative_incremental`
- 只允许在 active bank 上继续叠加新特征

这正是当前实验目录 `tmp_glm_probe_v4/by_model/glm-5.1/exp_20260510_181255/` 所采用的协议。

### 4.2 item 级数据是怎么给 LLM 的

在 `HybridSVM/src/feature_search.py` 中，LLM 拿到的上下文不只有聚合表 `agg_df`，还有 item 级表 `items_df`。
不是，这个表不是把原始数据（比如原始 JSON 或原始订单+物品级明细“原始表”）直接喂给 LLM，而是把经过基本清洗和结构化的 item 级长宽高/体积/属性数据（即 `items_df`），和当前聚合后的 dispatch 级特征（`agg_df` 或 feature bank），一起提供给 LLM。  
LLM 拿到的 `items_df` 是“逐件物品一条”，比如每个 dispatch 下有很多 item，每个 item 有 length/width/height/volume/flag 等专栏，数据类型和实际生产表一一对应，但通常已经做了必要的清洗（如极端异常值过滤、字段标准化、繁琐属性归一化等），便于直接用于特征工程。

关于数据规模和 LLM 上下文：
实际运行时肯定不能把全部原始 item 级数据直接塞给 LLM，否则无论是 token 数还是上下文长度都会远超模型容量（哪怕只塞一批 dispatch 的全部明细，也很快超限）。  
因此，当前的做法一般是——对于每个特征建议/实验轮次，只挑有限数量（如 8~32 个代表性 dispatch），每个 dispatch 跑抽样或采样（比如优先覆盖大件、极端/典型/异常 case），把它们的 item 级明细组装成简化版“上下文样本片段”，再和聚合特征一起提供给 LLM。  
这样既能让 LLM 理解 item 级 tail/interaction 等统计结构，又不会导致上下文爆炸，满足 glm 实际 token 长度限制。通常，如果发现输入长度超限，会进一步抽样或只提取极端 case 的 top-k item 子集，最大化上下文的信息密度。

简单理解：  
- 原始数据 → 数据清洗/标准化  
- 针对每次 LLM 特征建议，采样有限个 dispatch、有限数量的 item（重点保留尾部/极端/典型样本）  
- 组装成 item 表的“压缩快照”上下文，确保总 token 数可控  
- 喂给 LLM（和当前的聚合特征 bank 并行）

因此，实际不会让 glm 面对“全量原始数据”，而是严格控制输入规模、最大限度提高样本代表性和上下文利用率。


`items_df` 是逐件 item 的表，包含至少这些字段：

- `dispatch_id`
- `item_length`, `item_width`, `item_height`
- `dim_s`, `dim_m`, `dim_l`
- `item_volume`
- `item_flatness`
- `if_fragile`
- `load_parameter`
- `vehicle_capacity`

也就是说，LLM 不是只能在原始 40 个聚合特征上打转，而是可以直接利用单件 item 的长宽高分布、尾部、异质性和相对车厢尺寸关系来造新特征。

### 4.3 当前 glm 实验配置

来自 `tmp_glm_probe_v4/by_model/glm-5.1/exp_20260510_181255/summary.json`：

- LLM 模型：`glm-5.1`
- 数据：
  - `FunSearch_test/training_2orientations.csv`
  - `FunSearch_test/物品信息和dblf信息.csv`
- 固定切分：`test_size = 0.25`, `random_state = 42`
- 线性 SVM：`svm_c = 10.0`
- 单轮最多新增特征数：`6`
- 接收规则：
  - `auc_margin = 0.0005`
  - `tpr_margin = 0.005`
  - `accuracy_margin = 0.0005`

### 4.4 baseline 与总体结果

该路线自己的固定 baseline：

| 指标 | 数值 |
| --- | ---: |
| Accuracy | 0.9276 |
| AUC | 0.9651 |
| TPR@1% | 0.6347 |

当前主实验共 10 轮：

- `n_trials_total = 10`
- `n_trials_accepted = 5`
- `n_trials_rejected = 4`
- `n_trials_failed = 1`

失败轮次 `iter_006` 的原因是格式修复失败：

- `ValueError: Repair response did not include FEATURE_CODE python block`

### 4.5 被接受的关键 trial

| Iter | 新增/候选特征 | Accuracy | AUC | TPR@1% | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| `iter_000` | `dominant_type_share`, `p90_long_over_bin_long`, `p90_mid_over_bin_mid`, `thin_item_share`, `max_face_area_load_over_floor`, `tight_bin_large_piece_interaction` | 0.9332 | 0.9721 | 0.7139 | 初始 seed |
| `iter_003` | `dominant_type_share`, `p90_long_over_bin_long`, `p90_mid_over_bin_mid`, `max_face_area_load_over_floor`, `tight_bin_large_piece_interaction`, `count_near_long_limit` | 0.9344 | 0.9725 | 0.7134 | 早期整组候选的更优版本 |
| `iter_007` | `spare_per_item`, `p90_short_over_bin_short`, `count_near_height_limit` | 0.9348 | 0.9730 | 0.7119 | 切换到累计叠加后首次接受 |
| `iter_008` | `count_near_width_limit`, `max_2d_pressure`, `volume_tail_ratio` | 0.9336 | 0.9740 | 0.7439 | 当前最佳低误报召回 |
| `iter_009` | `multi_dim_tight_share`, `max_elongation`, `short_dim_sum_ratio` | 0.9332 | 0.9749 | 0.7296 | 当前最佳 AUC |

这里要特别说明协议演化：

- `iter_000` 和 `iter_003` 仍然带有早期“整组候选快照”的痕迹
- 从 `iter_007` 开始，实验已经切换为“在 active bank 上持续叠加”的主协议

### 4.6 当前 active feature bank

来自 `active_feature_bank.md`，当前 bank 共 15 个特征：

1. `dominant_type_share`
2. `p90_long_over_bin_long`
3. `p90_mid_over_bin_mid`
4. `max_face_area_load_over_floor`
5. `tight_bin_large_piece_interaction`
6. `count_near_long_limit`
7. `spare_per_item`
8. `p90_short_over_bin_short`
9. `count_near_height_limit`
10. `count_near_width_limit`
11. `max_2d_pressure`
12. `volume_tail_ratio`
13. `multi_dim_tight_share`
14. `max_elongation`
15. `short_dim_sum_ratio`

当前 active bank 指标：

| 指标 | 数值 | 相对该路线 baseline 增量 |
| --- | ---: | ---: |
| Accuracy | 0.9332 | +0.0056 |
| AUC | 0.9749 | +0.0097 |
| TPR@1% | 0.7296 | +0.0949 |

### 4.7 对这条路线的当前判断

这条线已经得到一个比较清楚的结论：

1. LLM + 线性 SVM 并不是靠“更多平滑统计量”起作用。
2. 真正有效的新增特征，几乎都在模拟树模型擅长的结构：
   - 尺寸尾部压力
   - 近边界计数
   - 多维同时紧张的交互
   - 低 slack 与大件压力的交互
   - item 异质性/重复度
3. 因此这条线确实在“把树模型吃到的非线性结构，翻译成可解释特征”。

但它还没有追平树模型：

- ensemble 中树模型单体已经达到 AUC `0.982` 到 `0.985`
- 当前 15 特征 bank 的线性 SVM 为 AUC `0.9749`

所以目前更准确的结论是：

> LLM 特征工程已经追回了树模型优势的一部分，而且这些特征是可解释的，但还没有完全追回。

## 5. 手工特征补充实验（不混入 glm 历史）

这部分实验用 `HybridSVM/scripts/run_manual_feature_trial.py` 单独执行，基准是当前 15 特征 active bank，而不是重写 glm 历史。

### 5.1 `wall_pressure_v1`

目录：

- `HybridSVM/experiments_feature_search/manual_incremental/wall_pressure_v1_20260510_212617/`

新增特征：

- `long_width_competition`
- `tight_wall_share`
- `slack_wall_interaction`

结果：

| 指标 | 数值 | 相对 active bank 变化 |
| --- | ---: | ---: |
| Accuracy | 0.9332 | +0.0000 |
| AUC | 0.9750 | +0.0001 |
| TPR@1% | 0.7291 | -0.0005 |

判断：

- 几乎没有实际收益
- 说明“wall pressure”方向并非完全无效，但当前版本信息量不够

### 5.2 `heterogeneity_v1`

目录：

- `HybridSVM/experiments_feature_search/manual_incremental/heterogeneity_v1_20260510_212617/`

新增特征：

- `volume_cv`
- `shape_mix_entropy`
- `p95_volume_over_median`

结果：

| 指标 | 数值 | 相对 active bank 变化 |
| --- | ---: | ---: |
| Accuracy | 0.9316 | -0.0016 |
| AUC | 0.9760 | +0.0012 |
| TPR@1% | 0.7670 | +0.0374 |

判断：

- 这是目前非常值得继续追的一组手工特征
- 它明显提高了 AUC 和 `TPR@1%`
- 代价是 Accuracy 略降

这说明“item 异质性/长尾结构”仍然是非常有潜力的方向，而且它和当前 bank 并不完全冗余。

## 6. 当前综合结论

### 6.1 关于 ensemble

- ensemble 的提升是真实的
- 主增益来自树模型，不来自线性模型互堆
- `rf/xgb` 抓住了聚合特征中的阈值和交互结构
- `svm/lr` 的附加价值更像是少量补边和校准

### 6.2 关于 HybridSVM + LLM 特征工程

- 这条线已经跑通完整闭环：`context -> policy/memory -> 新特征 -> SVM 测试 -> trials/summary -> 下一轮`
- item 级长宽高数据已经进入上下文与特征生成过程，不再局限于聚合描述
- 当前 15 特征 bank 已经稳定优于该路线 baseline
- 最有价值的信号不是“更多特征”，而是“更像树模型会用到的阈值/交互/尾部/异质性特征”

### 6.3 二者之间的联系

目前最有说服力的统一解释是：

1. tree ensemble 的优势说明数据里存在明显的非线性阈值结构
2. LLM 特征工程路线的任务不是复制树模型，而是把其中一部分结构显式化
3. 当前已证明这件事可行，而且得到的是可解释特征，不是黑盒模型增益

## 7. 参数与可视化分析

为方便报告展示，我额外生成了一组参数表和 SVG 图，统一放在：

- `research/feature_figures_20260511/`

索引文件：

- `research/feature_figures_20260511/README.md`

### 7.1 HybridSVM：当前 15-feature bank 的参数大小

核心图：

- `research/feature_figures_20260511/figures/hybridsvm_active_feature_coefficients.svg`
- `research/feature_figures_20260511/figures/hybridsvm_single_feature_auc_gain.svg`
- `research/feature_figures_20260511/figures/hybridsvm_leave_one_out_auc_drop.svg`

对应表：

- `research/feature_figures_20260511/tables/hybridsvm_active_feature_coefficients.csv`
- `research/feature_figures_20260511/tables/hybridsvm_single_feature_ablation.csv`
- `research/feature_figures_20260511/tables/hybridsvm_leave_one_out_ablation.csv`

需要先说明一个口径：

- 线性 SVM 的 `coef_scaled` 是对 **MinMax-scaled** 特征的系数
- 所以它适合比较“模型对哪些特征更敏感”
- 但不应把不同量纲下的 `approx_coef_raw_units` 当作严格可比较的物理斜率

当前 active bank 中，按 `|coef_scaled|` 排名前几的特征是：

1. `tight_bin_large_piece_interaction`：`-15.21`
2. `max_face_area_load_over_floor`：`-6.59`
3. `spare_per_item`：`+5.21`
4. `p90_long_over_bin_long`：`+2.67`
5. `multi_dim_tight_share`：`-2.03`
6. `volume_tail_ratio`：`-1.79`

这说明当前线性 SVM 最敏感的信号主要不是简单均值，而是：

- slack 与大件压力的交互
- 面积/占地压力
- 每件平均冗余空间
- 长边尾部逼近
- 多维同时紧张
- 体积长尾

### 7.2 HybridSVM：不是只看系数，还要看消融

只看系数还不够，所以我又做了两类补充：

1. single-feature 增益：单独加某个特征时，相对 baseline 能涨多少
2. leave-one-out 掉点：从完整 15-feature bank 里删掉某个特征，会掉多少

从 `hybridsvm_single_feature_ablation.csv` 看，单特征增益最强的是：

1. `tight_bin_large_piece_interaction`
   - `ΔAUC = +0.00536`
   - `ΔTPR@1% = +0.08555`
2. `volume_tail_ratio`
   - `ΔAUC = +0.00271`
   - `ΔTPR@1% = +0.04523`
3. `p90_long_over_bin_long`
   - `ΔAUC = +0.00119`
   - `ΔTPR@1% = +0.00639`

从 `hybridsvm_leave_one_out_ablation.csv` 看，删掉后损失最大的也是：

1. `tight_bin_large_piece_interaction`
   - AUC 掉 `0.00424`
   - `TPR@1%` 掉 `0.07227`
2. `p90_long_over_bin_long`
   - AUC 掉 `0.00188`
3. `volume_tail_ratio`
   - AUC 掉 `0.00111`

因此，当前 HybridSVM 路线里最稳的核心贡献者可以概括为三类：

- interaction：`tight_bin_large_piece_interaction`
- tail：`p90_long_over_bin_long`, `volume_tail_ratio`
- multi-pressure：`multi_dim_tight_share`

另一个值得记录的现象是：

- `count_near_long_limit`
- `count_near_height_limit`
- `count_near_width_limit`

在当前完整 bank 下，它们的单特征增益和 leave-one-out 掉点都接近 0。更合理的解释不是“这类特征没用”，而是：

- 这些 near-limit count 的信息，已经部分被更强的 interaction / tail 特征吸收了

### 7.3 Ensemble：线性模型到底在看什么

核心图：

- `research/feature_figures_20260511/figures/ensemble_single_svm_coefficients.svg`
- `research/feature_figures_20260511/figures/ensemble_single_lr_coefficients.svg`

对应表：

- `research/feature_figures_20260511/tables/ensemble_single_svm_coefficients.csv`
- `research/feature_figures_20260511/tables/ensemble_single_lr_coefficients.csv`

单模型 SVM 按 `|coef_scaled|` 排名前几的是：

1. `spare_capacity`：`+11.51`
2. `sku_average_volume`：`-9.28`
3. `sku_counts`：`-5.93`
4. `wl_to_vehicle_wl_std`：`-2.87`
5. `wl_to_vehicle_wl_min`：`-2.39`
6. `wl_to_vehicle_wl_total`：`-2.33`

单模型 LR 的排序与之非常接近，前几名仍然是：

1. `spare_capacity`
2. `sku_average_volume`
3. `sku_counts`
4. `wl_to_vehicle_wl_std`
5. `wl_to_vehicle_wl_min`
6. `sku_height_avg`
7. `wl_to_vehicle_wl_total`

这和前面的结论一致：

- 线性模型主要利用的是少数全局强信号
- 例如 spare capacity、平均体积、件数、以及 `wl_to_vehicle_wl_*` 这一类总量/比例统计
- `svm` 与 `lr` 的关注重点高度相似，所以二者互相 stacking 时提升很小

### 7.4 Ensemble：树模型在看什么

核心图：

- `research/feature_figures_20260511/figures/ensemble_single_rf_importance.svg`
- `research/feature_figures_20260511/figures/ensemble_single_gbdt_importance.svg`

对应表：

- `research/feature_figures_20260511/tables/ensemble_single_rf_importance.csv`
- `research/feature_figures_20260511/tables/ensemble_single_gbdt_importance.csv`

RF importance 前几名：

1. `spare_capacity`
2. `wl_to_vehicle_wl_total`
3. `sku_counts`
4. `sku_average_volume`
5. `wl_to_vehicle_wl_max`
6. `h_to_H_ratio_avg`
7. `sku_height_avg`

GBDT importance 前几名：

1. `sku_counts`
2. `spare_capacity`
3. `sku_length_avg`
4. `sku_height_var`
5. `wl_to_vehicle_wl_std`
6. `sku_average_volume`
7. `sku_length_var`
8. `sku_width_avg`

这组结果很有价值，因为它说明：

1. 树模型和线性模型都认可 `spare_capacity`、`sku_average_volume`、`sku_counts`
2. 但树模型更稳定地把：
   - `wl_to_vehicle_wl_total`
   - `wl_to_vehicle_wl_max`
   - `wl_to_vehicle_wl_std`
   - 尺寸均值/方差
   - 高度/长度分布波动
   用成了有效信号
3. 这正是树模型擅长的区域：阈值、分段、交互、尾部

也就是说，树模型给 `HybridSVM` 的启发是明确的：

- 不要只继续堆平滑均值
- 要把 `wl_to_vehicle_wl_*`、尺寸方差、tail pressure、interaction 显式化

### 7.5 Ensemble：stacking 的 meta 权重

核心图：

- `research/feature_figures_20260511/figures/ensemble_stacking_meta_weights.svg`

对应表：

- `research/feature_figures_20260511/tables/ensemble_stacking_meta_weights.csv`

在早期 full stack (`svm, lr, rf, xgb`) 下，meta logreg 权重为：

1. `rf`：`+4.71`
2. `lr`：`+2.85`
3. `gbdt_fallback`：`+2.53`
4. `svm`：`-1.29`

这个结果进一步支持前面的主结论：

- 最强支撑来自 `rf`
- `gbdt_fallback` 也显著有用
- `svm` 并不是主导者，甚至在 meta 层里是负权

如果看真实 `xgb` 的 full stack，权重格局也类似：

1. `rf`：`+3.8442`
2. `xgb`：`+3.4399`
3. `lr`：`+2.7137`
4. `svm`：`-1.3098`

说明在 fallback 和真实 `xgb` 两种 setting 下，meta 层的定性结论是一致的：

- 树模型是主力
- `svm` 更像校正项，不是主判别项

这里的负权不应简单理解为 “svm 没用”，更合理的解释是：

- 在已有 `rf/lr/gbdt` 的条件下，`svm` 输出与其他模型有较强相关性
- meta 模型把它作为一个校正项，而不是主判别项

### 7.6 哪几张图最适合直接放报告

如果只放最关键的 4 张，我建议是：

1. `figures/ensemble_metric_comparison.svg`
   - 用来先给出整体性能格局
2. `figures/ensemble_single_rf_importance.svg`
   - 用来解释树模型到底看到了什么
3. `figures/hybridsvm_active_feature_coefficients.svg`
   - 用来展示当前 LLM 特征工程 bank 的参数大小
4. `figures/hybridsvm_leave_one_out_auc_drop.svg`
   - 用来证明哪些 feature 真正在支撑性能

如果可以放到 6 张，再加：

5. `figures/ensemble_stacking_meta_weights.svg`
6. `figures/hybridsvm_single_feature_auc_gain.svg`

## 8. 建议的下一步实验

优先级建议如下。

### 第一优先级：对当前 15 特征 bank 做分组消融

建议按语义分组，而不是逐个删：

1. tail pressure 组
   - `p90_long_over_bin_long`
   - `p90_mid_over_bin_mid`
   - `p90_short_over_bin_short`
   - `volume_tail_ratio`
2. near-limit count 组
   - `count_near_long_limit`
   - `count_near_width_limit`
   - `count_near_height_limit`
3. slack / pressure interaction 组
   - `tight_bin_large_piece_interaction`
   - `spare_per_item`
   - `max_2d_pressure`
   - `multi_dim_tight_share`
4. heterogeneity / shape 组
   - `dominant_type_share`
   - `max_elongation`
   - `short_dim_sum_ratio`
   - `max_face_area_load_over_floor`

目标是回答：

- 哪组特征对 AUC 贡献最大
- 哪组特征对 `TPR@1%` 贡献最大
- 哪组特征实际上已经冗余

### 第二优先级：把 `heterogeneity_v1` 系列系统化

这条线目前最有希望继续涨：

- `volume_cv`
- `shape_mix_entropy`
- `p95_volume_over_median`

建议新开手工系列，例如：

- `heterogeneity_v2`
- `heterogeneity_tail_mix_v1`

先继续作为手工独立实验，不直接写回 glm 历史。

### 第三优先级：让 glm 在更窄的策略空间里继续搜索

下一阶段不应再让它大范围乱搜，而应明确偏向：

- 异质性和长尾
- 多维同时逼近边界
- slack 与局部拥挤的交互

也就是把树模型给出的经验更强地喂回 `policy.md`。

## 9. 本报告对应的关键引用文件

- `Ensemble_baseline/README.md`
- `Ensemble_baseline/run_ensemble_ablation.py`
- `Ensemble_baseline/experiments/ablation_20260509_170335/summary.json`
- `Ensemble_baseline/experiments/ablation_20260509_170103/summary.json`
- `tmp_glm_probe_v4/by_model/glm-5.1/exp_20260510_181255/summary.json`
- `tmp_glm_probe_v4/by_model/glm-5.1/exp_20260510_181255/trials.csv`
- `tmp_glm_probe_v4/by_model/glm-5.1/exp_20260510_181255/active_feature_bank.md`
- `HybridSVM/experiments_feature_search/manual_incremental/wall_pressure_v1_20260510_212617/summary.json`
- `HybridSVM/experiments_feature_search/manual_incremental/heterogeneity_v1_20260510_212617/summary.json`
- `HybridSVM/TREE_INSPIRED_FEATURE_HYPOTHESES.md`
- `research/feature_figures_20260511/README.md`
- `research/feature_figures_20260511/summary.json`
- `research/feature_figures_20260511/tables/hybridsvm_active_feature_coefficients.csv`
- `research/feature_figures_20260511/tables/hybridsvm_single_feature_ablation.csv`
- `research/feature_figures_20260511/tables/hybridsvm_leave_one_out_ablation.csv`
- `research/feature_figures_20260511/tables/ensemble_single_svm_coefficients.csv`
- `research/feature_figures_20260511/tables/ensemble_single_lr_coefficients.csv`
- `research/feature_figures_20260511/tables/ensemble_single_rf_importance.csv`
- `research/feature_figures_20260511/tables/ensemble_single_gbdt_importance.csv`
- `research/feature_figures_20260511/tables/ensemble_stacking_meta_weights.csv`
