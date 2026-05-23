# 3D-BPP 导师汇报版（2026-05-11）

完整版见：

- `research/2026-05-11_ensemble_and_hybridsvm_report.md`

本版只保留最关键的 4 张图和 6 条结论。

## 1. 结论先行

1. `ensemble` 的提升是真实的，但主增益来自树模型，尤其是 boosting，不来自线性模型之间互相 stacking。
2. 删掉 `rf/xgb` 后，只靠 `svm + lr` 并不能集成出强分类器。
3. 树模型之所以强，是因为它们更容易利用阈值、尾部和交互结构；真实 `xgb` 目前是最强单模型。
4. `HybridSVM` 这条线已经证明：可以用 LLM 生成可解释特征，把树模型的一部分优势“翻译”给线性 SVM。
5. 当前最有效的 HybridSVM 特征不是简单计数，而是：
   - slack × large-piece interaction
   - tail pressure
   - multi-dimension tightness
6. 下一步最值得做的不是盲目继续加特征，而是：
   - 对当前 15-feature bank 做分组消融
   - 继续沿 `heterogeneity / tail / interaction` 方向扩展

## 2. 最关键的 4 张图

### 图 1：整体性能格局

- 图：`research/feature_figures_20260511/figures/ensemble_metric_comparison.svg`

这张图回答“哪条路现在最强”。

关键数值：

| 模型 | Accuracy | AUC | TPR@1% |
| --- | ---: | ---: | ---: |
| `svm` | 0.9276 | 0.9644 | 0.6057 |
| `lr` | 0.9220 | 0.9649 | 0.6283 |
| `rf` | 0.9436 | 0.9823 | 0.8058 |
| `gbdt_fallback` | 0.9468 | 0.9845 | 0.8132 |
| `xgb` | 0.9524 | 0.9864 | 0.8550 |
| `xgb + h15` | 0.9516 | 0.9880 | 0.8633 |
| `full stack + xgb + h15` | 0.9548 | 0.9874 | 0.8736 |

> 线性模型明显落后，真实 `xgb` 已经是最强单模型；full ensemble 的作用更像是在 boosting 基础上再把极低误报段往上推一点。

### 图 2：树模型到底在看什么

- 图：`research/feature_figures_20260511/figures/ensemble_single_rf_importance.svg`

这张图回答“为什么 boosting / ensemble 会更好”。

RF importance 前几名：

1. `spare_capacity`
2. `wl_to_vehicle_wl_total`
3. `sku_counts`
4. `sku_average_volume`
5. `wl_to_vehicle_wl_max`

一句话解释：

> 树模型抓住了 spare capacity、总装载压力、件数和长宽组合压力这些阈值/交互信号；boosting 在这些结构上吃得最彻底。

### 图 3：HybridSVM 当前最重要的特征权重

- 图：`research/feature_figures_20260511/figures/hybridsvm_active_feature_coefficients.svg`

这张图回答“LLM 造出来的特征到底学到了什么”。

当前 15-feature bank 中，按 `|coef_scaled|` 最大的几个特征：

#### 主要特征公式解释：

1. **`tight_bin_large_piece_interaction`**  
   定义：衡量体积较大的单件物品在箱型空间逼仄（tight bin）时的装载影响。  
   公式示意：  
   ```
   tight_bin_large_piece_interaction = (max_item_volume / bin_volume) * indicator(bin_tight)
   ```
   - 其中 `max_item_volume` 是单件最大物品体积，`bin_volume` 为车厢或箱子的体积，`indicator(bin_tight)` 表示某种紧张（tight）条件下的标志变量（如空间填充率高于特定阈值）。

2. **`max_face_area_load_over_floor`**  
   定义：最大物品表面积与车厢底面积的比值，衡量大平面物体对底部空间压力。  
   公式示意：  
   ```
   max_face_area_load_over_floor = max(item_length * item_width) / (vehicle_length * vehicle_width)
   ```
   - 取所有物品中最大的长×宽，除以车厢地板面积。

3. **`spare_per_item`**  
   定义：平均每件物品的剩余空间量，代表装载的松弛程度。  
   公式示意：  
   ```
   spare_per_item = spare_capacity / sku_counts
   ```
   - 其中 `spare_capacity` 为剩余容积，`sku_counts` 为总件数。

4. **`p90_long_over_bin_long`**  
   定义：物品长度的90分位数与车厢有效长度的比值，反映极大件与车厢长边关系。  
   公式示意：  
   ```
   p90_long_over_bin_long = percentile(item_length, 90) / vehicle_length
   ```
   - 取所有物品length的90分位数。

5. **`multi_dim_tight_share`**  
   定义：多维紧张（在长、宽、高等维度中有至少两维接近极限）的物品比例。  
   公式示意：  
   ```
   multi_dim_tight_share = count(items where (item_dim_i / vehicle_dim_i > threshold) in >=2 dims) / sku_counts
   ```
   - threshold 常取 0.85~0.95。

6. **`volume_tail_ratio`**  
   定义：体积分布尾部（如90分位数以上）占总体积的比例，衡量大体积物品的影响。  
   公式示意：  
   ```
   volume_tail_ratio = sum(item_volume where item_volume > percentile(item_volume, 90)) / sum(item_volume)
   ```

> 这些特征核心在于直接刻画“极端值”、“相互作用”和“异质性”的装载压力——远超简单均值统计量。

1. `tight_bin_large_piece_interaction`
2. `max_face_area_load_over_floor`
3. `spare_per_item`
4. `p90_long_over_bin_long`
5. `multi_dim_tight_share`
6. `volume_tail_ratio`

一句话解释：

> 真正起作用的是 interaction、tail 和 pressure，而不是再加一批平滑均值统计量。

### 图 4：HybridSVM 哪些特征真的不能删

- 图：`research/feature_figures_20260511/figures/hybridsvm_leave_one_out_auc_drop.svg`

这张图回答“哪些 feature 真正在支撑提升”。

删掉后损失最大的特征：

1. `tight_bin_large_piece_interaction`
   - AUC 掉 `0.00424`
   - `TPR@1%` 掉 `0.07227`
2. `p90_long_over_bin_long`
   - AUC 掉 `0.00188`
3. `volume_tail_ratio`
   - AUC 掉 `0.00111`

一句话解释：

> 当前最核心的三类可解释信号是：interaction、tail、multi-pressure。

## 3. 两条主线的最简判断

### Ensemble 线

- 现在最强
- 但解释性较弱
- 提升主体来自树模型，尤其是 boosting
- 非树模型之间互相堆叠没有明显价值
- 真实 `xgb` 当前已经强到可以单独作为主力基线

### HybridSVM 线

- 现在还没追平树模型
- 但已经明显强于原始线性 SVM baseline
- 优势是可解释、可控、可做消融
- 已经证明 item 级长宽高数据确实能转化成有效特征

当前 HybridSVM 主结果：

| 模型 | Accuracy | AUC | TPR@1% |
| --- | ---: | ---: | ---: |
| baseline SVM | 0.9276 | 0.9651 | 0.6347 |
| 当前 15-feature bank | 0.9332 | 0.9749 | 0.7296 |

一句话解释：

> 这条线已经追回了树模型优势的一部分，而且追回来的部分是可解释的。

## 4. 最值得继续做的事

### 第一优先级：15-feature bank 分组消融

目标不是继续堆 feature，而是回答：

- 哪组特征对 AUC 贡献最大
- 哪组特征对 `TPR@1%` 贡献最大
- 哪些特征已经被更强特征吸收

### 第二优先级：继续做 heterogeneity 系列

目前最值得继续追的手工特征方向是：

- `volume_cv`
- `shape_mix_entropy`
- `p95_volume_over_median`

它们相对当前 active bank 的结果是：

- AUC `+0.0012`
- `TPR@1% +0.0374`
- Accuracy `-0.0016`

一句话解释：

> heterogeneity/long-tail 方向很可能是下一轮最有希望继续涨的方向。

## 5. 如果只讲 3 分钟

可以直接按下面顺序讲：

1. 先放 `ensemble_metric_comparison.svg`
   - 说明现阶段最强模型是谁
2. 再放 `ensemble_single_rf_importance.svg`
   - 说明 boosting/树模型为什么强
3. 再放 `hybridsvm_active_feature_coefficients.svg`
   - 说明我们已经把一部分树模型优势翻译成可解释特征
4. 最后放 `hybridsvm_leave_one_out_auc_drop.svg`
   - 说明当前最关键的 feature 是什么，以及下一步该如何做消融

## 6. 对应材料

- `research/2026-05-11_ensemble_and_hybridsvm_report.md`
- `research/feature_figures_20260511/README.md`
- `research/feature_figures_20260511/summary.json`
- `research/real_xgb_with_hybrid_features_20260511/summary.json`
