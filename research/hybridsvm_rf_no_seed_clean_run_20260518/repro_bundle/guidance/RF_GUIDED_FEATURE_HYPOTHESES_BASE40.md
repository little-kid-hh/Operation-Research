# RF-Guided Feature Hypotheses for Linear SVM (base40)

**Route:** RF base40 → GLM guidance → linear SVM feature search  
**Output file:** `RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md`  
**Date:** 2026-05-14

---

## 1. Reproducible Evidence Source

| Item | Path / Value |
|---|---|
| Training data | `/Users/zhongxiaochuan/Operation-Research/FunSearch_test/training_2orientations.csv` |
| RF config source | `/Users/zhongxiaochuan/Operation-Research/Ensemble_baseline/run_ensemble_ablation.py` |
| Analysis summary | `/Users/zhongxiaochuan/Operation-Research/research/rf_guidance_base40_20260514/summary.json` |
| Split | test_size=0.25, random_state=42 |
| Teacher scope | base40 only (no active-bank features) |
| RF hyper-params | n_estimators=200, max_depth=20, min_samples_leaf=2 |

---

## 2. Current Evidence

### 2.1 Metric Comparison (base40, same split)

| Metric | SVM | RF | Δ (RF − SVM) |
|---|---|---|---|
| Accuracy | 0.9276 | 0.9436 | +0.0160 |
| AUC | 0.9651 | 0.9823 | +0.0172 |
| TPR@FPR1% | 0.6347 | 0.8058 | **+0.1711** |
| FNR | 0.0320 | 0.0182 | −0.0138 |
| FPR | 0.2489 | 0.2232 | −0.0257 |
| Precision | 0.9444 | 0.9505 | +0.0061 |

**Key gap:** TPR@FPR1% shows the largest deficit (0.17). The linear SVM is substantially weaker than RF in the low-FPR operating regime, meaning it fails to rank the hardest positive cases high enough.

### 2.2 Recovery Analysis

- Test set: 2,500 rows
- SVM wrong, RF right: **66** cases (primary recovery opportunity)
- SVM right, RF wrong: **26** cases
- Net recoverable: 40 cases → potential accuracy gain ≈ +1.6 pp if all recovered

The 66 SVM-wrong/RF-right cases are the target. RF captures nonlinear patterns in these that the linear kernel cannot express with current features.

### 2.3 RF Feature Importance (top 12 of base40)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `spare_capacity` | 0.3117 |
| 2 | `wl_to_vehicle_wl_total` | 0.1409 |
| 3 | `sku_counts` | 0.0653 |
| 4 | `sku_average_volume` | 0.0638 |
| 5 | `wl_to_vehicle_wl_max` | 0.0345 |
| 6 | `h_to_H_ratio_avg` | 0.0300 |
| 7 | `sku_height_avg` | 0.0275 |
| 8 | `wl_to_vehicle_wl_std` | 0.0247 |
| 9 | `wl_to_vehicle_wl_avg` | 0.0231 |
| 10 | `sku_length_avg` | 0.0190 |
| 11 | `l_to_L_ratio_avg` | 0.0177 |
| 12 | `sku_concentration` | 0.0145 |

**Observations:**
- `spare_capacity` alone accounts for 31.2% of importance — a dominant single feature.
- The `wl_to_vehicle_wl_*` family (total, max, std, avg) collectively accounts for ~22.3% — RF leverages their joint structure.
- SKU dimension features (`sku_counts`, `sku_average_volume`, `sku_height_avg`, `sku_length_avg`) collectively ~17.6%.
- Item-to-vehicle ratio features (`h_to_H_ratio_avg`, `l_to_L_ratio_avg`) collectively ~4.8%.

---

## 3. Feature Directions to Test

The following hypotheses are derived from the importance structure and the SVM–RF performance gap. Each targets a specific nonlinear pattern that RF exploits but linear SVM cannot express without explicit feature construction.

### H1: `spare_capacity` Nonlinear Terms

**Rationale:** `spare_capacity` is the single most important feature (0.3117). RF can partition its range into step-wise regions via tree splits; linear SVM cannot.

- `spare_capacity_sq` = `spare_capacity²` — captures quadratic curvature in the utilization–feasibility relationship.
- `spare_capacity_cubed` = `spare_capacity³` — tests whether the relationship is steeper near zero spare capacity.
- `spare_capacity_log1p` = `log(1 + spare_capacity)` — compresses the right tail where additional spare capacity has diminishing effect.

### H2: `spare_capacity` Interactions with Top Features

**Rationale:** RF's top split is likely on `spare_capacity`, with subsequent splits on `wl_to_vehicle_wl_total` or `sku_counts` within each branch. Linear SVM needs explicit interaction terms.

- `spare_capacity_x_wl_total` = `spare_capacity × wl_to_vehicle_wl_total`
- `spare_capacity_x_sku_counts` = `spare_capacity × sku_counts`
- `spare_capacity_x_sku_avg_vol` = `spare_capacity × sku_average_volume`
- `spare_capacity_x_h_to_H` = `spare_capacity × h_to_H_ratio_avg`

### H3: `wl_to_vehicle_wl` Family Derived Ratios

**Rationale:** Four correlated `wl_to_vehicle_wl_*` features (total, max, std, avg) collectively contribute 22.3%. RF can pick conditional splits; SVM benefits from explicit ratio/dispersion features.

- `wl_cv` = `wl_to_vehicle_wl_std / (wl_to_vehicle_wl_avg + ε)` — coefficient of variation for weight-length utilization dispersion.
- `wl_max_to_avg` = `wl_to_vehicle_wl_max / (wl_to_vehicle_wl_avg + ε)` — captures whether a single SKU dominates weight-length utilization.
- `wl_max_residual` = `wl_to_vehicle_wl_max − wl_to_vehicle_wl_avg` — absolute version of the above, less sensitive to near-zero avg.

### H4: Cross-Dimension Ratio Interactions

**Rationale:** `h_to_H_ratio_avg` (0.0300) and `l_to_L_ratio_avg` (0.0177) measure vertical and longitudinal fit separately. Their interaction may indicate multi-axis tightness.

- `hH_x_lL` = `h_to_H_ratio_avg × l_to_L_ratio_avg` — joint tightness; high values mean both axes are tight.
- `max_dim_ratio` = `max(h_to_H_ratio_avg, l_to_L_ratio_avg)` — bottleneck dimension.
- `dim_ratio_range` = `abs(h_to_H_ratio_avg − l_to_L_ratio_avg)` — asymmetry between axes.

### H5: SKU Count × Dimension Interactions

**Rationale:** `sku_counts` (0.0653) and dimension features (`sku_average_volume`, `sku_height_avg`, `sku_length_avg`) are both important. More SKUs with large average dimensions may create different packing dynamics than few SKUs with large dimensions.

- `sku_counts_x_sku_avg_vol` = `sku_counts × sku_average_volume` — total volume proxy (if not already in base40).
- `sku_counts_x_hH` = `sku_counts × h_to_H_ratio_avg` — many SKUs in a tight vertical space.
- `sku_counts_x_lL` = `sku_counts × l_to_L_ratio_avg` — many SKUs in a tight longitudinal space.

### H6: Utilization Regime Indicators

**Rationale:** The large TPR@FPR1% gap suggests the SVM misses positives in specific utilization regimes. Binarized or binned indicators let the linear model assign regime-specific weights.

- `is_tight_fit` = `1 if spare_capacity < q25 else 0` — flag for the lowest spare-capacity quartile (where feasibility is hardest to predict).
- `is_high_wl` = `1 if wl_to_vehicle_wl_total > q75 else 0` — flag for high weight-length utilization.
- `tight_and_high_wl` = `is_tight_fit × is_high_wl` — interaction of the two regime flags.

---

## 4. What Not to Over-Invest In

| Avoid | Reason |
|---|---|
| Features derived from base40 variables with importance < 0.01 | Signal too weak; RF itself barely uses them. |
| High-degree polynomials (degree ≥ 4) on any single feature | Risk of overfitting on 2,500 test rows; RF's max_depth=20 already regularizes. |
| Complex multi-way interactions (3+ features multiplied) | Exponential candidate explosion; prioritize 2-way interactions from top-5 features first. |
| Features that duplicate existing base40 columns | Check `sku_counts × sku_average_volume` against any existing total-volume feature in base40 before adding. |
| Active-bank features | Teacher scope is base40 only. Do not leak features from later pipeline stages. |
| Extensive hyper-parameter tuning of SVM alongside feature search | Confounds feature attribution. Fix SVM C and kernel params; vary only features. |

---

## 5. Immediate Workflow Implication

1. **Priority order:** Test H1 and H2 first (target `spare_capacity` nonlinearity and interactions — 45%+ of RF importance). Then H3 (wl family ratios), then H4–H6.

2. **Evaluation protocol:** Use the same split (test_size=0.25, random_state=42). Report AUC and TPR@FPR1% as primary metrics; accuracy as secondary. The target is to close the TPR@FPR1% gap (currently 0.17) by at least 30% (i.e., reach ≥ 0.69).

3. **Feature search entry point:** Pass this document to `HybridSVM/scripts/run_feature_search_agent.py` via `--tree-guidance-path`. The agent should generate candidate features from the hypotheses above, evaluate them incrementally, and retain only those that improve TPR@FPR1% without degrading AUC.

4. **Reproducibility:** Save the prompt, raw LLM response, and final markdown at:
   - Prompt: `{analysis_dir}/feature_search_prompt.txt`
   - Raw response: `{analysis_dir}/feature_search_raw_response.txt`
   - Final markdown: `{analysis_dir}/RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

5. **Stopping criterion:** If cumulative feature additions recover ≥ 40 of the 66 SVM-wrong/RF-right cases, or if three consecutive candidate batches yield no TPR@FPR1% improvement, halt the search.
