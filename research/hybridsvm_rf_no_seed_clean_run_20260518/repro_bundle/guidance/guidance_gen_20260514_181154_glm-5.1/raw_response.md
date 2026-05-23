## GUIDANCE_MARKDOWN
```markdown
# RF Base40 → GLM Guidance → Linear SVM Feature Search

**Route:** RF base40 → GLM guidance → linear SVM feature search  
**Output file:** `RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md`  
**Date:** 2026-05-14

---

## 1. Reproducible Evidence Source

| Item | Value |
|---|---|
| Data path | `/Users/zhongxiaochuan/Operation-Research/FunSearch_test/training_2orientations.csv` |
| Feature set | base40 (no active-bank features) |
| Teacher model | RandomForest, n_estimators=200, max_depth=20, min_samples_leaf=2 |
| RF config source | `/Users/zhongxiaochuan/Operation-Research/Ensemble_baseline/run_ensemble_ablation.py` |
| Split | test_size=0.25, random_state=42 |
| Summary artifact | `/Users/zhongxiaochuan/Operation-Research/research/rf_guidance_base40_20260514/summary.json` |
| Analysis dir | `/Users/zhongxiaochuan/Operation-Research/research/rf_guidance_base40_20260514` |

Reproducibility requirement: save the prompt, raw LLM response, and final markdown alongside this guidance.

---

## 2. Current Evidence

### 2.1 Metric Gap (SVM base40 vs. RF base40)

| Metric | SVM | RF | Gap |
|---|---|---|---|
| Accuracy | 0.9276 | 0.9436 | +0.0160 |
| AUC | 0.9651 | 0.9823 | +0.0172 |
| TPR@FPR1% | 0.6347 | 0.8058 | **+0.1711** |
| Recall | 0.9680 | 0.9818 | +0.0138 |
| FPR (default threshold) | 0.2489 | 0.2232 | −0.0258 |
| FNR | 0.0320 | 0.0182 | −0.0138 |

The dominant gap is at the low-FPR operating point: RF recovers ~17 percentage points more true positives at FPR≤1%. The SVM threshold at FPR=1% is 0.934, indicating the SVM decision function is poorly calibrated near the boundary for hard negatives.

### 2.2 Recovery Asymmetry

- **SVM wrong, RF right:** 66 test samples (recovery targets)
- **SVM right, RF wrong:** 26 test samples

The 66 recovery samples are the primary optimization target. These are cases where nonlinear structure captured by RF is not expressible by the current linear SVM on base40 features.

### 2.3 Feature Importance (RF permutation/gini, base40)

| Rank | Feature | Importance |
|---|---|---|
| 1 | spare_capacity | 0.3117 |
| 2 | wl_to_vehicle_wl_total | 0.1409 |
| 3 | sku_counts | 0.0653 |
| 4 | sku_average_volume | 0.0638 |
| 5 | wl_to_vehicle_wl_max | 0.0345 |
| 6 | h_to_H_ratio_avg | 0.0300 |
| 7 | sku_height_avg | 0.0275 |
| 8 | wl_to_vehicle_wl_std | 0.0247 |
| 9 | wl_to_vehicle_wl_avg | 0.0231 |
| 10 | sku_length_avg | 0.0190 |
| 11 | l_to_L_ratio_avg | 0.0177 |
| 12 | sku_concentration | 0.0145 |

`spare_capacity` alone accounts for 31.2% of RF importance. The `wl_to_vehicle_wl_*` family (total, max, std, avg) collectively accounts for ~22.3%. Height-related features (`h_to_H_ratio_avg`, `sku_height_avg`) account for ~5.8%.

---

## 3. Feature Directions to Test

All hypotheses below are grounded in the evidence above. They aim to linearize nonlinear decision boundaries that RF exploits but SVM cannot express.

### 3.1 spare_capacity Interactions (highest priority)

`spare_capacity` dominates RF importance. Threshold and interaction effects around spare_capacity are the strongest hypothesis space.

- **H1:** `spare_capacity × sku_counts` — RF may capture that high spare capacity is only predictive when many SKUs remain unpacked.
- **H2:** `spare_capacity × sku_average_volume` — interaction between residual space and average item size.
- **H3:** `spare_capacity × wl_to_vehicle_wl_total` — residual space modulated by total weight-loading ratio.
- **H4:** `spare_capacity²` — quadratic term to capture diminishing returns or acceleration near capacity limits.
- **H5:** Indicator features: `spare_capacity < τ` for thresholds τ ∈ {0.05, 0.10, 0.15, 0.20} — RF can partition on spare_capacity via tree splits; explicit binarization gives SVM similar capability.

### 3.2 wl_to_vehicle_wl Family Interactions

The four wl_to_vehicle_wl variants suggest RF uses weight-loading dispersion and extremes, not just averages.

- **H6:** `wl_to_vehicle_wl_max − wl_to_vehicle_wl_avg` — dispersion of weight-loading across orientations (already partially captured by std, but max-minus-mean may capture skew).
- **H7:** `wl_to_vehicle_wl_total × h_to_H_ratio_avg` — total weight-loading modulated by vertical fill ratio.
- **H8:** `wl_to_vehicle_wl_std × sku_counts` — weight-loading dispersion becomes more informative with more SKUs.

### 3.3 Vertical/Height Constraints

`h_to_H_ratio_avg` ranks 6th, suggesting vertical packing tightness is a separate signal from overall capacity.

- **H9:** `h_to_H_ratio_avg × spare_capacity` — vertical constraint becomes critical only when overall spare capacity is low.
- **H10:** `h_to_H_ratio_avg²` — nonlinear vertical tightness effect.

### 3.4 SKU Count and Volume Interactions

- **H11:** `sku_counts × sku_average_volume` — joint effect of item count and size; RF splits on these jointly.
- **H12:** `sku_counts / spare_capacity` — packing density ratio (items per unit residual space).

### 3.5 Low-FPR Boundary Features

The TPR@FPR1% gap is the largest single metric gap. The 116 SVM false positives at default threshold (and the hard decision boundary at threshold 0.934) suggest features that better separate near-boundary negatives.

- **H13:** `spare_capacity × (1 − wl_to_vehicle_wl_total)` — interaction explicitly marking "space remains but weight-loading is low" (potential false-positive region).
- **H14:** `l_to_L_ratio_avg × spare_capacity` — horizontal fill ratio × residual space.

---

## 4. What Not to Over-Invest In

| Direction | Rationale |
|---|---|
| `sku_concentration` variants | Importance only 0.0145; unlikely to drive recovery of 66 samples. |
| Raw dimension averages (`sku_length_avg`, `sku_height_avg`) alone | Already in base40; low marginal importance (0.019, 0.028); ratio features dominate. |
| Active-bank features | Out of scope for this route; must not leak into guidance. |
| Optimizing for the 26 RF-wrong/SVM-right samples | These are 1% of test set and may reflect RF overfitting; not a productive target. |
| High-degree polynomial expansions (degree ≥ 3) | Risk of overfitting on 2500 test rows; limit to degree 2 and pairwise interactions. |
| Features requiring external data or non-deterministic computation | Must remain reproducible from the CSV at `data_path` with `random_state=42`. |

---

## 5. Immediate Workflow Implication

1. **Target metric:** TPR@FPR1% (current: 0.6347, RF ceiling: 0.8058). Secondary: AUC (current: 0.9651, RF ceiling: 0.9823).
2. **Target sample set:** The 66 SVM-wrong/RF-right instances define the recovery frontier.
3. **Feature search strategy:** Generate features from hypotheses H1–H14 above. Evaluate each feature by adding it to base40 and retraining linear SVM; record delta-TPR@FPR1% and delta-AUC. Retain features with positive contribution.
4. **Constraint:** All new features must be deterministic functions of columns already present in base40. No active-bank features. No external data.
5. **Reproducibility:** The consuming script (`HybridSVM/scripts/run_feature_search_agent.py`) will receive this guidance via `--tree-guidance-path`. Log the prompt, raw response, and final markdown in the analysis directory.
6. **Stopping criterion:** If no single hypothesis feature improves TPR@FPR1% by ≥ 0.005, consider pairwise combinations of the top 3 hypotheses before concluding the search.
```