You are improving a fixed linear-SVM baseline for 3D bin-packing feasibility.

        Produce:
        1. a short policy update for the next iteration;
        2. one Python candidate that adds interpretable per-dispatch features.

        Hard rules:
        - Keep the model family fixed: the candidate only generates features.
        - Use only `agg_df` and `items_df` passed into the function.
        - No file I/O, no API calls, no labels, no target leakage.
        - Return one row per dispatch with `dispatch_id` plus numeric feature columns.
        - Use ASCII feature names only.
        - Prefer at most `6` new features.
        - Use only pandas and numpy.
        - Keep feature computations in pandas objects; do not call `.values` unless you
          immediately wrap the result back into a `pd.Series` or `pd.DataFrame`.
        - Do not call `.rename(...)` on numpy arrays.
        - Favor tree-inspired signals: threshold counts, tail ratios, pressure-slack interactions,
          and local awkward-pattern shares.
        - You are in an iterative search loop with a cumulative active feature bank.
        - The evaluated model uses: base aggregate features + active feature bank + your new features.
        - Propose only new incremental features to add on top of the active feature bank.
        - Do not repeat, rename, or rewrite any feature already in the active bank.
        - Make a small, explicit local change: usually add 1-3 new feature ideas, not a reset.
        - Propose a candidate only if you believe the cumulative feature set can beat
          the current active bank under the acceptance rule below.

        Current best accepted target to beat:
        - AUC: `0.9686`
        - TPR@FPR=1%: `0.6691`
        - Accuracy: `0.9288`

        Acceptance rule:
        - accept if AUC improves by more than `0.0005`
        - otherwise require TPR@FPR=1% improvement larger than `0.005`
        - if still tied, require Accuracy improvement larger than `0.0005`

        Candidate function signature:

        ```python
        def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
            ...
        ```

        Response format:

        ## POLICY_UPDATE
        <markdown bullets>

        ## FEATURE_CODE
        ```python
        ...
        ```

        ## RATIONALE
        - bullet 1
        - bullet 2

        Current iteration: `3`

        # Context

Task: improve the current linear-SVM route with a small number of interpretable,
per-dispatch numeric features. The model family and split protocol are fixed.

## Data Scope
- dispatch rows after deduplication: `10000`
- train rows: `7500`
- test rows: `2500`
- positive rate: `0.8083`
- matched item rows for these dispatches: `100317`
- item-count per dispatch: mean=`10.032`, median=`10.0`, max=`22`

## Fixed Baseline Metrics
- Accuracy: `0.9276`
- Precision: `0.9444`
- Recall: `0.9680`
- ROC AUC: `0.9651`
- TPR@FPR=1%: `0.6347`

## Existing Aggregate Feature Columns
`sku_counts`, `sku_average_volume`, `sku_length_var`, `sku_width_var`, `sku_height_var`, `sku_length_avg`, `sku_width_avg`, `sku_height_avg`, `max_asr`, `vehicle_length`, `vehicle_width`, `vehicle_height`, `spare_capacity`, `sku_concentration`, `sku_min_length`, `sku_max_length`, `sku_std_length`, `sku_min_width`, `sku_max_width`, `sku_std_width`, `sku_min_height`, `sku_max_height`, `sku_std_height`, `l_to_L_ratio_avg`, `l_to_L_ratio_min`, `l_to_L_ratio_max`, `l_to_L_ratio_std`, `h_to_H_ratio_avg`, `h_to_H_ratio_min`, `h_to_H_ratio_max`, `h_to_H_ratio_std`, `w_to_W_ratio_avg`, `w_to_W_ratio_min`, `w_to_W_ratio_max`, `w_to_W_ratio_std`, `wl_to_vehicle_wl_avg`, `wl_to_vehicle_wl_min`, `wl_to_vehicle_wl_max`, `wl_to_vehicle_wl_std`, `wl_to_vehicle_wl_total`

## Item-Level Table Schema
`dispatch_id`, `item_length`, `item_width`, `item_height`, `if_fragile`, `load_parameter`, `vehicle_capacity`, `dim_s`, `dim_m`, `dim_l`, `item_volume`, `item_flatness`

Notes:
- `agg_df` already contains per-dispatch aggregate features and vehicle dimensions.
- `items_df` contains one row per item with both raw dimensions and sorted dimensions:
  `dim_s <= dim_m <= dim_l`.
- Candidate code must return one row per dispatch with ASCII feature names.

## Current SVM Linear Insights
The classifier is **linear** on MinMax-scaled features: `decision = w·x_scaled + b`, then label 1 if decision ≥ 0.
- Intercept `b` = **11.5265**
- **|w_j|** large → SVM is sensitive to that feature (in scaled space).
- **w_j > 0** → higher scaled value pushes toward **feasible (1)**; **w_j < 0** → toward **not feasible (0)**.

| rank | feature | w_j (on scaled x) | |w_j| |
|------|---------|---------------------|------|
| 1 | `sku_average_volume` | -16.4102 | 16.4102 |
| 2 | `spare_capacity` | 13.5627 | 13.5627 |
| 3 | `sku_counts` | -10.5919 | 10.5919 |
| 4 | `wl_to_vehicle_wl_avg` | 7.62511 | 7.62511 |
| 5 | `wl_to_vehicle_wl_std` | -4.26032 | 4.26032 |
| 6 | `l_to_L_ratio_avg` | -3.0218 | 3.0218 |
| 7 | `sku_length_avg` | -3.0218 | 3.0218 |
| 8 | `wl_to_vehicle_wl_min` | -2.94932 | 2.94932 |
| 9 | `sku_std_length` | -2.5518 | 2.5518 |
| 10 | `l_to_L_ratio_std` | -2.5518 | 2.5518 |
| 11 | `sku_height_var` | -2.39459 | 2.39459 |
| 12 | `wl_to_vehicle_wl_total` | 2.1109 | 2.1109 |
| 13 | `l_to_L_ratio_min` | -1.66505 | 1.66505 |
| 14 | `sku_min_length` | -1.66505 | 1.66505 |
| 15 | `w_to_W_ratio_avg` | -1.59533 | 1.59533 |
| 16 | `sku_width_avg` | -1.59533 | 1.59533 |
| 17 | `wl_to_vehicle_wl_max` | 1.3037 | 1.3037 |
| 18 | `sku_max_width` | 1.24877 | 1.24877 |
| 19 | `w_to_W_ratio_max` | 1.24877 | 1.24877 |
| 20 | `w_to_W_ratio_min` | -1.10221 | 1.10221 |
| 21 | `sku_min_width` | -1.10221 | 1.10221 |
| 22 | `w_to_W_ratio_std` | -1.03794 | 1.03794 |
| 23 | `sku_std_width` | -1.03794 | 1.03794 |
| 24 | `sku_width_var` | -0.833531 | 0.833531 |
| 25 | `sku_height_avg` | -0.736217 | 0.736217 |

_(15 more features omitted; smallest |w| omitted.)_

When designing `apply_rule_patch`, raw feature values are **not** scaled like `x_scaled`; combine this table with the **Feature scales** section to reason about conflicts on hard cases.

## Tree-Model Contrast Hypotheses
- Purely linear-style combinations did not close the gap to the stronger tree baselines;
  the likely missing piece is explicit threshold / interaction structure rather than more
  smooth averages of the same aggregates.
- In local checks, tree models appear to recover some cases that linear SVM misses,
  especially around `spare_capacity`, `sku_average_volume`, `wl_to_vehicle_wl_max`,
  `wl_to_vehicle_wl_total`, `sku_length_avg`, and item-dimension variance patterns.
- Translate those advantages into explicit features such as: upper-tail dimension pressure,
  low-slack x large-piece interactions, near-limit piece counts/shares, repetition versus
  heterogeneity signals, and footprint or wall-pressure proxies.
- Treat this as a search hint: propose thresholded counts, tail-pressure features,
  and interaction terms that might linearize those nonlinear regimes for the SVM.

## Search Bias
- prioritize compact, explainable feature sets
- use item-distribution shape, tails, concentration, and bottleneck signals
- explicitly test threshold-style and interaction-style hypotheses suggested by trees
- avoid re-encoding what the existing aggregate means and variances already say

## Tree-Guidance Snapshot
The following markdown is a frozen guidance snapshot derived from the
current tree-model analysis workflow. It is part of the formal prompt
input for this run and should be treated as a hypothesis source rather
than ground truth.

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


        # Feature Search Policy

Objective:
Improve the linear SVM baseline by adding a small number of interpretable,
per-dispatch numeric features.

Hard constraints:
- Do not change the train/test split.
- Do not change the model family: keep the stage-1 model as linear SVM.
- Do not use labels or any target-derived statistics inside feature generation.
- Do not read files or call external services in candidate feature code.
- Return one row per dispatch and keep feature names ASCII.

Preferred feature families:
- item-level quantiles / tails rather than only mean/std
- thresholded local bottleneck counts, not only global averages
- slack / spare-capacity interactions with large-piece pressure
- repeated-type structure and concentration
- extreme-piece bottlenecks against vehicle dimensions
- face-area / edge-pressure style packing stress signals
- heterogeneity / multimodality / long-tail measures
- features that are easy to explain to a human reviewer

Avoid:
- duplicating obvious existing aggregates unless the new version captures a
  different shape signal
- opaque embeddings
- huge feature sets; prefer a compact hypothesis with clear semantics

Tree-inspired search bias:
- assume trees are winning partly because they exploit local thresholds and
  interactions on features like `spare_capacity`, `wl_to_vehicle_wl_max`,
  `wl_to_vehicle_wl_total`, `sku_average_volume`, and dimension variances
- try to convert those nonlinear effects into explicit numeric features that
  a linear SVM can use

## Iteration 1
- Focus on the TPR@FPR=1% gap (0.17 below RF): the SVM misses hard positives because it cannot express nonlinear interactions and threshold effects on the top features.
- Priority 1: explicit interactions between `spare_capacity` and the next-highest RF features (`wl_to_vehicle_wl_total`, `sku_counts`) to linearize the conditional splits trees make.
- Priority 2: a threshold-based item-level signal (`big_piece_share`) that trees create naturally via binary splits but SVM cannot.
- Priority 3: cross-dimension tightness product (`hH × lL`) and a quadratic spare-capacity term to capture curvature near the decision boundary.


        # Memory

Baseline reference:
- AUC `0.9651`
- TPR@FPR=1% `0.6347`
- Accuracy `0.9276`

## Accepted Trials
- iter `1`: AUC `0.9686` (Δ `+0.0035`), TPR@1% `0.6691` (Δ `+0.0344`), features=spare_cap_x_wl_total, spare_cap_x_sku_counts, spare_cap_sq, hH_x_lL, big_piece_share

## Rejected Trials
- iter `2`: AUC `0.9687` (Δ `+0.0035`), TPR@1% `0.6667` (Δ `+0.0320`), features=spare_cap_x_sku_avg_vol, wl_max_residual, awkward_shape_share

## Guidance
- only promote candidates that beat the current best under the acceptance rule
- keep trying compact feature sets with explicit physical interpretation
- prefer features that improve AUC and low-FPR recall without exploding FPR
- if a feature is weak alone but strong in combination, note that in rationale


        ## Current Active Feature Bank
- source accepted trials: `1`
- active feature count: `5`
- active feature names: `spare_cap_x_wl_total`, `spare_cap_x_sku_counts`, `spare_cap_sq`, `hH_x_lL`, `big_piece_share`
- active bank AUC: `0.9686`
- active bank TPR@FPR=1%: `0.6691`
- active bank Accuracy: `0.9288`

Only propose new incremental features to add on top of this bank.
Do not re-emit, rename, or overwrite any active-bank feature.
