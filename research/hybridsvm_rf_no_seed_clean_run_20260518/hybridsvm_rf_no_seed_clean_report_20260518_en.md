# HybridSVM RF -> GLM No-Seed Clean Run Report (2026-05-18)

## 0. Scope and Status

This report documents one formal no-seed feature-search run:

- route: `RF -> GLM -> linear SVM`
- experiment directory: `HybridSVM/experiments_feature_search/by_model/glm-5.1/exp_20260518_194131/`

This run should be treated as the formal clean run because:

- no manual seed features were used;
- the run was not resumed on the same `exp_dir`;
- the budget was fixed at exactly `10` iterations;
- the only teacher guidance used was the frozen
  `RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md`.

Accordingly, this run is the current formal record for the no-seed
`RF -> GLM -> SVM` route.

---

## 1. End-to-End Pipeline

```text
raw dispatch table + raw item table
    |
    v
data loading and normalization
    |
    v
two internal views
  - agg_df: one row per dispatch
  - items_df: one row per item
    |
    v
fixed-split base40 SVM baseline
    |
    v
base40 RF teacher analysis
    |
    v
GLM writes frozen RF guidance markdown
    |
    v
feature-search prompt assembly:
context + policy + memory + active bank
    |
    v
GLM returns POLICY_UPDATE + FEATURE_CODE + RATIONALE
    |
    v
local execution of FEATURE_CODE on full agg_df and full items_df
    |
    v
candidate feature materialization + SVM evaluation + validator decision
    |
    v
accept / reject / fail
    |
    v
update trials, memory, active bank, and summary
```

---

## 2. Role of the Raw Data

### 2.1 Meaning of `dispatch`

In this repository, `dispatch` is the normalized name for the raw column
`发车号`.

The prediction target is defined at the dispatch level. Therefore:

- one sample = one `dispatch`;
- one label = whether that dispatch is feasible / loaded (`if_loaded`);
- multiple item rows may belong to the same `dispatch`.

### 2.2 Raw inputs

The pipeline starts from two raw tables:

- dispatch-level table: `FunSearch_test/training_2orientations.csv`
- item-level table: `FunSearch_test/物品信息和dblf信息.csv`

These two tables serve different roles.

### 2.3 How the raw dispatch table is used

The dispatch table is used to:

1. **define samples**
   - one row corresponds to one `dispatch`;
   - the train/test split is defined at this level.

2. **provide supervision**
   - the target label `if_loaded` is read from this table.

3. **provide the base40 aggregate representation**
   - after deduplication and column dropping, the remaining aggregate columns
     form the fixed baseline feature set for both the SVM baseline and the RF
     teacher.

4. **provide vehicle-level context**
   - fields such as `vehicle_length`, `vehicle_width`, `vehicle_height`, and
     `spare_capacity` are later used as denominators, thresholds, and
     interaction anchors in candidate features.

### 2.4 How the raw item table is used

The item table is not a second supervised-learning matrix. Its purpose is to
provide the fine-grained geometry and constraint information needed for feature
engineering.

It is normalized into:

- raw item dimensions: `item_length`, `item_width`, `item_height`
- sorted dimensions: `dim_s`, `dim_m`, `dim_l`
- item volume: `item_volume`
- shape proxy: `item_flatness`
- per-item attributes: e.g. `if_fragile`
- weight-related fields: `load_parameter`, `vehicle_capacity`

During feature generation, these rows are grouped by `dispatch_id` to compute:

- shares of long / tall / multi-axis-tight items;
- tail quantiles and tail ratios;
- fragile share;
- load utilization;
- interactions between dispatch-level slack and item-level geometric pressure.

### 2.5 What the raw data becomes inside the search loop

After loading and normalization, the raw inputs become two internal views:

1. **`agg_df`**
   - one row per `dispatch`;
   - contains base40 aggregate columns and vehicle dimensions;
   - serves as both the baseline feature frame and the dispatch-level input to
     candidate code.

2. **`items_df`**
   - one row per item;
   - retains fine-grained dimensions and attributes;
   - serves as the item-level input to candidate code.

The LLM does not receive the full raw tables in the prompt. It is only told
that locally executed feature code will later receive `agg_df` and `items_df`.
The returned code is then executed on the full local `agg_df` and `items_df`.

---

## 3. What Is Actually Passed to GLM

The implementation does not send the full dispatch matrix or full item table
to GLM.

Instead, GLM receives a prompt assembled from:

1. `context.md`
2. `policy.md`
3. `memory.md`
4. `active_feature_bank.md`

### 3.1 Data summary

The data-related part of `context.md` contains:

- the number of dispatch rows after deduplication;
- train/test split sizes;
- positive rate;
- the number of matched item rows;
- item-count statistics per dispatch;
- base40 column names;
- item schema names.

Thus GLM sees:

- table structure;
- dataset scale;
- available fields.

It does not see:

- the full `10000` dispatch rows;
- the full `100317` item rows;
- raw numeric rows for individual cases.

### 3.2 Student-model summary

The prompt also contains a compact summary of the current linear SVM:

- intercept;
- top coefficients by absolute magnitude;
- interpretation of coefficient signs.

This tells GLM which base40 directions the linear model already uses strongly,
and where explicit nonlinear recovery may still be required.

### 3.3 Teacher-model summary

The prompt inserts the frozen RF guidance file:

- `HybridSVM/RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

This contributes the teacher-side evidence:

- metric gap between SVM and RF on the fixed split;
- recovery counts (`SVM wrong, RF right`, etc.);
- top RF features;
- recommended feature directions;
- directions that are less worth investing in.

### 3.4 Iteration memory

The prompt also contains the evolving search state:

- `policy.md`: hard constraints and preferred feature families;
- `memory.md`: summaries of accepted and rejected trials;
- `active_feature_bank.md`: currently accepted incremental features and their
  resulting metrics.

Therefore, GLM reasons from:

- data availability;
- current linear-model behavior;
- RF teacher evidence;
- cumulative search history.

---

## 4. RF Teacher Analysis

### 4.1 Teacher package

The teacher package used in this run is:

- `research/rf_guidance_base40_20260514/`

Core files:

- `summary.json`
- `base40_rf_importance.csv`
- `rf_vs_svm_recovery_contrast_base40.csv`

### 4.2 Fixed protocol

The teacher analysis uses the following fixed protocol:

- data: `FunSearch_test/training_2orientations.csv`
- split: `test_size=0.25`, `random_state=42`
- student model: linear SVM with `C=10.0`
- teacher scope: base40 only

The RF configuration comes from:

- `Ensemble_baseline/run_ensemble_ablation.py`

The configuration used here is:

- `n_estimators=200`
- `max_depth=20`
- `min_samples_leaf=2`

### 4.3 Teacher-side findings

On the fixed split, base40 performance is:

| Model | Accuracy | AUC | TPR@FPR=1% |
|---|---:|---:|---:|
| linear SVM | `0.9276` | `0.9651` | `0.6347` |
| RF | `0.9436` | `0.9823` | `0.8058` |

Recovery counts:

- `SVM wrong, RF right = 66`
- `SVM right, RF wrong = 26`

The dominant RF signals concentrate on:

- `spare_capacity`
- `wl_to_vehicle_wl_total`
- `sku_counts`
- `sku_average_volume`
- `wl_to_vehicle_wl_max`
- `h_to_H_ratio_avg`
- `sku_height_avg`
- `wl_to_vehicle_wl_std`
- `wl_to_vehicle_wl_avg`

This suggests that feature search should prioritize explicit numeric encodings
of the nonlinear structures exploited by trees, especially:

- nonlinear and interaction terms around `spare_capacity`;
- dispersion and tail structure in the `wl` family;
- item-level threshold bottleneck counts / shares;
- multi-axis tightness and local crowding structure;
- weight-related pressure signals not explicitly represented in base40.

### 4.4 How RF knowledge is taught to GLM

This step is fully programmatic and reproducible; it does not rely on a human
writing the guidance manually.

The workflow is:

1. Run `research/analyze_rf_guidance_base40.py`
   - produces the structured RF teacher package.

2. Run `HybridSVM/scripts/generate_guidance_with_llm.py`
   - reads `summary.json` from the teacher package;
   - sends the following structured evidence to GLM:
     - teacher scope
     - split
     - data path
     - RF config
     - SVM / RF metric gap
     - recovery counts
     - top base40 RF features
   - asks GLM to write one frozen guidance markdown file.

3. Save the full trace:
   - prompt: `repro_bundle/guidance/guidance_gen_20260514_181154_glm-5.1/prompt.md`
   - raw response: `.../raw_response.md`
   - manifest: `.../manifest.json`
   - final guidance: `repro_bundle/guidance/RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

Accordingly, the step “RF teaches GLM how to search for features” is itself
reproducible.

---

## 5. Validator Logic

### 5.1 Required response format

Each GLM response must contain three parts:

- `POLICY_UPDATE`
- `FEATURE_CODE`
- `RATIONALE`

`FEATURE_CODE` must define:

```python
build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame
```

### 5.2 Local execution flow

Candidate code is executed locally rather than by the model itself:

1. validate the returned code block;
2. execute it in a restricted namespace;
3. call `build_candidate_features(agg_df.copy(), items_df.copy())`;
4. align the returned frame to one row per dispatch;
5. concatenate:
   - base40 features;
   - current active feature bank;
   - new candidate features;
6. retrain and evaluate the linear SVM on the fixed split.

### 5.3 `accept / reject / fail`

A trial is labeled:

- **fail**
  - no valid code block;
  - code not executable;
  - invalid return object;
  - missing or duplicated `dispatch_id`;
  - feature-name conflict or non-ASCII feature names.

- **reject**
  - code executed successfully;
  - but it did not beat the current best accepted trial under the acceptance
    rule.

- **accept**
  - code executed successfully;
  - and it beat the current best accepted trial under the acceptance rule.

Acceptance rule:

- accept if AUC improves by more than `0.0005`;
- otherwise require TPR@FPR=1% improvement larger than `0.005`;
- if still tied, require Accuracy improvement larger than `0.0005`.

### 5.4 Repair mechanism

If the first response violates format or code constraints, the pipeline sends a
repair prompt containing:

- the original prompt;
- the original response;
- the error message.

This RF clean run had no final failures, but `iter_007` and `iter_008` still
preserve repair traces, which confirms that the repair path was exercised.

---

## 6. Overall Results

### 6.1 Baseline

- Accuracy `0.9276`
- AUC `0.9651`
- TPR@FPR=1% `0.6347`

### 6.2 Final outcome

Trial counts:

- `10` total
- `6` accepted
- `4` rejected
- `0` failed

Accepted iterations:

- `1`, `3`, `5`, `6`, `9`, `10`

Final best accepted trial:

- iteration `10`
- features:
  - `load_util_ratio`
  - `multi_dim_tight_share`
  - `dim_l_tail_ratio`

Best accepted metrics:

- Accuracy `0.9312`
- AUC `0.9733`
- TPR@FPR=1% `0.6903`

Delta vs baseline:

- Accuracy `+0.0036`
- AUC `+0.0082`
- TPR@FPR=1% `+0.0556`

One additional point matters:

- the highest accepted `TPR@FPR=1%` occurred at `iter_006`, with `0.7040`;
- however, `iter_010` produced a larger AUC gain and cleared the acceptance
  margin, so the final active bank advanced to `iter_010`.

This shows that the current validator uses a lexicographic objective with AUC
first, rather than optimizing low-FPR recall in isolation.

---

## 7. Per-Iteration Results

| Iter | Status | New features | Result |
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

## 8. Interpretation

This run supports four conclusions.

### 8.1 The no-seed route is viable

Iteration `001` was already accepted. This means the RF-guided route can
bootstrap without any manually written seed features.

### 8.2 Item-level data contributes genuinely new information

A large share of accepted features comes from item-level statistics rather than
simple rewrites of base40 aggregates. Examples include:

- `big_piece_share`
- `item_vol_cv`
- `fragile_share`
- `near_h_limit_share`
- `multi_dim_tight_share`
- `dim_l_tail_ratio`

This shows that the item table is not auxiliary context; it is the main source
of new interpretable geometric information beyond base40.

### 8.3 RF guidance provides direction, not formulas

The RF teacher does not directly output candidate feature formulas. Instead, it
provides:

- which base40 variables matter most;
- which error cases RF recovers over SVM;
- which directions are worth exploring first.

GLM then converts that evidence into:

- interaction terms;
- threshold shares;
- tail ratios;
- explicit slack-pressure features.

So RF acts as a structured inductive bias, not as a hand-written feature
template library.

### 8.4 The acceptance rule favors stronger cumulative AUC

This run makes that trade-off explicit:

- `iter_006` achieved better `TPR@FPR=1%`;
- `iter_010` achieved better AUC and Accuracy;
- because the validator compares AUC first, the active bank ultimately moved to
  `iter_010`.

If future work wants to prioritize low-FPR recall more directly, the validator
itself may become part of the research question.

---

## 9. Key Files

Main report:

- `research/hybridsvm_rf_no_seed_clean_run_20260518/hybridsvm_rf_no_seed_clean_report_20260518.md`

English report:

- `research/hybridsvm_rf_no_seed_clean_run_20260518/hybridsvm_rf_no_seed_clean_report_20260518_en.md`

Concise package summary:

- `research/hybridsvm_rf_no_seed_clean_run_20260518/README.md`

Original experiment directory:

- `HybridSVM/experiments_feature_search/by_model/glm-5.1/exp_20260518_194131/`

Teacher analysis:

- `research/rf_guidance_base40_20260514/`

Frozen teacher guidance:

- `HybridSVM/RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

Guidance-generation trace:

- `research/hybridsvm_rf_no_seed_clean_run_20260518/repro_bundle/guidance/guidance_gen_20260514_181154_glm-5.1/`

Core implementation:

- `HybridSVM/scripts/generate_guidance_with_llm.py`
- `HybridSVM/scripts/run_feature_search_agent.py`
- `HybridSVM/src/feature_search.py`
- `HybridSVM/src/svm_train.py`
- `research/analyze_rf_guidance_base40.py`
