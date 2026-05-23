# HybridSVM XGB -> GLM No-Seed Exploratory Run Report (2026-05-18)

## 0. Scope and Status

This report documents a single no-seed feature-search run under the route
`XGB -> GLM -> linear SVM`.

Experiment directory:

- `trials_archive/exp_20260518_173257/`

This run should be treated as **exploratory**, not as the final formal
10-iteration result, because:

- it started as a fresh no-seed run;
- it was later resumed on the same `exp_dir`;
- in `code/HybridSVM/scripts/run_feature_search_agent.py`, `--n-iters` in resume
  mode means “append this many more iterations”, not “run to this total”.

Consequently, the directory accumulated `15` trials instead of the originally
intended `10`.

Even so, the run is methodologically important because it answers the following
question:

- can the `XGB -> GLM -> linear SVM` route bootstrap without any manually
  written seed features?

The answer from this run is **yes**: iteration `001` was already accepted.

---

## 1. End-to-End Pipeline

```text
raw dispatch table + raw item table
    |
    v
data loading and normalization
    |
    v
two parallel representations
  - agg_df: one row per dispatch
  - items_df: one row per item
    |
    v
fixed-split base40 SVM baseline
    |
    v
base40 XGB teacher analysis
    |
    v
GLM writes frozen tree-guidance markdown
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

Operationally, one `dispatch` corresponds to one concrete loading / departure
job. The prediction target is defined at this level. Therefore:

- one sample = one `dispatch`;
- one label = whether that dispatch is feasible / loaded (`if_loaded`);
- multiple item rows may belong to the same `dispatch`.

### 2.2 Raw inputs

The pipeline starts from two raw tables:

- dispatch-level table: `repro_bundle/data/training_2orientations.csv`
- item-level table: `repro_bundle/data/物品信息和dblf信息.csv`

These two tables serve distinct roles throughout the pipeline.

### 2.3 How the raw dispatch table is used

The dispatch table is the primary source for:

1. **sample definition**
   - one row corresponds to one `dispatch`;
   - the train/test split is defined at this level.

2. **supervision**
   - the target label `if_loaded` is read from this table.

3. **base40 aggregate features**
   - after deduplication and column dropping, the remaining aggregate columns
     form the fixed baseline representation used by SVM and teacher models.

4. **vehicle-level context**
   - fields such as `vehicle_length`, `vehicle_width`, `vehicle_height`, and
     `spare_capacity` come from this table and are later used as denominators,
     thresholds, and interaction anchors for candidate features.

Implementation references:

- `repro_bundle/code/HybridSVM/src/svm_train.py`
- `repro_bundle/code/HybridSVM/src/feature_search.py:374-394`

### 2.4 How the raw item table is used

The item table is not used as a separate prediction matrix. Its role is to
provide the fine-grained geometry required for feature engineering.

Specifically, it is used to derive:

- raw item dimensions: `item_length`, `item_width`, `item_height`
- sorted dimensions: `dim_s`, `dim_m`, `dim_l`
- item volume: `item_volume`
- shape proxy: `item_flatness`
- per-item attributes such as `if_fragile`

These rows are grouped by `dispatch_id` during feature generation, for example
to compute:

- top-k volume concentration;
- counts of long / wide / tall items;
- upper-tail quantiles;
- footprint pressure;
- interactions between dispatch-level slack and item-level geometric pressure.

Normalization reference:

- `repro_bundle/code/HybridSVM/src/feature_search.py:337-371`

### 2.5 What the raw data becomes inside the search loop

After loading, the raw inputs are transformed into two internal views:

1. **`agg_df`**
   - one row per `dispatch`;
   - contains aggregate columns and vehicle dimensions;
   - serves as the baseline feature frame and the dispatch-level side input for
     candidate code.

2. **`items_df`**
   - one row per item;
   - retains fine-grained item geometry;
   - serves as the item-level side input for candidate code.

This distinction is central:

- the LLM does **not** receive the full raw tables in the prompt;
- instead, the LLM writes feature code under the assumption that `agg_df` and
  `items_df` will be provided locally;
- the returned code is then executed on the **full local** `agg_df` and
  **full local** `items_df`.

The raw data therefore has two roles:

- it defines the fixed baseline representation;
- it remains the full execution substrate for every candidate feature proposal.

---

## 3. What Is Actually Passed to GLM

The current implementation does **not** send the full dispatch matrix or the
full item-level table to GLM.

Instead, GLM receives a text prompt assembled from:

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
- item-level schema names.

This means GLM sees:

- table structure;
- dataset scale;
- available fields.

It does **not** see:

- the full `10000` dispatch rows;
- the full `100317` item rows;
- raw per-case numeric examples.

### 3.2 Student-model summary

The prompt also contains a compact linear summary of the current baseline SVM:

- intercept;
- top coefficients by absolute magnitude;
- coefficient sign interpretation.

This is intended to tell GLM which base40 directions the linear model already
uses strongly, and where nonlinear recovery may still be required.

### 3.3 Teacher-model summary

The prompt includes the frozen XGB guidance file:

- `repro_bundle/guidance/XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

This file contributes the teacher-side evidence:

- metric gap between SVM and XGB on the fixed split;
- recovery counts (`SVM wrong, XGB right`, etc.);
- top-gain XGB features;
- recommended feature directions;
- discouraged directions.

### 3.4 Iteration memory

The prompt also contains the evolving search state:

- `policy.md`: hard constraints and preferred feature families;
- `memory.md`: summaries of accepted / rejected / failed trials;
- `active_feature_bank.md`: the currently accepted incremental features and
  their resulting metrics.

The search loop therefore asks GLM to reason from:

- data availability;
- current linear-model behavior;
- teacher-model evidence;
- cumulative search history.

---

## 4. Teacher Analysis

### 4.1 Teacher package

The teacher package for this run is:

- `repro_bundle/teacher_analysis/xgb_guidance_base40_20260514/`

Core files:

- `summary.json`
- `base40_gain_importance.csv`
- `xgb_vs_svm_recovery_contrast_base40.csv`

### 4.2 Fixed protocol

The teacher analysis uses:

- data: `repro_bundle/data/training_2orientations.csv`
- split: `test_size=0.25`, `random_state=42`
- student: linear SVM with `C=10.0`
- teacher scope: `base40` only

The XGB configuration comes from:

- `Ensemble_baseline/experiments/xgb_search_20260511_202834_base40/summary.json`

Best configuration:

- `n_estimators=700`
- `max_depth=8`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.9`
- `reg_lambda=0.5`
- `min_child_weight=2`
- `gamma=0.0`

### 4.3 Teacher findings

Base40 fixed-split performance:

| Model | Accuracy | AUC | TPR@FPR=1% |
|---|---:|---:|---:|
| linear SVM | `0.9276` | `0.9651` | `0.6347` |
| XGB | `0.9532` | `0.9868` | `0.8289` |

Recovery counts:

- `SVM wrong, XGB right = 92`
- `SVM right, XGB wrong = 28`

Most important XGB signals:

- `spare_capacity`
- `sku_average_volume`
- `wl_to_vehicle_wl_total`
- `wl_to_vehicle_wl_max`
- `sku_counts`
- `l_to_L_ratio_std`
- `sku_concentration`
- `h_to_H_ratio_max`

The practical implication for feature search is that GLM is encouraged to
translate tree-style nonlinear structure into explicit numeric features,
especially:

- slack-related nonlinearities;
- slack-pressure interactions;
- tail / threshold counts;
- local bottleneck structure;
- concentration and footprint pressure.

---

## 5. Validator Logic

### 5.1 Required response structure

Each GLM response must contain:

- `POLICY_UPDATE`
- `FEATURE_CODE`
- `RATIONALE`

`FEATURE_CODE` must define:

```python
build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame
```

### 5.2 Local execution

Candidate code is executed locally, not by the model itself. The execution path
is:

1. validate the returned code;
2. execute it in a restricted namespace;
3. call `build_candidate_features(agg_df.copy(), items_df.copy())`;
4. align the returned features to one row per `dispatch`;
5. concatenate:
   - base40 features;
   - active feature bank;
   - new candidate features;
6. retrain / evaluate linear SVM on the fixed split.

### 5.3 `accept / reject / fail`

A trial is marked as:

- **fail**
  - if the code block is missing;
  - if the code is invalid;
  - if the returned object is invalid;
  - if `dispatch_id` is missing or duplicated;
  - if feature names collide or are non-ASCII.

- **reject**
  - if the candidate executes correctly but does not improve beyond the
    current best accepted result under the acceptance rule.

- **accept**
  - if the candidate exceeds the current best accepted result under the
    acceptance rule.

Acceptance rule:

- accept if AUC improves by more than `0.0005`;
- otherwise require `TPR@FPR=1%` improvement larger than `0.005`;
- if still tied, require Accuracy improvement larger than `0.0005`.

### 5.4 Repair step

If validation fails, the pipeline issues one repair prompt containing:

- the original prompt;
- the original response;
- the error message.

If the repaired response still fails, the trial remains `fail`.

---

## 6. Run Outcome

### 6.1 Baseline

- Accuracy `0.9276`
- AUC `0.9651`
- TPR@FPR=1% `0.6347`

### 6.2 Summary

Total trials:

- `15` total
- `6` accepted
- `5` rejected
- `4` failed

Accepted iterations:

- `1`, `5`, `6`, `8`, `9`, `14`

Best accepted trial:

- iteration `14`
- features:
  - `q90_l_to_L`
  - `spare_cap_x_n_wide`
  - `height_tail_share`

Best accepted metrics:

- Accuracy `0.9372`
- AUC `0.9745`
- TPR@FPR=1% `0.7207`

Delta vs baseline:

- Accuracy `+0.0096`
- AUC `+0.0094`
- TPR@FPR=1% `+0.0860`

Two additional facts are important:

- the highest raw Accuracy occurred at iteration `12` (`0.9388`), but that
  trial was rejected because its low-FPR recall did not beat the current
  accepted best;
- the highest raw AUC occurred at iteration `15`, but that trial was also
  rejected for the same reason.

This confirms that the validator is not optimizing pure Accuracy or pure AUC in
isolation. It is explicitly protecting the low-FPR operating regime.

---

## 7. Per-Iteration Results

| Iter | Status | New features | Result |
|---|---|---|---|
| `1` | accept | `spare_cap_sq=spare_capacity^2`; `spare_cap_x_conc=spare_capacity*sku_concentration`; `spare_cap_x_wl_total=spare_capacity*wl_to_vehicle_wl_total`; `vol_top3_share=sum(top3 item_volume)/sum(item_volume)`; `n_high_asr_items=count(dim_l/dim_s>5)` | AUC `0.9681`, TPR@1% `0.6672`, ACC `0.9292` |
| `2` | fail | backend `429`; no executable candidate returned | fail |
| `3` | reject | `spare_cap_x_avg_vol=spare_capacity*sku_average_volume`; `wl_max_residual=wl_to_vehicle_wl_max-wl_to_vehicle_wl_total`; `n_long_items=count(dim_l>0.6*vehicle_length)` | AUC `0.9682`, TPR@1% `0.6608`, ACC `0.9292` |
| `4` | fail | repair response did not contain a valid `FEATURE_CODE` block | fail |
| `5` | accept | `spare_cap_log1p=log1p(max(spare_capacity,0))`; `h_to_H_max_sq=(h_to_H_ratio_max)^2`; `l_std_x_h_max=l_to_L_ratio_std*h_to_H_ratio_max`; `max_fp_ratio=max(dim_s*dim_m)/(vehicle_length*vehicle_width)` | AUC `0.9689`, TPR@1% `0.6593`, ACC `0.9292` |
| `6` | accept | `spare_cap_x_wl_max=spare_capacity*wl_to_vehicle_wl_max`; `wl_total_over_wl_max=wl_to_vehicle_wl_total/wl_to_vehicle_wl_max`; `max_dim_l_to_vL=max(dim_l)/vehicle_length` | AUC `0.9695`, TPR@1% `0.6647`, ACC `0.9292` |
| `7` | fail | backend `429`; no executable candidate returned | fail |
| `8` | accept | `spare_cap_x_sku_counts=spare_capacity*sku_counts`; `n_items_near_vL=count(dim_l>0.5*vehicle_length)`; `n_tall_items=count(item_height>0.5*vehicle_height)` | AUC `0.9730`, TPR@1% `0.7026`, ACC `0.9308` |
| `9` | accept | `n_wide_items=count(item_width>0.5*vehicle_width)`; `n_multi_dim_stress=count[(dim_l>0.4*L)+(dim_m>0.4*W)+(dim_s>0.4*H)>=2]`; `spare_cap_x_n_near_vL=spare_capacity*count(dim_l>0.5*vehicle_length)` | AUC `0.9736`, TPR@1% `0.7006`, ACC `0.9360` |
| `10` | reject | `n_fragile_items=sum(if_fragile)`; `flat_item_share=mean(item_flatness>3)`; `vol_top1_share=max(item_volume)/sum(item_volume)` | AUC `0.9737`, TPR@1% `0.6957`, ACC `0.9360` |
| `11` | reject | `spare_cap_x_h_to_H_max=spare_capacity*h_to_H_ratio_max`; `wl_total_x_h_to_H_max=wl_to_vehicle_wl_total*h_to_H_ratio_max`; `dim_s_avg_to_vmin=mean(dim_s/min(vehicle_length,vehicle_width,vehicle_height))` | AUC `0.9737`, TPR@1% `0.6957`, ACC `0.9348` |
| `12` | reject | `footprint_pressure=sum(dim_s*dim_m)/(vehicle_length*vehicle_width)`; `spare_cap_x_n_tall_items=spare_capacity*count(item_height>0.5*vehicle_height)`; `n_long_tall_items=count(dim_l>0.5*vehicle_length and item_height>0.5*vehicle_height)` | AUC `0.9741`, TPR@1% `0.6898`, ACC `0.9388` |
| `13` | fail | schema mismatch: the candidate referenced unavailable vehicle columns | fail |
| `14` | accept | `q90_l_to_L=Q90(dim_l/vehicle_length)`; `spare_cap_x_n_wide=spare_capacity*count(item_width>0.5*vehicle_width)`; `height_tail_share=mean(item_height>0.6*vehicle_height)` | AUC `0.9745`, TPR@1% `0.7207`, ACC `0.9372` |
| `15` | reject | `q90_w_to_W=Q90(item_width/vehicle_width)`; `spare_cap_x_height_tail=spare_capacity*mean(item_height/vehicle_height>0.5)`; `n_long_and_wide=count(dim_l>0.5*vehicle_length and dim_m>0.5*vehicle_width)` | AUC `0.9745`, TPR@1% `0.7193`, ACC `0.9376` |

---

## 8. Interpretation

This run supports four conclusions.

### 8.1 The no-seed route is viable

Iteration `001` was accepted immediately. Therefore, this route does not depend
on the earlier manually written six-feature seed bank merely to start moving.

### 8.2 The raw item table contributes materially

The accepted features are not simple rewrites of base40 aggregates. They make
substantial use of item-level structure:

- top-k volume concentration;
- long / tall / wide item counts;
- upper-tail quantiles;
- footprint-derived pressure measures.

This confirms that the fine-grained item table is not an auxiliary artifact. It
is the main source of geometric information beyond base40.

### 8.3 XGB guidance is directionally useful

The accepted features align with the teacher-side hypotheses:

- `spare_capacity` interactions;
- local threshold counts;
- tail pressure;
- bottleneck-style structure.

### 8.4 The current run is evidential, not canonical

Because the run exceeded its original budget through resume-on-same-directory,
its value is methodological rather than canonical:

- it demonstrates that pure GLM bootstrapping is feasible;
- it demonstrates that the no-seed route can improve the SVM materially;
- it does **not** replace a clean, fixed-budget formal rerun.

---

## 9. Key Files

Primary report:

- `research/hybridsvm_no_seed_exploratory_run_20260518/hybridsvm_xgb_no_seed_exploratory_report_20260518_en.md`

Companion run summary:

- `research/hybridsvm_no_seed_exploratory_run_20260518/README.md`

Original experiment directory:

- `trials_archive/exp_20260518_173257/`

Teacher analysis:

- `repro_bundle/teacher_analysis/xgb_guidance_base40_20260514/`

Frozen teacher guidance:

- `repro_bundle/guidance/XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

Core implementation:

- `code/HybridSVM/scripts/generate_guidance_with_llm.py`
- `code/HybridSVM/scripts/run_feature_search_agent.py`
- `repro_bundle/code/HybridSVM/src/feature_search.py`
- `repro_bundle/code/HybridSVM/src/svm_train.py`
- `repro_bundle/code/research/analyze_xgb_guidance_base40.py`

Recommended reading order:

1. this report
2. `README.md`
3. `summary.json`
4. `trials.csv`
5. `active_feature_bank.md`
6. `trials/iter_014/feature_candidate.py`
