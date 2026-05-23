# HybridSVM Feature Search Workflow

This document defines the current reproducible workflow for using tree-model
knowledge to guide LLM-driven feature generation for the linear-SVM route.

It covers one narrow question:

> how do we take RF/XGB findings, freeze them into promptable guidance, let an
> LLM generate new item-level / dispatch-level features, and keep the whole
> search auditably reproducible?

## 1. Scope

This workflow is only for the `HybridSVM` meaning used in this repo:

- stage-1 model family stays `linear SVM`
- LLM is used for feature generation, not for direct classification
- generated features are additive on top of the original aggregate feature set
- item-level geometry is allowed as input during feature generation

This is distinct from:

- `Ensemble_baseline/`: standalone ensemble route
- older boundary-rule / evolution routes in `run_experiment.py`
- `TabTreeFormer/`

## 2. Core idea

The practical hypothesis is:

1. tree models are stronger because they use threshold, tail, and interaction
   structure that linear SVM cannot express directly
2. if we summarize those tree-model advantages explicitly, an LLM can propose
   interpretable engineered features that expose part of that nonlinear signal
   to a linear SVM
3. the generated feature code must be evaluated under a fixed split and a fixed
   SVM protocol, so the effect of feature engineering remains attributable

## 3. Frozen inputs per run

Every feature-search run should freeze the following inputs:

1. aggregate dispatch table
   - `FunSearch_test/training_2orientations.csv`
2. item-level table
   - `FunSearch_test/物品信息和dblf信息.csv`
3. split protocol
   - deduplicated dispatch-level rows
   - `test_size = 0.25`
   - `random_state = 42`
4. stage-1 classifier family
   - linear SVM only
5. tree-guidance note
   - `HybridSVM/TREE_INSPIRED_FEATURE_HYPOTHESES.md`

The tree-guidance note is now a formal run input. It is copied into each run as
`tree_guidance.md` and also embedded into `context.md`.

## 4. Where tree knowledge comes from

The tree guidance does not come from the LLM itself. It is distilled from local
tree-model analysis, mainly:

- standalone ensemble ablations under `Ensemble_baseline/`
- feature-importance summaries in `research/feature_figures_20260511/`
- same-split comparisons between linear SVM and tree baselines

Current guidance emphasizes:

- `spare_capacity`
- `wl_to_vehicle_wl_total`
- `wl_to_vehicle_wl_max`
- `sku_average_volume`
- dimension variance / upper-tail pressure
- thresholded local bottleneck counts
- slack x pressure interactions
- heterogeneity versus repetition

That distilled guidance lives in:

- `HybridSVM/TREE_INSPIRED_FEATURE_HYPOTHESES.md`

This file should be updated when the tree-analysis conclusion changes. It is the
canonical bridge from tree-model findings to LLM prompt design.

## 5. Prompt construction

The current feature-search prompt is assembled from four frozen blocks:

1. `context.md`
2. `policy.md`
3. `memory.md`
4. current active feature bank summary

### 5.1 `context.md`

Built by `src/feature_search.py::build_context_markdown(...)`.

It contains:

- data size and split metadata
- baseline SVM metrics
- aggregate feature schema
- item-level schema
- current SVM linear coefficient summary
- hardcoded tree-contrast notes
- the full `tree_guidance.md` snapshot

### 5.2 `policy.md`

Starts from `DEFAULT_FEATURE_POLICY` and is only updated by accepted trials.

It encodes:

- hard constraints
- preferred feature families
- tree-inspired search bias

### 5.3 `memory.md`

Built from accepted / rejected / failed trial records.

It tells the model:

- what has already worked
- what failed
- current best metric target to beat

### 5.4 Active feature bank summary

Built from previously accepted incremental features.

This is what makes the search cumulative rather than replacement-based.

The model is told explicitly:

- base aggregate features are always present
- accepted custom features remain active
- new trials should add incremental features on top of the current bank

## 6. Execution workflow

Main entry:

```bash
python HybridSVM/scripts/run_feature_search_agent.py \
  --llm-model glm-5.1 \
  --seed-trial \
  --n-iters 10 \
  --max-new-features 6 \
  --tree-guidance-path HybridSVM/TREE_INSPIRED_FEATURE_HYPOTHESES.md
```

Per iteration, the script does:

1. load frozen aggregate + item tables
2. build baseline SVM metrics on the fixed split
3. load `policy.md`, `memory.md`, active-bank state
4. build the generation prompt
5. call the LLM
6. extract `FEATURE_CODE`
7. run the candidate code on `agg_df` + `items_df`
8. append candidate features to:
   - original 40 aggregate features
   - current accepted active feature bank
9. train/evaluate linear SVM on the same split
10. run feature ablation
11. accept / reject / fail the trial
12. persist all run-level artifacts

## 7. Acceptance rule

The current search protocol is cumulative incremental search.

A trial is accepted only if it beats the current best accepted bank under:

1. `AUC`
2. then `TPR@FPR=1%`
3. then `Accuracy`

with configured margins:

- `auc_margin = 5e-4`
- `tpr_margin = 5e-3`
- `accuracy_margin = 5e-4`

This rule is recorded in:

- run `summary.json`
- per-trial `decision.json`

## 8. Artifact layout

Canonical run location:

`HybridSVM/experiments_feature_search/by_model/<llm_model>/exp_<timestamp>/`

Run-level files:

- `context.md`
- `tree_guidance.md`
- `policy.md`
- `memory.md`
- `active_feature_bank.md`
- `active_feature_bank.csv`
- `summary.json`
- `trials.csv`

Per-trial files:

- `prompt.md`
- `raw_response.md`
- optional `repair_prompt.md`
- optional `repair_raw_response.md`
- `feature_candidate.py`
- `candidate_features.csv`
- `metrics.json`
- `candidate_feature_coefficients.json`
- `candidate_feature_coefficients.md`
- `ablation.json`
- `decision.json`
- optional `failure.json`

## 9. Canonical current GLM run

The current main GLM cumulative run has been normalized into the repo here:

- `HybridSVM/experiments_feature_search/by_model/glm-5.1/exp_20260513_235323/`

This is the reference experiment for:

- the formal rerun under the explicit `tree_guidance.md` protocol
- accepted / rejected / failed trial sequence
- cumulative incremental protocol

Use this as the default bank source for follow-up manual incremental trials.

## 10. Manual follow-up trials

Hand-written trials should not pollute LLM history.

Use:

```bash
python HybridSVM/scripts/run_manual_feature_trial.py \
  --bank-exp-dir HybridSVM/experiments_feature_search/by_model/glm-5.1/exp_20260513_235323 \
  --candidate-name heterogeneity_v1 \
  --tree-guidance-path HybridSVM/TREE_INSPIRED_FEATURE_HYPOTHESES.md
```

These runs are stored separately under:

`HybridSVM/experiments_feature_search/manual_incremental/`

## 11. Reproducibility checklist

For a reportable rerun, check all of the following:

1. same aggregate CSV
2. same item-level CSV
3. same dedup protocol
4. same `test_size`
5. same `random_state`
6. same `svm_c`
7. same `tree_guidance.md`
8. same `policy.md` starting state
9. same accepted active feature bank if resuming incrementally
10. same LLM model and API configuration

If any of these changes, the run should be treated as a different experiment
family rather than a continuation of the old one.
