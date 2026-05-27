# Loadability Model Pack (2026-05-26, v2)

Included models:

- `svm` (`models/svm_base40/`)
- `xgb` (`models/xgb_base40/`)
- `rf` (`models/rf_base40/`)
- `lg` (`models/lg_base40/`)
- feature-engineered `svm` (`models/svm_feature_engineered/`)

## How to use

Import `load_model_examples.py` and call:

- `load_svm_base40()`
- `load_svm_feature_engineered()`
- `load_lr_base40()`
- `load_rf_base40()`
- `load_xgb_base40()`

## Feature-engineered SVM

This model was trained with the cumulative active bank from the formal no-seed
GLM run.

To rebuild that bank, use:

- `feature_engineering/build_active_bank.py`
- `feature_engineering/accepted_trials/`

## Reuse note

The saved models expect the same feature columns used at training time.
