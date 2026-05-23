# HybridSVM RF -> GLM No-Seed Clean Run (2026-05-18)

This folder is a standalone package for the formal no-seed RF-guided run:

- route: `RF -> GLM -> linear SVM`
- experiment dir: `HybridSVM/experiments_feature_search/by_model/glm-5.1/exp_20260518_194131/`

This is a clean fixed-budget run:

- no manual seed trial
- `10` iterations exactly
- cumulative incremental feature bank
- frozen RF base40 teacher guidance

## Final outcome

Baseline fixed SVM:

- Accuracy: `0.9276`
- AUC: `0.9651`
- TPR@FPR=1%: `0.6347`

Final accepted best trial:

- iteration: `10`
- features:
  - `load_util_ratio`
  - `multi_dim_tight_share`
  - `dim_l_tail_ratio`

Best accepted metrics:

- Accuracy: `0.9312`
- AUC: `0.9733`
- TPR@FPR=1%: `0.6903`

Delta vs baseline:

- Accuracy `+0.0036`
- AUC `+0.0082`
- TPR@FPR=1% `+0.0556`

Trial counts:

- `10` total
- `6` accepted
- `4` rejected
- `0` failed

## Important interpretation

This run is the clean no-seed RF-guided result. It shows that:

- pure GLM generation can bootstrap the feature bank without manual seed features;
- RF teacher evidence can drive a stable 10-round search process;
- the accepted feature bank improved AUC and Accuracy materially;
- the best low-FPR recall trial by metric value is not the same as the final
  best accepted trial, because the acceptance rule is lexicographic on
  AUC first, then TPR@FPR=1%, then Accuracy.

In particular:

- iteration `6` reached the best TPR@FPR=1% among accepted trials: `0.7040`
- iteration `10` became the final best accepted trial because its AUC rose to
  `0.9733`, which exceeded the acceptance margin over the previous best

## What is included here

Top-level run artifacts:

- `summary.json`
- `trials.csv`
- `context.md`
- `tree_guidance.md`
- `policy.md`
- `memory.md`
- `active_feature_bank.csv`
- `active_feature_bank.md`
- `trials/iter_001` ... `trials/iter_010`

Human-readable reports:

- `hybridsvm_rf_no_seed_clean_report_20260518.md`
- `hybridsvm_rf_no_seed_clean_report_20260518_en.md`

Minimal repro bundle:

- `repro_bundle/README_REPRO.md`
- `repro_bundle/FILES_MAP.md`

## Reading order

If you want the shortest path:

1. `hybridsvm_rf_no_seed_clean_report_20260518.md`
2. `summary.json`
3. `trials.csv`
4. `active_feature_bank.md`
5. `trials/iter_010/feature_candidate.py`
6. `repro_bundle/README_REPRO.md`
