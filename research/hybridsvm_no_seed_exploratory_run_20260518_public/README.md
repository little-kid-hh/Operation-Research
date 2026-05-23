# HybridSVM No-Seed Exploratory Run (2026-05-18)

This folder is a clean viewing package for the no-seed `glm-5.1`
feature-search run under XGB teacher guidance.

Source experiment directory:

- `trials_archive/exp_20260518_173257/`

## Status

This run is **exploratory**, not the formal 10-round reportable run.

Reason:

- it was started as a fresh no-seed run
- later it was resumed on the same `exp_dir`
- in `code/HybridSVM/scripts/run_feature_search_agent.py`, `--n-iters` under
  resume means "append this many more iterations", not "run to this total"
- so the directory continued from `iter_011` onward and finally accumulated
  `15` trials instead of the originally intended `10`

There is no running process now; this folder is a frozen record of the stopped
run.

## Protocol snapshot

Terminology:

- `dispatch` = raw-data `发车号`
- one `dispatch` = one concrete loading / departure job
- the aggregate table is one row per `dispatch`
- the item table has multiple rows under the same `dispatch`

Core inputs:

- aggregate data: `repro_bundle/data/training_2orientations.csv`
- item-level data: `repro_bundle/data/物品信息和dblf信息.csv`
- teacher guidance: `repro_bundle/guidance/XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md`

Key settings:

- `llm_model = glm-5.1`
- `seed_trial = false`
- `max_new_features = 6`
- `feature_bank_mode = cumulative_incremental`
- `svm_c = 10.0`
- `test_size = 0.25`
- `random_state = 42`

Acceptance rule:

- AUC margin: `0.0005`
- TPR@FPR=1% margin: `0.005`
- Accuracy margin: `0.0005`
- policy update only if accepted

Baseline fixed SVM:

- Accuracy: `0.9276`
- AUC: `0.9651`
- TPR@FPR=1%: `0.6347`

## Final outcome

Trial counts:

- `15` total
- `6` accepted
- `5` rejected
- `4` failed

Accepted iterations:

- `1`, `5`, `6`, `8`, `9`, `14`

Best accepted trial:

- iteration: `14`
- features:
  - `q90_l_to_L`
  - `spare_cap_x_n_wide`
  - `height_tail_share`

Best accepted metrics:

- Accuracy: `0.9372`
- AUC: `0.9745`
- TPR@FPR=1%: `0.7207`

Delta vs baseline:

- Accuracy `+0.0096`
- AUC `+0.0094`
- TPR@FPR=1% `+0.0860`

## Important interpretation

This run is still useful because it proves that the route can start without the
manual 6-feature seed bank:

- `iter_001` was already accepted
- `iter_001` reached Accuracy `0.9292`
- therefore pure GLM generation can bootstrap the feature bank by itself

But the run should not be quoted as the formal 10-round result, because the
resume operation changed the trial budget midstream.

## Key accepted trials

| Iter | Features | Accuracy | AUC | TPR@FPR=1% |
| --- | --- | ---: | ---: | ---: |
| `1` | `spare_cap_sq`, `spare_cap_x_conc`, `spare_cap_x_wl_total`, `vol_top3_share`, `n_high_asr_items` | `0.9292` | `0.9681` | `0.6672` |
| `5` | `spare_cap_log1p`, `h_to_H_max_sq`, `l_std_x_h_max`, `max_fp_ratio` | `0.9292` | `0.9689` | `0.6593` |
| `6` | `spare_cap_x_wl_max`, `wl_total_over_wl_max`, `max_dim_l_to_vL` | `0.9292` | `0.9695` | `0.6647` |
| `8` | `spare_cap_x_sku_counts`, `n_items_near_vL`, `n_tall_items` | `0.9308` | `0.9730` | `0.7026` |
| `9` | `n_wide_items`, `n_multi_dim_stress`, `spare_cap_x_n_near_vL` | `0.9360` | `0.9736` | `0.7006` |
| `14` | `q90_l_to_L`, `spare_cap_x_n_wide`, `height_tail_share` | `0.9372` | `0.9745` | `0.7207` |

## Rejected but informative trials

- `iter_012` had the highest raw Accuracy in this run: `0.9388`, but it was
  rejected because its `TPR@FPR=1%` (`0.6898`) was worse than the current best
  accepted trial under the acceptance rule.
- `iter_015` had the highest raw AUC in this run: `0.9745`, but it was also
  rejected because its `TPR@FPR=1%` (`0.7193`) did not beat the accepted best
  by the required margin.

## Failure records

The failed trials are preserved under `trials/`:

- `iter_002`, `iter_007`: backend `429` / deployment unavailable
- `iter_004`: repair response missing `FEATURE_CODE` block
- `iter_013`: schema mismatch referencing unavailable vehicle columns

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

Full per-trial artifacts:

- `trials/iter_001` ... `trials/iter_015`

## Reading order

If you only want the shortest path:

1. `README.md`
2. `summary.json`
3. `trials.csv`
4. `active_feature_bank.md`
5. `trials/iter_014/decision.json`
6. `trials/iter_014/feature_candidate.py`
