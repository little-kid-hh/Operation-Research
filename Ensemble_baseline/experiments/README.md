# Ensemble Experiment Artifacts

This directory stores generated experiment outputs for the standalone
`ensemble` route.

## Policy

- Generated run folders are ignored by git by default.
- This `README.md` may be tracked.
- Source code for the current ablation runner lives in:
  - `Ensemble_baseline/run_ensemble_ablation.py`

## Purpose

This directory is meant for:

1. base-learner single-model baselines;
2. stacking subset ablations;
3. meta-combiner comparisons;
4. focused checks such as "remove RF/XGB and test only non-tree classifiers".
