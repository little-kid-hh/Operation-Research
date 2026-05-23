# Ensemble Baseline

This directory is the standalone `ensemble` route. It is separate from
`HybridSVM/`, which is reserved for the LLM+SVM line.

## Scope

Current code and records here answer two questions:

1. does stacking improve over the linear SVM baseline?
2. if it does, where does the gain actually come from?

The ablation runner is:

- `Ensemble_baseline/run_ensemble_ablation.py`

Generated run records live under:

- `Ensemble_baseline/experiments/`

## Canonical setup

Data:

- aggregate table: `FunSearch_test/training_2orientations.csv`

Split:

- fixed holdout split
- `test_size=0.25`
- `random_state=42`

Base feature set:

- 40 aggregate features

Default base learners:

- `svm`
- `lr`
- `rf`
- `xgb` if available, otherwise sklearn `HistGradientBoostingClassifier` fallback

Default meta learner:

- `logreg`

Cross-validation for stacking:

- `cv_folds=5`

## What is already recorded

Root-level experiment folders currently include:

- `ablation_20260509_164504`: full ablation with `svm,lr,rf,xgb` and meta `{logreg, mlp}`
- `ablation_20260509_170103`: focused ablation with only `svm,lr`
- `ablation_20260509_170335`: clean full ablation with `svm,lr,rf,xgb` and meta `logreg`

Each run folder records:

- input data path
- train/test row counts
- feature names
- base-model and meta-model configuration
- model class names
- model parameters
- training-round hints
- single-model metrics
- stacking metrics
- CSV summaries and structured `summary.json`

## Main results

On the fixed split:

- single `svm`: AUC `0.9644`, TPR@1% `0.6057`, ACC `0.9276`
- single `lr`: AUC `0.9649`, TPR@1% `0.6283`, ACC `0.9220`
- single `rf`: AUC `0.9823`, TPR@1% `0.8058`, ACC `0.9436`
- single `xgb` fallback (`gbdt_fallback`): AUC `0.9845`, TPR@1% `0.8132`, ACC `0.9468`

Best stacking variants observed:

- `svm,rf,xgb | logreg`: TPR@1% `0.8274`
- `svm,lr,rf,xgb | logreg`: AUC `0.9842`, TPR@1% `0.8171`, ACC `0.9476`
- `rf,xgb | mlp`: AUC `0.9847`

Focused non-tree check:

- `svm+lr | logreg`: AUC `0.9648`, TPR@1% `0.6219`, ACC `0.9248`
- `svm+lr | mlp`: AUC `0.9647`, TPR@1% `0.6150`, ACC `0.9276`

## Conclusion

The ensemble improvement is real, but it does **not** come from “stacking more
linear classifiers together”.

The gain comes mainly from the tree models:

- `rf`
- `xgb` / `gbdt_fallback`

Why ensemble keeps helping here:

1. tree models capture thresholded and interaction-heavy structure that linear
   `svm` / `lr` cannot express directly;
2. stacking combines complementary error patterns through out-of-fold
   probabilities rather than naive averaging;
3. the linear meta learner is useful mainly because the base learners are
   heterogeneous, not because the meta learner itself is complex.

What the `svm+lr` ablation shows:

- removing `rf` and `xgb` collapses performance back to the linear-model range;
- non-tree classifiers alone do not assemble into a much stronger classifier on
  this feature space.

## Implication for the SVM+LLM route

This is the useful takeaway for `HybridSVM`, not that we should copy the
ensemble route forever.

If tree models are the source of the gain, then the SVM+LLM feature-engineering
route should try to manufacture the kinds of signals that trees exploit well,
such as:

- upper-tail and extreme-piece pressure
- slack / spare-capacity interactions
- local bottleneck counts
- heterogeneous-vs-repeated item structure
- thresholded “bad pattern” indicators

That is the bridge back to the linear SVM line: use LLM-guided feature
engineering to linearize some of the nonlinear structure currently captured by
the tree learners.
