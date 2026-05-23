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

Search stance:
- rely only on the baseline SVM behavior, raw aggregate columns, item-level
  geometry, and general packing-domain reasoning
- do not assume access to any teacher model, tree-model contrast, or external
  guidance note

## Iteration 5
- Add dimensional pressure features capturing stacking difficulty and floor competition
- Focus on sum-based pressure signals rather than percentile/average ratios already in the bank
- height_layer_pressure captures vertical stacking demand via smallest dimensions
- footprint_sum_ratio captures floor-area bottleneck using sorted dims (best-case orientation)
- length_sum_pressure captures total length competition as a count×avg interaction

## Iteration 6
- Add non-linear interaction features that the linear SVM cannot capture on its own
- Focus on the multiplicative pressure interaction between vertical stacking and floor coverage
- Introduce a worst-case single-item bottleneck against the tighter vehicle floor dimension
- Encode the compounding difficulty of high item count combined with high volume fill
