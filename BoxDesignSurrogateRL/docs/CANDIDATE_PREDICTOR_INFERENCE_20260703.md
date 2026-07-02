# Candidate Predictor Inference Compression, 2026-07-03

## Question

Can we reduce candidate predictor inference time without changing the
MILP-verified acceptance rule?

## Implementation Notes

- `scripts/train_candidate_ranker.py` now exposes:
  - `--hgbt-max-iter`
  - `--hgbt-learning-rate`
- `CandidateRanker` has an optional numpy fast path for compatible sklearn
  pipelines, but it is disabled for `HistGradientBoostingRegressor`.

The reason is empirical: for the current HGBT ranker, preprocessing is not the
bottleneck. On a 60-candidate batch, `fast_transform` was about `0.225 ms`,
while sklearn preprocessing was about `1.766 ms`; HGBT `model.predict` was
about `48-64 ms`. Bypassing `ColumnTransformer` did not consistently reduce
end-to-end HGBT inference.

## Artifacts

Original 300-iteration HGBT:

```text
BoxDesignSurrogateRL\results\candidate_predictors_dev300_repaired_20260702\candidate_ranker_20260702_230711\candidate_ranker.joblib
```

Lightweight HGBT candidates:

```text
BoxDesignSurrogateRL\results\candidate_predictors_dev300_repaired_fast_20260702\candidate_ranker_20260702_235112\candidate_ranker.joblib
BoxDesignSurrogateRL\results\candidate_predictors_dev300_repaired_fast_20260702\candidate_ranker_20260702_235131\candidate_ranker.joblib
```

## Predictor-Only Timing

Benchmark setup:

- Windows/Gurobi experiment machine
- 60 candidates per call
- 500 repeated `CandidateRanker.predict_scores` calls
- Same candidate trace batch:
  `results\candidate_traces\dev300_train_repaired_exact05_from_kmeans.csv`

| Model | HGBT iter | ms / 60-candidate call | Speedup vs 300 iter |
|---|---:|---:|---:|
| Original HGBT | 300 | 55.826973 | 1.000x |
| Lightweight HGBT | 50 | 24.311524 | 2.296x |
| Lightweight HGBT | 100 | 34.776653 | 1.605x |

## Held-Out Step-Group Quality

Training data is the repaired dev300 train split only:

```text
results\candidate_traces\dev300_train_repaired_exact05_from_kmeans.csv
results\candidate_traces\dev300_train_repaired_exact025_from_05.csv
```

| Model | Exact-best capture@30 | Mean PF gap@30 | Interpretation |
|---|---:|---:|---|
| Original HGBT, 300 iter | 0.50000 | 0.0002525 | Reference predictor |
| HGBT, 50 iter | 0.25000 | 0.0010405 | Too much quality loss |
| HGBT, 100 iter | 0.53125 | 0.0003507 | Usable speed/quality tradeoff candidate |

## MILP-Verified Dev100 Validation

Both validation runs use:

- `--ranker-adaptive-top-k 10,30`
- `--ranker-noop-fallback`
- exact MILP final acceptance
- same dev100 checkpoints as the original candidate predictor validation

100-iteration HGBT validation artifacts:

```text
BoxDesignSurrogateRL\results\candidate_predictor_oos_dev100_fast_20260702\ranker_filtered_greedy\run_20260703_003029_555127
BoxDesignSurrogateRL\results\candidate_predictor_oos_dev100_fast_20260702\ranker_filtered_greedy\run_20260703_003055_202617
```

| Stage | Exact PF | 100-iter predictor PF | Coverage | MILP validated | Avoidance | Ranker seconds |
|---|---:|---:|---:|---:|---:|---:|
| dev100 0.25 | 1.7467902175053618 | 1.7467902175053618 | 1.0 | 420 / 720 | 41.6667% | 0.265859 |
| dev100 0.1 | 1.739963658999321 | 1.739963658999321 | 1.0 | 790 / 960 | 17.7083% | 0.311097 |

Reference 300-iteration predictor validation:

| Stage | Predictor PF | MILP validated | Avoidance | Ranker seconds |
|---|---:|---:|---:|---:|
| dev100 0.25 | 1.7467902175053618 | 350 / 720 | 51.3889% | 0.523547 |
| dev100 0.1 | 1.739963658999321 | 690 / 960 | 28.1250% | 0.546394 |

## Conclusion

Yes, predictor inference can be compressed, but the clean lever is model size,
not pandas/sklearn preprocessing. The 100-iteration HGBT reduces predictor-only
inference time by about `1.6x` and reduces measured ranker time in the dev100
validation by about `1.8-2.0x`, while preserving exact-MILP PF and coverage on
these two dev100 stages.

The tradeoff is that the lighter model validates more candidates with MILP:
`350 -> 420` on 0.25 and `690 -> 790` on 0.1. For the paper, this should be
reported as a predictor-inference speed/quality tradeoff, not yet as a stronger
end-to-end speed claim.
