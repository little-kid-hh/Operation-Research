# Candidate Status Prefetch Dev500 Check, 2026-07-03

## Purpose

Check whether the candidate-status prefetch optimization remains beneficial on
the 500-order dev fine stage used for the ranker frontier experiments.

## Setup

- Dataset: OR2023 dev split, first 500 orders.
- Initial boxes: exact converged `0.5` checkpoint from
  `candidate_predictor_dev500_limit2500dev_20260703/staged_greedy/run_20260703_010436_419840/best_boxes.json`.
- Algorithm: exact `staged_greedy`.
- Schedule: `0.25:1000`.
- K: 10.
- Oracle: Java MILP, `label_6ori`, 30s per MILP label time limit.
- Cache: fresh independent oracle cache per run.

## Result

| config | PF | coverage | validated candidates | uncached boxes | uncached batches | subprocess seconds | elapsed seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no prefetch | 2.3062720835 | 1.000 | 900 | 134 | 125 | 236.6537 | 259.8127 |
| prefetch | 2.3062720835 | 1.000 | 900 | 134 | 16 | 627.5705 | 644.2784 |

## Interpretation

The prefetch path preserved the exact final PF and coverage, but it made the
500-order exact run much slower: elapsed time increased by 148.0% even though
uncached subprocess batches fell by 87.2%. On this scale, fewer Java/Gurobi
batches is not sufficient; large batched labeler calls appear materially more
expensive than smaller incremental calls.

Therefore `--prefetch-candidate-statuses` should not be treated as the default
strong baseline for dev500. It remains an implementation-level ablation whose
effect is scale dependent. Main exact-vs-ranker claims should continue to
report `oracle_cache.uncached_boxes`, `oracle_cache.subprocess_seconds`, and
end-to-end elapsed time, and should keep the prefetch switch fixed within any
single comparison.

Run directories:

- no prefetch:
  `BoxDesignSurrogateRL/results/candidate_predictor_dev500_limit2500dev_20260703/staged_greedy/run_20260703_013535_387902`
- prefetch:
  `BoxDesignSurrogateRL/results/candidate_predictor_dev500_prefetch_20260703/staged_greedy/run_20260703_033033_802193`
