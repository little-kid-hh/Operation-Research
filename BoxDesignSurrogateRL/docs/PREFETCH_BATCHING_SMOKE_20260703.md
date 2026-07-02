# Candidate Status Prefetch Smoke, 2026-07-03

## Purpose

Validate that `--prefetch-candidate-statuses` only changes Java/Gurobi oracle
batching, not the exact local-search objective or accepted result.

## Setup

- Machine: Windows/Gurobi remote machine over Tailscale.
- Dataset: OR2023 dev split, first 20 orders.
- Initial boxes: `candidate_predictor_dev500_limit2500dev_20260703/staged_greedy/run_20260703_010436_419840/best_boxes.json`.
- Algorithm: `staged_greedy`.
- Schedule: `0.25:1`.
- K: 10.
- Oracle: Java MILP, `label_6ori`, 30s per MILP label time limit.
- Cache: fresh independent oracle cache per run.

## Result

| config | PF | coverage | validated candidates | uncached boxes | uncached batches | subprocess seconds | elapsed seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no prefetch | 2.4932110848 | 1.000 | 60 | 70 | 61 | 18.5724 | 21.2025 |
| prefetch | 2.4932110848 | 1.000 | 60 | 70 | 2 | 4.1580 | 5.5005 |

## Interpretation

Prefetch preserved the exact final PF and coverage while reducing Java/Gurobi
subprocess batches by 96.7%, subprocess time by 77.6%, and elapsed time by
74.1% on this smoke run. This is an oracle implementation optimization, not a
surrogate/ranker algorithmic claim. Future exact-vs-filtered comparisons should
either keep this switch fixed across all methods or report it as a separate
implementation-level ablation.

Run directories:

- `BoxDesignSurrogateRL/results/prefetch_smoke_20260703/staged_greedy/run_20260703_032740_118809`
- `BoxDesignSurrogateRL/results/prefetch_smoke_20260703/staged_greedy/run_20260703_032712_193116`
