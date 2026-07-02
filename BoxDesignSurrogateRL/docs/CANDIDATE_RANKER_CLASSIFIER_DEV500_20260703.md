# Candidate Classifier Ranker Dev500, 2026-07-03

## Purpose

Improve the query-budgeted ranker by training directly for low top-k capture of
the exact MILP local-search move. The previous pointwise regression targets
reduced candidate validations, but still missed too many exact accepted moves
at low top-k and needed an expensive exact audit.

## Method

The new target modes in `scripts/train_candidate_ranker.py` are:

- `exact_best_classifier`: positive label is the exact within-step best
  candidate.
- `accepted_classifier`: positive label is the exact best candidate only when
  it improves the current exact MILP score.

Runtime behavior is unchanged: the classifier only ranks candidates by
`-P(positive)`. The runner still evaluates selected candidates with the exact
Java/Gurobi MILP oracle and accepts a move only if exact PF/coverage improves
lexicographically.

## Data

Training traces:

- `BoxDesignSurrogateRL/results/candidate_traces/dev300_train_repaired_exact05_from_kmeans.csv`
- `BoxDesignSurrogateRL/results/candidate_traces/dev300_train_repaired_exact025_from_05.csv`

Held-out trace:

- `BoxDesignSurrogateRL/results/candidate_traces/dev500_limit2500dev_exact025_from_05_20260703.csv`

Live dev500 setup:

- Dataset: OR2023 dev split, first 500 orders.
- K: 10.
- Start point: exact converged `0.5` checkpoint.
- Fine stage: `0.25:1000`.
- Oracle: Java/Gurobi MILP, `label_6ori`, no candidate-status prefetch.

Exact baseline:

- `BoxDesignSurrogateRL/results/candidate_predictor_dev500_limit2500dev_20260703/staged_greedy/run_20260703_013535_387902`

## Offline Held-Out Trace Results

| model | target | accepted capture@10 | accepted capture@30 | accepted capture@50 | mean accepted PF gap@30 |
| --- | --- | ---: | ---: | ---: | ---: |
| hgbt | rank regression | 0.2143 | 0.3571 | n/a | n/a |
| hgbt | accepted classifier | 0.2857 | 0.5714 | 0.8571 | 0.0001556 |
| hgbt | exact-best classifier | 0.2143 | 0.5000 | 0.7143 | 0.0001952 |
| rf | accepted classifier | 0.3571 | 0.7143 | 1.0000 | 0.0000498 |

The RF accepted-move classifier was selected for live MILP evaluation:

```text
BoxDesignSurrogateRL/results/candidate_ranker_classifier_dev300_to_dev500_20260703/candidate_ranker_20260703_035111/candidate_ranker.joblib
```

## Live Query-Only Frontier

| budget sequence | PF | PF gap vs exact | coverage | MILP validations | uncached boxes | subprocess seconds | elapsed seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact staged 0.25 | 2.3062720835 | 0.0000000000 | 1.000 | 900 | 134 | 236.6537 | 259.8127 |
| top5 | 2.3228934482 | 0.0166213647 | 1.000 | 20 | 21 | 68.4431 | 69.8637 |
| top10 | 2.3216366946 | 0.0153646111 | 1.000 | 60 | 34 | 87.8394 | 90.6972 |
| top10,20 | 2.3198310282 | 0.0135589447 | 1.000 | 90 | 45 | 108.8753 | 112.7104 |
| top10,20,30 | 2.3114490614 | 0.0051769779 | 1.000 | 220 | 73 | 147.7901 | 155.7031 |
| top10,20,30,40 | 2.3086848736 | 0.0024127900 | 1.000 | 310 | 88 | 178.4662 | 188.7393 |
| top10,20,30,40,50 | 2.3062720835 | 0.0000000000 | 1.000 | 420 | 106 | 184.3803 | 196.8975 |

Run directories:

- Frontier:
  `BoxDesignSurrogateRL/results/query_budgeted_ranker_classifier_frontier_dev500_20260703/frontier_20260703_035556`
- Top50 query-only:
  `BoxDesignSurrogateRL/results/query_budgeted_ranker_classifier_dev500_20260703/ranker_filtered_greedy/run_20260703_035136_637863`

## Exact Audit Of Top50

An exact staged-greedy audit was run from the RF accepted-classifier top50 final
boxes while sharing the same oracle cache:

```text
BoxDesignSurrogateRL/results/query_budgeted_ranker_classifier_dev500_20260703/staged_greedy/run_20260703_035522_407221
```

The audit found no improvement and kept PF at `2.3062720835`. Incremental audit
cost was:

- 60 additional exact candidate validations;
- 8 additional uncached boxes;
- 12.1366 additional Java/Gurobi subprocess seconds;
- 13.8249 additional wall-clock seconds.

Combined query-only plus audit:

| metric | exact staged 0.25 | RF accepted top50 + exact audit |
| --- | ---: | ---: |
| PF | 2.3062720835 | 2.3062720835 |
| coverage | 1.000 | 1.000 |
| MILP validations | 900 | 480 |
| uncached boxes | 134 | 114 |
| subprocess seconds | 236.6537 | 196.5169 |
| elapsed seconds | 259.8127 | 210.7224 |

This is a 18.9% wall-clock reduction after exact audit, with the same PF and
coverage as the exact staged baseline. Query-only top50 already reaches the
exact baseline PF with a 24.2% wall-clock reduction.

## Interpretation

The classifier target is a materially better fit for the scientific claim than
standalone feasibility prediction or pointwise objective regression. It learns
the search decision that matters: which local-search candidates are worth
spending exact MILP queries on first. Because every accepted move is still
verified by MILP, the method avoids surrogate-only false improvements.

This result is currently dev500, seed 0, fine stage `0.25` only. It supports a
promising paper claim on query-efficient exact-MILP local search, but it still
needs replicated seeds and a larger/full OR2023 run before being treated as a
main result.
