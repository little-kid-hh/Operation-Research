# Candidate Ranker Target Study, 2026-07-03

## Question

The dev500 query-budgeted frontier showed that the main weakness is not ranker
inference time. The weakness is low top-k exact-best capture: the query-only
ranker is fast, but it can miss local improvements and leave a PF gap that an
exact audit later has to repair.

This study tests whether changing the training target improves exact-best
capture and live query-budgeted search.

## Implemented Target Modes

`scripts/train_candidate_ranker.py` now supports:

- `--target-mode objective`: original scalarized exact objective regression.
- `--target-mode objective_gap`: within-step objective gap to the exact-best
  candidate.
- `--target-mode rank`: `log1p(candidate_rank - 1)`, directly emphasizing the
  step-level rank order.

The script also supports `--eval-trace-csv`, so a model can train on one trace
set and report metrics on a held-out trace set.

## Offline Trace Findings

On a same-trace dev500 split, `rank` looked much better than `objective`:

| Target | exact-best@5 | exact-best@10 | exact-best@30 | accepted@5 | accepted@10 | accepted@30 |
|---|---:|---:|---:|---:|---:|---:|
| objective | 0.0000 | 0.1667 | 0.3333 | 0.0000 | 0.2000 | 0.4000 |
| objective_gap | 0.0000 | 0.1667 | 0.3333 | 0.0000 | 0.2000 | 0.4000 |
| rank | 0.5000 | 0.5000 | 0.6667 | 0.4000 | 0.4000 | 0.6000 |

This did not hold under a stricter train/eval split. Training on dev300
repaired traces and evaluating on the dev500 0.25 trace gave:

| Target | Train traces | exact-best@5 | exact-best@10 | exact-best@30 | accepted@5 | accepted@10 | accepted@30 |
|---|---|---:|---:|---:|---:|---:|---:|
| objective | dev300 0.25 | 0.0000 | 0.1333 | 0.6000 | 0.0000 | 0.0714 | 0.5714 |
| objective_gap | dev300 0.25 | 0.0000 | 0.1333 | 0.6000 | 0.0000 | 0.0714 | 0.5714 |
| rank | dev300 0.25 | 0.0000 | 0.0000 | 0.0667 | 0.0000 | 0.0000 | 0.0000 |
| objective | dev300 0.5+0.25 | 0.0667 | 0.0667 | 0.4667 | 0.0000 | 0.0000 | 0.4286 |
| objective_gap | dev300 0.5+0.25 | 0.0667 | 0.0667 | 0.4667 | 0.0000 | 0.0000 | 0.4286 |
| rank | dev300 0.5+0.25 | 0.0000 | 0.2000 | 0.4000 | 0.0000 | 0.2143 | 0.3571 |

Interpretation: `rank` can improve some low-budget held-out trace metrics when
trained on the fuller dev300 trace set, but it is not uniformly better and is
unstable with less training data.

## Live MILP Query-Only Check

The best-looking held-out `rank` model from the full dev300 traces was then
tested in the live dev500 query-budgeted search without exact audit:

```text
C:\Users\Lenovo\Downloads\Operation-Research\BoxDesignSurrogateRL\results\query_budgeted_ranker_ranktarget_dev500_20260703\frontier_20260703_022633
```

| Budget | Model | PF | PF gap vs exact | Coverage | Uncached boxes | Wall-clock seconds |
|---|---|---:|---:|---:|---:|---:|
| top10 | original objective ranker | 2.3215656006 | +0.0152935171 | 1.0000 | 33 | 91.2342 |
| top10 | rank target | 2.3225014816 | +0.0162293980 | 1.0000 | 28 | 85.2698 |
| top10,30 | original objective ranker | 2.3174115903 | +0.0111395067 | 1.0000 | 63 | 136.4738 |
| top10,30 | rank target | 2.3206060859 | +0.0143340024 | 1.0000 | 54 | 125.1039 |

The rank-target model reduces uncached boxes slightly, but solution quality is
worse at both budgets. It should not replace the current objective-trained
ranker as the main method.

## Decision

Keep the new target modes and held-out evaluation support as diagnostics, but
do not promote `rank` or `objective_gap` as the next main algorithmic claim.
The next useful improvement is likely not a scalar target swap. We need either:

1. richer training data covering the query-budgeted trajectories that the ranker
   actually induces; or
2. a hybrid policy that uses the ranker for cheap early tiers but triggers an
   exact fallback selectively, based on an uncertainty or expected-gain rule,
   rather than a fixed no-op fallback.
