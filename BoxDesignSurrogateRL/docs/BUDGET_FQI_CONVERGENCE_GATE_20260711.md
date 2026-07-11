# Budget FQI Convergence Gate, 2026-07-11

## Question

Can a fitted-Q controller choose the number of ranker-prioritized candidates
to verify at each search state, match exact local-search quality, and reduce the
real Java/Gurobi cost beyond a fixed adaptive ranker?

This is a development gate, not a test-set claim.

## Data Isolation

- Candidate ranker: assignment-aware HGBT trained on the historical repaired
  300-order development traces (`7,680` candidate rows, `128` step groups).
- Budget-policy training episodes: OR2023 train split orders `[0,100)` and
  `[100,200)`, each with 250 complete exact states and 60 candidates per state.
- Budget-policy held-out episode and online gate: OR2023 train split orders
  `[200,300)`. The entire episode was excluded from FQI fitting.
- K: 10; step: 0.25; seed: 1; exact oracle: Java/Gurobi, `label_6ori`.
- All methods use the same orders and initial boxes. Coverage repair is
  deterministic `geometric_expand`.

The frozen FQI configuration was selected before the held-out episode was
used online: budgets `5,10,20,30,40,50,60`, gamma `0.99`, miss penalty `20`,
10 fitted-Q iterations. The shared state-action regressor includes normalized
budget as an action feature.

## Offline Gate

On the complete held-out `[200,300)` exact trace:

| policy | mean selected budget | exact-action preservation |
| --- | ---: | ---: |
| fixed Top-20 | 20.00 | 79.6% |
| fitted-Q budget | 21.00 | 80.0% |

This is effectively a tie and is not evidence of an online speedup. The
reported oracle-hindsight audit cost is diagnostic only: a miss cannot be
identified online without the exact labels.

## Online 250-Round Checkpoint

All methods reached the 250-round cap with 100% coverage, so these are anytime
checkpoints rather than convergence results.

| method | PF | validations | uncached boxes | subprocess s | wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact staged | 2.079471 | 15,000 | 1,374 | 562.09 | 786.93 |
| fixed ranker `10->30->full` | 2.107916 | 2,500 | 916 | 416.49 | 517.80 |
| fitted-Q budget + full fallback | 2.087026 | 5,780 | 1,045 | 515.59 | 695.13 |

The budget policy improves the fixed ranker's capped PF, but spends more
oracle work. It remains slightly worse than exact at the same iteration cap.

## Convergence Result

Each checkpoint was resumed from its own iteration-250 boxes with iteration
offset 250 and horizon 500. All three searches naturally stopped at total
iteration 355 after an exact no-improvement check. Metrics below sum the first
and continuation processes, including cache misses and startup cost in both
segments.

| method | terminal PF | coverage | validations | uncached boxes | subprocess s | wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact staged | 2.006760 | 100% | 21,300 | 1,963 | 840.91 | 1,219.42 |
| fixed ranker `10->30->full` | 2.006760 | 100% | 6,920 | 1,452 | 644.28 | 855.41 |
| fitted-Q budget + full fallback | 2.006760 | 100% | 10,760 | 1,566 | 742.16 | 1,032.25 |

Relative to exact, the budget policy preserves terminal PF and coverage while
reducing validations by 49.5%, uncached boxes by 20.2%, subprocess time by
11.7%, and wall time by 15.3%.

However, relative to the fixed ranker, the budget policy increases validations
by 55.5%, uncached boxes by 7.9%, subprocess time by 15.2%, and wall time by
20.7%, with identical terminal PF. It therefore fails the required ablation.

## Decision

Do not promote the current budget FQI as the main method and do not run it on
the untouched test set. The exact-baseline reduction is real, but the simpler
supervised ranker explains a stronger reduction without RL.

The next RL iteration should target candidate long-term value or search-path
length, where a sequential policy can add information beyond one-step ranker
ordering. Further tuning of the scalar budget miss penalty is not justified by
this gate.

## Artifacts

- Exact first segment:
  `results/budget_rl_train_exact_20260710/staged_greedy/run_20260710_211046_690364`
- Fixed/budget first segments:
  `results/budget_rl_online_dev_o200_20260710`
- Convergence continuations:
  `results/budget_rl_convergence_dev_o200_20260711`
- FQI artifact and metrics:
  `results/budget_fqi_train_o0_o100_dev_o200_20260710/run_20260710_212712`
- Code: `aac28f0` for the first online segments and `1527d68` for continuation
  iteration accounting.
