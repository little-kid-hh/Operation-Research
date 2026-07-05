# Ranker Frontier Budget Ablation: seed3:test[400,500), 2026-07-05

## Scope

This focused ablation tests whether a wider ranker frontier improves the
certified ranker+audit result on the heaviest seed3 validation window.

- Dataset window: OR2023 test split `seed3:test[400,500)`.
- K: 10.
- Schedule: `0.25:1000`.
- Oracle: Java/Gurobi MILP with `label_6ori`.
- Coverage repair: `geometric_expand`.
- Ranker artifact:
  `BoxDesignSurrogateRL/results/candidate_ranker_assignment_aware_20260705/candidate_ranker_20260705_084240/candidate_ranker.joblib`.
- Final quality is measured after exact staged-greedy audit from the ranker
  result.
- Exact baseline was reused from:
  `BoxDesignSurrogateRL/results/aa_s3_ranker900_o200_o400/protocol_20260705_132216`.

## Result

The wider `20,50,100,200` frontier does not improve the certified final PF
relative to the current main `20,30,40,50` frontier. Both reach the same
ranker+audit PF, but the wider frontier costs more.

| metric | exact staged | main `20,30,40,50` | ablation `20,50,100,200` |
| --- | ---: | ---: | ---: |
| ranker-only PF | n/a | 1.8342269911 | 1.8355592927 |
| ranker+audit PF | 1.8313420040 | 1.8251207019 | 1.8251207019 |
| PF delta vs exact | n/a | -0.0062213021 | -0.0062213021 |
| coverage | 1.0000 | 1.0000 | 1.0000 |
| uncovered orders | 0 | 0 | 0 |
| validations | 20760 | 12660 | 14070 |
| uncached MILP boxes | 1919 | 1626 | 1654 |
| Java/Gurobi subprocess seconds | 1047.5676 | 848.1555 | 869.9215 |
| wall-clock seconds | 1409.2608 | 1131.4878 | 1166.0600 |
| ranker stop reason | n/a | `time_limit` | `time_limit` |

Compared with the current main frontier, `20,50,100,200` adds:

- 1410 additional MILP validations.
- 28 additional uncached MILP box queries.
- 21.7660 additional Java/Gurobi subprocess seconds.
- 34.5722 additional wall-clock seconds.

## Interpretation

This ablation supports keeping `20,30,40,50` as the current main frontier
policy. On this heavy window, widening the frontier spends more oracle budget
in the ranker stage but does not reduce audit work enough to compensate, nor
does it improve the certified final PF.

The useful next robustness ablation is not simply a larger frontier. A better
next test is `ranker-safety-policy=all_expansions`, because it changes which
candidate moves are protected rather than only spending more budget on the
same learned ranking.

## Source Records

Remote result root:

- `BoxDesignSurrogateRL/results/aa_s3_o400_ablation_budget_20_50_100_200/manifest_frontier_20260705_153342`

Main comparison source:

- `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED3_CERTIFIED_SUMMARY_20260705.json`

