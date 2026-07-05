# Ranker Time-Budget Ablation: seed3:test[400,500), 2026-07-05

## Scope

This focused ablation tests whether the main 900-second ranker cap is longer
than needed on the heaviest seed3 window. The experiment keeps the exact staged
audit, so final quality is still certified by the same Java/Gurobi MILP oracle
and local-search objective as the exact baseline.

- Dataset window: OR2023 test split `seed3:test[400,500)`.
- K: 10.
- Schedule: `0.25:1000`.
- Oracle: Java/Gurobi MILP with `label_6ori`.
- Coverage repair: `geometric_expand`.
- Ranker budget sequence: `20,30,40,50`.
- Ranker safety policy: `none`.
- Ranker max elapsed seconds tested: `300`, `600`, `900`.
- Final quality is measured after exact staged-greedy audit from the ranker
  result.
- Exact baseline was reused from:
  `BoxDesignSurrogateRL/results/aa_s3_ranker900_o200_o400/protocol_20260705_132216`.

## Result

All tested ranker time caps recover the same certified final PF after exact
audit and preserve 100% coverage. Shorter ranker budgets, however, leave the
ranker-only solution farther from the audited solution, which shifts work into
the exact audit. The 600-second cap is slightly faster in wall-clock time but
uses more validations, more uncached MILP box queries, and slightly more
Java/Gurobi subprocess time than the 900-second main policy. The 300-second cap
is worse than the 900-second main policy on every main cost metric.

| metric | exact staged | ranker cap 300 | ranker cap 600 | main ranker cap 900 |
| --- | ---: | ---: | ---: | ---: |
| ranker-only PF | n/a | 1.9814697311 | 1.9071373816 | 1.8342269911 |
| ranker+audit PF | 1.8313420040 | 1.8251207019 | 1.8251207019 | 1.8251207019 |
| PF delta vs exact | n/a | -0.0062213021 | -0.0062213021 | -0.0062213021 |
| coverage | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| uncovered orders | 0 | 0 | 0 | 0 |
| validations | 20760 | 18400 | 13980 | 12660 |
| uncached MILP boxes | 1919 | 1807 | 1639 | 1626 |
| Java/Gurobi subprocess seconds | 1047.5676 | 947.4663 | 849.1336 | 848.1555 |
| wall-clock seconds | 1409.2608 | 1311.3109 | 1119.0861 | 1131.4878 |
| ranker stop reason | n/a | `time_limit` | `time_limit` | `time_limit` |

Compared with the current main 900-second ranker cap, the 600-second cap
changes:

- +1320 validations.
- +13 uncached MILP box queries.
- +0.9781 Java/Gurobi subprocess seconds.
- -12.4017 wall-clock seconds.

Compared with the current main 900-second ranker cap, the 300-second cap
changes:

- +5740 validations.
- +181 uncached MILP box queries.
- +99.3108 Java/Gurobi subprocess seconds.
- +179.8231 wall-clock seconds.

## Interpretation

This ablation does not justify replacing the 900-second cap as the main policy
on this heavy window. The 600-second cap is a wall-clock-only tradeoff, not a
clean oracle-work improvement. Because the paper's primary systems metrics
include uncached MILP boxes and subprocess seconds, the 900-second policy
remains the stronger default for this window.

The 300-second result explains the mechanism: stopping the ranker too early
leaves the exact audit with many more local-search candidates to validate. In
other words, the learned ranker is useful not only because it avoids candidate
queries during its own phase, but also because enough ranker progress can make
the subsequent exact audit cheaper.

The next useful direction is not a fixed shorter ranker cap. It is an adaptive
ranker-audit controller that estimates whether another ranker iteration is
likely to reduce exact audit cost enough to justify its own oracle work. That
controller should still be evaluated with final exact audit before making any
quality claim.

## Source Records

Remote result roots:

- cap=300:
  `BoxDesignSurrogateRL/results/aa_s3_o400_ranker_budget300/manifest_frontier_20260705_174127`
- cap=600:
  `BoxDesignSurrogateRL/results/aa_s3_o400_ranker_budget600/manifest_frontier_20260705_172212`
- cap=900 main:
  `BoxDesignSurrogateRL/results/aa_s3_ranker900_o200_o400/protocol_20260705_132216`

Main comparison source:

- `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED3_CERTIFIED_SUMMARY_20260705.json`

