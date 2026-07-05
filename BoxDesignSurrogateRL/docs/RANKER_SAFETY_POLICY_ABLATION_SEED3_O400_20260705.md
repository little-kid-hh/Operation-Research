# Ranker Safety Policy Ablation: seed3:test[400,500), 2026-07-05

## Scope

This focused ablation tests whether adding the `all_expansions` ranker safety
set improves robustness or reduces exact-audit cost on the heaviest seed3
validation window.

- Dataset window: OR2023 test split `seed3:test[400,500)`.
- K: 10.
- Schedule: `0.25:1000`.
- Oracle: Java/Gurobi MILP with `label_6ori`.
- Coverage repair: `geometric_expand`.
- Ranker budget sequence: `20,30,40,50`.
- Ranker max elapsed seconds: `900`.
- Ranker artifact:
  `BoxDesignSurrogateRL/results/candidate_ranker_assignment_aware_20260705/candidate_ranker_20260705_084240/candidate_ranker.joblib`.
- Final quality is measured after exact staged-greedy audit from the ranker
  result.
- Exact baseline was reused from:
  `BoxDesignSurrogateRL/results/aa_s3_ranker900_o200_o400/protocol_20260705_132216`.

## Result

The `all_expansions` safety policy does not improve certified final PF on this
window. It reaches the same ranker+audit PF as the current main policy, but it
spends substantially more oracle budget and is slower than both the current
main ranker+audit configuration and the exact staged baseline on the main
runtime metrics.

| metric | exact staged | main safety `none` | ablation `all_expansions` |
| --- | ---: | ---: | ---: |
| ranker-only PF | n/a | 1.8342269911 | 1.9113500551 |
| ranker+audit PF | 1.8313420040 | 1.8251207019 | 1.8251207019 |
| PF delta vs exact | n/a | -0.0062213021 | -0.0062213021 |
| coverage | 1.0000 | 1.0000 | 1.0000 |
| uncovered orders | 0 | 0 | 0 |
| validations | 20760 | 12660 | 19841 |
| uncached MILP boxes | 1919 | 1626 | 1987 |
| Java/Gurobi subprocess seconds | 1047.5676 | 848.1555 | 1058.0048 |
| wall-clock seconds | 1409.2608 | 1131.4878 | 1469.6137 |
| ranker stop reason | n/a | `time_limit` | `time_limit` |

Compared with the current main safety policy `none`, `all_expansions` adds:

- 7181 additional MILP validations.
- 361 additional uncached MILP box queries.
- 209.8493 additional Java/Gurobi subprocess seconds.
- 338.1259 additional wall-clock seconds.

The raw frontier summary reports `ranker_safety_candidates=6510`, so the
safety set is very large on this window. The extra protected moves do not
reduce downstream exact-audit work enough to compensate.

## Interpretation

This ablation supports keeping `ranker-safety-policy=none` for the current main
method. On this heavy window, protecting all expansion moves is too blunt: it
turns the ranker stage into a much larger exact validation workload, while the
final exact audit still has to do substantial work.

Together with the wider-frontier ablation, this suggests that simply spending
more MILP budget is not the right improvement path. The next useful direction
is a targeted certification rule that protects a much smaller, evidence-based
subset of high-risk moves, or a stopping/audit policy that uses ranker trace
features to decide when exact audit will be cheap.

## Source Records

Remote result root:

- `BoxDesignSurrogateRL/results/aa_s3_o400_ablation_safety_all_expansions/manifest_frontier_20260705_155643`

Main comparison source:

- `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED3_CERTIFIED_SUMMARY_20260705.json`

