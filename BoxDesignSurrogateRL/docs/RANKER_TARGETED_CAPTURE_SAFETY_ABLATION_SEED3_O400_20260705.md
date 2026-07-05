# Ranker Targeted Capture Safety Ablation: seed3:test[400,500), 2026-07-05

## Scope

This focused ablation tests whether a smaller, assignment-aware safety set can
improve on the current main ranker-audit policy for the heaviest seed3 window.

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

The tested policy is `ranker-safety-policy=targeted_expansion_capture`. It
protects expansion moves with positive assignment-capture value, using existing
assignment-aware candidate features, and caps the number of protected moves per
iteration with `--ranker-safety-max-candidates`.

## Result

Both targeted settings reach the same certified final PF as the current main
policy and preserve 100% coverage. Neither setting clearly improves the main
policy. The cap-1 variant is time-comparable but uses slightly more exact oracle
work. The cap-3 variant is uniformly worse than the main policy on cost metrics.

| metric | exact staged | main safety `none` | targeted cap=1 | targeted cap=3 | wide frontier `20,50,100,200` | safety `all_expansions` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ranker-only PF | n/a | 1.8342269911 | 1.8344934514 | 1.8348931419 | 1.8355592927 | 1.9113500551 |
| ranker+audit PF | 1.8313420040 | 1.8251207019 | 1.8251207019 | 1.8251207019 | 1.8251207019 | 1.8251207019 |
| PF delta vs exact | n/a | -0.0062213021 | -0.0062213021 | -0.0062213021 | -0.0062213021 | -0.0062213021 |
| coverage | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| uncovered orders | 0 | 0 | 0 | 0 | 0 | 0 |
| validations | 20760 | 12660 | 12754 | 13002 | 14070 | 19841 |
| uncached MILP boxes | 1919 | 1626 | 1627 | 1658 | 1654 | 1987 |
| Java/Gurobi subprocess seconds | 1047.5676 | 848.1555 | 843.2772 | 862.4527 | 869.9215 | 1058.0048 |
| wall-clock seconds | 1409.2608 | 1131.4878 | 1128.5435 | 1143.5167 | 1166.0600 | 1469.6137 |
| ranker safety candidates | n/a | 0 | 311 | 924 | 0 | 6510 |
| ranker stop reason | n/a | `time_limit` | `time_limit` | `time_limit` | `time_limit` | `time_limit` |

Compared with the current main safety policy `none`, targeted cap=1 changes:

- +94 validations.
- +1 uncached MILP box query.
- -4.8783 Java/Gurobi subprocess seconds.
- -2.9443 wall-clock seconds.

Compared with the current main safety policy `none`, targeted cap=3 changes:

- +342 validations.
- +32 uncached MILP box queries.
- +14.2972 Java/Gurobi subprocess seconds.
- +12.0289 wall-clock seconds.

## Interpretation

This ablation does not justify replacing the main `ranker-safety-policy=none`
configuration. The cap-1 result is essentially comparable to the main policy,
but it does not reduce the cleanest oracle-work metric: uncached MILP box
queries. The cap-3 result shows that protecting more targeted expansion moves
quickly becomes worse.

Together with the wider-frontier and `all_expansions` ablations, this narrows
the next research direction. The main method should remain certified
ranker-audit with safety `none`. The next iteration should focus on audit
stopping or trace-calibrated audit prediction rather than adding more protected
candidates during the ranker stage.

## Source Records

Remote result roots:

- cap=1:
  `BoxDesignSurrogateRL/results/aa_s3_o400_ablation_targeted_capture_cap1/manifest_frontier_20260705_165606`
- cap=3:
  `BoxDesignSurrogateRL/results/aa_s3_o400_ablation_targeted_capture/manifest_frontier_20260705_163613`

Main comparison sources:

- `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED3_CERTIFIED_SUMMARY_20260705.json`
- `BoxDesignSurrogateRL/docs/RANKER_FRONTIER_BUDGET_ABLATION_SEED3_O400_20260705.md`
- `BoxDesignSurrogateRL/docs/RANKER_SAFETY_POLICY_ABLATION_SEED3_O400_20260705.md`

