# Certified Ranker-Audit Claim Audit

- Verdict: PASS
- Summary JSON: `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_SUMMARY_20260706.json`
- Windows: 20
- Bootstrap iterations: 10000
- Bootstrap seed: 20260705
- Bootstrap intervals are descriptive window resampling checks, not iid guarantees.

## Acceptance Checks

| check | result | detail |
| --- | ---: | --- |
| paired_window_count | PASS | windows=20, required>=15 |
| pf_complete | PASS | PF deltas=20, windows=20 |
| no_pf_regression_after_exact_audit | PASS | matched/improved/regressed=19/1/0, max_regression=0.0000000000, tolerance=1e-12 |
| exact_baseline_feasible | PASS | min_coverage=1.0000, uncovered=0 |
| ranker_audit_feasible | PASS | min_coverage=1.0000, uncovered=0 |
| cost_validations | PASS | n=20, aggregate_reduction=11.12%, bootstrap95=[7.88%, 15.47%], reduced/tied/increased=19/0/1, sign_p=0.000040 |
| cost_uncached_boxes | PASS | n=20, aggregate_reduction=4.84%, bootstrap95=[2.97%, 7.40%], reduced/tied/increased=18/1/1, sign_p=0.000076 |
| cost_subprocess_seconds | PASS | n=20, aggregate_reduction=5.57%, bootstrap95=[3.73%, 8.18%], reduced/tied/increased=19/0/1, sign_p=0.000040 |
| cost_elapsed_seconds | PASS | n=20, aggregate_reduction=5.87%, bootstrap95=[3.62%, 8.69%], reduced/tied/increased=19/0/1, sign_p=0.000040 |

## Quality And Coverage

- PF matched/improved/regressed: 19/1/0
- Mean PF delta: -0.0003110651
- Max PF regression: 0.0000000000
- Max PF improvement: -0.0062213021
- Min exact coverage: 1.0000
- Min ranker+audit coverage: 1.0000
- Total exact uncovered: 0
- Total ranker+audit uncovered: 0

## Cost Metrics

| metric | aggregate reduction | bootstrap 95% CI | reduced/tied/increased | sign-test p |
| --- | ---: | ---: | ---: | ---: |
| validations | 11.12% | [7.88%, 15.47%] | 19/0/1 | 0.000040 |
| uncached_boxes | 4.84% | [2.97%, 7.40%] | 18/1/1 | 0.000076 |
| subprocess_seconds | 5.57% | [3.73%, 8.18%] | 19/0/1 | 0.000040 |
| elapsed_seconds | 5.87% | [3.62%, 8.69%] | 19/0/1 | 0.000040 |
