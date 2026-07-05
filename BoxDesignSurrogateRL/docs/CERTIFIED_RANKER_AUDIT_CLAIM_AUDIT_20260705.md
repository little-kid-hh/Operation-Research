# Certified Ranker-Audit Claim Audit

- Verdict: PASS
- Summary JSON: `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_CERTIFIED_SUMMARY_20260705.json`
- Windows: 15
- Bootstrap iterations: 10000
- Bootstrap seed: 20260705
- Bootstrap intervals are descriptive window resampling checks, not iid guarantees.

## Acceptance Checks

| check | result | detail |
| --- | ---: | --- |
| paired_window_count | PASS | windows=15, required>=15 |
| pf_complete | PASS | PF deltas=15, windows=15 |
| no_pf_regression_after_exact_audit | PASS | matched/improved/regressed=14/1/0, max_regression=0.0000000000, tolerance=1e-12 |
| exact_baseline_feasible | PASS | min_coverage=1.0000, uncovered=0 |
| ranker_audit_feasible | PASS | min_coverage=1.0000, uncovered=0 |
| cost_validations | PASS | n=15, aggregate_reduction=26.02%, bootstrap95=[16.31%, 37.13%], reduced/tied/increased=15/0/0, sign_p=0.000061 |
| cost_uncached_boxes | PASS | n=15, aggregate_reduction=11.71%, bootstrap95=[7.30%, 16.79%], reduced/tied/increased=15/0/0, sign_p=0.000061 |
| cost_subprocess_seconds | PASS | n=15, aggregate_reduction=12.35%, bootstrap95=[7.62%, 17.50%], reduced/tied/increased=15/0/0, sign_p=0.000061 |
| cost_elapsed_seconds | PASS | n=15, aggregate_reduction=13.93%, bootstrap95=[8.86%, 19.62%], reduced/tied/increased=15/0/0, sign_p=0.000061 |

## Quality And Coverage

- PF matched/improved/regressed: 14/1/0
- Mean PF delta: -0.0004147535
- Max PF regression: 0.0000000000
- Max PF improvement: -0.0062213021
- Min exact coverage: 1.0000
- Min ranker+audit coverage: 1.0000
- Total exact uncovered: 0
- Total ranker+audit uncovered: 0

## Cost Metrics

| metric | aggregate reduction | bootstrap 95% CI | reduced/tied/increased | sign-test p |
| --- | ---: | ---: | ---: | ---: |
| validations | 26.02% | [16.31%, 37.13%] | 15/0/0 | 0.000061 |
| uncached_boxes | 11.71% | [7.30%, 16.79%] | 15/0/0 | 0.000061 |
| subprocess_seconds | 12.35% | [7.62%, 17.50%] | 15/0/0 | 0.000061 |
| elapsed_seconds | 13.93% | [8.86%, 19.62%] | 15/0/0 | 0.000061 |
