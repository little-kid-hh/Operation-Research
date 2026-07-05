# Certified Ranker-Audit Claim Audit

- Verdict: FAIL
- Summary JSON: `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_COMBINED_SUMMARY_20260706.json`
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
| cost_validations | PASS | n=20, aggregate_reduction=22.49%, bootstrap95=[14.56%, 31.76%], reduced/tied/increased=20/0/0, sign_p=0.000002 |
| cost_uncached_boxes | PASS | n=20, aggregate_reduction=9.89%, bootstrap95=[6.13%, 14.15%], reduced/tied/increased=19/1/0, sign_p=0.000004 |
| cost_subprocess_seconds | FAIL | n=20, aggregate_reduction=10.56%, bootstrap95=[6.72%, 14.99%], reduced/tied/increased=19/0/1, sign_p=0.000040 |
| cost_elapsed_seconds | PASS | n=20, aggregate_reduction=11.87%, bootstrap95=[7.58%, 16.72%], reduced/tied/increased=20/0/0, sign_p=0.000002 |

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
| validations | 22.49% | [14.56%, 31.76%] | 20/0/0 | 0.000002 |
| uncached_boxes | 9.89% | [6.13%, 14.15%] | 19/1/0 | 0.000004 |
| subprocess_seconds | 10.56% | [6.72%, 14.99%] | 19/0/1 | 0.000040 |
| elapsed_seconds | 11.87% | [7.58%, 16.72%] | 20/0/0 | 0.000002 |
