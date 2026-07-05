# Ranker Window Statistical Summary

- Windows: 5
- Bootstrap iterations: 10000
- Bootstrap seed: 20260705
- Confidence intervals are descriptive window-bootstrap intervals, not formal iid statistical guarantees.

## Quality

- PF matched/improved/regressed: 4/1/0
- Mean PF delta: -0.0012442604
- Max PF regression: 0.0000000000
- Max PF improvement: -0.0062213021
- Mean PF delta bootstrap 95% CI: [-0.0037327813, 0.0000000000]

## Coverage

- Min exact coverage: 1.0000
- Min ranker+audit coverage: 1.0000
- Total exact uncovered: 0
- Total ranker+audit uncovered: 0

## Cost

| metric | aggregate reduction | bootstrap 95% CI | windows reduced/tied/increased | sign-test p |
| --- | ---: | ---: | ---: | ---: |
| validations | 49.04% | [42.80%, 57.05%] | 5/0/0 | 0.0625 |
| uncached_boxes | 20.77% | [16.62%, 24.44%] | 5/0/0 | 0.0625 |
| subprocess_seconds | 22.13% | [19.00%, 25.33%] | 5/0/0 | 0.0625 |
| elapsed_seconds | 25.01% | [21.68%, 28.86%] | 5/0/0 | 0.0625 |
