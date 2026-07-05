# Ranker Window Statistical Summary

- Windows: 15
- Bootstrap iterations: 10000
- Bootstrap seed: 20260705
- Confidence intervals are descriptive window-bootstrap intervals, not formal iid statistical guarantees.

## Quality

- PF matched/improved/regressed: 14/1/0
- Mean PF delta: -0.0004147535
- Max PF regression: 0.0000000000
- Max PF improvement: -0.0062213021
- Mean PF delta bootstrap 95% CI: [-0.0012442604, 0.0000000000]

## Coverage

- Min exact coverage: 1.0000
- Min ranker+audit coverage: 1.0000
- Total exact uncovered: 0
- Total ranker+audit uncovered: 0

## Cost

| metric | aggregate reduction | bootstrap 95% CI | windows reduced/tied/increased | sign-test p |
| --- | ---: | ---: | ---: | ---: |
| validations | 26.02% | [16.31%, 37.13%] | 15/0/0 | 0.0001 |
| uncached_boxes | 11.71% | [7.30%, 16.79%] | 15/0/0 | 0.0001 |
| subprocess_seconds | 12.35% | [7.62%, 17.50%] | 15/0/0 | 0.0001 |
| elapsed_seconds | 13.93% | [8.86%, 19.62%] | 15/0/0 | 0.0001 |
