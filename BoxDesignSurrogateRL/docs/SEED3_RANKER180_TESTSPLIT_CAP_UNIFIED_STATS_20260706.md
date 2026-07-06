# Ranker Window Statistical Summary

- Windows: 5
- Bootstrap iterations: 10000
- Bootstrap seed: 20260706
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
| validations | 7.07% | [2.04%, 13.84%] | 4/0/1 | 0.3750 |
| uncached_boxes | 2.22% | [-0.43%, 4.31%] | 4/0/1 | 0.3750 |
| subprocess_seconds | 3.85% | [1.93%, 5.96%] | 5/0/0 | 0.0625 |
| elapsed_seconds | 2.92% | [-0.61%, 7.16%] | 4/0/1 | 0.3750 |
