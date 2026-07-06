# Ranker Window Statistical Summary

- Windows: 20
- Bootstrap iterations: 10000
- Bootstrap seed: 20260706
- Confidence intervals are descriptive window-bootstrap intervals, not formal iid statistical guarantees.

## Quality

- PF matched/improved/regressed: 19/1/0
- Mean PF delta: -0.0003110651
- Max PF regression: 0.0000000000
- Max PF improvement: -0.0062213021
- Mean PF delta bootstrap 95% CI: [-0.0009331953, 0.0000000000]

## Coverage

- Min exact coverage: 1.0000
- Min ranker+audit coverage: 1.0000
- Total exact uncovered: 0
- Total ranker+audit uncovered: 0

## Cost

| metric | aggregate reduction | bootstrap 95% CI | windows reduced/tied/increased | sign-test p |
| --- | ---: | ---: | ---: | ---: |
| validations | 11.12% | [7.84%, 15.50%] | 19/0/1 | 0.0000 |
| uncached_boxes | 4.84% | [2.93%, 7.42%] | 18/1/1 | 0.0001 |
| subprocess_seconds | 5.57% | [3.75%, 8.17%] | 19/0/1 | 0.0000 |
| elapsed_seconds | 5.87% | [3.65%, 8.78%] | 19/0/1 | 0.0000 |
