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
| validations | 22.49% | [14.52%, 31.62%] | 20/0/0 | 0.0000 |
| uncached_boxes | 9.89% | [6.14%, 14.09%] | 19/1/0 | 0.0000 |
| subprocess_seconds | 10.56% | [6.63%, 14.99%] | 19/0/1 | 0.0000 |
| elapsed_seconds | 11.87% | [7.58%, 16.69%] | 20/0/0 | 0.0000 |
