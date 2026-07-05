# Ranker Window Statistical Summary

- Windows: 5
- Bootstrap iterations: 10000
- Bootstrap seed: 20260706
- Confidence intervals are descriptive window-bootstrap intervals, not formal iid statistical guarantees.

## Quality

- PF matched/improved/regressed: 5/0/0
- Mean PF delta: 0.0000000000
- Max PF regression: 0.0000000000
- Max PF improvement: 0.0000000000
- Mean PF delta bootstrap 95% CI: [0.0000000000, 0.0000000000]

## Coverage

- Min exact coverage: 1.0000
- Min ranker+audit coverage: 1.0000
- Total exact uncovered: 0
- Total ranker+audit uncovered: 0

## Cost

| metric | aggregate reduction | bootstrap 95% CI | windows reduced/tied/increased | sign-test p |
| --- | ---: | ---: | ---: | ---: |
| validations | 10.76% | [6.34%, 18.93%] | 5/0/0 | 0.0625 |
| uncached_boxes | 3.95% | [1.36%, 8.59%] | 4/1/0 | 0.1250 |
| subprocess_seconds | 4.90% | [1.42%, 10.13%] | 4/0/1 | 0.3750 |
| elapsed_seconds | 5.39% | [1.66%, 10.94%] | 5/0/0 | 0.0625 |
