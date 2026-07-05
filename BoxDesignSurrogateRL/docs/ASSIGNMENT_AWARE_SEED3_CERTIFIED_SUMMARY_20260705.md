# Assignment-Aware Certified Ranker+Audit Seed3 Summary, 2026-07-05

Scope:

- Dataset: OR2023 test split, five 100-order windows, seed3 initial conditions.
- K: 10.
- Schedule: `0.25:1000`.
- Ranker budget sequence: `20,30,40,50`.
- Ranker max elapsed seconds: `900`.
- Safety policy: `none`.
- Final quality is measured after exact staged-greedy audit from the ranker result.

## Window Table

| window | exact PF | ranker+audit PF | coverage | exact validations | ranker validations | exact uncached | ranker uncached | exact subprocess s | ranker subprocess s | exact elapsed s | ranker elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed3:test[0,100) | 1.9780321464 | 1.9780321464 | 1.0000 | 8580 | 3730 | 804 | 685 | 438.0489 | 369.5694 | 600.8094 | 460.6840 |
| seed3:test[100,200) | 2.1883726050 | 2.1883726050 | 1.0000 | 19980 | 10460 | 1894 | 1442 | 1019.8061 | 793.9161 | 1374.9555 | 1035.2469 |
| seed3:test[200,300) | 1.9324569101 | 1.9324569101 | 1.0000 | 13800 | 6560 | 1316 | 982 | 753.5055 | 548.8925 | 992.4090 | 699.0928 |
| seed3:test[300,400) | 2.0360690769 | 2.0360690769 | 1.0000 | 11040 | 4380 | 1129 | 860 | 670.9197 | 499.6697 | 885.0929 | 619.9253 |
| seed3:test[400,500) | 1.8313420040 | 1.8251207019 | 1.0000 | 20760 | 12660 | 1919 | 1626 | 1047.5676 | 848.1555 | 1409.2608 | 1131.4878 |
| seed3 certified ranker+audit window total | window-wise | window-wise | 1.0000 | 74160 | 37790 | 7062 | 5595 | 3929.8478 | 3060.2033 | 5262.5276 | 3946.4368 |

| window | PF delta | validation reduction | uncached-box reduction | subprocess reduction | wall-clock reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed3:test[0,100) | 0.0000000000 | 56.5% | 14.8% | 15.6% | 23.3% |
| seed3:test[100,200) | 0.0000000000 | 47.6% | 23.9% | 22.2% | 24.7% |
| seed3:test[200,300) | 0.0000000000 | 52.5% | 25.4% | 27.2% | 29.6% |
| seed3:test[300,400) | 0.0000000000 | 60.3% | 23.8% | 25.5% | 30.0% |
| seed3:test[400,500) | -0.0062213021 | 39.0% | 15.3% | 19.0% | 19.7% |
| seed3 certified ranker+audit window total | window-wise | 49.0% | 20.8% | 22.1% | 25.0% |

## Interpretation

- Seed3 has 4 matched windows, 1 improved window, and 0 regressions after exact audit.
- Coverage is 100% for exact and ranker+audit in all five windows.
- Aggregate reductions: validations 49.0%, uncached boxes 20.8%, subprocess seconds 22.1%, wall-clock seconds 25.0%.
- Ranker-only remains an ablation/diagnostic; the main method is the certified ranker+audit pipeline.
