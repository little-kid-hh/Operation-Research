# Assignment-Aware Certified Ranker+Audit Seed1/Seed2/Seed3 Summary, 2026-07-05

Scope:

- Dataset: OR2023 test split, fifteen 100-order windows across seed1, seed2, and seed3 initial conditions.
- K: 10.
- Schedule: `0.25:1000`.
- Seed1/seed2 use the assignment-aware HGBT ranker+audit runs documented earlier.
- Seed3 uses the certified ranker+audit protocol with a 900 second ranker cap.
- Final quality is measured after exact staged-greedy audit from the ranker result.

## Window Table

| window | exact PF | ranker+audit PF | coverage | exact validations | ranker validations | exact uncached | ranker uncached | exact subprocess s | ranker subprocess s | exact elapsed s | ranker elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed1:test[0,100) | 1.8908695471 | 1.8908695471 | 1.0000 | 10620 | 8620 | 976 | 932 | 511.3663 | 492.2234 | 696.5627 | 646.8383 |
| seed1:test[100,200) | 2.2084188336 | 2.2084188336 | 1.0000 | 26280 | 25120 | 2421 | 2338 | 1307.7590 | 1269.9778 | 1814.9036 | 1742.7348 |
| seed1:test[200,300) | 2.0315201381 | 2.0315201381 | 1.0000 | 9420 | 8220 | 951 | 931 | 524.6430 | 515.5745 | 692.0440 | 671.0918 |
| seed1:test[300,400) | 1.8968685620 | 1.8968685620 | 1.0000 | 15540 | 13620 | 1411 | 1323 | 864.6081 | 799.2195 | 1144.2333 | 1038.2689 |
| seed1:test[400,500) | 1.9910783173 | 1.9910783173 | 1.0000 | 7920 | 4280 | 716 | 480 | 397.6499 | 265.5042 | 528.6209 | 349.4464 |
| seed2:test[0,100) | 1.9380009674 | 1.9380009674 | 1.0000 | 9840 | 7800 | 906 | 853 | 471.6141 | 442.3405 | 641.3276 | 585.9211 |
| seed2:test[100,200) | 2.2095968714 | 2.2095968714 | 1.0000 | 16200 | 15120 | 1579 | 1523 | 853.3800 | 824.3480 | 1155.2793 | 1088.9027 |
| seed2:test[200,300) | 1.9186301368 | 1.9186301368 | 1.0000 | 9480 | 8280 | 964 | 928 | 553.5630 | 529.5180 | 728.1239 | 694.4606 |
| seed2:test[300,400) | 1.9960143397 | 1.9960143397 | 1.0000 | 7860 | 5820 | 793 | 672 | 475.6722 | 407.0212 | 620.7365 | 530.2212 |
| seed2:test[400,500) | 1.9929555750 | 1.9929555750 | 1.0000 | 23160 | 21040 | 2063 | 1943 | 1043.8999 | 977.5555 | 1425.8324 | 1367.2486 |
| seed3:test[0,100) | 1.9780321464 | 1.9780321464 | 1.0000 | 8580 | 3730 | 804 | 685 | 438.0489 | 369.5694 | 600.8094 | 460.6840 |
| seed3:test[100,200) | 2.1883726050 | 2.1883726050 | 1.0000 | 19980 | 10460 | 1894 | 1442 | 1019.8061 | 793.9161 | 1374.9555 | 1035.2469 |
| seed3:test[200,300) | 1.9324569101 | 1.9324569101 | 1.0000 | 13800 | 6560 | 1316 | 982 | 753.5055 | 548.8925 | 992.4090 | 699.0928 |
| seed3:test[300,400) | 2.0360690769 | 2.0360690769 | 1.0000 | 11040 | 4380 | 1129 | 860 | 670.9197 | 499.6697 | 885.0929 | 619.9253 |
| seed3:test[400,500) | 1.8313420040 | 1.8251207019 | 1.0000 | 20760 | 12660 | 1919 | 1626 | 1047.5676 | 848.1555 | 1409.2608 | 1131.4878 |
| seed1+seed2+seed3 certified ranker+audit window total | window-wise | window-wise | 1.0000 | 210480 | 155710 | 19842 | 17518 | 10934.0034 | 9583.4858 | 14710.1918 | 12661.5712 |

| window | PF delta | validation reduction | uncached-box reduction | subprocess reduction | wall-clock reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed1:test[0,100) | 0.0000000000 | 18.8% | 4.5% | 3.7% | 7.1% |
| seed1:test[100,200) | 0.0000000000 | 4.4% | 3.4% | 2.9% | 4.0% |
| seed1:test[200,300) | 0.0000000000 | 12.7% | 2.1% | 1.7% | 3.0% |
| seed1:test[300,400) | 0.0000000000 | 12.4% | 6.2% | 7.6% | 9.3% |
| seed1:test[400,500) | 0.0000000000 | 46.0% | 33.0% | 33.2% | 33.9% |
| seed2:test[0,100) | 0.0000000000 | 20.7% | 5.8% | 6.2% | 8.6% |
| seed2:test[100,200) | 0.0000000000 | 6.7% | 3.5% | 3.4% | 5.7% |
| seed2:test[200,300) | 0.0000000000 | 12.7% | 3.7% | 4.3% | 4.6% |
| seed2:test[300,400) | 0.0000000000 | 26.0% | 15.3% | 14.4% | 14.6% |
| seed2:test[400,500) | 0.0000000000 | 9.2% | 5.8% | 6.4% | 4.1% |
| seed3:test[0,100) | 0.0000000000 | 56.5% | 14.8% | 15.6% | 23.3% |
| seed3:test[100,200) | 0.0000000000 | 47.6% | 23.9% | 22.2% | 24.7% |
| seed3:test[200,300) | 0.0000000000 | 52.5% | 25.4% | 27.2% | 29.6% |
| seed3:test[300,400) | 0.0000000000 | 60.3% | 23.8% | 25.5% | 30.0% |
| seed3:test[400,500) | -0.0062213021 | 39.0% | 15.3% | 19.0% | 19.7% |
| seed1+seed2+seed3 certified ranker+audit window total | window-wise | 26.0% | 11.7% | 12.4% | 13.9% |

## Interpretation

- Across 15 paired windows, certified ranker+audit has 14 matched windows, 1 improved window, and 0 regressions after exact audit.
- Coverage is 100% for exact and ranker+audit in all windows.
- Aggregate reductions: validations 26.0%, uncached boxes 11.7%, subprocess seconds 12.4%, wall-clock seconds 13.9%.
- This supports the conservative paper claim: learned candidate ordering plus exact audit preserves or improves exact staged quality while reducing MILP-oracle work under the current Java/Gurobi subprocess implementation.
