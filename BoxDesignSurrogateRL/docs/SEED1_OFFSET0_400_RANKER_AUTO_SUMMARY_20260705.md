# Seed1 Offset0-400 Ranker Auto Summary, 2026-07-05

Generated from committed manifest:

```text
python BoxDesignSurrogateRL/scripts/summarize_ranker_window_results.py \
  --manifest BoxDesignSurrogateRL/docs/SEED1_OFFSET0_400_RANKER_RESULT_MANIFEST_20260705.json \
  --total-label "seed1:test[0,500) window total"
```

| window | exact PF | ranker+audit PF | coverage | exact validations | ranker validations | exact uncached | ranker uncached | exact subprocess s | ranker subprocess s | exact elapsed s | ranker elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed1:test[0,100) | 1.8908695471 | 1.8892309108 | 1.0000 | 10620 | 8020 | 976 | 843 | 511.3663 | 443.1656 | 696.5627 | 601.0076 |
| seed1:test[100,200) | 2.2084188336 | 2.2084188336 | 1.0000 | 26280 | 24180 | 2421 | 2291 | 1307.7590 | 1242.7996 | 1814.9036 | 1718.0409 |
| seed1:test[200,300) | 2.0315201381 | 2.0315201381 | 1.0000 | 9420 | 7580 | 951 | 923 | 524.6430 | 506.5391 | 692.0440 | 647.2780 |
| seed1:test[300,400) | 1.8968685620 | 1.8968685620 | 1.0000 | 15540 | 13230 | 1411 | 1343 | 864.6081 | 808.4010 | 1144.2333 | 1074.5126 |
| seed1:test[400,500) | 1.9910783173 | 1.9910783173 | 1.0000 | 7920 | 6770 | 716 | 660 | 397.6499 | 363.2582 | 528.6209 | 483.5014 |
| seed1:test[0,500) window total | window-wise | window-wise | 1.0000 | 69780 | 59780 | 6475 | 6060 | 3606.0264 | 3364.1635 | 4876.3645 | 4524.3405 |

| window | PF delta | validation reduction | uncached-box reduction | subprocess reduction | wall-clock reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed1:test[0,100) | -0.0016386364 | 24.5% | 13.6% | 13.3% | 13.7% |
| seed1:test[100,200) | 0.0000000000 | 8.0% | 5.4% | 5.0% | 5.3% |
| seed1:test[200,300) | 0.0000000000 | 19.5% | 2.9% | 3.5% | 6.5% |
| seed1:test[300,400) | 0.0000000000 | 14.9% | 4.8% | 6.5% | 6.1% |
| seed1:test[400,500) | 0.0000000000 | 14.5% | 7.8% | 8.6% | 8.5% |
| seed1:test[0,500) window total | window-wise | 14.3% | 6.4% | 6.7% | 7.2% |
