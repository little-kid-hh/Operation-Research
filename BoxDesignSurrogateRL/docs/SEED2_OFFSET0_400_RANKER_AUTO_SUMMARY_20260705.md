# Seed2 Offset0-400 Ranker Auto Summary, 2026-07-05

Generated from committed manifest:

```text
python BoxDesignSurrogateRL/scripts/summarize_ranker_window_results.py \
  --manifest BoxDesignSurrogateRL/docs/SEED2_OFFSET0_400_RANKER_RESULT_MANIFEST_20260705.json \
  --total-label "seed2:test[0,500) window total"
```

| window | exact PF | ranker+audit PF | coverage | exact validations | ranker validations | exact uncached | ranker uncached | exact subprocess s | ranker subprocess s | exact elapsed s | ranker elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed2:test[0,100) | 1.9380009674 | 1.9380009674 | 1.0000 | 9840 | 6980 | 906 | 750 | 471.6141 | 390.3809 | 641.3276 | 534.3316 |
| seed2:test[100,200) | 2.2095968714 | 2.2095968714 | 1.0000 | 16200 | 14830 | 1579 | 1548 | 853.3800 | 830.7895 | 1155.2793 | 1092.2495 |
| seed2:test[200,300) | 1.9186301368 | 1.9186301368 | 1.0000 | 9480 | 7110 | 964 | 882 | 553.5630 | 495.5806 | 728.1239 | 644.5394 |
| seed2:test[300,400) | 1.9960143397 | 1.9726992574 | 1.0000 | 7860 | 7370 | 793 | 794 | 475.6722 | 471.3637 | 620.7365 | 613.5968 |
| seed2:test[400,500) | 1.9929555750 | 2.0025546412 | 1.0000 | 23160 | 21510 | 2063 | 2016 | 1043.8999 | 1012.4072 | 1425.8324 | 1432.5585 |
| seeds=2 window total | window-wise | window-wise | 1.0000 | 66540 | 57800 | 6305 | 5990 | 3398.1293 | 3200.5220 | 4571.2997 | 4317.2759 |

| window | PF delta | validation reduction | uncached-box reduction | subprocess reduction | wall-clock reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed2:test[0,100) | 0.0000000000 | 29.1% | 17.2% | 17.2% | 16.7% |
| seed2:test[100,200) | 0.0000000000 | 8.5% | 2.0% | 2.6% | 5.5% |
| seed2:test[200,300) | 0.0000000000 | 25.0% | 8.5% | 10.5% | 11.5% |
| seed2:test[300,400) | -0.0233150822 | 6.2% | -0.1% | 0.9% | 1.2% |
| seed2:test[400,500) | 0.0095990662 | 7.1% | 2.3% | 3.0% | -0.5% |
| seeds=2 window total | window-wise | 13.1% | 5.0% | 5.8% | 5.6% |
