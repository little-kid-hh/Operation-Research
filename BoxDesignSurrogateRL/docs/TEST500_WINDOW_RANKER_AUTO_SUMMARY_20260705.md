# Test500 Window Ranker Auto Summary, 2026-07-05

Generated from committed manifest:

```text
python BoxDesignSurrogateRL/scripts/summarize_ranker_window_results.py \
  --manifest BoxDesignSurrogateRL/docs/TEST500_WINDOW_RANKER_RESULT_MANIFEST_20260705.json \
  --total-label "test[0,500) window total"
```

| window | exact PF | ranker+audit PF | coverage | exact validations | ranker validations | exact uncached | ranker uncached | exact subprocess s | ranker subprocess s | exact elapsed s | ranker elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test[0,100) | 1.8121077375 | 1.8121077375 | 1.0000 | 31140 | 26190 | 2647 | 2374 | 1560.0587 | 1366.4305 | 2164.8369 | 1842.0075 |
| test[100,200) | 2.0757308361 | 2.0757308361 | 1.0000 | 28980 | 26200 | 2521 | 2416 | 1469.8335 | 1419.6851 | 2047.5268 | 1944.8361 |
| test[200,300) | 1.9098141289 | 1.9098141289 | 1.0000 | 43020 | 39220 | 3714 | 3485 | 2114.5624 | 1955.3046 | 2954.3013 | 2729.9786 |
| test[300,400) | 1.8590426956 | 1.8590426956 | 1.0000 | 28560 | 25460 | 2478 | 2301 | 1275.1553 | 1143.9761 | 1830.4731 | 1596.6522 |
| test[400,500) | 1.9889904647 | 1.9889904647 | 1.0000 | 36360 | 31860 | 3173 | 2853 | 1857.8720 | 1660.0325 | 2488.9937 | 2277.6879 |
| test[0,500) window total | window-wise | window-wise | 1.0000 | 168060 | 148930 | 14533 | 13429 | 8277.4819 | 7545.4288 | 11486.1318 | 10391.1623 |

| window | PF delta | validation reduction | uncached-box reduction | subprocess reduction | wall-clock reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| test[0,100) | 0.0000000000 | 15.9% | 10.3% | 12.4% | 14.9% |
| test[100,200) | 0.0000000000 | 9.6% | 4.2% | 3.4% | 5.0% |
| test[200,300) | 0.0000000000 | 8.8% | 6.2% | 7.5% | 7.6% |
| test[300,400) | 0.0000000000 | 10.9% | 7.1% | 10.3% | 12.8% |
| test[400,500) | 0.0000000000 | 12.4% | 10.1% | 10.6% | 8.5% |
| test[0,500) window total | 0.0000000000 | 11.4% | 7.6% | 8.8% | 9.5% |
