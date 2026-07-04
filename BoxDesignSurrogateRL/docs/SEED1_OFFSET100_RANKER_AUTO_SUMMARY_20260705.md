# Seed1 Offset100 Ranker Auto Summary, 2026-07-05

Generated from committed manifest:

```text
python BoxDesignSurrogateRL/scripts/summarize_ranker_window_results.py \
  --manifest BoxDesignSurrogateRL/docs/SEED1_OFFSET100_RANKER_RESULT_MANIFEST_20260705.json \
  --total-label "seed1:test[100,200) total"
```

| window | exact PF | ranker+audit PF | coverage | exact validations | ranker validations | exact uncached | ranker uncached | exact subprocess s | ranker subprocess s | exact elapsed s | ranker elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed1:test[100,200) | 2.2084188336 | 2.2084188336 | 1.0000 | 26280 | 24180 | 2421 | 2291 | 1307.7590 | 1242.7996 | 1814.9036 | 1718.0409 |
| seeds=1 window total | window-wise | window-wise | 1.0000 | 26280 | 24180 | 2421 | 2291 | 1307.7590 | 1242.7996 | 1814.9036 | 1718.0409 |

| window | PF delta | validation reduction | uncached-box reduction | subprocess reduction | wall-clock reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed1:test[100,200) | 0.0000000000 | 8.0% | 5.4% | 5.0% | 5.3% |
| seeds=1 window total | 0.0000000000 | 8.0% | 5.4% | 5.0% | 5.3% |
