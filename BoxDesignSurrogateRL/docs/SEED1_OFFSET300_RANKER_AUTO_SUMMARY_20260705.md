# Seed1 Offset300 Ranker Auto Summary, 2026-07-05

Generated from committed manifest:

```text
python BoxDesignSurrogateRL/scripts/summarize_ranker_window_results.py \
  --manifest BoxDesignSurrogateRL/docs/SEED1_OFFSET300_RANKER_RESULT_MANIFEST_20260705.json \
  --total-label "seed1:test[300,400) total"
```

| window | exact PF | ranker+audit PF | coverage | exact validations | ranker validations | exact uncached | ranker uncached | exact subprocess s | ranker subprocess s | exact elapsed s | ranker elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed1:test[300,400) | 1.8968685620 | 1.8968685620 | 1.0000 | 15540 | 13230 | 1411 | 1343 | 864.6081 | 808.4010 | 1144.2333 | 1074.5126 |
| seeds=1 window total | window-wise | window-wise | 1.0000 | 15540 | 13230 | 1411 | 1343 | 864.6081 | 808.4010 | 1144.2333 | 1074.5126 |

| window | PF delta | validation reduction | uncached-box reduction | subprocess reduction | wall-clock reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed1:test[300,400) | 0.0000000000 | 14.9% | 4.8% | 6.5% | 6.1% |
| seeds=1 window total | 0.0000000000 | 14.9% | 4.8% | 6.5% | 6.1% |
