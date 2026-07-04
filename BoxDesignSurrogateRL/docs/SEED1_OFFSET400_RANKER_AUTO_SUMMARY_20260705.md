# Seed1 Offset400 Ranker Auto Summary, 2026-07-05

Generated from committed manifest:

```text
python BoxDesignSurrogateRL/scripts/summarize_ranker_window_results.py \
  --manifest BoxDesignSurrogateRL/docs/SEED1_OFFSET400_RANKER_RESULT_MANIFEST_20260705.json \
  --total-label "seed1:test[400,500) total"
```

| window | exact PF | ranker+audit PF | coverage | exact validations | ranker validations | exact uncached | ranker uncached | exact subprocess s | ranker subprocess s | exact elapsed s | ranker elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed1:test[400,500) | 1.9910783173 | 1.9910783173 | 1.0000 | 7920 | 6770 | 716 | 660 | 397.6499 | 363.2582 | 528.6209 | 483.5014 |
| seeds=1 window total | window-wise | window-wise | 1.0000 | 7920 | 6770 | 716 | 660 | 397.6499 | 363.2582 | 528.6209 | 483.5014 |

| window | PF delta | validation reduction | uncached-box reduction | subprocess reduction | wall-clock reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed1:test[400,500) | 0.0000000000 | 14.5% | 7.8% | 8.6% | 8.5% |
| seeds=1 window total | 0.0000000000 | 14.5% | 7.8% | 8.6% | 8.5% |
