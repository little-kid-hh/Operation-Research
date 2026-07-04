# Seed1 Offset200 Ranker Auto Summary, 2026-07-05

Generated from committed manifest:

```text
python BoxDesignSurrogateRL/scripts/summarize_ranker_window_results.py \
  --manifest BoxDesignSurrogateRL/docs/SEED1_OFFSET200_RANKER_RESULT_MANIFEST_20260705.json \
  --total-label "seed1:test[200,300) total"
```

| window | exact PF | ranker+audit PF | coverage | exact validations | ranker validations | exact uncached | ranker uncached | exact subprocess s | ranker subprocess s | exact elapsed s | ranker elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed1:test[200,300) | 2.0315201381 | 2.0315201381 | 1.0000 | 9420 | 7580 | 951 | 923 | 524.6430 | 506.5391 | 692.0440 | 647.2780 |
| seeds=1 window total | window-wise | window-wise | 1.0000 | 9420 | 7580 | 951 | 923 | 524.6430 | 506.5391 | 692.0440 | 647.2780 |

| window | PF delta | validation reduction | uncached-box reduction | subprocess reduction | wall-clock reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed1:test[200,300) | 0.0000000000 | 19.5% | 2.9% | 3.5% | 6.5% |
| seeds=1 window total | 0.0000000000 | 19.5% | 2.9% | 3.5% | 6.5% |
