# Seed1 Offset0 Ranker Auto Summary, 2026-07-05

Generated from committed manifest:

```text
python BoxDesignSurrogateRL/scripts/summarize_ranker_window_results.py \
  --manifest BoxDesignSurrogateRL/docs/SEED1_OFFSET0_RANKER_RESULT_MANIFEST_20260705.json \
  --total-label "seed1:test[0,100) total"
```

| window | exact PF | ranker+audit PF | coverage | exact validations | ranker validations | exact uncached | ranker uncached | exact subprocess s | ranker subprocess s | exact elapsed s | ranker elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed1:test[0,100) | 1.8908695471 | 1.8892309108 | 1.0000 | 10620 | 8020 | 976 | 843 | 511.3663 | 443.1656 | 696.5627 | 601.0076 |
| seed1:test[0,100) total | window-wise | window-wise | 1.0000 | 10620 | 8020 | 976 | 843 | 511.3663 | 443.1656 | 696.5627 | 601.0076 |

| window | PF delta | validation reduction | uncached-box reduction | subprocess reduction | wall-clock reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed1:test[0,100) | -0.0016386364 | 24.5% | 13.6% | 13.3% | 13.7% |
| seed1:test[0,100) total | window-wise | 24.5% | 13.6% | 13.3% | 13.7% |
