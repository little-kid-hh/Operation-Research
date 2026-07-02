# Dev500 Limit2500 Validation, 2026-07-03

## Split Clarification

The earlier `or2023_seed20260701_limit500` split is not a 500-order dev set.
It contains:

| Split | Orders |
|---|---:|
| train | 300 |
| dev | 100 |
| test | 100 |

Therefore, runs using
`results\splits_calibration\or2023_seed20260701_limit500\or2023_bsp_unique_orders_dev.xml`
are dev100 runs, even if `--orders-limit 500` is passed.

For a true 500-order dev validation set, a new split was materialized on the
Windows/Gurobi experiment machine:

```text
BoxDesignSurrogateRL\results\splits_calibration\or2023_seed20260701_limit2500
```

Its counts are:

| Split | Orders |
|---|---:|
| train | 1500 |
| dev | 500 |
| test | 500 |

The validation below uses the 500-order dev XML:

```text
BoxDesignSurrogateRL\results\splits_calibration\or2023_seed20260701_limit2500\or2023_bsp_unique_orders_dev.xml
```

## Exact 0.5 Checkpoint

Command family:

- `--algorithm staged_greedy`
- `--k 10`
- `--seed 0`
- `--orders-limit 500`
- `--schedule 0.5:600`
- `--coverage-repair geometric_expand`
- Java/Gurobi MILP oracle
- cold cache:
  `results\oracle_cache_dev500_limit2500dev_exact_0p5_20260703`

Artifact:

```text
BoxDesignSurrogateRL\results\candidate_predictor_dev500_limit2500dev_20260703\staged_greedy\run_20260703_010436_419840
```

Result:

| Metric | Value |
|---|---:|
| PF | 2.3273668585124985 |
| Coverage | 1.0 |
| Uncovered orders | 0 |
| Generated candidates | 9120 |
| MILP validated candidates | 9120 |
| Elapsed seconds | 1830.6246638000011 |

## Fine Stage 0.25: Exact vs Ranker

Both runs start from the exact 0.5 checkpoint above and use the same 500-order
dev XML.

Exact artifact:

```text
BoxDesignSurrogateRL\results\candidate_predictor_dev500_limit2500dev_20260703\staged_greedy\run_20260703_013535_387902
```

Ranker artifact:

```text
BoxDesignSurrogateRL\results\candidate_predictor_dev500_limit2500dev_20260703\ranker_filtered_greedy\run_20260703_014015_997884
```

Ranker model:

```text
BoxDesignSurrogateRL\results\candidate_predictors_dev300_repaired_20260702\candidate_ranker_20260702_230711\candidate_ranker.joblib
```

Ranker filter controls:

- `--ranker-adaptive-top-k 10,30`
- `--ranker-noop-fallback`
- exact MILP final acceptance

| Metric | Exact 0.25 | Ranker-filtered 0.25 |
|---|---:|---:|
| PF | 2.3062720835413355 | 2.3062720835413355 |
| Coverage | 1.0 | 1.0 |
| Uncovered orders | 0 | 0 |
| Generated candidates | 900 | 900 |
| MILP validated candidates | 900 | 660 |
| MILP candidate evaluations avoided | 0 | 240 |
| MILP avoidance rate | 0.0 | 0.26666666666666666 |
| Ranker-scored candidates | 0 | 900 |
| Ranker eval seconds | 0.0 | 1.1018391004763544 |
| MILP eval seconds | 204.02312940033153 | 192.63178289681673 |
| Elapsed seconds | 259.81265099998564 | 251.70497150020674 |

## Takeaway

On this true 500-order dev split, the ranker-filtered 0.25 stage preserved the
exact MILP solution quality (`PF=2.3062720835413355`, 100% coverage) while
reducing MILP candidate validations from `900` to `660` (`26.7%` avoided).

The wall-clock improvement in this cold-cache run was modest (`259.8s` to
`251.7s`). The stronger current claim is therefore reduction in exact MILP
candidate validations with preserved solution quality, not yet large end-to-end
runtime acceleration on this 0.25 stage.
