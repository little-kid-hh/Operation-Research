# Query-Budgeted Ranker Dev500 Pilot, 2026-07-03

## Working Narrative Contract

The research target is exact-MILP-backed box design under fine-grained box
dimension search. The core bottleneck is not the number of generated local
search candidates by itself, but the number of uncached order-box feasibility
queries sent to the Java/Gurobi MILP oracle.

The method direction is therefore:

1. use a learned candidate ranker to decide which local-search candidates to
   ask the MILP oracle first;
2. accept a move only when the exact MILP oracle verifies that it improves the
   current exact objective;
3. report query-budgeted search as an anytime Pareto point;
4. optionally run an exact staged-greedy audit from the query-budgeted final
   boxes and report the audit PF gap plus added oracle cost.

This is a stronger and cleaner claim than "ML replaces MILP": the current
method uses ML to reduce oracle queries while preserving exact-MILP acceptance
and making any missed local improvement measurable.

## Protocol

All runs below use the true 500-order OR2023 dev split:

```text
BoxDesignSurrogateRL\results\splits_calibration\or2023_seed20260701_limit2500\or2023_bsp_unique_orders_dev.xml
```

Shared setup:

- `K=10`
- seed `0`
- Java/Gurobi MILP oracle, `label_6ori`
- start from the exact converged `0.5` checkpoint:
  `BoxDesignSurrogateRL\results\candidate_predictor_dev500_limit2500dev_20260703\staged_greedy\run_20260703_010436_419840\best_boxes.json`
- fine stage schedule: `0.25:1000`
- candidate ranker:
  `BoxDesignSurrogateRL\results\candidate_predictors_dev300_repaired_20260702\candidate_ranker_20260702_230711\candidate_ranker.joblib`

Exact `0.25` baseline artifact:

```text
BoxDesignSurrogateRL\results\candidate_predictor_dev500_limit2500dev_20260703\staged_greedy\run_20260703_013535_387902
```

Ranker with exact-preserving no-op fallback:

```text
BoxDesignSurrogateRL\results\candidate_predictor_dev500_limit2500dev_20260703\ranker_filtered_greedy\run_20260703_014015_997884
```

Query-budgeted ranker without no-op fallback:

```text
BoxDesignSurrogateRL\results\candidate_predictor_dev500_limit2500dev_query_budget_20260703\ranker_filtered_greedy\run_20260703_015356_409202
```

Exact audit after query-budgeted ranker:

```text
BoxDesignSurrogateRL\results\candidate_predictor_dev500_limit2500dev_query_budget_20260703\staged_greedy\run_20260703_015640_581111
```

## Results

### Exact-Preserving And Top10,30 Pilot

| Metric | Exact 0.25 | Ranker + fallback | Query-budgeted ranker | Query-budgeted + exact audit |
|---|---:|---:|---:|---:|
| Final PF | 2.3062720835 | 2.3062720835 | 2.3174115903 | 2.3062720835 |
| PF gap vs exact | 0.0000000000 | 0.0000000000 | +0.0111395067 | 0.0000000000 |
| Relative PF gap vs exact | 0.00% | 0.00% | +0.48% | 0.00% |
| Coverage | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Uncovered orders | 0 | 0 | 0 | 0 |
| Generated candidates | 900 | 900 | 420 | 960 |
| MILP-validated candidates | 900 | 660 | 150 | 690 |
| Candidate validations avoided | 0 | 240 | 270 | 210 |
| Oracle uncached boxes | 134 | 130 | 63 | 130 |
| Oracle subprocess seconds | 236.6537 | 233.1558 | 130.3505 | 231.9624 |
| Wall-clock seconds | 259.8127 | 251.7050 | 136.4738 | 252.4288 |
| Ranker eval seconds | 0.0000 | 1.1018 | 0.6102 | 0.6102 |

For the `Query-budgeted + exact audit` column, candidate, uncached-box,
subprocess, elapsed, and ranker times are summed across the query-budgeted run
and the subsequent exact audit run. The final PF and coverage are the audited
final exact values.

### Budget Frontier Smoke On Dev500

The helper script was then run on two additional budget points, each with an
independent cold cache and an exact audit sharing only that budget's cache:

```text
C:\Users\Lenovo\Downloads\Operation-Research\BoxDesignSurrogateRL\results\query_budgeted_ranker_frontier_dev500_20260703\frontier_20260703_020935
```

| Budget | Query-only PF | PF gap vs exact | Query-only coverage | Query-only uncached boxes | Query-only wall-clock seconds | Audited PF | Combined uncached boxes | Combined wall-clock seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top5 | 2.3250495869 | +0.0187775034 | 1.0000 | 18 | 62.5147 | 2.3062720835 | 133 | 251.6819 |
| top10 | 2.3215656006 | +0.0152935171 | 1.0000 | 33 | 91.2342 | 2.3062720835 | 133 | 255.5133 |
| top10,30 | 2.3174115903 | +0.0111395067 | 1.0000 | 63 | 136.4738 | 2.3062720835 | 130 | 252.4288 |

Against the exact `0.25` baseline (`134` uncached boxes, `259.8127s`), the
query-only frontier is monotonic in the expected direction: larger budget
spends more uncached MILP queries and reduces PF gap. After exact audit, all
three budget points recover the exact PF, but the combined wall-clock savings
remain small (`1.7%` to `3.1%`).

## Interpretation

The exact-preserving fallback mode is scientifically clean but currently not a
strong speed result. It reduces candidate validations by `26.7%` while
preserving the exact PF, but real uncached oracle boxes drop only from `134` to
`130` because the Java MILP oracle cache already reuses many order-box labels.
Wall-clock time improves only from `259.8s` to `251.7s`.

The query-budgeted mode is the first result that shows the intended advantage
on the true bottleneck. It keeps 100% coverage and exact-MILP-verified accepted
moves, while reducing uncached oracle boxes from `134` to `63` and wall-clock
time from `259.8s` to `136.5s`. The cost is a PF gap of `+0.01114`, about
`+0.48%` relative to the exact `0.25` local optimum.

The exact audit recovers the exact `0.25` PF from the query-budgeted final box
set, but consumes most of the saved runtime. End-to-end audited wall-clock time
is `252.4s`, only `2.8%` faster than exact. This is useful as a rigor mechanism,
not yet as the main speed claim.

The frontier result strengthens the diagnosis: query-budgeted search alone
shows a real cost-quality tradeoff, but exact audit still dominates the
end-to-end cost. The next model improvement should target exact-best capture at
low top-k so that the audit gap and audit cost shrink, not merely make ranker
inference faster.

## Claim Status

Supported now:

- The learned ranker can be used as a query-budgeting policy without accepting
  surrogate-only improvements; every accepted move is still exact-MILP
  verified.
- On this dev500 fine stage, query-budgeted search gives a meaningful Pareto
  frontier: top5/top10/top10,30 reduce uncached oracle boxes by `86.57%`,
  `75.37%`, and `52.99%`, respectively, with PF gaps of `0.81%`, `0.66%`, and
  `0.48%`, and no coverage loss.
- Exact audit can certify and repair the remaining local-search gap.

Not yet supported:

- A strong claim of faster exact-equivalent convergence. The audited run reaches
  the exact PF, but its end-to-end speedup is only about `2.8%` in this pilot.
- A full-dataset claim. These numbers are dev500, seed `0`, fine stage `0.25`
  only.

## Next Iteration

Use `scripts/run_query_budgeted_ranker_frontier.py` to produce a Pareto frontier
over several ranker top-k budgets, always paired with an exact audit:

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python3 BoxDesignSurrogateRL/scripts/run_query_budgeted_ranker_frontier.py \
  --xml-path BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_dev.xml \
  --initial-boxes-json BoxDesignSurrogateRL/results/candidate_predictor_dev500_limit2500dev_20260703/staged_greedy/run_20260703_010436_419840/best_boxes.json \
  --candidate-ranker-path BoxDesignSurrogateRL/results/candidate_predictors_dev300_repaired_20260702/candidate_ranker_20260702_230711/candidate_ranker.joblib \
  --exact-baseline-summary BoxDesignSurrogateRL/results/candidate_predictor_dev500_limit2500dev_20260703/staged_greedy/run_20260703_013535_387902/summary.json \
  --ranker-budget-sequence 5 \
  --ranker-budget-sequence 10 \
  --ranker-budget-sequence 10,30 \
  --schedule 0.25:1000 \
  --orders-limit 500 \
  --run-audit
```

The next modeling target should be a listwise or pairwise candidate ranker
trained directly to capture the exact best-improvement move at low top-k. The
primary validation metric should be audit gap per uncached oracle box, not
standalone feasibility accuracy.
