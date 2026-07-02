# MILP-Backed OR2023 Box Algorithms

This is the current clean entrypoint for comparing two box-design algorithms on
OR2023 while using MILP feasibility instead of the earlier aggregate proxy.
It implements **Problem B: OR2023 exact-MILP box design** from
`BoxDesignSurrogateRL/docs/FORMAL_PROBLEM.md`.

## Entrypoint

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python3 BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm paper_fixed_step \
  --orders-limit 20 \
  --k 10 \
  --fixed-step 0.5 \
  --iterations 3
```

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python3 BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm staged_greedy \
  --orders-limit 20 \
  --k 10 \
  --schedule 0.5:2,0.25:2
```

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python3 BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm surrogate_filtered_greedy \
  --orders-limit 20 \
  --k 10 \
  --schedule 0.25:20 \
  --surrogate-top-k 10
```

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python3 BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm staged_greedy \
  --orders-limit 100 \
  --k 10 \
  --schedule 0.25:50 \
  --candidate-trace-csv BoxDesignSurrogateRL/results/candidate_traces/dev100_exact025.csv
```

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python3 BoxDesignSurrogateRL/scripts/train_candidate_ranker.py \
  --trace-csv BoxDesignSurrogateRL/results/candidate_traces/dev100_exact025.csv \
  --out-dir BoxDesignSurrogateRL/results/candidate_rankers \
  --top-k 10,30
```

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python3 BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm ranker_filtered_greedy \
  --orders-limit 100 \
  --k 10 \
  --schedule 0.25:50 \
  --candidate-ranker-path BoxDesignSurrogateRL/results/candidate_rankers/<run>/candidate_ranker.joblib \
  --ranker-adaptive-top-k 10,30 \
  --ranker-noop-fallback
```

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python3 BoxDesignSurrogateRL/scripts/run_query_budgeted_ranker_frontier.py \
  --xml-path BoxDesignSurrogateRL/results/splits_calibration/<split>/or2023_bsp_unique_orders_dev.xml \
  --initial-boxes-json BoxDesignSurrogateRL/results/<exact_0p5_run>/best_boxes.json \
  --candidate-ranker-path BoxDesignSurrogateRL/results/<ranker_run>/candidate_ranker.joblib \
  --exact-baseline-summary BoxDesignSurrogateRL/results/<exact_0p25_run>/summary.json \
  --ranker-budget-sequence 10 \
  --ranker-budget-sequence 10,30 \
  --run-audit
```

The default oracle is `--oracle java`, which invokes
`org.example.GeneratePerminPackageLabels` through Java/Gurobi. This supports
arbitrary generated box dimensions and is the correct oracle for searching new
box sets.

## Algorithm A: `paper_fixed_step`

This keeps the paper-style action space: for each of `K` boxes, change one of
`length`, `width`, or `height` by `+step` or `-step`, plus a no-op. The default
step is `0.5`.

Current deliberate differences from the paper:

- Dataset: OR2023 unique orders instead of the paper's private SKU-order data.
- Feasibility: online MILP labels for each order-box pair instead of the prior
  aggregate max-dimension/volume proxy.
- Search loop: small deterministic best-improvement passes over the same fixed
  action set, so MILP calls remain tractable while the feasibility objective is
  being validated.
- Parameters: small `--orders-limit` and `--iterations` by default; increase
  only after the MILP runtime is understood.

## Algorithm B: `staged_greedy`

This uses the same MILP feasibility oracle and coordinate action set, but the
step size is adjusted manually across stages. Example:

```text
0.5:2,0.25:2
```

means run at most two best-improvement iterations at step `0.5`, then at most
two iterations at step `0.25`.

This is the tuned greedy variant, not the original paper baseline.

## Algorithm C: `surrogate_filtered_greedy`

This algorithm uses the same coordinate action set as `staged_greedy`, but it
does not send every generated candidate directly to MILP. For each iteration:

1. generate all `6K` coordinate-move candidates;
2. score all generated candidates with the learned feasibility surrogate;
3. keep the top `--surrogate-top-k` candidates under the configured surrogate
   ranking mode;
4. evaluate only those kept candidates with the exact MILP oracle;
5. accept a move only if the MILP score improves the current MILP score.

The surrogate is therefore a candidate filter, not the final feasibility
oracle. Final `packaging_factor`, `coverage_rate`, `uncovered_orders`, and
`unknown_pairs` remain exact-MILP metrics.

The batch surrogate evaluator deduplicates repeated box dimensions across the
candidate batch before calling `predict_proba`, then reconstructs each
candidate score from the shared probability matrix. This avoids repeatedly
scoring unchanged boxes in local-search neighborhoods.

Two optional controls reduce false surrogate no-ops:

```text
--surrogate-adaptive-top-k 10,30
--surrogate-noop-fallback
```

With an adaptive top-k sequence, the runner first validates only the smallest
surrogate-ranked tier. If that tier contains no MILP-improving move, it widens
to the next tier. With `--surrogate-noop-fallback`, a surrogate-filter no-op
triggers exact MILP validation of the remaining generated candidates before the
runner records a true no-op. This preserves exact-MILP acceptance while avoiding
the false local optima caused by a missed candidate in the initial top-k.

The first supported ranking mode is:

```text
--surrogate-rank-mode paper_pf_surrogate
```

which ranks candidates by surrogate-covered assigned box volume with a hard
surrogate feasibility threshold. `risk_aware_surrogate` is available for
diagnostics, but should not be the primary claim until its thresholding and
calibration are frozen.

## Algorithm D: `ranker_filtered_greedy`

This is the candidate-level version of the filter idea. Instead of training on
standalone order-box feasibility labels, it trains from exact local-search
candidate traces: for each search step, every generated coordinate candidate is
evaluated by MILP and labeled with its exact rank, PF delta, and whether it was
the accepted best-improvement move.

The trained ranker predicts a scalar candidate objective used only to order the
generated `6K` candidates. The runner validates the top-ranked candidates with
the exact MILP oracle and accepts a move only if the exact MILP score improves
the current exact score. This preserves the same final acceptance rule as
`staged_greedy`.

Use `--candidate-trace-csv` on exact `paper_fixed_step` or `staged_greedy`
runs to export training data. The trace rows include:

- current exact score and box-set summary;
- moved box dimensions and coordinate action;
- candidate exact score;
- exact candidate rank within the step;
- improvement and accepted-move labels.

The primary ranker diagnostics are:

- exact-best capture at top-k;
- accepted-move capture at top-k;
- exact step preservation at top-k;
- mean predicted rank of the exact-best candidate.

These diagnostics test whether the learned filter would have kept the
candidate that exact greedy would have selected.

There are two supported ranker modes:

- **Exact-preserving filter mode**: use `--ranker-noop-fallback`. The ranker
  checks small top-k tiers first, but if they contain no improving move, the
  runner validates the remaining candidates before declaring no-op. This should
  match exact `staged_greedy` at the same step schedule, but real runtime gains
  can be small when the Java MILP oracle cache already reuses many order-box
  labels.
- **Query-budgeted mode**: use `--no-ranker-noop-fallback`. The ranker validates
  only the configured top-k tiers and accepts a move only after exact MILP
  verification. This cannot accept a surrogate-only false improvement, but it
  can stop before exact local convergence if the ranker misses the best
  improving candidate. Therefore it must be reported with an exact audit:
  restart `staged_greedy` from the query-budgeted final boxes, then report the
  PF audit gap and the additional uncached oracle cost.

The helper entrypoint `scripts/run_query_budgeted_ranker_frontier.py` runs this
budget/audit protocol across multiple top-k sequences and writes
`frontier_summary.csv`, `frontier_summary.json`, and `raw_summaries.json`.
Each top-k sequence gets an independent oracle cache, while the ranker run and
its exact audit share that sequence's cache. This keeps different budget points
comparable while measuring the true incremental cost of the audit.
The scientific comparison should emphasize `oracle_cache.uncached_boxes`,
`oracle_cache.subprocess_seconds`, and end-to-end elapsed time, not only
`milp_validated_candidates`, because the online Java MILP oracle caches
order-box feasibility labels across candidate box sets.

## Outputs

Each run writes a timestamped directory under:

```text
BoxDesignSurrogateRL/results/milp_box_algorithms/<algorithm>/run_YYYYMMDD_HHMMSS/
```

Files:

- `manifest.json`: configuration and oracle settings.
- `initial_boxes.json`: KMeans-initialized boxes.
- `trace.csv`: per-iteration PF, coverage, uncovered orders, and selected action.
- `best_boxes.json`: final selected box set.
- `summary.json`: run configuration plus initial and best scores.

Main metrics:

- `packaging_factor`: mean assigned box volume divided by mean order item volume.
- `coverage_rate`: fraction of orders with at least one MILP-feasible box.
- `uncovered_orders`: orders with no MILP-feasible box; these must be inspected
  before claiming a PF win.
- `mean_box_volume`: mean volume of the selected feasible box per order, with a
  large penalty for uncovered orders.
- `mean_order_volume`: mean total item volume per order.

Runtime and filtering metrics:

- `generated_candidates`: full candidate set size before surrogate filtering.
- `milp_validated_candidates`: candidates actually evaluated by exact MILP.
- `milp_candidate_evaluations_avoided`: generated candidates skipped by the
  surrogate filter.
- `milp_avoidance_rate`: avoided divided by generated candidates.
- `surrogate_eval_seconds`: time spent scoring candidates with the surrogate.
- `ranker_eval_seconds`: time spent scoring candidates with the candidate
  ranker.
- `milp_eval_seconds`: Python-side elapsed time around exact oracle calls.
- `oracle_cache`: includes cache hits/misses, uncached Java/Gurobi batches,
  uncached boxes, and subprocess wall time.

## Environment Notes

The online oracle requires:

- a Java runtime;
- compiled `MILP_3DBPP/target/classes`;
- Gurobi Java dependencies on the classpath, usually via `GUROBI_JAR`.

The fallback `--oracle labels` reads the precomputed OR2023 labels for the 90
original package dimensions only. It is useful for evaluating existing package
sets, but it cannot search continuous generated boxes unless those boxes exactly
match pre-labeled package dimensions.
