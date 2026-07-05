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
  --run-audit \
  --prefetch-candidate-statuses
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

- **False-no-op fallback mode**: use `--ranker-noop-fallback`. The ranker
  checks small top-k tiers first, but if they contain no improving move, the
  runner validates the remaining candidates before declaring no-op. This should
  reduce false local optima caused by a missed improving candidate. It does
  not theoretically guarantee the same step as exact `staged_greedy`: if an
  early tier contains an improving move, the runner accepts the best
  MILP-verified move in that tier without validating every lower-ranked
  candidate. Any exact-equivalence claim must therefore be empirical for a
  reported run, or certified by a subsequent exact audit.
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
Use `--ranker-max-elapsed-seconds` and `--audit-max-elapsed-seconds` when the
ranker and audit stages need separate graceful wall-clock budgets.
The scientific comparison should emphasize `oracle_cache.uncached_boxes`,
`oracle_cache.subprocess_seconds`, and end-to-end elapsed time, not only
`milp_validated_candidates`, because the online Java MILP oracle caches
order-box feasibility labels across candidate box sets.

For multi-window or multi-seed runs, use:

```text
python3 BoxDesignSurrogateRL/scripts/run_ranker_window_protocol.py \
  --xml-path BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_test.xml \
  --candidate-ranker-path BoxDesignSurrogateRL/results/candidate_predictors_dev300_repaired_20260702/candidate_ranker_20260702_230711/candidate_ranker.joblib \
  --orders-offsets 0,100,200,300,400 \
  --orders-limit 100 \
  --seeds 1,2 \
  --schedule 0.25:1000 \
  --coverage-repair geometric_expand \
  --ranker-max-elapsed-seconds 180
```

This wrapper first runs exact staged convergence, then runs ranker search and
exact audit from the same initial boxes, and finally writes a manifest that
`scripts/summarize_ranker_window_results.py` can turn into JSON/CSV/Markdown
tables. To run a true seed replicate, omit `--initial-boxes-json`; the exact
baseline will generate seed-specific k-means initial boxes and the ranker path
will reuse the exact run's `initial_boxes.json`. Supplying
`--initial-boxes-json` intentionally fixes the initial boxes and is therefore
not a true initial-condition seed replicate.

The candidate ranker can be trained with either pointwise regression targets
or direct classifier targets. The classifier targets are:

```text
--target-mode exact_best_classifier
--target-mode accepted_classifier
```

These train a probability model for the exact within-step best candidate or the
accepted exact-improving candidate. At runtime the ranker orders candidates by
negative positive-class probability, then the runner still validates candidates
with the exact MILP oracle before accepting any move.

Ranker safety policies are controlled by:

```text
--ranker-safety-policy none
--ranker-safety-policy all_expansions
--ranker-safety-policy targeted_expansion_capture
--ranker-safety-max-candidates <n>
```

`none` is the current main policy. `all_expansions` validates every expansion
move in each ranker tier and is intentionally broad. `targeted_expansion_capture`
uses assignment-aware candidate features to validate only high capture-value
expansion moves, capped by `--ranker-safety-max-candidates`; when the cap is
omitted, the runner uses a conservative default of five targeted safety
candidates per iteration. These safety policies affect only the ranker stage.
Final PF and coverage must still be reported after exact staged-greedy audit.

## Time-Budgeted Runs

Long held-out or full-data exact runs can take much longer than the dev fine
stage. Use:

```text
--max-elapsed-seconds <seconds>
```

to request a graceful wall-clock budget. The runner checks the budget before
starting each new local-search iteration. When the budget is reached, it writes
the current best boxes, trace, and summary with `stop_reason: time_limit`.
The budget is therefore not a hard interrupt inside a Java/Gurobi candidate
batch; elapsed time can exceed the requested value by the duration of the
current iteration.

To analyze a ranker/audit handoff after running one or more frontier summaries,
use:

```text
python3 BoxDesignSurrogateRL/scripts/analyze_ranker_handoff_traces.py \
  --frontier-summary cap300=BoxDesignSurrogateRL/results/<cap300>/frontier_summary.json \
  --frontier-summary cap600=BoxDesignSurrogateRL/results/<cap600>/frontier_summary.json \
  --frontier-summary cap900=BoxDesignSurrogateRL/results/<cap900>/frontier_summary.json \
  --tail-iterations 10 \
  --out-json BoxDesignSurrogateRL/results/<analysis>/handoff_trace_analysis.json \
  --out-md BoxDesignSurrogateRL/results/<analysis>/handoff_trace_analysis.md
```

The handoff analyzer reads each frontier row's ranker `trace.csv` and reports
ranker-only PF, exact-audited PF, combined oracle work, last improving
iteration, and recent tail progress. Use this for diagnosing adaptive
ranker-audit controllers; it is not a substitute for exact audit.

## Order Windows

Use `--orders-offset` together with `--orders-limit` to run non-overlapping
held-out windows from the same XML file. For example, `--orders-offset 100
--orders-limit 100` evaluates orders with zero-based indices `[100, 200)`.
The runner records `available_orders`, `orders_offset`, `orders_limit`,
`selected_orders`, and `orders_end_exclusive` in `manifest.json` and
`summary.json`. With the online Java MILP oracle, the runner passes the same
window to `GeneratePerminPackageLabels` by reading through the window end and
skipping the order-box tasks before `orders_offset`; offset runs are therefore
not treated as XML prefixes.

## Candidate Status Prefetching

The Java oracle caches exact order-box feasibility by order signature and box
dimensions. Use:

```text
--prefetch-candidate-statuses
```

to batch-prefetch exact statuses for all unique box dimensions in the candidate
set or filtered tier before per-candidate scoring. This does not change the
accepted objective, feasibility oracle, or exact-MILP acceptance rule; it only
changes how many Java/Gurobi subprocess batches are launched to populate the
same cache.

Report this switch explicitly. Fair comparisons should either keep it fixed
across exact and filtered methods, or present a separate implementation-level
ablation. The main method claims should still prioritize exact final PF,
coverage, uncached oracle boxes, oracle subprocess time, and end-to-end wall
time.

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
- `prefetch_eval_seconds`: Python-side elapsed time spent prefetching exact
  candidate statuses before per-candidate scoring.
- `milp_eval_seconds`: Python-side elapsed time around exact oracle calls,
  including prefetch time when `--prefetch-candidate-statuses` is enabled.
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
