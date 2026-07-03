# Candidate Classifier Ranker Evidence And Claim Boundary, 2026-07-03

## Working Narrative Contract

Working thesis: exact MILP feasibility makes fine-grained box-design local
search expensive, but a learned candidate ranker can spend exact oracle calls
on more promising local moves while preserving exact-MILP verification for
every accepted box-set update.

The method should be described as query-efficient exact-MILP local search, not
as a surrogate-only replacement for the packing oracle. The ranker proposes an
evaluation order; the Java/Gurobi MILP oracle still decides whether a candidate
move is accepted.

In scope:

- OR2023 order-geometry box design with K=10.
- Local-search methods that use the same objective, candidate generator, and
  exact MILP feasibility oracle.
- Solution quality measured by packaging factor (PF), reported together with
  coverage and uncovered orders.
- Oracle cost measured by uncached order-box queries, Java/Gurobi subprocess
  time, and end-to-end wall-clock time.

Out of scope for the current evidence:

- Claiming numerical reproduction of a private-data SKU-to-box paper result.
- Claiming that ML replaces exact feasibility checks.
- Claiming full OR2023 or multi-seed superiority before those runs exist.
- Claiming universal uncached-query reduction from the held-out time-budget
  probes, where test50 and test100 improve quality but touch about the same or
  slightly more uncached boxes.

## Method Under Evaluation

The current learned method is the RF accepted-move classifier ranker:

```text
BoxDesignSurrogateRL/results/candidate_ranker_classifier_dev300_to_dev500_20260703/candidate_ranker_20260703_035111/candidate_ranker.joblib
```

Training target:

- `accepted_classifier`: positive label is the exact within-step best
  candidate only when that candidate improves the current exact MILP objective.

Runtime rule:

- Rank generated local-search candidates by `-P(accepted exact improvement)`.
- Validate only the configured top-k tiers unless an exact audit is being run.
- Accept a move only after exact Java/Gurobi MILP verification.

This separates the scientific contribution from pure feasibility prediction:
the model learns which exact local-search candidates are worth querying first,
not whether to trust a candidate without exact verification.

## Main Exact-Audited Evidence: Dev500

Setup:

- Dataset: OR2023 dev split, first 500 orders.
- K: 10.
- Start point: exact converged `0.5` checkpoint.
- Fine stage: `0.25:1000`.
- Oracle: Java/Gurobi MILP, `label_6ori`.
- Seed: 0.

Run directories:

```text
Exact staged:
BoxDesignSurrogateRL/results/candidate_predictor_dev500_limit2500dev_20260703/staged_greedy/run_20260703_013535_387902

RF accepted top50 query-only:
BoxDesignSurrogateRL/results/query_budgeted_ranker_classifier_dev500_20260703/ranker_filtered_greedy/run_20260703_035136_637863

Exact audit from RF top50 final boxes:
BoxDesignSurrogateRL/results/query_budgeted_ranker_classifier_dev500_20260703/staged_greedy/run_20260703_035522_407221
```

| metric | exact staged 0.25 | RF top50 query-only | RF top50 + exact audit |
| --- | ---: | ---: | ---: |
| PF | 2.3062720835 | 2.3062720835 | 2.3062720835 |
| coverage | 1.000 | 1.000 | 1.000 |
| MILP candidate validations | 900 | 420 | 480 |
| uncached boxes | 134 | 106 | 114 |
| subprocess seconds | 236.6537 | 184.3803 | 196.5169 |
| elapsed seconds | 259.8127 | 196.8975 | 210.7224 |

The query-only run already reaches the exact staged PF with 100% coverage and
24.2% lower wall-clock time. After the exact audit, the result still matches
the exact staged PF and coverage while reducing MILP validations by 46.7%,
uncached boxes by 14.9%, Java/Gurobi subprocess time by 17.0%, and wall-clock
time by 18.9%.

Supported claim from this run:

> On dev500 seed 0, the accepted-move classifier ranker reaches the same
> exact-audited local-search solution as exact staged greedy, with fewer
> uncached MILP oracle queries and lower wall-clock time.

Boundary:

- This is the strongest current evidence because it includes an exact audit.
- It is still one split size and one seed, so it is not yet a final main-table
  claim.

## Dev500 Budget Frontier

The same dev500 setup gives a clean quality-cost frontier:

| budget sequence | PF | PF gap vs exact | coverage | uncached boxes | elapsed seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact staged 0.25 | 2.3062720835 | 0.0000000000 | 1.000 | 134 | 259.8127 |
| top5 | 2.3228934482 | 0.0166213647 | 1.000 | 21 | 69.8637 |
| top10 | 2.3216366946 | 0.0153646111 | 1.000 | 34 | 90.6972 |
| top10,20 | 2.3198310282 | 0.0135589447 | 1.000 | 45 | 112.7104 |
| top10,20,30 | 2.3114490614 | 0.0051769779 | 1.000 | 73 | 155.7031 |
| top10,20,30,40 | 2.3086848736 | 0.0024127900 | 1.000 | 88 | 188.7393 |
| top10,20,30,40,50 | 2.3062720835 | 0.0000000000 | 1.000 | 106 | 196.8975 |

This supports a tunable tradeoff claim: lower top-k budgets can save much more
oracle work with a small PF gap, while top50 recovers the exact local optimum
on this dev500 run.

## Held-Out Time-Budget Evidence

Held-out runs use independent test orders from:

```text
BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_test.xml
```

Common setup:

- Start point: same dev500 exact `0.5` checkpoint.
- Fine stage: `0.25:1000`.
- Wall-clock budget: `--max-elapsed-seconds 180`.
- Oracle: Java/Gurobi MILP, `label_6ori`.
- No candidate-status prefetch.

Unrepaired held-out probes:

| held-out slice | method | PF at stop | coverage | uncovered | validations | uncached boxes | subprocess sec | elapsed sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test50 | exact staged | 2.1931578038 | 1.000 | 0 | 2820 | 300 | 129.3814 | 180.4813 |
| test50 | RF accepted top50 | 2.0875865358 | 1.000 | 0 | 990 | 319 | 142.3415 | 180.8429 |
| test100 | exact staged | 2.2313691575 | 1.000 | 0 | 2040 | 235 | 140.9532 | 182.2379 |
| test100 | RF accepted top50 | 2.1038561898 | 1.000 | 0 | 970 | 237 | 144.4647 | 181.9608 |
| test250 | exact staged | 11.9165101124 | 0.992 | 2 | 960 | 145 | 162.1439 | 184.3934 |
| test250 | RF accepted top50 | 11.3901214442 | 0.992 | 2 | 600 | 132 | 158.4936 | 180.3225 |

These runs support an anytime-search claim: under the same time budget, the
ranker finds better exact-verified boxes on all held-out slices tested so far.
The PF reduction versus exact is 4.8% on test50, 5.7% on test100, and 4.4% on
unrepaired test250. The test250 PF values are dominated by uncovered-order
penalties, so they must be reported with coverage and uncovered counts.

The oracle-efficiency claim is mixed on unrepaired held-out probes. The ranker
validates fewer full candidate box sets on test50 and test100, but because it
makes faster progress through more iterations, it touches about the same or
slightly more distinct uncached order-box dimensions. Therefore these two runs
are quality wins, not uncached-query wins.

An exact staged-greedy audit was later run from the test50 ranker final boxes
without a wall-clock limit:

```text
BoxDesignSurrogateRL/results/test50_classifier_exact_audit_20260703/staged_greedy/run_20260703_154327_814314
```

The audit improved PF from `2.0875865358` to `1.7180912327` with 100% coverage,
using 16140 exact candidate validations and 821.3924 elapsed seconds. This
confirms that the held-out test50 result is an anytime advantage, not a
converged exact-equivalence result. The audit summary reported `disk_hits=0`,
so its oracle cost should be interpreted as a standalone cold-cache convergence
audit rather than an incremental audit sharing the ranker run's cache.

The matching exact staged convergence baseline from the same initial boxes was
then run without a wall-clock limit:

```text
BoxDesignSurrogateRL/results/test50_exact_convergence_20260703/staged_greedy/run_20260703_160207_147284
```

It converged to PF `1.7182177961` with 100% coverage after 22080 exact
candidate validations, 1892 uncached boxes, 756.5954 Java/Gurobi subprocess
seconds, and 1166.3584 elapsed seconds.

A shared-cache ranker plus exact-audit frontier was then run so the ranker
stage and audit stage reused the same oracle cache:

```text
BoxDesignSurrogateRL/results/test50_shared_cache_ranker_frontier_20260703/frontier_20260703_163254
```

The shared-cache ranker 180s run plus exact audit reached essentially the same
final PF (`1.7180912327`) with 16980 validations, 1695 uncached boxes, 675.3390
subprocess seconds, and 981.7580 elapsed seconds. This is a 23.1% validation
reduction, 10.4% uncached-box reduction, 10.7% subprocess-time reduction, and
15.8% wall-clock reduction on test50. The earlier cold-cache ranker/audit sum
was slightly more expensive, so the shared-cache frontier is the cleaner cost
accounting.

The same shared-cache convergence-path protocol was then replicated on the
first 100 held-out test orders:

```text
Exact convergence:
BoxDesignSurrogateRL/results/test100_exact_convergence_20260703/staged_greedy/run_20260703_165709_188504

Shared-cache ranker plus audit:
BoxDesignSurrogateRL/results/test100_shared_cache_ranker_frontier_20260703/frontier_20260703_173350
```

Exact staged converged to PF `1.8121077375` with 100% coverage after 31140
candidate validations, 2647 uncached boxes, 1560.0587 Java/Gurobi subprocess
seconds, and 2164.8369 elapsed seconds. The shared-cache ranker 180s run plus
exact audit reached the same final PF and coverage with 26190 validations,
2374 uncached boxes, 1366.4305 subprocess seconds, and 1842.0075 elapsed
seconds. This is a 15.9% validation reduction, 10.3% uncached-box reduction,
12.4% subprocess-time reduction, and 14.9% wall-clock reduction on test100.

## Coverage-Controlled Held-Out Probe

The test250 slice was also run after applying the same `geometric_expand`
coverage repair to both methods:

| held-out slice | method | PF at stop | coverage | uncovered | validations | uncached boxes | subprocess sec | elapsed sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test250 repaired | exact staged | 2.4763186692 | 1.000 | 0 | 60 | 176 | 189.1293 | 198.6728 |
| test250 repaired | RF accepted top50 | 2.4559321217 | 1.000 | 0 | 120 | 152 | 169.6840 | 180.9510 |

This is the cleanest held-out comparison so far: both methods end with 100%
coverage and zero uncovered orders. The classifier ranker has lower PF, 13.6%
fewer uncached boxes, 10.3% lower Java/Gurobi subprocess time, and 8.9% lower
wall-clock time.

The PF improvement is modest after repair, but it is meaningful because
coverage is equal and complete for both methods.

## Supported Claims

The current evidence supports these claims:

1. The method is exact-verified: every accepted move is checked by the
   Java/Gurobi MILP oracle, so reported PF improvements are not surrogate-only
   false positives.
2. On dev500 seed 0, RF accepted top50 reaches the same exact-audited PF and
   coverage as exact staged greedy while reducing true uncached oracle work and
   wall-clock time.
3. On held-out time-budget probes, the ranker transfers as a better anytime
   search policy: it spends the same budget on moves that reduce PF more
   quickly.
4. On held-out test50 and test100, the shared-cache ranker 180s path plus exact
   audit reaches the same or essentially the same converged quality as exact
   staged from the same initial boxes, with lower measured oracle and
   wall-clock cost.
5. On repaired test250, where coverage is controlled at 100% for both methods,
   the ranker improves PF and reduces uncached oracle cost.

## Claims Not Yet Supported

The current evidence does not yet support these claims:

1. Full OR2023 superiority over the tuned baseline or over the exact staged
   MILP baseline.
2. Multi-seed statistical significance.
3. Universal reduction in uncached MILP labels on every held-out slice.
4. Converged exact-equivalence across held-out test slices without running
   shared-cache exact audits from the ranker final boxes.
5. A claim that a feasibility predictor alone can solve the optimization
   problem.

These should remain non-claims in paper-facing prose until additional runs
exist.

## Recommended Next Experiments

The next experiments should be selected to turn the strongest current evidence
into a publishable result:

1. Replicate dev500 exact-audited runs across additional seeds or independent
   dev splits.
2. Extend the shared-cache ranker plus exact-audit protocol to a
   coverage-controlled larger slice, starting from repaired test250 or a
   train-split initial box set, before claiming larger held-out generality.
3. Standardize coverage handling before larger held-out comparisons: either
   train initial boxes on the corresponding train split or apply the same
   explicit repair step to both methods.
4. Scale to a larger held-out slice or full OR2023 only after the above protocol
   is fixed, because full runs are expensive and ambiguous coverage handling
   would weaken the claim.
5. Report prefetch batching only as an implementation ablation: it helped a
   20-order smoke test but hurt dev500, so it should not be part of the main
   method claim.

## Source Documents

- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_DEV500_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_HELDOUT_SUMMARY_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST50_TIMEBUDGET_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST50_EXACT_AUDIT_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST50_CONVERGENCE_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST100_TIMEBUDGET_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST100_CONVERGENCE_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST250_TIMEBUDGET_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST250_REPAIRED_TIMEBUDGET_20260703.md`
- `BoxDesignSurrogateRL/docs/MILP_BOX_ALGORITHMS.md`

Repository evidence snapshot before this document: `ce6e309`.
