# Candidate Classifier Ranker Evidence And Claim Boundary, 2026-07-05

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

The repaired test250 slice was then run through the same convergence-path
protocol:

```text
Exact repaired convergence:
BoxDesignSurrogateRL/results/test250_exact_repaired_convergence_20260703/staged_greedy/run_20260703_181235_291147

Shared-cache ranker plus audit:
BoxDesignSurrogateRL/results/test250_repaired_shared_cache_ranker_frontier_20260703/frontier_20260703_185904
```

Exact repaired staged convergence reached PF `2.1590677627` with 100% coverage
after 24180 validations, 2175 uncached boxes, 2284.5554 Java/Gurobi subprocess
seconds, and 2763.3551 elapsed seconds. The shared-cache ranker 180s run plus
exact audit reached the same PF and coverage with 23630 validations, 2140
uncached boxes, 2236.8717 subprocess seconds, and 2712.6450 elapsed seconds.
This is a 2.3% validation reduction, 1.6% uncached-box reduction, 2.1%
subprocess-time reduction, and 1.8% wall-clock reduction on repaired test250.
The larger repaired slice therefore supports same-final-quality behavior, but
with much smaller efficiency gains than test50/test100.

## Window-Wise Held-Out Test Split Checks

After adding `--orders-offset` support to the Python runner and fixing the
Java MILP oracle windowing, the shared-cache convergence-path protocol was
replicated across the full 500-order OR2023 test split as five 100-order
windows. The first window had complete initial coverage and uses the original
no-repair test100 protocol; the remaining repaired-window runs use
`geometric_expand` coverage repair for exact staged and ranker+audit.

| held-out window | method | final PF | coverage | uncovered | validations | uncached boxes | subprocess sec | elapsed sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test[0,100) | exact staged | 1.8121077375 | 1.000 | 0 | 31140 | 2647 | 1560.0587 | 2164.8369 |
| test[0,100) | RF top50 180s + exact audit | 1.8121077375 | 1.000 | 0 | 26190 | 2374 | 1366.4305 | 1842.0075 |
| test[100,200) repaired | exact staged | 2.0757308361 | 1.000 | 0 | 28980 | 2521 | 1469.8335 | 2047.5268 |
| test[100,200) repaired | RF top50 180s + exact audit | 2.0757308361 | 1.000 | 0 | 26200 | 2416 | 1419.6851 | 1944.8361 |
| test[200,300) repaired | exact staged | 1.9098141289 | 1.000 | 0 | 43020 | 3714 | 2114.5624 | 2954.3013 |
| test[200,300) repaired | RF top50 180s + exact audit | 1.9098141289 | 1.000 | 0 | 39220 | 3485 | 1955.3046 | 2729.9786 |
| test[300,400) repaired | exact staged | 1.8590426956 | 1.000 | 0 | 28560 | 2478 | 1275.1553 | 1830.4731 |
| test[300,400) repaired | RF top50 180s + exact audit | 1.8590426956 | 1.000 | 0 | 25460 | 2301 | 1143.9761 | 1596.6522 |
| test[400,500) repaired | exact staged | 1.9889904647 | 1.000 | 0 | 36360 | 3173 | 1857.8720 | 2488.9937 |
| test[400,500) repaired | RF top50 180s + exact audit | 1.9889904647 | 1.000 | 0 | 31860 | 2853 | 1660.0325 | 2277.6879 |
| test[0,500) window total | exact staged | window-wise same | 1.000 | 0 | 168060 | 14533 | 8277.4819 | 11486.1318 |
| test[0,500) window total | RF top50 180s + exact audit | window-wise same | 1.000 | 0 | 148930 | 13429 | 7545.4288 | 10391.1623 |

For test[0,100), the ranker path reaches the exact same final PF and coverage
with 15.9% fewer validations, 10.3% fewer uncached boxes, 12.4% lower
Java/Gurobi subprocess time, and 14.9% lower wall-clock time.

For test[100,200), the ranker path reaches the exact same final PF and
coverage with 9.6% fewer validations, 4.2% fewer uncached boxes, 3.4% lower
Java/Gurobi subprocess time, and 5.0% lower wall-clock time.

For test[200,300), the ranker path again reaches the exact same final PF and
coverage with 8.8% fewer validations, 6.2% fewer uncached boxes, 7.5% lower
subprocess time, and 7.6% lower wall-clock time. This second window is harder
than offset100: exact staged requires 43020 candidate validations and 2954.3
wall-clock seconds.

For test[300,400), the ranker path also reaches the exact same final PF and
coverage with 10.9% fewer validations, 7.1% fewer uncached boxes, 10.3% lower
subprocess time, and 12.8% lower wall-clock time.

For test[400,500), the ranker path reaches the exact same final PF and
coverage with 12.4% fewer validations, 10.1% fewer uncached boxes, 10.6% lower
subprocess time, and 8.5% lower wall-clock time.

Across all five 100-order windows, ranker+audit preserves exact staged final
PF and coverage in every window while reducing validations by 11.4%, uncached
boxes by 7.6%, Java/Gurobi subprocess time by 8.8%, and wall-clock time by
9.5%.

The four non-prefix windows are stronger evidence than nested prefix slices
because they do not repeatedly evaluate only the earliest orders in the XML
ordering. Together with the prefix test[0,100) run, they provide window-wise
coverage of the full 500-order test split.

## Seed-Specific Initial-Condition Replicate

The full seed-1 true seed-specific initial-condition replicate was run on the
500-order test split as five 100-order windows. These runs omit
`--initial-boxes-json`, so exact staged generates the initial K=10 box set by
k-means with `--seed 1`; the ranker+audit path then reuses each exact run's
`initial_boxes.json`.

| held-out window | method | final PF | coverage | uncovered | validations | uncached boxes | subprocess sec | elapsed sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed1:test[0,100) | exact staged | 1.8908695471 | 1.000 | 0 | 10620 | 976 | 511.3663 | 696.5627 |
| seed1:test[0,100) | RF top50 180s + exact audit | 1.8892309108 | 1.000 | 0 | 8020 | 843 | 443.1656 | 601.0076 |
| seed1:test[100,200) | exact staged | 2.2084188336 | 1.000 | 0 | 26280 | 2421 | 1307.7590 | 1814.9036 |
| seed1:test[100,200) | RF top50 180s + exact audit | 2.2084188336 | 1.000 | 0 | 24180 | 2291 | 1242.7996 | 1718.0409 |
| seed1:test[200,300) | exact staged | 2.0315201381 | 1.000 | 0 | 9420 | 951 | 524.6430 | 692.0440 |
| seed1:test[200,300) | RF top50 180s + exact audit | 2.0315201381 | 1.000 | 0 | 7580 | 923 | 506.5391 | 647.2780 |
| seed1:test[300,400) | exact staged | 1.8968685620 | 1.000 | 0 | 15540 | 1411 | 864.6081 | 1144.2333 |
| seed1:test[300,400) | RF top50 180s + exact audit | 1.8968685620 | 1.000 | 0 | 13230 | 1343 | 808.4010 | 1074.5126 |
| seed1:test[400,500) | exact staged | 1.9910783173 | 1.000 | 0 | 7920 | 716 | 397.6499 | 528.6209 |
| seed1:test[400,500) | RF top50 180s + exact audit | 1.9910783173 | 1.000 | 0 | 6770 | 660 | 363.2582 | 483.5014 |
| seed1:test[0,500) window total | exact staged | window-wise | 1.000 | 0 | 69780 | 6475 | 3606.0264 | 4876.3645 |
| seed1:test[0,500) window total | RF top50 180s + exact audit | window-wise | 1.000 | 0 | 59780 | 6060 | 3364.1635 | 4524.3405 |

For seed1:test[0,100), the ranker path reaches slightly better final PF than
exact staged from the same seed-1 initial boxes, with 24.5% fewer validations,
13.6% fewer uncached boxes, 13.3% lower subprocess time, and 13.7% lower
wall-clock time.

For seed1:test[100,200), the ranker path reaches the same final PF and full
coverage, with 8.0% fewer validations, 5.4% fewer uncached boxes, 5.0% lower
subprocess time, and 5.3% lower wall-clock time.

For seed1:test[200,300), the ranker path reaches the same final PF and full
coverage, with 19.5% fewer validations, 2.9% fewer uncached boxes, 3.5% lower
subprocess time, and 6.5% lower wall-clock time.

For seed1:test[300,400), the ranker path reaches the same final PF and full
coverage, with 14.9% fewer validations, 4.8% fewer uncached boxes, 6.5% lower
subprocess time, and 6.1% lower wall-clock time.

For seed1:test[400,500), the ranker path reaches the same final PF and full
coverage, with 14.5% fewer validations, 7.8% fewer uncached boxes, 8.6% lower
subprocess time, and 8.5% lower wall-clock time.

Across the complete seed-1 five-window replicate, ranker+audit preserves or
improves final PF and full coverage, with aggregate reductions of 14.3% in
validations, 6.4% in uncached boxes, 6.7% in Java/Gurobi subprocess time, and
7.2% in wall-clock time. This is positive initial-condition evidence, but it
should not be described as statistical significance by itself.

The full seed-2 initial-condition replicate was then run with the same
protocol. Unlike seed1, seed2 is not a strict same-or-better result in every
window, so it should be treated as aggregate efficiency evidence with one
quality tradeoff.

| held-out window | method | final PF | coverage | uncovered | validations | uncached boxes | subprocess sec | elapsed sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed2:test[0,100) | exact staged | 1.9380009674 | 1.000 | 0 | 9840 | 906 | 471.6141 | 641.3276 |
| seed2:test[0,100) | RF top50 180s + exact audit | 1.9380009674 | 1.000 | 0 | 6980 | 750 | 390.3809 | 534.3316 |
| seed2:test[100,200) | exact staged | 2.2095968714 | 1.000 | 0 | 16200 | 1579 | 853.3800 | 1155.2793 |
| seed2:test[100,200) | RF top50 180s + exact audit | 2.2095968714 | 1.000 | 0 | 14830 | 1548 | 830.7895 | 1092.2495 |
| seed2:test[200,300) | exact staged | 1.9186301368 | 1.000 | 0 | 9480 | 964 | 553.5630 | 728.1239 |
| seed2:test[200,300) | RF top50 180s + exact audit | 1.9186301368 | 1.000 | 0 | 7110 | 882 | 495.5806 | 644.5394 |
| seed2:test[300,400) | exact staged | 1.9960143397 | 1.000 | 0 | 7860 | 793 | 475.6722 | 620.7365 |
| seed2:test[300,400) | RF top50 180s + exact audit | 1.9726992574 | 1.000 | 0 | 7370 | 794 | 471.3637 | 613.5968 |
| seed2:test[400,500) | exact staged | 1.9929555750 | 1.000 | 0 | 23160 | 2063 | 1043.8999 | 1425.8324 |
| seed2:test[400,500) | RF top50 180s + exact audit | 2.0025546412 | 1.000 | 0 | 21510 | 2016 | 1012.4072 | 1432.5585 |
| seed2:test[0,500) window total | exact staged | window-wise | 1.000 | 0 | 66540 | 6305 | 3398.1293 | 4571.2997 |
| seed2:test[0,500) window total | RF top50 180s + exact audit | window-wise | 1.000 | 0 | 57800 | 5990 | 3200.5220 | 4317.2759 |

For seed2, ranker+audit matches exact staged PF in three windows, improves PF
in seed2:test[300,400) by `0.0233150822`, and is worse in seed2:test[400,500)
by `0.0095990662`. Coverage remains 100% in all windows. Across all five
seed2 windows, ranker+audit reduces validations by 13.1%, uncached boxes by
5.0%, Java/Gurobi subprocess time by 5.8%, and wall-clock time by 5.6%.

Across the two complete seed-specific replicates, seed1 and seed2, the 10
paired windows have 7 equal-PF outcomes, 2 ranker+audit improvements, and 1
ranker+audit regression. Aggregated over those 10 windows, ranker+audit
reduces validations by 13.7%, uncached boxes by 5.7%, Java/Gurobi subprocess
time by 6.3%, and wall-clock time by 6.4%, with full coverage in every window.

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
4. On held-out test50, test100, and repaired test250, the shared-cache ranker
   180s path plus exact audit reaches the same or essentially the same
   converged quality as exact staged from the same initial boxes, with lower
   measured oracle and wall-clock cost.
5. Across the five 100-order windows covering the full 500-order test split,
   the same shared-cache ranker-plus-audit protocol reaches identical final PF
   and coverage in every window with lower measured oracle and wall-clock cost.
6. On the complete seed-1 five-window initial-condition replicate,
   ranker+audit preserves or improves final PF and full coverage with lower
   measured oracle and wall-clock cost.
7. On the complete seed-2 five-window initial-condition replicate,
   ranker+audit preserves full coverage and lowers aggregate oracle and
   wall-clock cost, but includes one small PF-regression window.
8. The cost reduction is substantial on test50/test100, modest on repaired
   test250, and positive on every 100-order test window, so the current
   evidence supports a consistent efficiency advantage, not unconditional
   quality dominance or a uniform large speedup across all slices.

## Claims Not Yet Supported

The current evidence does not yet support these claims:

1. Full OR2023 superiority over the tuned baseline or over the exact staged
   MILP baseline.
2. Multi-seed statistical significance.
3. Strict same-or-better final PF in every seed/window; seed2:test[400,500)
   is a documented PF-regression case.
4. Universal reduction in uncached MILP labels on every held-out slice.
5. A single monolithic full-held-out run, or statistical significance across
   many seeds, without running shared-cache exact audits from the ranker final
   boxes.
6. A claim that a feasibility predictor alone can solve the optimization
   problem.

These should remain non-claims in paper-facing prose until additional runs
exist.

## Recommended Next Experiments

The next experiments should be selected to turn the strongest current evidence
into a publishable result:

1. Add more seed-specific five-window replicates, or compute a paired
   window-level statistical summary over the existing seed1/seed2 replicates
   plus any compatible seed0 protocol rows.
2. Analyze the seed2:test[400,500) PF-regression case to identify whether it
   is an acceptable small quality tradeoff, a ranker/audit search-path failure,
   or a sign that the audit budget/schedule needs a robustness variant.
3. Replicate dev500 exact-audited runs across additional seeds or independent
   dev splits.
4. Standardize coverage handling before larger held-out/full comparisons: either
   train initial boxes on the corresponding train split or apply the same
   explicit repair step to both methods.
5. Scale to a larger held-out slice or full OR2023 only after the above protocol
   is fixed, because full runs are expensive and ambiguous coverage handling
   would weaken the claim.
6. Report prefetch batching only as an implementation ablation: it helped a
   20-order smoke test but hurt dev500, so it should not be part of the main
   method claim.

## Source Documents

- `BoxDesignSurrogateRL/scripts/summarize_ranker_window_results.py`
- `BoxDesignSurrogateRL/docs/TEST500_WINDOW_RANKER_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/TEST500_WINDOW_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET0_RANKER_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET0_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_SEED1_OFFSET0_CONVERGENCE_20260705.md`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET100_RANKER_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET100_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_SEED1_OFFSET100_CONVERGENCE_20260705.md`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET0_100_RANKER_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET0_100_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET200_RANKER_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET200_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_SEED1_OFFSET200_CONVERGENCE_20260705.md`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET0_200_RANKER_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET0_200_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET300_RANKER_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET300_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_SEED1_OFFSET300_CONVERGENCE_20260705.md`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET0_300_RANKER_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET0_300_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET400_RANKER_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET400_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_SEED1_OFFSET400_CONVERGENCE_20260705.md`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET0_400_RANKER_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET0_400_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/SEED2_OFFSET0_400_RANKER_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/SEED2_OFFSET0_400_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_SEED2_CONVERGENCE_20260705.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_SEED1_SEED2_WINDOW_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_PAPER_SUMMARY_20260704.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_DEV500_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_HELDOUT_SUMMARY_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST50_TIMEBUDGET_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST50_EXACT_AUDIT_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST50_CONVERGENCE_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST100_TIMEBUDGET_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST100_CONVERGENCE_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST250_TIMEBUDGET_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST250_REPAIRED_TIMEBUDGET_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST250_REPAIRED_CONVERGENCE_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_OFFSET100_REPAIRED_CONVERGENCE_20260704.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_OFFSET200_REPAIRED_CONVERGENCE_20260704.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_OFFSET300_REPAIRED_CONVERGENCE_20260704.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_OFFSET400_REPAIRED_CONVERGENCE_20260705.md`
- `BoxDesignSurrogateRL/docs/MILP_BOX_ALGORITHMS.md`

Current documented evidence covers results through the five-window seed-0 test
split plus the complete five-window seed-1 and seed-2 initial-condition
replicates completed on 2026-07-05.
