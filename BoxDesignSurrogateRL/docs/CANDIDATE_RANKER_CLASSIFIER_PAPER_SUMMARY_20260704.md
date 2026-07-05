# Candidate Classifier Ranker Paper Summary, 2026-07-05

## Paper-Facing Claim

The current strongest contribution is query-efficient exact-MILP local search
for fine-grained box design. A learned accepted-move classifier ranks local
box-set candidates, but every accepted move is still verified by the exact
Java/Gurobi MILP oracle. The method should not be described as replacing exact
packing feasibility.

The claim supported by current evidence is:

> A learned candidate ranker can guide exact-MILP local search to the same or
> essentially the same local-search solution as exact staged greedy while
> reducing measured exact-oracle work and wall-clock time on dev500, three
> prefix held-out slice protocols, and a complete 500-order test split evaluated
> as five 100-order windows, with a complete seed-1 five-window
> initial-condition replicate showing the same pattern. A seed-2
> replicate preserves full coverage and reduces aggregate oracle effort, but
> includes one window with a small PF regression.

This is a window-wise 500-order test-split claim across the seed-0 fixed
checkpoint protocol plus full seed-1 and seed-2 initial-condition replicates.
It is not yet a full-OR2023 statistical claim.

The complete seed-1 five-window initial-condition replicate also preserves or
improves final PF and full coverage in every window. Across seed1:test[0,500),
ranker+audit reduces validations by 14.3%, uncached boxes by 6.4%, subprocess
time by 6.7%, and wall-clock time by 7.2%. This is positive
initial-condition evidence, but it is not a statistical result by itself.

The complete seed-2 five-window initial-condition replicate preserves full
coverage in every window and reduces aggregate oracle effort, but it is not a
strict dominance result: ranker+audit matches exact PF in three windows,
improves one window by `0.0233150822`, and is worse in one window by
`0.0095990662`. Across seed2:test[0,500), ranker+audit reduces validations by
13.1%, uncached boxes by 5.0%, subprocess time by 5.8%, and wall-clock time by
5.6%.

The seed2:test[400,500) regression has now been diagnosed. The exact first
move is an expansion, `5:height:+0.250000`, but the ranker predicted it at rank
48/60 and stopped after top10 because smaller verified improvements already
existed. An `all_expansions` safety variant recovers exact PF on this window,
but it is a robustness tradeoff rather than a speedup: the 180-second safety
run matches PF and coverage while increasing wall-clock by 2.1%, and the
60-second safety run increases wall-clock by 5.3%. This supports adding
assignment-aware ranker features next, not promoting expansion safety as the
main method.

Across the 10 seed-specific windows in seed1 and seed2, ranker+audit matches
exact PF in 7 windows, improves PF in 2 windows, and is worse in 1 window,
while preserving 100% coverage in all windows. Aggregated over those 10
windows, it reduces validations by 13.7%, uncached boxes by 5.7%,
Java/Gurobi subprocess time by 6.3%, and wall-clock time by 6.4%.

## Main Convergence And Audit Table

All rows use K=10, the Java/Gurobi MILP oracle with `label_6ori`, and exact
verification for every accepted move. The learned method is the RF
accepted-move classifier with adaptive top-k `10,20,30,40,50`, followed by an
exact staged-greedy audit when needed.

| slice | protocol | exact PF | ranker+audit PF | coverage | exact validations | ranker+audit validations | exact uncached boxes | ranker+audit uncached boxes | exact subprocess sec | ranker+audit subprocess sec | exact elapsed sec | ranker+audit elapsed sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev500 | exact audit after top50 | 2.3062720835 | 2.3062720835 | 1.000 | 900 | 480 | 134 | 114 | 236.6537 | 196.5169 | 259.8127 | 210.7224 |
| test50 | shared-cache convergence-path audit | 1.7182177961 | 1.7180912327 | 1.000 | 22080 | 16980 | 1892 | 1695 | 756.5954 | 675.3390 | 1166.3584 | 981.7580 |
| test100 | shared-cache convergence-path audit | 1.8121077375 | 1.8121077375 | 1.000 | 31140 | 26190 | 2647 | 2374 | 1560.0587 | 1366.4305 | 2164.8369 | 1842.0075 |
| test250 repaired | shared-cache repaired convergence-path audit | 2.1590677627 | 2.1590677627 | 1.000 | 24180 | 23630 | 2175 | 2140 | 2284.5554 | 2236.8717 | 2763.3551 | 2712.6450 |
| test100 offset100 repaired | non-prefix shared-cache repaired convergence-path audit | 2.0757308361 | 2.0757308361 | 1.000 | 28980 | 26200 | 2521 | 2416 | 1469.8335 | 1419.6851 | 2047.5268 | 1944.8361 |
| test100 offset200 repaired | non-prefix shared-cache repaired convergence-path audit | 1.9098141289 | 1.9098141289 | 1.000 | 43020 | 39220 | 3714 | 3485 | 2114.5624 | 1955.3046 | 2954.3013 | 2729.9786 |
| test100 offset300 repaired | non-prefix shared-cache repaired convergence-path audit | 1.8590426956 | 1.8590426956 | 1.000 | 28560 | 25460 | 2478 | 2301 | 1275.1553 | 1143.9761 | 1830.4731 | 1596.6522 |
| test100 offset400 repaired | non-prefix shared-cache repaired convergence-path audit | 1.9889904647 | 1.9889904647 | 1.000 | 36360 | 31860 | 3173 | 2853 | 1857.8720 | 1660.0325 | 2488.9937 | 2277.6879 |
| test500 window total | five 100-order repaired convergence-path audits | window-wise same | window-wise same | 1.000 | 168060 | 148930 | 14533 | 13429 | 8277.4819 | 7545.4288 | 11486.1318 | 10391.1623 |
| seed1:test[0,500) window total | five 100-order seed-specific audits | window-wise same/better | window-wise same/better | 1.000 | 69780 | 59780 | 6475 | 6060 | 3606.0264 | 3364.1635 | 4876.3645 | 4524.3405 |
| seed2:test[0,500) window total | five 100-order seed-specific audits | mixed window-wise | mixed window-wise | 1.000 | 66540 | 57800 | 6305 | 5990 | 3398.1293 | 3200.5220 | 4571.2997 | 4317.2759 |

Cost reductions relative to exact staged:

| slice | PF delta | validation reduction | uncached-box reduction | subprocess-time reduction | wall-clock reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev500 | 0.0000000000 | 46.7% | 14.9% | 17.0% | 18.9% |
| test50 | -0.0001265634 | 23.1% | 10.4% | 10.7% | 15.8% |
| test100 | 0.0000000000 | 15.9% | 10.3% | 12.4% | 14.9% |
| test250 repaired | 0.0000000000 | 2.3% | 1.6% | 2.1% | 1.8% |
| test100 offset100 repaired | 0.0000000000 | 9.6% | 4.2% | 3.4% | 5.0% |
| test100 offset200 repaired | 0.0000000000 | 8.8% | 6.2% | 7.5% | 7.6% |
| test100 offset300 repaired | 0.0000000000 | 10.9% | 7.1% | 10.3% | 12.8% |
| test100 offset400 repaired | 0.0000000000 | 12.4% | 10.1% | 10.6% | 8.5% |
| test500 window total | 0.0000000000 window-wise | 11.4% | 7.6% | 8.8% | 9.5% |
| seed1:test[0,500) window total | same/better window-wise | 14.3% | 6.4% | 6.7% | 7.2% |
| seed2:test[0,500) window total | mixed; mean -0.0027432032 | 13.1% | 5.0% | 5.8% | 5.6% |

Interpretation:

- dev500, test50, and test100 show clear oracle-efficiency gains at equal or
  essentially equal final PF.
- repaired test250 also preserves final PF and coverage, but the cost reduction
  is small. It should be reported as positive but not as a large speedup.
- test100 offset100 repaired is the cleanest non-prefix window check so far:
  it preserves final PF and coverage with modest but positive cost reductions.
- test100 offset200 repaired is a harder non-prefix window and also preserves
  final PF and coverage with positive reductions across all primary cost
  metrics.
- test100 offset300 repaired preserves final PF and coverage and has the
  largest non-prefix wall-clock reduction so far.
- test100 offset400 repaired completes the five-window test split coverage and
  again preserves final PF and coverage with lower cost.
- Across the five 100-order test windows, ranker+audit preserves exact staged
  final PF and coverage in every window while reducing validations by 11.4%,
  uncached boxes by 7.6%, Java/Gurobi subprocess time by 8.8%, and wall-clock
  time by 9.5%.
- Across the complete seed-1 five-window replicate, ranker+audit preserves or
  improves final PF and full coverage while reducing validations by 14.3%,
  uncached boxes by 6.4%, Java/Gurobi subprocess time by 6.7%, and wall-clock
  time by 7.2%.
- Across the complete seed-2 five-window replicate, ranker+audit preserves
  full coverage and reduces aggregate oracle effort, but the quality result is
  mixed: three windows match exact PF, one improves PF, and one is worse by
  `0.0095990662`.
- The seed2:test[400,500) regression is caused by a low-ranked expansion move:
  exact chooses `5:height:+0.250000` first, but the ranker places it at rank
  48/60 and accepts smaller top10 improvements. The expansion-safety variant
  fixes PF on this window but does not improve wall-clock time, so it is a
  robustness check rather than the main acceleration claim.
- The strongest empirical pattern is consistent same-quality convergence with
  fewer exact oracle calls on seed0/seed1 and aggregate oracle-effort reduction
  with a small quality tradeoff on seed2, not uniformly large acceleration or
  unconditional dominance.

## Anytime Results

The 180-second held-out probes test time-budgeted search quality before exact
audit:

| slice | exact PF at about 180s | ranker PF at about 180s | coverage | interpretation |
| --- | ---: | ---: | ---: | --- |
| test50 | 2.1931578038 | 2.0875865358 | 1.000 | ranker reaches better PF under the same time budget |
| test100 | 2.2313691575 | 2.1038561898 | 1.000 | ranker reaches better PF under the same time budget |
| test250 unrepaired | 11.9165101124 | 11.3901214442 | 0.992 | PF is dominated by two uncovered orders; report only with coverage |
| test250 repaired | 2.4763186692 | 2.4559321217 | 1.000 | ranker reaches better repaired PF under the same time budget |

These are anytime-search results, not convergence claims by themselves. The
convergence-path audits above are required before claiming same-quality local
convergence.

## Claim Matrix

Supported:

1. Exact-verified learned ranking: the classifier only orders candidate
   evaluations; exact Java/Gurobi MILP decides accepted moves.
2. Same or essentially same local-search quality with lower measured oracle
   cost on dev500, test50, test100, repaired test250, and the non-prefix
   five-window repaired test split under the recorded protocols.
3. The same seed-specific protocol on the complete seed1:test[0,500)
   five-window replicate preserves or improves final PF and full coverage with
   lower measured oracle and wall-clock cost.
4. The complete seed2:test[0,500) replicate preserves full coverage and lowers
   aggregate validations, uncached boxes, subprocess time, and wall-clock time,
   while exposing one small PF-regression window.
5. Better time-budgeted search quality on test50, test100, unrepaired test250
   when reported with coverage, and repaired test250.
6. The benefit is strongest on test50/test100 and smaller on repaired test250.

Not supported yet:

1. Full OR2023 superiority.
2. Multi-seed statistical significance.
3. A strict claim that ranker+audit matches or improves exact staged final PF
   in every seed and every window.
4. A claim that ML replaces exact feasibility.
5. A universal large speedup across all slice sizes and coverage conditions.
6. Exact reproduction of the private-data SKU-to-box paper; this project uses
   OR2023 order geometry and an exact MILP order-feasibility oracle.

## Next Experiments

The next experiments that would most improve paper rigor are:

1. Build a paired window-level statistical summary over the seed-specific
   replicates, and add more seeds if the confidence interval is still too wide.
2. Turn the seed2:test[400,500) diagnosis into an assignment-aware ranker
   feature set, because the current feature set under-ranks expansions that
   improve order-to-box assignment.
3. Standardize coverage handling for any larger/full held-out comparison.
4. If full OR2023 is attempted, treat it as a long systems experiment and
   report coverage, final PF, uncached boxes, subprocess time, and wall-clock
   time together.

## Source Result Records

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
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_SEED2_O400_FAILURE_AND_SAFETY_20260705.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_DEV500_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST50_CONVERGENCE_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST100_CONVERGENCE_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_TEST250_REPAIRED_CONVERGENCE_20260703.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_OFFSET100_REPAIRED_CONVERGENCE_20260704.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_OFFSET200_REPAIRED_CONVERGENCE_20260704.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_OFFSET300_REPAIRED_CONVERGENCE_20260704.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_OFFSET400_REPAIRED_CONVERGENCE_20260705.md`
- `BoxDesignSurrogateRL/docs/CANDIDATE_RANKER_CLASSIFIER_HELDOUT_SUMMARY_20260703.md`
