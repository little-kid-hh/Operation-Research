# Assignment-Aware Candidate Ranker Seed1/Seed2 Window Summary, 2026-07-05

## Scope

This is the paired-window validation for the assignment-aware HGBT
accepted-move ranker after the seed2:test[400,500) failure diagnosis.

- Dataset: OR2023 test split windows, two seed-specific initial-condition
  replicates.
- Windows: seed1:test[0,500) and seed2:test[0,500), evaluated as ten
  100-order windows.
- K: 10.
- Step schedule: `0.25:1000`.
- Oracle: Java/Gurobi MILP with `label_6ori`.
- Coverage repair: `geometric_expand`.
- Ranker budget sequence: `20,30,40,50`.
- Safety policy: `none`.
- Final quality check: exact staged-greedy audit after ranker search.
- Ranker artifact:
  `BoxDesignSurrogateRL/results/candidate_ranker_assignment_aware_20260705/candidate_ranker_20260705_084240/candidate_ranker.joblib`.

The exact baseline summaries and initial boxes are reused from the already
fixed seed1/seed2 exact staged runs, so each ranker window is paired with the
same seed-specific initial condition as its exact baseline.

## Aggregate Result

| metric | exact staged | assignment-aware ranker+audit | reduction |
| --- | ---: | ---: | ---: |
| MILP validations | 136320 | 117920 | 13.5% |
| uncached boxes | 12780 | 11923 | 6.7% |
| Java/Gurobi subprocess seconds | 7004.1556 | 6523.2825 | 6.9% |
| wall-clock seconds | 9447.6642 | 8715.1345 | 7.8% |

Descriptive paired-window uncertainty estimates are in
`ASSIGNMENT_AWARE_SEED1_SEED2_STATS_20260705.md`. With 10,000 window-bootstrap
resamples, the 95% descriptive CIs for aggregate reduction are:

| metric | aggregate reduction | bootstrap 95% CI | windows reduced/tied/increased |
| --- | ---: | ---: | ---: |
| validations | 13.50% | [8.93%, 20.94%] | 10/0/0 |
| uncached boxes | 6.71% | [4.12%, 11.40%] | 10/0/0 |
| Java/Gurobi subprocess seconds | 6.87% | [4.05%, 11.64%] | 10/0/0 |
| wall-clock seconds | 7.75% | [4.99%, 12.64%] | 10/0/0 |

These intervals are descriptive window-bootstrap intervals, not formal iid
statistical guarantees.

## Quality And Coverage

Across all 10 seed-specific windows:

- 10/10 windows match exact staged PF exactly at the reported precision.
- 0/10 windows regress relative to exact staged PF.
- Exact staged and ranker+audit both have 100% coverage in all windows.
- The previously failing seed2:test[400,500) window is repaired: PF returns
  from the old ranker+audit value `2.0025546412` to exact PF `1.9929555750`.

The assignment-aware ranker alone is not a good final optimizer: in every
window, the ranker-stage PF is worse than the audited PF. The result should be
claimed as query-efficient exact local search with a learned candidate order,
not as replacing exact feasibility or exact audit.

## Window Table

| window | exact PF | ranker+audit PF | coverage | exact validations | ranker validations | exact uncached | ranker uncached | exact subprocess s | ranker subprocess s | exact elapsed s | ranker elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed1:test[0,100) | 1.8908695471 | 1.8908695471 | 1.0000 | 10620 | 8620 | 976 | 932 | 511.3663 | 492.2234 | 696.5627 | 646.8383 |
| seed1:test[100,200) | 2.2084188336 | 2.2084188336 | 1.0000 | 26280 | 25120 | 2421 | 2338 | 1307.7590 | 1269.9778 | 1814.9036 | 1742.7348 |
| seed1:test[200,300) | 2.0315201381 | 2.0315201381 | 1.0000 | 9420 | 8220 | 951 | 931 | 524.6430 | 515.5745 | 692.0440 | 671.0918 |
| seed1:test[300,400) | 1.8968685620 | 1.8968685620 | 1.0000 | 15540 | 13620 | 1411 | 1323 | 864.6081 | 799.2195 | 1144.2333 | 1038.2689 |
| seed1:test[400,500) | 1.9910783173 | 1.9910783173 | 1.0000 | 7920 | 4280 | 716 | 480 | 397.6499 | 265.5042 | 528.6209 | 349.4464 |
| seed2:test[0,100) | 1.9380009674 | 1.9380009674 | 1.0000 | 9840 | 7800 | 906 | 853 | 471.6141 | 442.3405 | 641.3276 | 585.9211 |
| seed2:test[100,200) | 2.2095968714 | 2.2095968714 | 1.0000 | 16200 | 15120 | 1579 | 1523 | 853.3800 | 824.3480 | 1155.2793 | 1088.9027 |
| seed2:test[200,300) | 1.9186301368 | 1.9186301368 | 1.0000 | 9480 | 8280 | 964 | 928 | 553.5630 | 529.5180 | 728.1239 | 694.4606 |
| seed2:test[300,400) | 1.9960143397 | 1.9960143397 | 1.0000 | 7860 | 5820 | 793 | 672 | 475.6722 | 407.0212 | 620.7365 | 530.2212 |
| seed2:test[400,500) | 1.9929555750 | 1.9929555750 | 1.0000 | 23160 | 21040 | 2063 | 1943 | 1043.8999 | 977.5555 | 1425.8324 | 1367.2486 |
| seed1+seed2 assignment-aware window total | window-wise | window-wise | 1.0000 | 136320 | 117920 | 12780 | 11923 | 7004.1556 | 6523.2825 | 9447.6642 | 8715.1345 |

| window | PF delta | validation reduction | uncached-box reduction | subprocess reduction | wall-clock reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed1:test[0,100) | 0.0000000000 | 18.8% | 4.5% | 3.7% | 7.1% |
| seed1:test[100,200) | 0.0000000000 | 4.4% | 3.4% | 2.9% | 4.0% |
| seed1:test[200,300) | 0.0000000000 | 12.7% | 2.1% | 1.7% | 3.0% |
| seed1:test[300,400) | 0.0000000000 | 12.4% | 6.2% | 7.6% | 9.3% |
| seed1:test[400,500) | 0.0000000000 | 46.0% | 33.0% | 33.2% | 33.9% |
| seed2:test[0,100) | 0.0000000000 | 20.7% | 5.8% | 6.2% | 8.6% |
| seed2:test[100,200) | 0.0000000000 | 6.7% | 3.5% | 3.4% | 5.7% |
| seed2:test[200,300) | 0.0000000000 | 12.7% | 3.7% | 4.3% | 4.6% |
| seed2:test[300,400) | 0.0000000000 | 26.0% | 15.3% | 14.4% | 14.6% |
| seed2:test[400,500) | 0.0000000000 | 9.2% | 5.8% | 6.4% | 4.1% |
| seed1+seed2 assignment-aware window total | 0.0000000000 | 13.5% | 6.7% | 6.9% | 7.8% |

## Comparison To The Previous RF Ranker

The previous RF accepted-move ranker with `10,20,30,40,50` had this aggregate
seed1+seed2 profile:

- 7 windows matched exact PF.
- 2 windows improved PF.
- 1 window regressed PF, seed2:test[400,500), by `+0.0095990662`.
- Total reductions: validations 13.7%, uncached boxes 5.7%, subprocess time
  6.3%, wall-clock time 6.4%.

The assignment-aware HGBT profile is different:

- 10 windows match exact PF.
- 0 windows improve PF.
- 0 windows regress PF.
- Total reductions: validations 13.5%, uncached boxes 6.7%, subprocess time
  6.9%, wall-clock time 7.8%.

So the assignment-aware version is not uniformly better on every metric: it
loses the two opportunistic PF improvements and validates 340 more candidates
than the previous RF run. But it removes the only quality regression and
improves the measured uncached-oracle, subprocess, and wall-clock aggregates.
For a conservative paper claim, this is the cleaner main method because the
comparison to exact staged is no-regression on all 10 paired windows.

## Interpretation

This supports a stronger seed-specific claim than the previous ranker result:
learned assignment-aware candidate ordering plus exact audit can match exact
staged local-search PF and full coverage on these paired seed1/seed2 windows
while reducing exact-oracle work and wall-clock time.

The claim is still not full-OR2023 superiority or statistical significance.
It is a paired-window systems result on 10 windows, and all costs are measured
under the current Java/Gurobi subprocess oracle implementation.

## Source Records

Remote result roots on `laptop-29k27sem`:

- `BoxDesignSurrogateRL/results/aa_s1r/manifest_frontier_20260705_103217`
- `BoxDesignSurrogateRL/results/aa_s2r/manifest_frontier_20260705_092835`
- `BoxDesignSurrogateRL/results/ranker_assignment_aware_seed2_manifest_20260705/manifest_frontier_20260705_091502`
- `BoxDesignSurrogateRL/results/aa_s2_combined`
- `BoxDesignSurrogateRL/results/aa_s1s2_combined`

Code support:

- `BoxDesignSurrogateRL/scripts/run_ranker_frontier_from_exact_manifest.py`
- `BoxDesignSurrogateRL/scripts/summarize_ranker_window_results.py`
- `BoxDesignSurrogateRL/scripts/analyze_ranker_window_statistics.py`
- `BoxDesignSurrogateRL/box_design_surrogate/candidate_ranker.py`

Committed generated records:

- `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_AUTO_SUMMARY_20260705.json`
- `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_STATS_20260705.json`
- `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_STATS_20260705.md`
