# Certified Ranker-Audit Paper Claim Package, 2026-07-05

Updated with cap-unified ranker180 test-split evidence on 2026-07-06.

This document fixes the current paper-facing story for the OR2023
MILP-backed box design experiments. It is a claim and evidence package, not a
final paper section.

## Narrative Contract

- Working title: Certified Learned Candidate Ordering for MILP-Grounded Box
  Design Search.
- Target style: operations research, ML-for-optimization, or systems evaluation
  paper.
- Primary audience: reviewers who care about exact feasibility, fair baselines,
  and reproducible optimization protocols.
- One-sentence thesis: learned assignment-aware candidate ordering can reduce
  exact MILP oracle work in fine-grained box-design local search while an exact
  audit preserves or improves the final packaging factor and coverage.

## Problem And Gap

The fixed problem is OR2023 exact-MILP box design. The algorithm chooses
`K=10` box types for OR2023 order geometries. A candidate box set is evaluated
with an exact Java/Gurobi MILP order-feasibility oracle, and the primary
quality metric is packaging factor, where lower is better. Full coverage and
zero uncovered orders are required for a valid main result.

The practical bottleneck is not only the number of local-search iterations. In
the current implementation, fine-grained local search spends substantial time
on uncached MILP box queries and Java/Gurobi subprocess calls. A pure learned
surrogate is unsafe because missed improving moves can change the local-search
trajectory and can leave the final solution worse than exact staged search.

The gap is therefore narrower and more defensible than "replace the optimizer
with ML": the paper should ask whether learning can prioritize exact oracle
queries so that the same exact objective and feasibility rule are retained, but
less oracle work is needed.

## Method Fixed For Current Evidence

The main method is certified ranker-audit:

1. Generate the same local move candidates as the exact staged baseline.
2. Use an assignment-aware HGBT ranker to order candidate moves.
3. Validate a bounded ranked frontier with the same Java/Gurobi MILP oracle.
4. Continue search from the ranker-selected solution.
5. Run an exact staged-greedy audit from the ranker result.
6. Report final PF and coverage only after that exact audit.

The current main ranker policy is:

- ranker frontier budgets: `20,30,40,50`;
- ranker safety policy: `none`;
- ranker wall-clock cap: `180` seconds for the cap-unified main evidence;
- final exact audit: required;
- ranker-only output: diagnostic or ablation only.

This makes the central contribution a learned ordering policy with exact
certification, not a standalone feasibility predictor.

The previous mixed-cap table is retained as diagnostic evidence only: seed1,
seed2, and seed4 used a 180-second ranker cap, while seed3 used a 900-second
cap. The current main evidence fixes that protocol issue by rerunning seed3
with the same 180-second ranker cap on the same OR2023 test split XML used by
seed1, seed2, and seed4.

## Baseline Contract

The main paired baseline for the current evidence is exact staged greedy under
the same OR2023 test windows, the same `K=10`, the same schedule `0.25:1000`,
the same Java/Gurobi MILP oracle, and the same coverage repair rule.

This should be described as the exact staged baseline in this implementation.
It should not be described as a strict reproduction of the original private-data
paper baseline unless the original paper's data, feasibility model, and tuning
protocol are separately verified.

## Metrics

- PF: packaging factor. Lower is better.
- Coverage: fraction of orders covered by at least one selected box. Main
  results require `1.0000`.
- Uncovered orders: must be zero for main wins.
- Validations: exact candidate evaluations attempted by the search protocol.
- Uncached MILP boxes: distinct box feasibility queries that were not served by
  cache. This is the cleanest implementation-level proxy for new oracle work.
- Java/Gurobi subprocess seconds: time spent in the current MILP subprocess
  oracle.
- Wall-clock seconds: end-to-end measured runtime for the paired run.

Quality comparisons should be paired by window and initial condition. Runtime
claims should report validations, uncached boxes, subprocess seconds, and
wall-clock seconds together because each captures a different part of the
current oracle implementation.

## Main Evidence

The current main evidence is the cap-unified ranker180 test-split evaluation:
twenty 100-order OR2023 test windows across seed1, seed2, seed3, and seed4
initial conditions.

Configuration:

- dataset: OR2023 test split windows;
- XML: `BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_test.xml`;
- window size: 100 orders;
- seeds: 1, 2, 3, 4 initial conditions;
- schedule: `0.25:1000`;
- K: 10;
- ranker cap: 180 seconds for every seed;
- ranker frontier budgets: `20,30,40,50`;
- ranker safety policy: `none`;
- final quality: exact staged-greedy audit from the ranker result.

Current main result:

- PF matched/improved/regressed: 19/1/0.
- Mean PF delta: `-0.0003110651`.
- Max PF regression: `0.0000000000`.
- Max PF improvement: `-0.0062213021`.
- Minimum exact coverage: `1.0000`.
- Minimum ranker-audit coverage: `1.0000`.
- Total exact uncovered orders: `0`.
- Total ranker-audit uncovered orders: `0`.

Current main aggregate cost reductions:

| metric | aggregate reduction | descriptive bootstrap 95% CI | windows reduced/tied/increased | sign-test p |
| --- | ---: | ---: | ---: | ---: |
| validations | 11.12% | [7.84%, 15.50%] | 19/0/1 | 0.0000 |
| uncached boxes | 4.84% | [2.93%, 7.42%] | 18/1/1 | 0.0001 |
| subprocess seconds | 5.57% | [3.75%, 8.17%] | 19/0/1 | 0.0000 |
| elapsed seconds | 5.87% | [3.65%, 8.78%] | 19/0/1 | 0.0000 |

The current strict audit in
`ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_STRICT_CLAIM_AUDIT_20260706.md`
fails because it requires zero per-window cost increases. The aggregate claim
audit in
`ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_AGGREGATE_CLAIM_AUDIT_20260706.md`
passes when the claim is stated as no certified PF regression, complete
coverage, positive aggregate cost reductions, and at most two per-metric cost
increase windows.

The important hard-window diagnostic is seed3:test[400,500). With the
cap-unified 180-second policy, ranker-audit improves PF by `-0.0062213021` but
uses `0.8%` more validations, `2.1%` more uncached boxes, and `3.0%` more
wall-clock time on that window. The focused adaptive-budget comparison in
`ADAPTIVE_RANKER_BUDGET_SEED3_O400_COMPARISON_20260706.md` shows that all
tested 900-second ranker diagnostics reach the same audited PF, complete
coverage, and zero uncovered orders. The best wall-clock variant,
`adaptive_3e-6`, keeps the same audited PF and reduces cost relative to
`fixed_ranker180_audit`: `38.8%` fewer validations, `17.0%` fewer uncached
MILP boxes, `19.2%` less subprocess time, and `24.6%` lower wall-clock time.
This is single-window diagnostic evidence against a fixed short ranker cap and
for an adaptive ranker/audit budget controller, not yet a broad main claim.

During this rerun, an initial seed3 180-second attempt accidentally used the
full unique-order XML instead of the calibrated test-split XML. That run was
discarded and is not included in any main evidence. The accepted seed3
ranker180 rerun matches the test-split protocol: for seed3:test[0,100), the
raw exact summary has `mean_order_volume=35630.33`,
`milp_validated_candidates=8580`, and exact PF `1.9780321464`, matching the
prior test-split seed3 exact baseline.

Earlier mixed-cap evidence covers fifteen 100-order OR2023 test windows across
seed1, seed2, and seed3 initial conditions. It is useful as diagnostic evidence
but should not be used as the current main table because seed3 used a longer
900-second ranker cap.

Configuration:

- dataset: OR2023 test split windows;
- window size: 100 orders;
- seeds: 1, 2, 3 initial conditions;
- schedule: `0.25:1000`;
- K: 10;
- ranker cap: seed1 and seed2 use earlier 180-second assignment-aware runs;
  seed3 uses the certified 900-second protocol;
- final quality: exact staged-greedy audit from the ranker result.

Mixed-cap 15-window result:

- PF matched/improved/regressed: 14/1/0.
- Mean PF delta: `-0.0004147535`.
- Max PF regression: `0.0000000000`.
- Max PF improvement: `-0.0062213021`.
- Minimum exact coverage: `1.0000`.
- Minimum ranker-audit coverage: `1.0000`.
- Total exact uncovered orders: `0`.
- Total ranker-audit uncovered orders: `0`.

Mixed-cap aggregate cost reductions:

| metric | aggregate reduction | descriptive bootstrap 95% CI | windows reduced/tied/increased | sign-test p |
| --- | ---: | ---: | ---: | ---: |
| validations | 26.02% | [16.31%, 37.13%] | 15/0/0 | 0.0001 |
| uncached boxes | 11.71% | [7.30%, 16.79%] | 15/0/0 | 0.0001 |
| subprocess seconds | 12.35% | [7.62%, 17.50%] | 15/0/0 | 0.0001 |
| elapsed seconds | 13.93% | [8.86%, 19.62%] | 15/0/0 | 0.0001 |

The confidence intervals are descriptive window-bootstrap intervals, not formal
iid guarantees.

The stricter claim audit in
`CERTIFIED_RANKER_AUDIT_CLAIM_AUDIT_20260705.md` passes with the following
gates: at least 15 paired windows, complete PF deltas, no post-audit PF
regression, exact and ranker-audit feasibility, positive aggregate reductions
for all four cost metrics, no per-window cost increases, positive descriptive
bootstrap lower bounds, and sign-test p-values below 0.05 for the four cost
metrics.

The earlier expanded mixed-cap evidence covers twenty 100-order OR2023 test
windows after adding seed4 with the same `K=10`, `0.25:1000` exact staged
baseline, frontier budgets, safety policy, and exact-audit requirement. It
mixes the 900-second seed3 diagnostic with 180-second seed1, seed2, and seed4
runs, so it should not be the main table.

Mixed-cap 20-window result:

- PF matched/improved/regressed: 19/1/0.
- Mean PF delta: `-0.0003110651`.
- Max PF regression: `0.0000000000`.
- Max PF improvement: `-0.0062213021`.
- Minimum exact coverage: `1.0000`.
- Minimum ranker-audit coverage: `1.0000`.
- Total exact uncovered orders: `0`.
- Total ranker-audit uncovered orders: `0`.

Mixed-cap aggregate cost reductions:

| metric | aggregate reduction | descriptive bootstrap 95% CI | windows reduced/tied/increased | sign-test p |
| --- | ---: | ---: | ---: | ---: |
| validations | 22.49% | [14.52%, 31.62%] | 20/0/0 | 0.0000 |
| uncached boxes | 9.89% | [6.14%, 14.09%] | 19/1/0 | 0.0000 |
| subprocess seconds | 10.56% | [6.63%, 14.99%] | 19/0/1 | 0.0000 |
| elapsed seconds | 11.87% | [7.58%, 16.69%] | 20/0/0 | 0.0000 |

The mixed-cap strict audit in
`ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_STRICT_CLAIM_AUDIT_20260706.md`
fails only because it requires zero per-window increases for every cost metric:
seed4:test[100,200) has a `0.5%` subprocess-time increase and an uncached-box
tie while still matching PF and keeping full coverage. The aggregate claim audit
in
`ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_AGGREGATE_CLAIM_AUDIT_20260706.md`
passes when the paper claim is stated as no certified PF regression, complete
coverage, and positive aggregate cost reductions with at most one per-metric
window increase.

The anytime trace replay in
`RANKER_AUDIT_ANYTIME_BUDGET_ANALYSIS_20260705.md` gives the mechanism-level
evidence. At 25%, 50%, and 75% of each exact baseline's validation budget,
ranker-audit has lower PF in all 15 windows. It reaches the exact baseline's
final PF with 152,360 validations versus 209,580 validations, a 27.30%
aggregate reduction, with reductions in all 15 windows.

## Negative And Diagnostic Evidence

Ranker-only should not be the main optimizer. Earlier ranker-only runs can stop
by time limit and leave exact audit to recover quality. This is useful evidence
for the paper because it motivates the certified method design.

Widening the frontier is not currently useful. On seed3:test[400,500), the
`20,50,100,200` frontier reaches the same certified final PF as the main
`20,30,40,50` frontier, but adds 1410 validations, 28 uncached box queries,
21.7660 subprocess seconds, and 34.5722 wall-clock seconds.

The broad `all_expansions` safety policy is also not useful. On the same
window, it reaches the same certified final PF as the main safety policy, but
adds 7181 validations, 361 uncached box queries, 209.8493 subprocess seconds,
and 338.1259 wall-clock seconds. It is also slower than exact staged on the
main runtime metrics for that window.

The narrower `targeted_expansion_capture` safety policy is also not a clear
replacement for safety `none` on the same window. With cap=1 it reaches the
same certified final PF and is time-comparable, but it uses one additional
uncached MILP box query and 94 additional validations. With cap=3 it is worse
than safety `none` on all main cost metrics.

Fixed shorter ranker time budgets were tested on the heavy seed3:test[400,500)
diagnostic. A 600-second cap reaches the same certified final PF as the
900-second diagnostic and is 12.4017 seconds faster wall-clock, but uses 13
more uncached box queries, 1320 more validations, and 0.9781 more subprocess
seconds. A 300-second cap is worse than the 900-second diagnostic on every main
cost metric.

These ablations support keeping safety `none` and treating ranker budget as an
adaptive control problem. Simply adding protected candidates or choosing a
single fixed shorter ranker cap is not the strongest next direction.

## Defensible Claims

The current evidence supports the following conservative claims:

1. In the evaluated OR2023 paired windows, certified learned candidate ordering
   preserves or improves exact staged final PF after exact audit.
2. In the same paired windows, coverage remains complete and uncovered orders
   remain zero.
3. In the cap-unified 20-window ranker180 test-split evidence, the certified
   ranker-audit path has no PF regression and reduces aggregate oracle cost:
   `11.12%` fewer validations, `4.84%` fewer uncached MILP box queries,
   `5.57%` less subprocess time, and `5.87%` lower wall-clock time.
4. The current main evidence should not be worded as "every cost metric
   decreases in every window." Most windows reduce every main cost metric, but
   seed3:test[400,500) trades extra validations, uncached boxes, and wall-clock
   time for a better PF, and seed4:test[100,200) has a small subprocess-time
   increase with unchanged PF.
5. Exact audit is essential: the paper should not claim that the learned ranker
   alone is a reliable replacement for exact optimization.
6. A fixed short ranker cap is not automatically best. The seed3:test[400,500)
   comparison between 180-second and 900-second ranker diagnostics indicates
   that hard windows can benefit from spending more ranker-side effort before
   exact audit.

## Non-Claims

Do not claim the following from current evidence:

- global optimality of the selected box set;
- strict reproduction of the original private-data paper baseline;
- full-OR2023 superiority until the larger run is completed;
- that ML replaces MILP feasibility;
- that the bootstrap intervals are formal iid statistical guarantees;
- that the current wall-clock reductions would transfer unchanged to an
  optimized native oracle integration.

## Next Method Iteration

The next scientifically useful method direction is an adaptive ranker-audit
controller. The principle is to keep exact certification, but use trace signals
to decide whether another ranker iteration is likely to reduce later exact
audit cost enough to justify its own oracle work.

Candidate signals for audit control:

- ranker score margin between the top accepted move and nearby alternatives;
- disagreement between ranker score and simple geometric/PF improvement
  features;
- candidate move type, especially expansion or dimension-balanced changes;
- no-improvement streak length in the ranker phase;
- number of candidates that the ranker rejected before the accepted move;
- cache-hit and uncached-query profile in the current window.

Trace-calibrated audit control should be tested against five controls:

1. Main certified ranker-audit: `20,30,40,50`, safety `none`.
2. Wide frontier: `20,50,100,200`.
3. Broad safety: `all_expansions`.
4. Targeted safety: `targeted_expansion_capture`.
5. Fixed shorter ranker caps: `300` and `600` seconds.

Acceptance rule:

- no PF regression after exact audit;
- coverage `1.0000`;
- zero uncovered orders;
- lower uncached boxes and lower subprocess seconds than the main ranker-audit
  policy on paired windows, or a clear quality improvement at comparable cost.

The handoff trace evidence in
`RANKER_HANDOFF_TRACE_ANALYSIS_20260705.md` should guide this controller. In
the 14-window trace table, every analyzed run has its last ranker improvement
one iteration before the stopping row. This rules out simple patience as the
main adaptive mechanism and points to a marginal-value controller: keep running
the ranker only while its expected reduction in subsequent exact-audit burden
is larger than its own expected oracle cost.

The first implementation of that controller is documented in
`ADAPTIVE_RANKER_HANDOFF_CONTROLLER_20260705.md`. It is experimental and
disabled by default; any paper claim still requires paired exact-audit results.
The current threshold evidence is single-window only. Relative to the paired
900-second no-handoff diagnostic, `3e-6` preserves audited quality and reduces
subprocess and wall-clock time, but it ties uncached MILP queries and increases
validations. Relative to the cap-unified fixed-180 main policy on the same
hard window, it preserves audited quality and reduces all four tracked cost
metrics. It should remain an ablation candidate until broader paired-window
evidence exists.
The 14-window trace simulation shows the tested thresholds would trigger only
on that heavy window, so the current controller is not yet a general paper
claim.

This next iteration would strengthen the paper because it directly targets the
remaining weakness in the cap-unified ranker180 result: exact audit is retained
as the certification mechanism, but the ranker/audit handoff is chosen by
expected incremental oracle cost rather than by a fixed time cap or by
validating more candidates during the ranker stage.

## Evidence Map

- Mixed-cap 15-window summary:
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_CERTIFIED_SUMMARY_20260705.md`
- Mixed-cap 15-window statistical summary:
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_CERTIFIED_STATS_20260705.md`
- Mixed-cap 15-window claim audit:
  `BoxDesignSurrogateRL/docs/CERTIFIED_RANKER_AUDIT_CLAIM_AUDIT_20260705.md`
- Mixed-cap 15-window anytime budget analysis:
  `BoxDesignSurrogateRL/docs/RANKER_AUDIT_ANYTIME_BUDGET_ANALYSIS_20260705.md`
- Seed3 ranker180 test-split cap-unified summary:
  `BoxDesignSurrogateRL/docs/SEED3_RANKER180_TESTSPLIT_CAP_UNIFIED_SUMMARY_20260706.md`
- Seed3 ranker180 test-split cap-unified statistical summary:
  `BoxDesignSurrogateRL/docs/SEED3_RANKER180_TESTSPLIT_CAP_UNIFIED_STATS_20260706.md`
- Cap-unified 20-window ranker180 test-split summary:
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_SUMMARY_20260706.md`
- Cap-unified 20-window ranker180 test-split statistical summary:
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_STATS_20260706.md`
- Cap-unified 20-window ranker180 strict claim audit:
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_STRICT_CLAIM_AUDIT_20260706.md`
- Cap-unified 20-window ranker180 aggregate claim audit:
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_AGGREGATE_CLAIM_AUDIT_20260706.md`
- Seed4 extension summary:
  `BoxDesignSurrogateRL/docs/SEED4_RANKER180_EXTENSION_SUMMARY_20260706.md`
- Seed4 extension statistical summary:
  `BoxDesignSurrogateRL/docs/SEED4_RANKER180_EXTENSION_STATS_20260706.md`
- Combined 20-window summary:
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_COMBINED_SUMMARY_20260706.md`
- Combined 20-window statistical summary:
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_COMBINED_STATS_20260706.md`
- Combined 20-window strict claim audit:
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_STRICT_CLAIM_AUDIT_20260706.md`
- Combined 20-window aggregate claim audit:
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_AGGREGATE_CLAIM_AUDIT_20260706.md`
- Seed3 certified summary:
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED3_CERTIFIED_SUMMARY_20260705.md`
- Certified method direction:
  `BoxDesignSurrogateRL/docs/CERTIFIED_RANKER_AUDIT_DIRECTION_20260705.md`
- Frontier budget ablation:
  `BoxDesignSurrogateRL/docs/RANKER_FRONTIER_BUDGET_ABLATION_SEED3_O400_20260705.md`
- Safety policy ablation:
  `BoxDesignSurrogateRL/docs/RANKER_SAFETY_POLICY_ABLATION_SEED3_O400_20260705.md`
- Targeted capture safety ablation:
  `BoxDesignSurrogateRL/docs/RANKER_TARGETED_CAPTURE_SAFETY_ABLATION_SEED3_O400_20260705.md`
- Ranker time-budget ablation:
  `BoxDesignSurrogateRL/docs/RANKER_TIME_BUDGET_ABLATION_SEED3_O400_20260705.md`
- Adaptive ranker budget comparison:
  `BoxDesignSurrogateRL/docs/ADAPTIVE_RANKER_BUDGET_SEED3_O400_COMPARISON_20260706.md`
- Algorithm/protocol documentation:
  `BoxDesignSurrogateRL/docs/MILP_BOX_ALGORITHMS.md`
  `BoxDesignSurrogateRL/docs/EXPERIMENT_PROTOCOL.md`

## Claim Audit

| Claim | Evidence | Status | Caveat |
| --- | --- | --- | --- |
| Ranker-audit has no certified PF regression on current paired windows. | Cap-unified 20-window ranker180 aggregate audit, summary, and stats. | Supported for current windows. | Not yet full OR2023. |
| Ranker-audit reduces aggregate uncached MILP box queries. | Cap-unified 20-window ranker180 aggregate audit, summary, and stats. | Supported for current windows. | One window increases and one ties on uncached boxes; uses current cache and subprocess implementation. |
| Ranker-audit reduces aggregate wall-clock time. | Cap-unified 20-window ranker180 aggregate audit, summary, and stats. | Supported for current windows. | One window increases wall-clock because it reaches a better PF; hardware and process-launch overhead should be reported in appendix. |
| Ranker-audit reduces every cost metric in every evaluated window. | Cap-unified 20-window ranker180 strict audit. | Unsupported as a general claim. | Fails because hard windows can trade extra oracle work for equal or better PF. |
| Mixed-cap ranker-audit has larger aggregate reductions. | Mixed-cap 20-window summary and stats. | Diagnostic only. | Not a main claim because seed3 uses a 900-second ranker cap while other seeds use 180 seconds. |
| Ranker-audit reaches better anytime PF under partial exact-validation budgets. | 15-window anytime budget analysis. | Supported for current windows. | Trace replay uses completed-run logs, not interrupted live runs. |
| Exact audit is required. | Ranker-only diagnostics and certified direction doc. | Supported as method rationale. | Need concise main-text wording. |
| Wide frontier is not a better default. | Seed3 heavy-window frontier ablation. | Supported as focused ablation. | Single heavy window, not broad proof. |
| Broad expansion safety is not a better default. | Seed3 heavy-window safety ablation. | Supported as focused ablation. | Single heavy window, not broad proof. |
| Targeted capture safety is not a better default. | Seed3 heavy-window targeted capture ablation. | Supported as focused ablation. | Single heavy window; does not rule out audit-control variants. |
| Fixed shorter ranker caps are not a better default. | Seed3 heavy-window time-budget ablation. | Supported as focused ablation. | Single heavy window; adaptive handoff remains open. |
| Adaptive budget control can repair the fixed-180 hard-window cost increase. | Seed3:test[400,500) adaptive budget comparison. | Supported as focused diagnostic. | Single hard window; not yet broad main evidence. |
| Original paper uses an identical baseline. | Not established in current evidence. | Unsupported. | Requires paper/data/protocol verification. |

## Same-Agent AC-Style Review

Decision: revise before paper submission, usable as an internal claim package.

Score estimate: 6/10 as paper-ready evidence, 8/10 as an experiment-direction
memo.

Major risks:

- The current empirical evidence covers 20 paired 100-order windows,
  not the full OR2023 dataset.
- The method claim must stay centered on certified candidate ordering. A claim
  that ML replaces MILP would be misleading.
- The current cap-unified evidence does not support saying every cost metric
  decreases in every window; it supports no PF regression, full coverage, and
  positive aggregate reductions.
- The fixed 180-second cap is scientifically useful as a controlled protocol
  but is not necessarily the best method policy on hard windows.
- Runtime reductions are partly implementation-dependent because the current
  oracle uses Java/Gurobi subprocess calls.
- Literature claims about the original baseline and prior box-sizing work still
  need verified citations.

Required revisions before main-text use:

- Add a concise formal definition of PF, coverage, and uncached oracle work.
- Verify and cite the original paper and OR2023 data source.
- Add hardware and oracle implementation details in appendix.
- Run a larger paired evaluation or clearly label the current result as a
  multi-window pilot.
