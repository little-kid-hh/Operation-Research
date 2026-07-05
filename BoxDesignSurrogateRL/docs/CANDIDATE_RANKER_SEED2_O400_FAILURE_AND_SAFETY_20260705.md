# Candidate Ranker Seed2 Offset400 Failure Analysis And Safety Check, 2026-07-05

## Purpose

Diagnose the only seed1+seed2 paired-window PF regression:
`seed2:test[400,500)`.

Baseline exact staged PF is `1.9929555750`. The original ranker+audit result
preserved coverage but ended at PF `2.0025546412`, a regression of
`0.0095990662`.

## Failure Mechanism

The first exact local-search move is:

```text
5:height:+0.250000
```

This expansion lowers PF from `2.2542956315` to `2.2319070382`, because it
allows orders to be assigned to a smaller box rather than a larger box. The
candidate ranker assigned this exact-best move predicted rank 48 out of 60.
The adaptive top-k policy stopped after top10 because top10 already contained
smaller MILP-verified improvements. Therefore the ranker never evaluated the
true best first move.

The final exact-audit solution differs from exact baseline mainly in box 5:

```text
exact box 5 height: 26.2124705093
ranker+audit box 5 height: 25.2124705093
```

The smaller box has lower raw volume but causes a worse assignment pattern,
raising PF. This is a path-dependence failure, not an infeasibility failure:
coverage remains 100%.

## Expansion Safety Variant

I added an explicit `--ranker-safety-policy all_expansions` option. It keeps
ranker sorting but always adds expansion moves to the exact MILP audit set for
each tier. This is a robustness variant motivated by the observation that
expansions can reduce PF by improving assignment choices.

Two focused runs on `seed2:test[400,500)`:

| config | final PF | PF gap vs exact | coverage | validations | uncached boxes | subprocess s | elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact staged baseline | 1.9929555750 | 0.0000000000 | 1.000 | 23160 | 2063 | 1043.8999 | 1425.8324 |
| original ranker+audit, 180s | 2.0025546412 | 0.0095990662 | 1.000 | 21510 | 2016 | 1012.4072 | 1432.5585 |
| expansion-safety ranker+audit, 180s | 1.9929555750 | 0.0000000000 | 1.000 | 22371 | 2057 | 1032.0711 | 1456.2263 |
| expansion-safety ranker+audit, 60s | 1.9929555750 | 0.0000000000 | 1.000 | 23137 | 2061 | 1029.9482 | 1500.8812 |

## Interpretation

Expansion safety fixes the quality regression on this known failure window:
both 180s and 60s variants recover exact PF and full coverage.

It is not a wall-clock win on this window. The 180s variant reduces validations
by 3.4%, uncached boxes by 0.3%, and Java/Gurobi subprocess time by 1.1%, but
wall-clock time increases by 2.1%. The 60s variant reduces subprocess time by
1.3% but increases wall-clock time by 5.3%.

This result should be reported as failure analysis and a robustness check, not
as the main acceleration method. The better next step is an assignment-aware
ranker feature set: the current ranker features do not include enough
information about current order-to-box assignments to detect expansions that
move orders from larger boxes into smaller boxes.

## Source Records

Remote result directories:

```text
BoxDesignSurrogateRL/results/ranker_safety_all_expansions_seed2_o400_20260705/frontier_20260705_073044
BoxDesignSurrogateRL/results/ranker_safety_all_expansions_seed2_o400_60s_20260705/frontier_20260705_075540
```

Diagnostic one-step candidate trace:

```text
BoxDesignSurrogateRL/results/ranker_failure_diag_20260705/seed2_o400_iter1_candidate_trace.csv
```
