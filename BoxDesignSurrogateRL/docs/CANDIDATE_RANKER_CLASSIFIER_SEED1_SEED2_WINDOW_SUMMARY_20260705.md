# Candidate Classifier Ranker Seed1/Seed2 Window Summary, 2026-07-05

## Scope

This is a descriptive paired-window summary over the two complete
seed-specific initial-condition replicates:

- seed1:test[0,500), evaluated as five 100-order windows.
- seed2:test[0,500), evaluated as five 100-order windows.

This document records the previous RF accepted-move ranker. The later
assignment-aware HGBT result is in
`CANDIDATE_RANKER_ASSIGNMENT_AWARE_SEED1_SEED2_WINDOW_SUMMARY_20260705.md`.

All rows use K=10, `0.25:1000`, the Java/Gurobi MILP oracle with `label_6ori`,
`geometric_expand` coverage repair, and ranker budget sequence
`10,20,30,40,50` followed by exact staged audit.

This is not a formal statistical significance test. It is intended to make the
current evidence easier to audit before deciding whether to run additional
seeds or a larger/full OR2023 experiment.

## Aggregate Cost

| metric | exact staged | ranker+audit | reduction |
| --- | ---: | ---: | ---: |
| MILP validations | 136320 | 117580 | 13.7% |
| uncached boxes | 12780 | 12050 | 5.7% |
| Java/Gurobi subprocess seconds | 7004.1557 | 6564.6855 | 6.3% |
| wall-clock seconds | 9447.6642 | 8841.6164 | 6.4% |

## Quality And Coverage

Across 10 paired seed-specific windows:

- 7 windows match exact staged PF.
- 2 windows improve PF.
- 1 window is worse than exact staged PF.
- Mean PF delta is `-0.0015354652` by unweighted window average.
- Exact staged and ranker+audit both have 100% coverage in all 10 windows.

The single worse window is seed2:test[400,500), where PF increases by
`0.0095990662` and wall-clock time increases by 0.5%, while validations,
uncached boxes, and Java/Gurobi subprocess time still decrease.

## Interpretation

The current seed-specific evidence supports a query-efficiency claim: learned
candidate ranking with exact audit reduces exact MILP oracle work across
multiple initial conditions while preserving full coverage.

The current evidence does not support strict dominance. The paper-facing claim
should state that final PF is usually equal or better but can regress slightly
in an individual window, and should report PF, coverage, uncached oracle work,
subprocess time, and wall-clock time together.

## Source Records

- `BoxDesignSurrogateRL/docs/SEED1_OFFSET0_400_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/SEED1_OFFSET0_400_RANKER_RESULT_MANIFEST_20260705.json`
- `BoxDesignSurrogateRL/docs/SEED2_OFFSET0_400_RANKER_AUTO_SUMMARY_20260705.md`
- `BoxDesignSurrogateRL/docs/SEED2_OFFSET0_400_RANKER_RESULT_MANIFEST_20260705.json`
