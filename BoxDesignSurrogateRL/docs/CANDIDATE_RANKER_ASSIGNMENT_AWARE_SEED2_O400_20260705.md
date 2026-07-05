# Assignment-Aware Candidate Ranker Seed2 Offset400, 2026-07-05

## Purpose

Turn the seed2:test[400,500) failure diagnosis into a model-side fix rather
than a broad safety fallback. The original ranker missed the exact first move:

```text
5:height:+0.250000
```

That move was predicted rank 48/60 by the previous ranker, causing the
adaptive top10 policy to accept smaller improvements before seeing the true
best expansion.

## Method Change

Candidate feature rows now include assignment-aware features computed from the
current exact MILP assignment and candidate geometry:

- orders currently assigned to the moved box;
- their total and mean order volume;
- orders assigned to larger boxes that the candidate moved box could
  aggregate-fit;
- aggregate volume delta between the current larger assigned box and the
  candidate moved box;
- shrink at-risk orders currently assigned to the moved box.

These features are available at both training and inference because they use
only the current exact score, current boxes, candidate boxes, and order
geometry. They do not use candidate exact MILP labels.

I also added:

```text
scripts/augment_candidate_trace_assignment_features.py
```

This script upgrades older exact candidate traces by re-evaluating each step
group's current box set once to recover current assignments, avoiding a full
candidate-level MILP rerun.

## Training And Probe

Assignment-aware training traces:

```text
results/candidate_traces_assignment_20260705/dev300_train_repaired_exact05_from_kmeans_assignment.csv
results/candidate_traces_assignment_20260705/dev300_train_repaired_exact025_from_05_assignment.csv
```

Trace augmentation cost:

| trace | step groups | rows | elapsed s | uncached boxes | subprocess s |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev300 exact0.25 from 0.5 | 8 | 480 | 32.7502 | 17 | 31.6258 |
| dev300 exact0.5 from kmeans | 120 | 7200 | 173.0480 | 119 | 157.2389 |

Selected model:

```text
results/candidate_ranker_assignment_aware_20260705/candidate_ranker_20260705_084240/candidate_ranker.joblib
```

Configuration:

- model: HGBT
- target: accepted-move classifier
- training rows/groups: 7680 rows / 128 groups
- eval probe: `seed2:test[400,500)` first exact step

The exact-best failure move improves from predicted rank 48/60 with the old
ranker to rank 14/60 with the assignment-aware HGBT accepted classifier.

## Focused MILP Verification

Live focused run:

```text
results/ranker_assignment_aware_seed2_o400_20260705/frontier_20260705_084308
```

The live run uses:

- ranker budget sequence: `20,30,40,50`
- ranker wall-clock budget: 180s
- exact audit after ranker phase
- no expansion safety fallback

| method | final PF | PF gap vs exact | coverage | validations | uncached boxes | subprocess s | elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact staged baseline | 1.9929555750 | 0.0000000000 | 1.000 | 23160 | 2063 | 1043.8999 | 1425.8324 |
| original ranker+audit | 2.0025546412 | 0.0095990662 | 1.000 | 21510 | 2016 | 1012.4072 | 1432.5585 |
| all-expansions safety | 1.9929555750 | 0.0000000000 | 1.000 | 22371 | 2057 | 1032.0711 | 1456.2263 |
| assignment-aware HGBT + audit | 1.9929555750 | 0.0000000000 | 1.000 | 20920 | 1934 | 972.2854 | 1391.4455 |

Relative to exact staged, assignment-aware HGBT + audit:

- preserves exact PF and full coverage;
- reduces validations by 9.7%;
- reduces uncached boxes by 6.3%;
- reduces Java/Gurobi subprocess time by 6.9%;
- reduces wall-clock time by 2.4%.

Relative to the original ranker+audit, it both fixes the PF regression and
reduces measured oracle work and wall-clock time.

## Interpretation

This is a focused repair of the known seed2 regression window, not yet a broad
multi-window claim. It is stronger than the all-expansions safety fallback
because it fixes quality without validating every expansion move and produces
a net wall-clock reduction on the failure window.

The next required check is to run the same assignment-aware HGBT configuration
on the full seed1+seed2 paired-window protocol. The paper-facing claim should
not be upgraded until that broader check confirms that the fix does not damage
the windows where the original ranker already matched or improved exact PF.
