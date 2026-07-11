# Multiscale Ranker Development Gate, 2026-07-11

## Motivation

On OR2023 train `[200,300)`, all converged fixed-0.25 methods accepted the
same multiset of 354 improving coordinate actions and reached byte-identical
boxes. Candidate ordering alone could not shorten that path.

This gate expands each state from 60 actions to 240 actions:

```text
10 boxes x 3 dimensions x 2 directions x {2.0, 1.0, 0.5, 0.25}
```

Every accepted action is still evaluated by the Java/Gurobi MILP oracle.

## Protocol

- Data: OR2023 train split orders `[200,300)`; this is a development window,
  not the untouched test set.
- K: 10; seed: 1; `label_6ori`; deterministic geometric coverage repair.
- Shared initial boxes across all methods.
- Ranker: historical assignment-aware HGBT trained on repaired 300-order
  development traces at 0.5/0.25 steps. It was not retrained on 1.0/2.0
  candidates for this gate.
- Multiscale ranker tiers: `10 -> 30 -> 60 -> full 240` on no improvement.
- The 600-second first segment was resumed with iteration context preserved;
  the second segment stopped naturally.

This method is supervised learned search, not reinforcement learning. It is a
new baseline that a subsequent RL policy must beat.

## Results

| method | status | PF | coverage | rounds | validations | uncached boxes | subprocess s | wall s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fine exact, fixed 0.25 | converged | 2.006760 | 100% | 355 | 21,300 | 1,963 | 840.91 | 1,219.42 |
| fine ranker `10->30->full` | converged | 2.006760 | 100% | 355 | 6,920 | 1,452 | 644.28 | 855.41 |
| global exact `0.5 stop -> 0.25 stop` | converged | 2.013791 | 100% | 184 | 11,040 | 1,085 | 501.46 | 662.48 |
| exact multiscale | 600 s cap | 2.147520 | 100% | 14 | 3,360 | 595 | 561.65 | 610.54 |
| multiscale ranker | converged | **1.957672** | 100% | **116** | **6,090** | **1,181** | 710.47 | **827.25** |

Against converged fine exact, multiscale ranker improves PF by 2.45% while
reducing validations by 71.4%, uncached boxes by 39.8%, subprocess time by
15.5%, and wall time by 32.2%.

Against converged fine ranker, it improves PF by 2.45%, reduces validations by
12.0%, uncached boxes by 18.7%, and wall time by 3.3%. Subprocess time is 10.3%
higher because wide fallback tiers create expensive Java/Gurobi batches.

Against global staged exact, it improves PF by 2.79% and uses 44.8% fewer
validations, but spends 8.8% more uncached boxes, 41.7% more subprocess time,
and 24.9% more wall time. This is a quality/cost tradeoff, not domination.

## Interpretation

The result validates the richer action-space hypothesis: learned filtering
makes 240-action search practical, and state-local access to multiple step
sizes reaches a better basin than both fine-only and globally staged exact
search on this window.

It does not establish an RL contribution. The assignment-aware ranker alone
explains the result. The remaining cost is concentrated in late wide/full
fallbacks, which is the next target for a sequential policy that jointly
chooses move magnitude and verification depth.

## Independent Development Replication

After freezing the action steps and ranker tiers above, the same configuration
was run on the non-overlapping OR2023 train `[300,400)` window. Both methods
were resumed after matched 600-second checkpoints and then stopped naturally.

| method | terminal PF | coverage | rounds | validations | uncached boxes | subprocess s | wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fine exact, fixed 0.25 | 1.832082 | 100% | 298 | 17,880 | 1,728 | 698.40 | 1,052.83 |
| multiscale ranker | **1.771564** | 100% | **74** | **4,240** | **1,072** | 757.11 | **893.58** |

On this second window, multiscale ranker improves PF by 3.30%, reduces rounds
by 75.2%, validations by 76.3%, uncached boxes by 38.0%, and wall time by
15.1%. Subprocess time increases by 8.4%, again identifying wide/full fallback
batches as the remaining systems bottleneck.

The result now has two development-window replications, but this is still too
small for a final statistical claim and still does not establish an RL gain.

The frozen configuration was then evaluated on a third non-overlapping window,
OR2023 train `[400,500)`:

| method | terminal PF | coverage | rounds | validations | uncached boxes | subprocess s | wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fine exact, fixed 0.25 | 2.114560 | 100% | 304 | 18,240 | 1,741 | 587.23 | 852.12 |
| multiscale ranker | **1.937439** | 100% | **87** | **7,240** | **1,131** | **428.91** | **554.80** |

Here multiscale ranker improves PF by 8.38%, reduces rounds by 71.4%,
validations by 60.3%, uncached boxes by 35.0%, subprocess time by 27.0%, and
wall time by 34.9%.

Across the three converged development windows `[200,500)`:

| aggregate metric | fine exact | multiscale ranker | change |
| --- | ---: | ---: | ---: |
| mean PF | 1.984467 | **1.888892** | **-4.82%** |
| coverage | 100% on 3/3 | 100% on 3/3 | tied |
| rounds | 957 | **277** | **-71.1%** |
| validations | 57,420 | **17,570** | **-69.4%** |
| uncached boxes | 5,432 | **3,384** | **-37.7%** |
| subprocess seconds | 2,126.54 | **1,896.49** | **-10.8%** |
| wall seconds | 3,124.38 | **2,275.63** | **-27.2%** |

These are paired descriptive development results. Three windows are not enough
for inferential uncertainty or a final generalization claim.

## Acceptance State

- Passed: three-window dev comparison against converged fine exact baselines.
- In progress: broader replication and an RL ablation against this frozen
  supervised baseline.
- Not passed: RL ablation against the multiscale supervised ranker.
- Not run: untouched test set.

## Artifacts

- Global staged exact:
  `results/adaptive_step_dev_o200_20260711/staged_greedy/run_20260711_171337_357114`
- Exact multiscale 600-second diagnostic:
  `results/adaptive_step_dev_o200_20260711/multiscale_greedy/run_20260711_181738_319621`
- Multiscale ranker first segment:
  `results/adaptive_step_dev_o200_20260711/multiscale_ranker_greedy/run_20260711_182810_954151`
- Multiscale ranker continuation:
  `results/adaptive_step_dev_o200_20260711/multiscale_ranker_greedy/run_20260711_183844_918692`
- Implementation commit: `cd38360`.
- Independent replication root:
  `results/adaptive_step_dev_replication_20260711`.
