# Counterfactual Branch-and-Follow Pilot (2026-07-12)

## Purpose

Test whether two nearly tied actions at the same fixed-step state have different
long-horizon outcomes. This is a training-window protocol pilot, not a held-out
test result.

The state is iteration 1 of the OR2023 training-order window `[0, 100)`, with
`K=10` and fixed step `0.25`. Each branch forcibly executes one independently
MILP-evaluated action and then uses the same frozen ranker plus exact MILP
verification until its no-improvement audit converges.

## Branches

| Branch | Forced action | One-step PF | Terminal PF | Coverage | Uncached boxes | Subprocess | Wall |
|---|---|---:|---:|---:|---:|---:|---:|
| Immediate rank 1 | box 4 height `-0.25` | 2.0318205874 | 1.8025812361 | 100% | 1030 | 457.45 s | 586.38 s |
| Immediate rank 2 | box 7 height `-0.25` | 2.0318952986 | 1.8025812361 | 100% | 1034 | 456.39 s | 601.14 s |

Both branches ran 286 continuation iterations. Their terminal `best_boxes.json`
files are byte-identical with SHA256
`5F841FE2B58A1C0D98F958F496B015A5929F31F4CC413AB79558EAE816864821`.

## Interpretation

The two distinct early actions commute under the later fixed-step coordinate
search: their trajectories merge into the same terminal box set. A long-horizon
policy cannot improve terminal PF by distinguishing this pair because no
terminal quality difference exists. The lower-ranked branch only incurs four
additional uncached box queries and 14.76 seconds more wall time.

This pilot therefore rejects fixed-step coordinate reordering as the primary RL
decision problem. The next counterfactual experiment should expose a
higher-level action with real path consequences: choose the first move scale
from `{2, 1, 0.5, 0.25}`, let the same ranker choose the coordinate at that
scale, and then run the same multiscale continuation. Terminal PF and exact
query cost jointly define the high-level action value.

## Provenance

- Infrastructure commits: `8ea62ea`, `55d4b6c`
- Rank-1 run: `counterfactual_branch_pilot_20260712/ranker_filtered_greedy/run_20260712_160841_675883`
- Rank-2 run: `counterfactual_branch_pilot_20260712/ranker_filtered_greedy/run_20260712_161845_713503`
