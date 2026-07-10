# Multi-window RL Dev Gate

Date: 2026-07-10

## Protocol

The policy was trained only on the frozen OR2023 train split. The 1,500 train
orders were partitioned into 15 non-overlapping 100-order environments. PPO ran
for 150 episodes, so every environment was visited exactly ten times in a
seeded shuffled order.

Configuration:

- `K=10`, step `0.25`, horizon 50;
- `paper_pf_surrogate` proposal environment with `tau=0.5`;
- decomposed reward: uncovered-count delta plus PF delta;
- shared observation normalization scale saved in the checkpoint;
- no dev or test order was used for policy updates.

The lower surrogate threshold is used only to train a proposal policy. Every
reported box-search result below uses the exact Java/Gurobi MILP oracle.

## Surrogate Dev Screening

Deterministic policy rollout was run on five non-overlapping 100-order dev
windows. It improved PF on one initially covered window, repaired one uncovered
window, made no improvement on one covered window, and left one uncovered order
on each of two windows. This mixed result passed only the minimal gate for one
exact dev experiment; it is not evidence of generalization.

## Exact Dev Gate

All methods used dev orders `[0, 100)`, seed 1, the same exact-run
`initial_boxes.json`, `K=10`, and schedule `0.25:50`. Coverage was 100% and
uncovered/unknown counts were zero for every final result.

| Method | PF | MILP validations | Rollout validations | Uncached boxes | Subprocess s | Wall s |
|---|---:|---:|---:|---:|---:|---:|
| Exact staged | 2.472933 | 3000 | 0 | 582 | 317.82 | 406.07 |
| Ranker-only | **2.472933** | **1000** | 0 | **492** | **264.34** | **314.49** |
| Ranker + multi-window PPO rollout | **2.472933** | 2215 | 1215 | 531 | 293.05 | 373.75 |

Relative to exact, ranker-only preserved PF while reducing validations by
66.7%, uncached boxes by 15.5%, subprocess time by 16.8%, and wall time by
22.6%.

The RL rollout changed 12 of 50 selected actions relative to ranker-only, but
the changes were equivalent width/height moves early in the trajectory or a
temporarily worse ordering that caught up by iteration 48. Final PF and boxes
were not improved, while rollout added 1,215 exact validations and 59.3 seconds
of wall time over ranker-only.

## Decision

This RL variant fails the dev gate and must not be expanded to other dev or test
windows. More episodes with the same state are not justified. The current state
contains only `K x 3` box dimensions, so a shared policy cannot directly observe
which order distribution it is optimizing. The next variant should add a fixed,
train-derived order-distribution summary to the RL state, then repeat surrogate
dev screening before any exact MILP gate.

## Order-context Follow-up

An explicit learned-state variant appended eight normalized train-derived
statistics: mean and p90 of order maximum length/width/height, plus mean and p90
total order volume. The original paper state remains the default; the extended
checkpoint records `obs_dim=38` and the exact context schema.

With the same 15x100 train environments and 150-episode budget, deterministic
surrogate dev rollout produced:

| Dev offset | Initial uncovered | No-context best uncovered / PF | Context best uncovered / PF |
|---:|---:|---:|---:|
| 0 | 0 | 0 / 1.603867 | 0 / 1.608200 |
| 100 | 1 | **0 / 1.674458** | 1 / 1.582700 |
| 200 | 0 | 0 / 1.723306 | **0 / 1.712249** |
| 300 | 1 | 1 / **1.499941** | 1 / 1.502450 |
| 400 | 1 | 1 / 1.597798 | 1 / **1.596667** |

The context policy is better on PF in two windows but loses the no-context
policy's coverage repair at offset 100. More importantly, its deterministic
rollout collapses to the same repeated action 10 on all five windows. It fails
the surrogate dev gate, so no exact-MILP experiment was run for this variant.

This result weakens the direct-action PPO direction. The next RL formulation
should target sequential oracle-budget control (candidate tier, audit, or stop),
where the existing ranker already supplies useful candidate ordering and a
terminal exact audit can preserve solution quality.

Artifacts:

- Policy: `results/kandula_surrogate_policy_train15x100_step0p25_tau0p5_rewardfix_ep150_seed1/run_20260710_192245`
- Exact: `results/rl_rollout_dev_gate_20260710/staged_greedy/run_20260710_193018_351696`
- Ranker-only: `results/rl_rollout_dev_gate_20260710/ranker_filtered_greedy/run_20260710_193726_315675`
- RL rollout: `results/rl_rollout_dev_gate_20260710/ranker_policy_rollout_greedy/run_20260710_194311_046189`
