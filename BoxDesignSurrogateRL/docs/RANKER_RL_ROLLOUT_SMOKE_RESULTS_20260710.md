# Ranker + RL Rollout Smoke Results

Date: 2026-07-10

## Scope

This is a paired smoke test, not a final result. All four runs use the same:

- frozen OR2023 test split, orders `[0, 100)`;
- `K=10`, seed `1`, and exact run `initial_boxes.json`;
- exact Java/Gurobi MILP feasibility oracle;
- `0.25:50` search schedule and geometric coverage repair.

All runs finished with 100% coverage and zero uncovered or unknown orders.

## Results

| Method | PF | MILP validations | Rollout validations | Uncached boxes | Subprocess s | Wall s |
|---|---:|---:|---:|---:|---:|---:|
| Exact staged | 1.993125 | 3000 | 0 | 347 | 148.69 | 193.82 |
| Ranker-only | **1.986088** | 1000 | 0 | **275** | 111.04 | **133.03** |
| Ranker + legacy RL transfer | 1.986956 | 1090 | 90 | 276 | **110.88** | 135.33 |
| Ranker + matched-step train-only RL | **1.986088** | 1291 | 291 | 306 | 134.13 | 174.39 |

Relative to exact staged, ranker-only obtains 0.353% lower PF while reducing:

- candidate validations by 66.7%;
- uncached box queries by 20.7%;
- Java/Gurobi subprocess time by 25.3%;
- wall-clock time by 31.4%.

## RL Ablation

The legacy checkpoint was trained in `paper` mode on the full OR2023 data with
step `0.5`, then transferred to a `0.25` exact-MILP test. It differs from the
ranker-only action sequence only at iteration 43 and produces a slightly worse
final PF. This run has both step-size and feasibility-oracle distribution shift,
and the full-data training source overlaps the frozen test population. It is
diagnostic only and is not a publishable held-out result.

The matched checkpoint was trained only on the first 500 orders of the frozen
train split, uses step `0.25`, and carries its training XML hash and observation
normalization scale in the checkpoint. Its exact action sequence is identical
to ranker-only. It therefore gives no independent RL quality gain, while adding
291 exact rollout evaluations.

## Conclusion

This smoke supports the learned ranker as an effective exact-oracle budget
controller. It does not yet support a claim that PPO rollout improves the box
design policy. The matched policy's surrogate training curve was unstable
because surrogate uncovered-order penalties dominated PF rewards; increasing
episodes without changing that objective is not justified.

The next scientifically valid step is to fix the RL training signal using only
the frozen train split, select hyperparameters on dev, and return to test only
after the policy changes candidate choices on dev. In parallel, the exact and
ranker baselines must be run beyond 50 iterations because every iteration in
this smoke still improved PF; these runs do not establish convergence.

## Convergence Follow-up

The exact and ranker-only paths were continued from their respective 50-step
checkpoints with step `0.25` and a 200-iteration cap. Both stopped before the
cap. Because the first ranker stop did not inspect the final ten candidates, a
separate run enabled `ranker_noop_fallback` to exact-audit the terminal state.
The audit found four additional improving moves and then certified a noop.

| Full path | Final PF | Validations | Uncached boxes | Subprocess s | Wall s |
|---|---:|---:|---:|---:|---:|
| Exact to local convergence | **1.890870** | 10620 | 1042 | 410.38 | 645.15 |
| Ranker + terminal exact audit | **1.890870** | **6300** | **938** | **357.68** | **515.51** |

The final `best_boxes.json` files are byte-identical. Relative to exact, the
audited ranker path reduces candidate validations by 40.7%, uncached box queries
by 10.0%, Java/Gurobi subprocess time by 12.8%, and wall-clock time by 20.1%.

This is the strongest result from the current experiment: learned candidate
ordering plus a terminal exact audit preserves the exact local-search endpoint
at lower oracle cost. It remains a ranker result, not evidence of an independent
RL gain, and must be repeated across held-out windows before a general claim.

## Artifacts

- Exact: `results/rl_rollout_smoke_20260710/staged_greedy/run_20260710_181518_008013`
- Ranker-only: `results/rl_rollout_smoke_20260710/ranker_filtered_greedy/run_20260710_182159_581400`
- Legacy transfer: `results/rl_rollout_smoke_20260710/ranker_policy_rollout_greedy/run_20260710_181913_153236`
- Matched train-only: `results/rl_rollout_smoke_20260710/ranker_policy_rollout_greedy/run_20260710_183911_165510`
- Exact continuation: `results/convergence_continuation_20260710/staged_greedy/run_20260710_184502_682970`
- Ranker continuation: `results/convergence_continuation_20260710/ranker_filtered_greedy/run_20260710_185256_865978`
- Ranker terminal audit: `results/convergence_continuation_20260710/ranker_filtered_greedy/run_20260710_185914_691679`
