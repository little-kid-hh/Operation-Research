# Multiscale PPO Candidate-Union Smoke Gate (2026-07-12)

## Question

Does a multiscale PPO policy add useful candidates beyond the frozen supervised
ranker's top 10, when every shortlisted candidate is still certified by the
same exact MILP oracle?

This is a development smoke test, not a test-set result. The evaluated window
contains 100 OR2023 unique orders at training-file offset 200. Both arms use
the same initial boxes, `K=10`, action steps `{2, 1, 0.5, 0.25}`, five search
iterations, coverage repair, and exact six-orientation MILP feasibility.

## Compared arms

- Ranker only: validate the ranker's top 10 actions per iteration.
- Ranker + PPO union: validate the union of the ranker's top 10 and PPO top 5
  actions per iteration. The PPO checkpoint was trained for 300 surrogate-mode
  episodes on a disjoint 100-order training window.

## Result

| Metric | Ranker only | Ranker + PPO | PPO delta |
|---|---:|---:|---:|
| Initial search PF | 2.4295257274 | 2.4295257274 | 0 |
| PF after 5 iterations | 2.3566543532 | 2.3566543532 | 0 |
| Coverage | 100% | 100% | 0 pp |
| MILP-validated candidates | 50 | 75 | +25 |
| Uncached box queries | 113 | 118 | +5 |
| MILP subprocess time | 72.21 s | 76.71 s | +4.49 s |
| Wall-clock time | 77.80 s | 82.36 s | +4.56 s |

The PPO top-5 and ranker shortlist had zero overlap in every iteration, so PPO
did propose genuinely different candidates. None was selected by the exact
MILP objective. Both arms selected the identical five-action sequence and
therefore reached the identical PF.

## Gate decision

**Fail.** This checkpoint increases exact verification cost without changing a
decision or improving PF. It must not be described as an RL improvement and
must not be promoted to a larger development or untouched-test experiment.

The result does not establish that long-horizon RL is intrinsically useless.
It establishes that this particular surrogate-mode PPO training objective and
candidate-union use do not provide useful information beyond the current
ranker in this smoke test. A future RL experiment needs a training target and
evaluation that explicitly reward long-horizon terminal PF under an exact-query
budget; merely adding policy-logit candidates is insufficient.

## Provenance

- Code version: `c230d88`
- PPO run: `multiscale_ppo_train_o0_o100_20260712/run_local_20260712_143249`
- PPO-union run: `multiscale_ppo_union_smoke_20260712/multiscale_ranker_policy_union_greedy/run_20260712_145003_257132`
- Ranker-only run: `multiscale_ranker_only_smoke_20260712/multiscale_ranker_greedy/run_20260712_145331_796662`
