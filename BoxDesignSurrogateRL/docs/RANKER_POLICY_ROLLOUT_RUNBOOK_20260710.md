# Ranker Policy Rollout Runbook

Date: 2026-07-10

## Purpose

Test the next RL-aligned variant:

```text
ranker_policy_rollout_greedy
```

This keeps the current ranker as a MILP query-budget controller, but changes
candidate selection after exact MILP validation:

```text
ranker ranks the 6K one-step moves
MILP validates the selected tier
policy rollout scores exact-verified improving child states
the selected action is the child with best rollout score
```

The final accepted child still has exact MILP feasibility/PF. The rollout also
uses exact MILP evaluations, and those extra oracle calls are reported as
`policy_rollout_milp_validations`.

## Smoke Pair

Run this on the Windows/Gurobi machine. First create the exact baseline for the
same window and initial boxes:

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm staged_greedy \
  --xml-path BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_test.xml \
  --orders-offset 0 \
  --orders-limit 100 \
  --seed 1 \
  --k 10 \
  --schedule 0.25:50 \
  --milp-time-limit-seconds 30 \
  --coverage-repair geometric_expand \
  --prefetch-candidate-statuses \
  --out-root BoxDesignSurrogateRL/results/rl_rollout_smoke_20260710 \
  --config-label exact_staged_0p25x50
```

Then rerun the same window with ranker + policy rollout, reusing the exact
run's `initial_boxes.json`:

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm ranker_policy_rollout_greedy \
  --xml-path BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_test.xml \
  --orders-offset 0 \
  --orders-limit 100 \
  --seed 1 \
  --k 10 \
  --initial-boxes-json <exact_run_dir>/initial_boxes.json \
  --schedule 0.25:50 \
  --milp-time-limit-seconds 30 \
  --coverage-repair geometric_expand \
  --candidate-ranker-path BoxDesignSurrogateRL/results/candidate_ranker_assignment_aware_20260705/candidate_ranker_20260705_084240/candidate_ranker.joblib \
  --ranker-adaptive-top-k 20,30,40,50 \
  --ranker-safety-policy none \
  --policy-path BoxDesignSurrogateRL/results/kandula_paper_policy_or2023_full_step0p5_seed1/run_20260615_182939/policy_final.pt \
  --policy-rollout-steps 3 \
  --policy-rollout-samples 1 \
  --policy-rollout-beta 0.95 \
  --no-sample-policy-rollout \
  --prefetch-candidate-statuses \
  --out-root BoxDesignSurrogateRL/results/rl_rollout_smoke_20260710 \
  --config-label ranker_policy_rollout_top20_30_40_50_r3
```

## Acceptance For Smoke

Compare `summary.json` from the two runs:

- coverage must be `1.0`;
- uncovered orders must be `0`;
- PF must not regress before scaling this variant;
- report both total `milp_validated_candidates` and
  `policy_rollout_milp_validations`;
- if PF improves but oracle cost rises, this is a quality-oriented RL signal,
  not yet a speed claim;
- if PF is equal and oracle cost rises, do not continue broad validation until
  policy quality is improved.

## Next Matrix If Smoke Passes

Run paired windows for seeds that have matching policy checkpoints first:

```text
seed1 offsets 0,100,200,300,400
seed2 offsets 0,100,200,300,400
```

For seed3/seed4 windows, either train matching `paper` mode policy checkpoints
with the same `K=10`, `step=0.5`, and full OR2023 data, or explicitly report
that a shared policy is being transferred across seeds.
