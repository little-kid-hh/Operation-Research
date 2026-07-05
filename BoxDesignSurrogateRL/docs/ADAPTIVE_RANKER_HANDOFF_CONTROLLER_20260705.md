# Adaptive Ranker-Handoff Controller, 2026-07-05

## Purpose

This controller is the next experimental method direction after the certified
ranker-audit result. It does not change the objective, the MILP feasibility
oracle, the accepted-move rule, or the final exact audit requirement. It only
changes when the ranker phase hands the current box set to exact staged audit.

The research question is:

Can the learned ranker stop spending exact oracle work once its recent marginal
PF improvement per validation is too low, while exact audit still certifies
final PF and coverage?

## Implemented Policy

Policy name:

```text
marginal_pf_per_validation
```

At the end of each improving ranker iteration, the runner computes over a
recent window of ranker iterations:

```text
recent_pf_improvement = PF_before_window - PF_after_window
pf_per_validation = recent_pf_improvement / recent_milp_validations
```

If all of the following hold, the ranker phase stops and writes
`stop_reason=ranker_marginal_pf_handoff`:

- ranker iterations so far are at least `ranker_handoff_min_iterations`;
- the recent window has at least `ranker_handoff_window` iterations;
- current coverage is complete and `unknown_pairs == 0`;
- `pf_per_validation < ranker_handoff_min_pf_improvement_per_validation`.

The exact audit should then run from the ranker checkpoint. Final paper-facing
quality claims still come only from the audited result.

## CLI

Core runner:

```bash
python BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm ranker_filtered_greedy \
  --ranker-handoff-policy marginal_pf_per_validation \
  --ranker-handoff-min-iterations 20 \
  --ranker-handoff-window 10 \
  --ranker-handoff-min-pf-improvement-per-validation 0.00001
```

Frontier wrapper:

```bash
python BoxDesignSurrogateRL/scripts/run_query_budgeted_ranker_frontier.py \
  --ranker-handoff-policy marginal_pf_per_validation \
  --ranker-handoff-min-iterations 20 \
  --ranker-handoff-window 10 \
  --ranker-handoff-min-pf-improvement-per-validation 0.00001
```

The same handoff arguments are also exposed through the paired-window protocol
entrypoints:

- `BoxDesignSurrogateRL/scripts/run_ranker_window_protocol.py`
- `BoxDesignSurrogateRL/scripts/run_ranker_frontier_from_exact_manifest.py`

The default remains:

```text
--ranker-handoff-policy none
```

so existing baseline and certified ranker-audit results are unchanged unless
the new policy is explicitly enabled.

## Trace Fields

When the policy triggers, `trace.csv` includes:

- `ranker_handoff_policy`;
- `ranker_handoff_min_iterations`;
- `ranker_handoff_window`;
- `ranker_handoff_recent_pf_improvement`;
- `ranker_handoff_recent_validations`;
- `ranker_handoff_pf_per_validation`;
- `ranker_handoff_threshold`.

The frontier summary also records the handoff policy and threshold parameters
for reproducibility.

## Smoke Validation

Remote Windows/Gurobi smoke checks were run on
`laptop-29k27sem / 100.115.236.89`.

Core runner stop-path smoke:

- dataset: OR2023 unique orders;
- window: `seed3:test[400,500)`;
- K: 10;
- schedule: `0.25:20`;
- ranker budgets: `20,30,40,50`;
- handoff policy: `marginal_pf_per_validation`;
- handoff threshold: intentionally high (`999`) to force stop-path validation;
- result: `stop_reason=ranker_marginal_pf_handoff`;
- run dir:
  `BoxDesignSurrogateRL/results/adaptive_handoff_smoke/ranker_filtered_greedy/run_20260705_183338_160117`.

Wrapper parameter-passthrough smoke:

- dataset: OR2023 unique orders;
- window: first 30 orders;
- audit disabled for smoke only;
- result: frontier summary records the handoff policy and threshold fields;
- run dir:
  `BoxDesignSurrogateRL/results/adaptive_handoff_wrapper_smoke/frontier_20260705_183540`.

These smoke runs are not paper evidence. They only validate code paths and
output schema.

## Experimental Plan

The first real ablation should be limited to paired windows where exact and
main ranker-audit results already exist.

Candidate thresholds:

- `0.0000025`
- `0.000005`
- `0.00001`
- `0.00002`

Initial protocol:

- window: `seed3:test[400,500)` because it is the heaviest known case;
- schedule: `0.25:1000`;
- ranker budgets: `20,30,40,50`;
- safety: `none`;
- ranker max elapsed: keep `900` as a hard outer guard;
- run exact audit after handoff.

Acceptance rule versus the current main ranker-audit policy:

- audited PF does not regress;
- coverage remains `1.0000`;
- uncovered orders remain zero;
- uncached MILP box queries decrease, or subprocess seconds decrease;
- wall-clock-only wins are reported separately from oracle-work wins.

If a threshold passes on `seed3:test[400,500)`, evaluate it on the remaining
main paired windows. If no threshold passes, the result is still useful: it
shows that the current fixed ranker budget is already near the best safe
handoff point under this simple marginal-value signal.

## First Threshold Result

The first formal threshold probe was run on `seed3:test[400,500)` with:

- threshold: `0.00001`;
- min iterations: `20`;
- window: `10`;
- ranker max elapsed: `900`;
- exact audit enabled.

Result:

| metric | exact staged | current main ranker+audit | adaptive threshold 1e-5 |
| --- | ---: | ---: | ---: |
| audited PF | 1.8313420040 | 1.8251207019 | 1.9221927464 |
| PF delta vs exact | n/a | -0.0062213021 | +0.0908507424 |
| coverage | 1.0000 | 1.0000 | 1.0000 |
| validations | 20760 | 12660 | 24680 |
| uncached MILP boxes | 1919 | 1626 | 2679 |
| subprocess seconds | 1047.5676 | 848.1555 | 2190.4414 |
| wall-clock seconds | 1409.2608 | 1131.4878 | 2670.3534 |
| ranker stop reason | n/a | `time_limit` | `time_limit` |

This is a negative result. The threshold did not trigger the adaptive handoff:
the ranker still stopped by the 900-second time limit. The ranker reached only
PF `2.2760985121`, leaving the exact audit with a much harder recovery problem.
The audit did not recover the exact or current main ranker-audit PF and instead
finished at PF `1.9221927464`.

Trace diagnostic:

- ranker iterations: `194`;
- ranker improvements: `193`;
- last improvement iteration: `193`;
- tail-10 PF improvement: `0.0139445897`;
- tail-10 validations: `180`;
- tail PF improvement per validation: about `7.7e-5`, above the `1e-5`
  threshold.

Interpretation:

- `1e-5` is too low to trigger on this heavy window.
- Recent PF-per-validation alone is not a reliable safety signal because it can
  remain positive while the ranker is still far from a state that makes audit
  cheap.
- Because this run used a new cold cache/time-limited rerun, the wall-clock
  comparison to the historical main run should be treated cautiously; the
  quality regression and increased oracle work are still enough to reject this
  threshold.

Source records:

- adaptive threshold run:
  `BoxDesignSurrogateRL/results/adaptive_handoff_threshold_1e5_s3_o400/manifest_frontier_20260705_184031`
- handoff trace analysis:
  `BoxDesignSurrogateRL/results/adaptive_handoff_threshold_1e5_s3_o400/manifest_frontier_20260705_184031/handoff_trace.json`
