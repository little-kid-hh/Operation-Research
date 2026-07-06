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

## Invalid Full-XML Diagnostic

An initial threshold probe was accidentally run against the full OR2023 unique
orders XML instead of the held-out test-split XML used by the certified
seed3:test[400,500) evidence. That run is not comparable to the paper-facing
main result and must not be used as formal algorithm evidence.

The invalid run is still useful as a reproducibility warning:

- wrong XML: `or2023_bsp_data/xml_unique/or2023_bsp_unique_orders.xml`;
- correct XML:
  `BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_test.xml`;
- invalid adaptive result dir:
  `BoxDesignSurrogateRL/results/adaptive_handoff_threshold_1e5_s3_o400/manifest_frontier_20260705_184031`;
- invalid no-handoff control dir:
  `BoxDesignSurrogateRL/results/nohandoff_current_s3_o400_control/manifest_frontier_20260705_192854`.

The invalid full-XML adaptive and no-handoff reruns both finished at audited PF
`1.9221927464`, far from the certified test-split PF `1.8251207019`. This
confirmed that the discrepancy came from the data window, not from the handoff
controller itself.

## Correct Test-Split Ablation

The comparable ablation uses the same test-split XML, initial boxes, exact
baseline summary, schedule, ranker artifact, ranker frontier, and exact audit
as the certified main result.

Configuration:

- window: `seed3:test[400,500)`;
- XML:
  `BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_test.xml`;
- K: 10;
- schedule: `0.25:1000`;
- ranker budgets: `20,30,40,50`;
- safety: `none`;
- ranker max elapsed: `900`;
- exact audit enabled;
- min iterations: `20`;
- window: `10`.

Result:

| metric | exact staged | no-handoff control | threshold 1e-5 | threshold 5e-6 | threshold 3e-6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| audited PF | 1.8313420040 | 1.8251207019 | 1.8251207019 | 1.8251207019 | 1.8251207019 |
| PF delta vs exact | n/a | -0.0062213021 | -0.0062213021 | -0.0062213021 | -0.0062213021 |
| coverage | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validations | 20760 | 12650 | 13090 | 12850 | 12800 |
| uncached MILP boxes | 1919 | 1626 | 1626 | 1626 | 1626 |
| subprocess seconds | 1047.5676 | 850.9569 | 842.7632 | 837.5591 | 839.2221 |
| wall-clock seconds | 1409.2608 | 1115.0527 | 1119.9636 | 1115.1521 | 1100.4007 |
| ranker stop reason | n/a | `time_limit` | `ranker_marginal_pf_handoff` | `ranker_marginal_pf_handoff` | `ranker_marginal_pf_handoff` |

Interpretation:

- All tested thresholds preserve audited PF, 100% coverage, and zero uncovered
  orders on this window.
- All tested thresholds tie the no-handoff control on uncached MILP box
  queries.
- `1e-5` hands off too early: it saves ranker work but shifts too much work to
  audit, increasing validations and slightly increasing wall-clock time versus
  the paired no-handoff control.
- `5e-6` is better but still not clean: same audited quality and uncached
  queries, lower subprocess seconds, near-tied wall-clock time, but more
  validations.
- `3e-6` is the best tested threshold on this single window: same audited
  quality, same uncached queries, lower subprocess time, and lower wall-clock
  time versus the paired current no-handoff control, with a small validation
  increase.

Comparison of `3e-6` against paired no-handoff control:

- PF delta: `0.0000000000`;
- coverage delta: `0.0000`;
- uncovered order delta: `0`;
- validations: `+150`;
- uncached MILP boxes: `0`;
- subprocess seconds: `-11.7347`;
- wall-clock seconds: `-14.6520`.

## Comparison Against Fixed Ranker180

A follow-up comparison also evaluates the same hard window against the
cap-unified fixed-180 main policy:
`ADAPTIVE_RANKER_BUDGET_SEED3_O400_COMPARISON_20260706.md`.

This comparison uses `fixed_ranker180_audit` as the reference. On
`seed3:test[400,500)`, `fixed_ranker180_audit` reaches audited PF
`1.8251207019`, which is better than exact staged PF `1.8313420040`, but it
uses more validations, more uncached MILP boxes, and more wall-clock time than
exact staged. The adaptive 900-second diagnostics keep the same audited PF and
full coverage while reducing the fixed-180 cost on that same window.

Best wall-clock variant in this comparison:

| metric | fixed ranker180 | adaptive 3e-6 | change vs fixed |
| --- | ---: | ---: | ---: |
| audited PF | 1.8251207019 | 1.8251207019 | 0.0000000000 |
| coverage | 1.0000 | 1.0000 | 0.0000 |
| uncovered orders | 0 | 0 | 0 |
| validations | 20920 | 12800 | -38.8% |
| uncached MILP boxes | 1960 | 1626 | -17.0% |
| subprocess seconds | 1038.2414 | 839.2221 | -19.2% |
| wall-clock seconds | 1458.5628 | 1100.4007 | -24.6% |

Interpretation:

- The fixed-180 cap is too short for this hard window: exact audit recovers
  quality, but the combined path can spend extra oracle work.
- Allowing a larger outer ranker budget and handing off by marginal PF gain can
  reduce the downstream exact-audit burden on this same window.
- This remains a single-window diagnostic. It supports adaptive budget control
  as the next method direction, but it is not a paper-wide result until tested
  on the remaining paired windows.

This is a modest single-window runtime win, not yet a main-paper claim. The
adaptive controller has not reduced uncached MILP query count on this window,
and the validation increase means it should be treated as an ablation candidate
rather than a replacement for the certified ranker-audit main method.

## Threshold Generalization Check

The existing no-handoff ranker traces were also used to simulate when the
tested thresholds would trigger across the available main-window frontier
summaries. The simulation covers 14 of the 15 main windows; `seed3:test[100,200)`
is excluded because the historical record is stored as split ranker/audit
summaries rather than a `frontier_summary.json`.

Simulation configuration:

- thresholds: `1e-5`, `5e-6`, `3e-6`;
- window: `10` ranker iterations;
- min iterations: `20`;
- source traces: current certified ranker-audit no-handoff runs.

Result:

- `seed3:test[400,500)` is the only analyzed window where any of these
  thresholds would trigger.
- `1e-5` triggers at iteration `270`.
- `5e-6` triggers at iteration `294`.
- `3e-6` triggers at iteration `299`.
- All other analyzed windows do not trigger for these thresholds before the
  existing ranker stop point.

This means the current marginal PF-per-validation controller is a targeted
heavy-window runtime optimization, not a broad replacement for the fixed
ranker cap. Running `3e-6` across all current windows would mostly reproduce
the no-handoff policy, with a change only on the heavy seed3 offset-400 window.

Source records:

- threshold simulation:
  `BoxDesignSurrogateRL/results/adaptive_handoff_threshold_simulation_20260705/threshold_simulation_14of15.json`

Source records:

- no-handoff control:
  `BoxDesignSurrogateRL/results/nohandoff_current_s3_o400_control_testsplit/manifest_frontier_20260705_201445`;
- threshold `1e-5`:
  `BoxDesignSurrogateRL/results/ah1e5_ts/manifest_frontier_20260705_203600`;
- threshold `5e-6`:
  `BoxDesignSurrogateRL/results/ah5e6_ts/manifest_frontier_20260705_205801`;
- threshold `3e-6`:
  `BoxDesignSurrogateRL/results/ah3e6_ts/manifest_frontier_20260705_211746`.

On Windows, keep `--out-root` short for these runs. Long result roots can push
per-box JSON cache files past path-length limits during MILP oracle cache
writes.
