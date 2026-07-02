# Candidate-Quality Predictor Results, 2026-07-02

This note records the current MILP-verified candidate-filter experiment. It is
not a full paper result yet; it fixes the current evidence, run artifacts, and
claim boundaries.

## Protocol

Problem setting follows `docs/EXPERIMENT_PROTOCOL.md`:

- Dataset family: OR2023 BSP unique orders.
- Feasibility oracle: Java/Gurobi exact MILP, `label_6ori`.
- Search objective: minimize exact-MILP score lexicographically by
  `uncovered_orders`, `unknown_pairs`, then packaging factor.
- Candidate predictor role: predict local-search candidate quality only for
  ranking/filtering. Final move acceptance remains exact-MILP verified.
- Main filter controls: `--ranker-adaptive-top-k 10,30` and
  `--ranker-noop-fallback`.

Terminology: the code artifact is currently named `candidate_ranker`, but the
scientific term used here is **candidate-quality predictor**.

## Training Data

The no-dev100-leakage predictor uses repaired dev300 train-split exact traces:

| Stage | Source | PF | Coverage | Generated candidates | Candidate trace rows |
|---|---:|---:|---:|---:|---:|
| 0.5 exact, repaired | `candidate_predictor_dev300_repaired_20260702/staged_greedy/run_20260702_222448` | 2.2845333823712695 | 1.0 | 7200 | 7200 |
| 0.25 exact from 0.5 checkpoint | `candidate_predictor_dev300_repaired_20260702/staged_greedy/run_20260702_230416` | 2.27942167748558 | 1.0 | 480 | 480 |

The selected no-leakage predictor:

```text
BoxDesignSurrogateRL\results\candidate_predictors_dev300_repaired_20260702\candidate_ranker_20260702_230711\candidate_ranker.joblib
```

Training command:

```bash
python BoxDesignSurrogateRL/scripts/train_candidate_ranker.py \
  --trace-csv \
    BoxDesignSurrogateRL/results/candidate_traces/dev300_train_repaired_exact05_from_kmeans.csv \
    BoxDesignSurrogateRL/results/candidate_traces/dev300_train_repaired_exact025_from_05.csv \
  --out-dir BoxDesignSurrogateRL/results/candidate_predictors_dev300_repaired_20260702 \
  --model hgbt \
  --top-k 5,10,20,30 \
  --test-fraction 0.25 \
  --random-state 1
```

Held-out step-group diagnostics:

| Metric | Value |
|---|---:|
| Train rows / groups | 5760 / 96 |
| Eval rows / groups | 1920 / 32 |
| Exact-best capture@10 | 0.4375 |
| Exact-best capture@20 | 0.5000 |
| Exact-best capture@30 | 0.5000 |
| Mean PF gap@30 | 0.00025249766954377195 |

## Out-of-Sample Dev100 Validation

The predictor above is trained only on the repaired dev300 train split. The
following validation runs use dev100 checkpoints and exact-MILP final
acceptance.

| Validation stage | Exact PF | Predictor-filter PF | Coverage | Exact generated | MILP validated | Avoided | Avoidance |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev100 0.25 from exact 0.5 checkpoint | 1.7467902175053618 | 1.7467902175053618 | 1.0 | 720 | 350 | 370 | 51.3889% |
| dev100 0.1 from exact 0.25 checkpoint | 1.739963658999321 | 1.739963658999321 | 1.0 | 960 | 690 | 270 | 28.1250% |

Exact baseline artifacts:

```text
BoxDesignSurrogateRL\results\milp_convergence_dev100_20260702\staged_greedy\run_20260702_183342
BoxDesignSurrogateRL\results\milp_convergence_dev100_20260702\staged_greedy\run_20260702_195233
```

Validation artifacts:

```text
BoxDesignSurrogateRL\results\candidate_predictor_oos_dev100_20260702\ranker_filtered_greedy\run_20260702_230741
BoxDesignSurrogateRL\results\candidate_predictor_oos_dev100_20260702\ranker_filtered_greedy\run_20260702_230806
```

Both runs match the exact baseline PF and preserve 100% coverage with zero
unknown labels. Therefore, on this single dev100 validation split and seed, the
candidate-quality predictor reduced exact MILP candidate validations while
preserving exact-MILP solution quality.

## Diagnostic Variants

A predictor trained only on repaired dev300 0.5 trace also preserves exact PF
on dev100:

| Validation stage | Predictor artifact | MILP validated | Exact generated | Avoidance |
|---|---|---:|---:|---:|
| dev100 0.25 | `candidate_ranker_20260702_230450` | 530 | 720 | 26.3889% |
| dev100 0.1 | `candidate_ranker_20260702_230450` | 660 | 960 | 31.2500% |

This variant is useful diagnostically: adding 0.25 train traces improved dev100
0.25 filtering but did not improve dev100 0.1 filtering. This suggests the
predictor is not yet a robust one-size-fits-all filter across fine stages.

## Claim Boundary

Supported by current evidence:

- The candidate-quality predictor can be trained without dev100 candidate-label
  leakage.
- On dev100 out-of-sample validation, MILP-verified predictor filtering matched
  exact fine-stage PF at 0.25 and 0.1 while reducing MILP candidate validations.
- The final solution quality is exact-MILP verified; predictor scores are not
  reported as final feasibility or quality metrics.

Not yet supported:

- Full OR2023 or multi-seed statistical claims.
- Wall-clock speedup claims. These validation runs used warm oracle caches, so
  elapsed seconds are not comparable speed evidence.
- A universal predictor claim across all step sizes. Fine-stage performance
  varies by training trace mix.

## Implementation Note

During validation, two parallel runs collided on a timestamp-only run directory.
`run_milp_box_algorithms.py` now includes microseconds in `run_id` to avoid
parallel run collisions.
