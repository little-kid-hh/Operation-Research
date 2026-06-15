# Reproducibility Notes

## Kandula-Style Baseline

Run a deterministic small baseline:

```bash
python3 BoxDesignSurrogateRL/scripts/run_kandula_repro_experiment.py \
  --orders-limit 2000 \
  --ks 10 \
  --seeds 42 \
  --step 0.5 \
  --max-iters 20 \
  --beam-depth 20 \
  --beam-width 4
```

The script writes:

```text
BoxDesignSurrogateRL/results/kandula_repro/run_<timestamp>/
  manifest.json
  summary.csv
  boxes/
```

For a broader baseline grid:

```bash
python3 BoxDesignSurrogateRL/scripts/run_kandula_repro_experiment.py \
  --orders-limit 2000 \
  --ks 10 20 30 \
  --seeds 0 1 2 \
  --step 0.5 \
  --max-iters 20 \
  --beam-depth 20 \
  --beam-width 4
```

## Determinism

- K-means uses explicit `random_state`.
- The exact command parameters are stored in `manifest.json`.
- Git commit and short worktree status are stored in `manifest.json`.
- Box dimensions for every stage are stored as JSON.

## Current Scope

These are framework-reproduction baselines on local OR2023 BSP data, not exact
numerical reproduction of Kandula et al.'s proprietary experiments.

