# MILP-Backed OR2023 Box Algorithms

This is the current clean entrypoint for comparing two box-design algorithms on
OR2023 while using MILP feasibility instead of the earlier aggregate proxy.

## Entrypoint

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python3 BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm paper_fixed_step \
  --orders-limit 20 \
  --k 10 \
  --fixed-step 0.5 \
  --iterations 3
```

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python3 BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm staged_greedy \
  --orders-limit 20 \
  --k 10 \
  --schedule 0.5:2,0.25:2
```

The default oracle is `--oracle java`, which invokes
`org.example.GeneratePerminPackageLabels` through Java/Gurobi. This supports
arbitrary generated box dimensions and is the correct oracle for searching new
box sets.

## Algorithm A: `paper_fixed_step`

This keeps the paper-style action space: for each of `K` boxes, change one of
`length`, `width`, or `height` by `+step` or `-step`, plus a no-op. The default
step is `0.5`.

Current deliberate differences from the paper:

- Dataset: OR2023 unique orders instead of the paper's private SKU-order data.
- Feasibility: online MILP labels for each order-box pair instead of the prior
  aggregate max-dimension/volume proxy.
- Search loop: small deterministic best-improvement passes over the same fixed
  action set, so MILP calls remain tractable while the feasibility objective is
  being validated.
- Parameters: small `--orders-limit` and `--iterations` by default; increase
  only after the MILP runtime is understood.

## Algorithm B: `staged_greedy`

This uses the same MILP feasibility oracle and coordinate action set, but the
step size is adjusted manually across stages. Example:

```text
0.5:2,0.25:2
```

means run at most two best-improvement iterations at step `0.5`, then at most
two iterations at step `0.25`.

This is the tuned greedy variant, not the original paper baseline.

## Outputs

Each run writes a timestamped directory under:

```text
BoxDesignSurrogateRL/results/milp_box_algorithms/<algorithm>/run_YYYYMMDD_HHMMSS/
```

Files:

- `manifest.json`: configuration and oracle settings.
- `initial_boxes.json`: KMeans-initialized boxes.
- `trace.csv`: per-iteration PF, coverage, uncovered orders, and selected action.
- `best_boxes.json`: final selected box set.
- `summary.json`: run configuration plus initial and best scores.

Main metrics:

- `packaging_factor`: mean assigned box volume divided by mean order item volume.
- `coverage_rate`: fraction of orders with at least one MILP-feasible box.
- `uncovered_orders`: orders with no MILP-feasible box; these must be inspected
  before claiming a PF win.
- `mean_box_volume`: mean volume of the selected feasible box per order, with a
  large penalty for uncovered orders.
- `mean_order_volume`: mean total item volume per order.

## Environment Notes

The online oracle requires:

- a Java runtime;
- compiled `MILP_3DBPP/target/classes`;
- Gurobi Java dependencies on the classpath, usually via `GUROBI_JAR`.

The fallback `--oracle labels` reads the precomputed OR2023 labels for the 90
original package dimensions only. It is useful for evaluating existing package
sets, but it cannot search continuous generated boxes unless those boxes exactly
match pre-labeled package dimensions.
