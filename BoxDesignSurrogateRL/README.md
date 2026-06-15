# BoxDesignSurrogateRL

This project extends the Kandula-style e-commerce box-sizing framework with a
learned loadability evaluator.

## Goal

Design a state-of-the-art set of carton sizes for OR 2023 / Fontaine-Minner
box-size design data, with carton dimension moves as fine as `0.01`.

The intended contribution is not just a smaller move size. The main change is
that each candidate box assortment is evaluated through a package feasibility
surrogate trained from MILP labels, so the optimizer does not need to freeze one
packing/assignment simplification after the initial solution.

## Core Idea

Kandula et al. use clustering to get an initial assortment, then improve box
dimensions through an RL policy and tree search. Their evaluation is simplified
by a fixed fit/assignment structure.

Our version keeps clustering only for the initial feasible seed:

1. Build an initial feasible box assortment with clustering.
2. At every box-dimension move, recompute order-to-box feasibility using the
   learned loadability model.
3. Assign each order to the cheapest feasible carton under a confidence-aware
   cost function.
4. Use RL, tree search, or hybrid local search to explore box dimensions at
   `0.01` resolution.
5. Send elite and high-risk candidates to exact MILP verification to avoid
   reporting surrogate-induced false positives.

## Reused Assets

Large existing assets are linked under `assets/` rather than copied.

- `assets/or2023_bsp_data`: main box-size design/order geometry data.
- `assets/or2023_bpp_data`: loadability benchmark data.
- `assets/processed_features`: generated base40/FE111 package features.
- `assets/milp_labels`: existing MILP feasibility labels.
- `assets/loadability_model_pack`: reusable SVM/ensemble/TabTreeFormer models.

See `assets/README.md` for data boundaries.

## Initial Project Structure

```text
BoxDesignSurrogateRL/
  box_design_surrogate/
    features.py      # order parsing and base40 feature generation
    evaluator.py     # confidence-aware box-set evaluator
    search.py        # search/RL interfaces
  docs/
    TECHNICAL_PLAN.md
    KANDULA_DELTA.md
  scripts/
    smoke_evaluate_existing_boxes.py
```

## First Experimental Milestone

The first milestone is a deterministic evaluator:

```text
box_set -> all order-box features -> model probabilities -> assignment -> cost/risk
```

Once this is stable, RL/tree search can be added without changing the objective
definition.

## MILP-Backed Baseline Entrypoints

The current clean entrypoint for exact-feasibility experiments is:

```bash
PYTHONPATH=BoxDesignSurrogateRL \
python3 BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm paper_fixed_step \
  --orders-limit 20 \
  --k 10 \
  --fixed-step 0.5 \
  --iterations 3
```

Use `--algorithm paper_fixed_step` for the fixed-step paper action framework,
and `--algorithm staged_greedy` for the manually adjusted greedy step schedule.
Both use the MILP oracle by default. See `docs/MILP_BOX_ALGORITHMS.md` for the
exact assumptions and runtime requirements.
