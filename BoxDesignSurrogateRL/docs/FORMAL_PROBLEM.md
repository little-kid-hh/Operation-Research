# Formal Problem Definitions

This document fixes the optimization problems used in the BoxDesignSurrogateRL
project. Every experiment must state which problem it targets. Do not mix
metrics, feasibility rules, or baselines across these problem definitions.

## Problem A: Kandula-Style Framework Reproduction

Purpose: reproduce the algorithmic shape of Kandula et al. on local OR2023 data.
This is a framework reproduction, not an exact numerical reproduction of the
paper's private-data experiments.

### Inputs

- A set of local OR2023 order geometries.
- Fixed number of carton types `K`.
- Initial carton set from KMeans over order geometry vectors.

### Decision Variables

- Box dimensions:

```text
B = {(L_j, W_j, H_j)} for j = 1..K
```

### Feasibility Rule

Problem A uses the aggregate geometric proxy:

```text
order i feasible in box j iff
  max item sorted length <= L_j
  max item sorted width  <= W_j
  max item sorted height <= H_j
  total item volume      <= L_j * W_j * H_j
```

This proxy is intentionally weaker than exact multi-item 3D packing. It is used
only to reproduce the paper-style box-sizing game on data available in this
repository.

### Assignment Rule

Each order is assigned to the smallest-volume feasible box:

```text
a(i) = argmin_j volume(B_j) subject to feasible_proxy(i, j)
```

If no feasible box exists, the order is uncovered and receives the configured
large uncovered penalty.

### Objective

Minimize packaging factor:

```text
PF(B) = mean_i volume(B_a(i)) / mean_i order_volume(i)
```

with uncovered orders penalized. Candidate states that are infeasible or worse
than the initial state are terminal failures in the paper-style game.

### Algorithmic Scaffold

This problem supports Kandula-style stages:

- KMeans initialization;
- `K x 3` box-dimension state;
- `6K + 1` actions: each box dimension can increase or decrease by a fixed step,
  plus a resignation action;
- reward based on PF improvement;
- policy/tree-search variants.

Use this problem when discussing "Kandula framework reproduction."

## Problem B: OR2023 Exact-MILP Box Design

Purpose: define the main publishable optimization target for this project. This
is the problem used for exact-MILP baselines, staged greedy search, adaptive
search, and any learned surrogate that claims final exact feasibility.

### Inputs

- OR2023 BSP unique order geometries.
- Fixed number of carton types `K`, currently `K=10` unless explicitly stated.
- A deterministic order split for dev/test when tuning or reporting held-out
  results.
- KMeans seed, which fixes the shared initial box set.
- Java/Gurobi MILP loading oracle.

### Decision Variables

- Continuous positive box dimensions:

```text
B = {(L_j, W_j, H_j)} for j = 1..K
L_j, W_j, H_j > 0
```

The current search algorithms optimize these dimensions through coordinate
actions, not by solving a single monolithic continuous optimization model.

### Exact Feasibility Constraint

For every order-box pair, feasibility is defined by the Java/Gurobi MILP oracle:

```text
z_ij = MILP_6ori(order i, box j)
```

where:

- `z_ij = 1`: feasible;
- `z_ij = 0`: proven infeasible;
- `z_ij < 0`: unknown/error.

For main results, unknown labels are not feasible. Valid final comparisons must
have:

```text
uncovered_orders = 0
unknown_pairs = 0
```

or else the run must be reported as incomplete/risky rather than a valid PF
claim.

### Assignment Rule

Given a box set `B`, assign every order to the smallest-volume MILP-feasible
box:

```text
a(i) = argmin_j volume(B_j) subject to z_ij = 1
```

If no such box exists, the order is uncovered and receives the configured hard
penalty. PF comparisons are meaningful only after coverage and unknown-label
gates are satisfied.

### Objective

The primary objective is exact-MILP packaging factor:

```text
PF_MILP(B) = mean_i volume(B_a(i)) / mean_i order_volume(i)
```

where assigned box volumes use the smallest feasible MILP box for each order.

The optimization ranking used by current exact-MILP search is lexicographic:

```text
1. minimize uncovered_orders
2. minimize unknown_pairs
3. minimize PF_MILP
```

This makes coverage and oracle certainty hard gates before volume optimization.

### Common Coverage Repair

KMeans boxes built from aggregate order geometry may fail exact MILP feasibility
for multi-item orders. Therefore Problem B permits an explicit shared
initialization repair:

```text
--coverage-repair geometric_expand
```

The repair expands existing boxes for currently uncovered orders, evaluates
each candidate with the same MILP oracle, and accepts only score improvements.
It is not an algorithm-specific advantage. When enabled, every compared method
must use the same repair settings and must report both:

- `initial_score`: raw KMeans score;
- `search_initial_score`: score after shared repair.

### Baseline and Candidate Comparison Rules

A valid Problem B comparison must hold fixed:

- order split;
- `K`;
- KMeans seed;
- MILP oracle and orientation label, primary `label_6ori`;
- MILP time limit and unknown-label policy;
- coverage-repair setting;
- search budget, reported as candidate evaluations and oracle cache misses.

The current active exact-MILP baseline is:

```text
fixed05_i2:
  algorithm = paper_fixed_step
  fixed_step = 0.5
  iterations = 2
  shared coverage repair enabled
```

The name `paper_fixed_step` means "paper-action fixed-step greedy adapted to
Problem B." It does not mean exact reproduction of Kandula's private-data
RL/tree-search pipeline.

## Reporting Rule

Reports must state explicitly:

```text
Problem A: Kandula-style framework reproduction
```

or:

```text
Problem B: OR2023 exact-MILP box design
```

Historical aggregate-proxy results are development evidence only and must not
be compared directly against Problem B exact-MILP results.
