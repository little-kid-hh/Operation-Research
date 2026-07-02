# Experiment Protocol for MILP-Backed Box Design

This document is the locked protocol for turning the OR2023 box-design work into
a scientifically comparable study. Older aggregate-proxy results remain useful
for development, but they are not main-result evidence.

This protocol targets **Problem B: OR2023 exact-MILP box design**, as fixed in
`BoxDesignSurrogateRL/docs/FORMAL_PROBLEM.md`. Kandula-style framework
reproduction is a separate **Problem A** track and must not be mixed with these
exact-MILP results.

## Dataset Scope

Primary unit: OR2023 BSP unique order geometries.

- XML: `or2023_bsp_data/xml_unique/or2023_bsp_unique_orders.xml`
- Size: 12,864 unique orders
- Package reference set: `or2023_bsp_data/packages.txt`

BPP package-loading data and learned loadability models are not part of the
main exact-MILP comparison unless a transfer/generalization experiment is
explicitly introduced.

## Main Comparison

The main A/B comparison must hold fixed:

- same order set;
- same `K`;
- same KMeans seed and therefore same initial boxes;
- same MILP feasibility oracle;
- same orientation label, primary `label_6ori`;
- same time limit and unknown-label policy;
- same coverage-repair setting, if enabled;
- matched search budget, reported as actual candidate evaluations and oracle
  cache misses.

### Algorithm A: Paper-Action Fixed-Step Greedy

Entrypoint:

```bash
python BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm paper_fixed_step \
  --fixed-step 0.5
```

This is the fixed-step paper action framework adapted to OR2023 and exact MILP
feasibility. It is not an exact reproduction of Kandula's private-data
RL/tree-search pipeline. Use this name in reports:

```text
paper-action fixed-step greedy
```

### Algorithm B: Staged Greedy Step Schedule

Entrypoint:

```bash
python BoxDesignSurrogateRL/scripts/run_milp_box_algorithms.py \
  --algorithm staged_greedy \
  --schedule 0.5:2,0.25:2
```

This uses the same coordinate action set and MILP oracle as Algorithm A. The
only intended algorithmic difference is the step schedule.

## Common Coverage Repair

Exact MILP feasibility is stricter than the aggregate initialization used to
construct KMeans boxes. A box set can satisfy aggregate sorted dimensions and
total volume while still failing to load a multi-item order under the MILP
packing model. Because uncovered orders dominate PF through the hard penalty,
main comparisons should either:

- report the raw initialization/search result as infeasible when uncovered
  orders remain; or
- apply the same explicit MILP-verified coverage repair to every algorithm
  before comparing volume optimization.

The current common repair is:

```bash
--coverage-repair geometric_expand \
--repair-margins 1.0,1.05,1.1,1.25,1.5,2.0 \
--repair-max-rounds 3
```

For each uncovered order, it generates candidate replacements by expanding one
existing box at a time to meet aggregate requirements at the listed margins,
then accepts only improvements verified by the same MILP oracle. This repair is
a shared initialization step, not an algorithm-specific advantage. Reports must
include both `initial_score` and `search_initial_score` so the repair effect is
auditable.

## Reference Baselines

These are not replacements for the A/B comparison:

- KMeans initial boxes only.
- Existing 90 OR2023 package dimensions evaluated with exact labels.
- Historical aggregate-proxy Kandula/staged results.

Historical aggregate-proxy numbers must be labeled as proxy development results,
not as exact-MILP baselines.

## Feasibility Oracle

Generated box dimensions must use:

```text
--oracle java
```

The label-table oracle only applies to package dimensions already present in the
precomputed label table. It must not be used for continuous box search.

Label policy:

- `1`: feasible
- `0`: proven infeasible
- negative values such as `-1` or `-2`: unknown/error

Unknown labels are not counted as feasible. A formal comparison is valid only
when both algorithms have zero unknown labels on the reported final evaluation,
or when unknown handling is explicitly rerun with a larger MILP time limit until
resolved.

## Metrics

Primary:

- MILP packaging factor:

```text
mean assigned feasible box volume / mean order item volume
```

Coverage is a hard gate. PF improvements are not meaningful if uncovered orders
or unknown labels differ.

The exact-MILP search ranking is:

```text
1. minimize uncovered_orders
2. minimize unknown_pairs
3. minimize MILP packaging factor
```

Always report:

- packaging factor;
- mean assigned box volume;
- mean order volume;
- coverage rate;
- uncovered order count;
- unknown pair count;
- orders with at least one unknown label;
- final box dimensions;
- assignment counts per box;
- elapsed seconds;
- oracle cache entries, hits, misses, disk hits;
- candidate-evaluation budget.

## Statistical Reporting

Use paired seeds: the same seed initializes both algorithms.

Report:

- per-seed results;
- paired B-A deltas;
- mean delta;
- standard deviation;
- 95% confidence interval;
- a paired nonparametric test when enough seeds are available.

Target: at least 5 seeds for a defensible internal result, 10 if runtime allows.

## Validity Gates

Do not claim Algorithm B beats Algorithm A unless all are true:

- same order set, `K`, seeds, oracle, and orientation label;
- same or explicitly budget-matched candidate evaluation count;
- zero uncovered orders for both methods, or uncovered differences are reported
  as infeasibility rather than PF wins;
- zero unknown labels in final reported evaluations;
- staged schedule chosen on dev and frozen before test;
- final order-box assignments are exactly MILP verified.

## Development Run Ladder

Small-prefix runs are plumbing and runtime calibration, not final evidence.
Use deterministic hash materialized split XMLs for dev/test comparisons.

1. Environment gate on Windows/Gurobi:
   - `orders-limit=10`, `K=10`, `iterations=0`.
2. Runtime calibration:
   - initial-only runs at `orders-limit=20,100,500`.
3. Search micro-benchmark:
   - A/B, seeds `0,1,2`, deterministic hash dev split, `iterations=1` or
     `0.5:1`.
4. Run dev tuning with the coverage-repair setting fixed across algorithms.
5. Freeze schedule and budget.
6. Run held-out exact-MILP final comparison.
7. Separately report all-12,864 in-sample benchmark when runtime allows.

## Current Calibration Evidence

On `2026-07-01`, code version `7b15578`, hash split
`or2023_seed20260701_limit250` dev set (`50` orders), `K=10`, seed `0`,
`label_6ori`, and MILP time limit `1s`:

- raw KMeans initialization had `coverage=98%`, `uncovered=1`, `PF=24.6438`;
- shared `geometric_expand` repair improved this to `coverage=100%`,
  `uncovered=0`, `PF=2.4513`;
- repaired fixed `0.5` one-sweep baseline reached `PF=2.4447`;
- repaired `0.25` one-sweep candidate reached `PF=2.4480`;
- both repaired searches had `unknown=0` and `119` candidate evaluations.
- repaired fixed `0.5` two-sweep baseline reached `PF=2.4380`;
- repaired staged `0.5:1,0.25:1` two-sweep candidate reached `PF=2.4413`;
- both repaired two-sweep searches had `unknown=0` and `179` candidate
  evaluations.

This is a calibration result only. It supports using common MILP coverage
repair before volume comparisons, but it is not a final performance claim.

The detailed dev100 repaired three-seed calibration is recorded in
`BoxDesignSurrogateRL/docs/CALIBRATION_RESULTS_20260701.md`.
