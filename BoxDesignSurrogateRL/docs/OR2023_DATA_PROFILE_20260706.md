# OR2023 Data Profile For Box Design

Date: 2026-07-06

This note records the dataset facts used for the current BoxDesignSurrogateRL
briefing. The goal is to keep the reporting language simple and reproducible.

## Source Dataset

The source table is the OR2023 BSP unique-order XML:

```text
or2023_bsp_unique_orders.xml
```

It contains unique e-commerce order geometries. Each order contains one or more
items. Each item has three integer dimensions:

```xml
<order id="165">
  <item id="0"><p>40</p><q>37</q><r>22</r></item>
  <item id="1"><p>40</p><q>37</q><r>22</r></item>
</order>
```

The dimensions `p`, `q`, and `r` are item side lengths. Loading feasibility is
not inferred only from these scalar dimensions in the main exact-MILP problem;
it is checked by the Java/Gurobi loading oracle.

## Full Unique-Order Dataset

| metric | value |
| --- | ---: |
| orders | 12,864 |
| items | 20,632 |
| mean items per order | 1.604 |
| single-item orders | 7,905 |
| multi-item orders | 4,959 |
| multi-item share | 38.55% |
| max items per order | 6 |
| all item dimensions are integer-valued | true |
| non-integer dimension values | 0 |
| dimension value range | 2 to 118 |
| unique dimension values | 105 |
| mean order volume | 34,263.758 |
| median order volume | 16,470 |
| p90 order volume | 84,000 |
| max order volume | 864,000 |

Item-count distribution:

| items in order | orders |
| --- | ---: |
| 1 | 7,905 |
| 2 | 3,377 |
| 3 | 856 |
| 4 | 392 |
| 5 | 167 |
| 6 | 167 |

Dimension distribution over all items:

| dimension | min | median | mean | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| p | 8 | 34 | 38.160 | 65 | 118 |
| q | 6 | 23 | 26.364 | 45 | 93 |
| r | 2 | 13 | 14.919 | 27 | 66 |

## Current Small-Scale Experimental Subset

The current pilot experiments use a deterministic hash sample/split from the
full unique-order dataset:

```text
seed = 20260701
limit = 2500 orders
split = 1500 train / 500 dev / 500 test
```

The practical reporting shortcut is:

```text
current ranker training traces: generated from 300 development orders
current final evaluation: separate 500 held-out test orders
```

The 500 test orders are disjoint from the 300 orders used to generate training
traces.

## Held-Out Test Orders

| metric | value |
| --- | ---: |
| orders | 500 |
| items | 797 |
| mean items per order | 1.594 |
| single-item orders | 309 |
| multi-item orders | 191 |
| multi-item share | 38.20% |
| max items per order | 6 |
| all item dimensions are integer-valued | true |
| non-integer dimension values | 0 |
| dimension value range | 2 to 105 |
| unique dimension values | 89 |
| mean order volume | 35,168.692 |
| median order volume | 16,121 |
| p90 order volume | 92,020.5 |
| max order volume | 373,248 |

Held-out test dimension distribution:

| dimension | min | median | mean | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| p | 13 | 35 | 38.499 | 66 | 105 |
| q | 10 | 23 | 26.659 | 45 | 93 |
| r | 2 | 13 | 14.961 | 27 | 63 |

## Simple Reporting Language

Use this wording in slides:

> The data are OR2023 e-commerce order geometries. Each order contains one to
> six items; about 39% of orders contain multiple items. Each item has three
> integer side lengths. The full unique-order dataset has 12,864 orders. In the
> current pilot, we train the learned ranker from exact-search traces generated
> on 300 orders and evaluate on a separate 500-order held-out test set.
