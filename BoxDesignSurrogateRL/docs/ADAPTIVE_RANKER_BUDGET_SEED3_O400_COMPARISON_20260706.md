# Ranker budget comparison: seed3:test[400,500)

This is a focused diagnostic comparison on one hard validation window. It is not a replacement for the multi-window main result.

| variant | kind | final PF | PF delta vs exact | coverage | uncovered | validations | uncached | subprocess s | elapsed s | ranker stop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_staged_baseline | exact | 1.8313420040 | 0.0000000000 | 1.0000 | 0 | 20760 | 1919 | 1044.7523 | 1416.1872 |  |
| fixed_ranker180_audit | ranker_audit | 1.8251207019 | -0.0062213021 | 1.0000 | 0 | 20920 | 1960 | 1038.2414 | 1458.5628 | time_limit |
| no_handoff_ranker900 | ranker_audit | 1.8251207019 | -0.0062213021 | 1.0000 | 0 | 12650 | 1626 | 850.9569 | 1115.0527 | time_limit |
| adaptive_1e-5 | ranker_audit | 1.8251207019 | -0.0062213021 | 1.0000 | 0 | 13090 | 1626 | 842.7632 | 1119.9636 | ranker_marginal_pf_handoff |
| adaptive_5e-6 | ranker_audit | 1.8251207019 | -0.0062213021 | 1.0000 | 0 | 12850 | 1626 | 837.5591 | 1115.1521 | ranker_marginal_pf_handoff |
| adaptive_3e-6 | ranker_audit | 1.8251207019 | -0.0062213021 | 1.0000 | 0 | 12800 | 1626 | 839.2221 | 1100.4007 | ranker_marginal_pf_handoff |

Cost reductions use `fixed_ranker180_audit` as the fixed-ranker reference.

| variant | val vs exact | uncached vs exact | subprocess vs exact | elapsed vs exact | val vs reference | uncached vs reference | subprocess vs reference | elapsed vs reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_ranker180_audit | -0.8% | -2.1% | 0.6% | -3.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| no_handoff_ranker900 | 39.1% | 15.3% | 18.5% | 21.3% | 39.5% | 17.0% | 18.0% | 23.6% |
| adaptive_1e-5 | 36.9% | 15.3% | 19.3% | 20.9% | 37.4% | 17.0% | 18.8% | 23.2% |
| adaptive_5e-6 | 38.1% | 15.3% | 19.8% | 21.3% | 38.6% | 17.0% | 19.3% | 23.5% |
| adaptive_3e-6 | 38.3% | 15.3% | 19.7% | 22.3% | 38.8% | 17.0% | 19.2% | 24.6% |

## Interpretation

- Exact audit keeps the final PF and coverage measurable under the original MILP oracle.
- Fixed short ranker caps can be too short on hard windows: they may preserve or improve PF after audit but still increase some costs.
- Adaptive handoff is a budget-control diagnostic: it tests whether spending more ranker time before exact audit reduces the downstream exact workload on the same hard window.
