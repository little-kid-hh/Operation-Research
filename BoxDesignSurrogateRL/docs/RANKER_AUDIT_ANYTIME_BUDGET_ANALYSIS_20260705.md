# Ranker-Audit Anytime Budget Analysis

- Windows in summary: 15
- Windows with complete traces: 15
- PF tolerance: `1e-12`

## Validation Budget Comparison

| exact validation budget | n | ranker better/tied/worse | mean PF delta | min PF delta | max PF delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25% | 15 | 15/0/0 | -0.0611865119 | -0.1366151300 | -0.0222638296 |
| 50% | 15 | 15/0/0 | -0.0513849825 | -0.1319209297 | -0.0086565969 |
| 75% | 15 | 15/0/0 | -0.0282371740 | -0.0710819043 | -0.0041595101 |
| 100% | 15 | 1/14/0 | -0.0004147535 | -0.0062213021 | 0.0000000000 |

## Time To Exact Final PF

- Windows: 15
- Exact validations: 209580
- Ranker-audit validations: 152360
- Aggregate reduction: 27.30%
- Windows reduced/tied/increased: 15/0/0

## Per-Window Final Validation Cost

| window | exact validations | ranker-audit validations | reduction | final PF delta |
| --- | ---: | ---: | ---: | ---: |
| seed1:test[0,100) | 10620 | 8620 | 18.8% | 0.0000000000 |
| seed1:test[100,200) | 26280 | 25120 | 4.4% | 0.0000000000 |
| seed1:test[200,300) | 9420 | 8220 | 12.7% | 0.0000000000 |
| seed1:test[300,400) | 15540 | 13620 | 12.4% | 0.0000000000 |
| seed1:test[400,500) | 7920 | 4280 | 46.0% | 0.0000000000 |
| seed2:test[0,100) | 9840 | 7800 | 20.7% | 0.0000000000 |
| seed2:test[100,200) | 16200 | 15120 | 6.7% | 0.0000000000 |
| seed2:test[200,300) | 9480 | 8280 | 12.7% | 0.0000000000 |
| seed2:test[300,400) | 7860 | 5820 | 26.0% | 0.0000000000 |
| seed2:test[400,500) | 23160 | 21040 | 9.2% | 0.0000000000 |
| seed3:test[0,100) | 8580 | 3730 | 56.5% | 0.0000000000 |
| seed3:test[100,200) | 19980 | 10460 | 47.6% | 0.0000000000 |
| seed3:test[200,300) | 13800 | 6560 | 52.5% | 0.0000000000 |
| seed3:test[300,400) | 11040 | 4380 | 60.3% | 0.0000000000 |
| seed3:test[400,500) | 20760 | 12660 | 39.0% | -0.0062213021 |
