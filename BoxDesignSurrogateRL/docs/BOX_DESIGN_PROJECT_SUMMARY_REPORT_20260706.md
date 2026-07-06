# Box Design Surrogate RL 项目阶段性总结汇报

日期：2026-07-06

## 一句话结论

当前最严谨、最可发表的主张不是“机器学习替代 MILP”，而是：

> 在 OR2023 MILP-grounded box design 的细粒度局部搜索中，学习型候选排序可以减少 exact MILP oracle 工作量；最终解仍由 exact staged-greedy audit 认证，因此 PF 和 coverage 的比较保持可比。

当前 cap-unified 20-window 主实验显示，certified ranker-audit 在不降低最终可行性的前提下，相比 exact staged baseline 实现了正向 aggregate 加速：

| 指标 | 主实验结果 |
| --- | ---: |
| paired windows | 20 |
| PF matched / improved / regressed | 19 / 1 / 0 |
| mean PF delta | -0.0003110651 |
| coverage | 100% |
| uncovered orders | 0 |
| MILP validations reduction | 11.12% |
| uncached MILP boxes reduction | 4.84% |
| Java/Gurobi subprocess time reduction | 5.57% |
| wall-clock time reduction | 5.87% |

这些是当前可以作为 paper-facing pilot evidence 的主结果。更强的 adaptive budget 结果目前只在一个 hard window 上成立，应作为下一步方法方向，而不是主结论。

## 问题定义

任务是选择 `K=10` 个箱型，使 OR2023 order geometries 的 packaging factor 尽量低，同时所有订单都必须被至少一个箱型覆盖。

当前固定的评估口径：

- 数据：OR2023 test split windows。
- 箱型数量：`K=10`。
- 搜索 schedule：`0.25:1000`。
- 可行性判断：Java/Gurobi MILP oracle。
- 主质量指标：PF，越低越好。
- 可行性要求：coverage = 1.0000，uncovered orders = 0。
- 成本指标：MILP validations、uncached MILP boxes、subprocess seconds、wall-clock seconds。

这个问题和原论文私有数据上的 baseline 不能直接等同。我们现在的严谨说法应该是 exact staged baseline under our OR2023 MILP oracle，而不是“严格复现原论文全部实验条件”。

## Baseline

主 paired baseline 是 exact staged greedy：

- same OR2023 test windows；
- same initial condition；
- same `K=10`；
- same `0.25:1000` schedule；
- same Java/Gurobi MILP feasibility oracle；
- same coverage repair rule。

因此，我们的比较不是用 surrogate 改 feasibility，也不是换目标函数，而是在相同 exact objective 和 exact oracle 下比较搜索路径的 oracle 使用量。

## 当前方法

当前主方法是 certified ranker-audit：

1. 生成和 exact staged baseline 相同的 local move candidates。
2. 用 assignment-aware HGBT ranker 给候选 move 排序。
3. 只对排序靠前的 bounded frontier 调用同一个 MILP oracle。
4. 从 ranker 选出的 box set 继续。
5. 最后从 ranker 结果出发运行 exact staged-greedy audit。
6. 只报告 audit 后的 PF 和 coverage。

主实验配置：

- ranker frontier budgets：`20,30,40,50`。
- ranker safety policy：`none`。
- ranker wall-clock cap：所有 seed 统一为 180 seconds。
- final exact audit：required。

这使得方法贡献更清楚：我们不是让 ML 判断最终可行性，而是让 ML 决定哪些候选更值得先被 exact oracle 检查。

## 主实验结果

主证据来自 20 个 100-order paired windows，覆盖 seed1、seed2、seed3、seed4 initial conditions。

质量结果：

- PF matched / improved / regressed：19 / 1 / 0。
- mean PF delta：`-0.0003110651`。
- max PF regression：`0.0000000000`。
- max PF improvement：`-0.0062213021`。
- exact coverage min：`1.0000`。
- ranker-audit coverage min：`1.0000`。
- exact uncovered orders total：`0`。
- ranker-audit uncovered orders total：`0`。

成本结果：

| metric | aggregate reduction | descriptive bootstrap 95% CI | windows reduced / tied / increased |
| --- | ---: | ---: | ---: |
| validations | 11.12% | [7.84%, 15.50%] | 19 / 0 / 1 |
| uncached boxes | 4.84% | [2.93%, 7.42%] | 18 / 1 / 1 |
| subprocess seconds | 5.57% | [3.75%, 8.17%] | 19 / 0 / 1 |
| elapsed seconds | 5.87% | [3.65%, 8.78%] | 19 / 0 / 1 |

解释：

- ranker-audit 没有出现 certified PF regression。
- 所有主窗口最终都保持 100% coverage 和 0 uncovered orders。
- 成本优势是 aggregate claim，不是 every-window claim。
- 有个别 hard window 会出现成本增加，说明固定 180 秒 ranker cap 不是最优预算控制策略。

## Hard Window 发现

最关键的 failure/diagnostic case 是 `seed3:test[400,500)`。

在 fixed 180-second ranker cap 下：

| variant | PF | coverage | validations | uncached boxes | subprocess s | elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact staged baseline | 1.8313420040 | 1.0000 | 20760 | 1919 | 1044.7523 | 1416.1872 |
| fixed ranker180 + audit | 1.8251207019 | 1.0000 | 20920 | 1960 | 1038.2414 | 1458.5628 |

这个窗口很重要，因为它说明：

- ranker-audit 质量更好，PF 下降 `0.0062213021`。
- 但 fixed 180 cap 下 validations、uncached boxes、wall-clock 反而增加。
- 所以问题不在 exact audit，也不在 feasibility correctness，而在 ranker phase 的 budget handoff。

## Adaptive Budget 诊断

我们随后在同一个 hard window 上做了 budget-control 诊断，比较 fixed ranker180、ranker900 no-handoff、以及 marginal PF per validation handoff。

最佳 wall-clock 结果是 `adaptive_3e-6`：

| metric | fixed ranker180 | adaptive 3e-6 | change vs fixed |
| --- | ---: | ---: | ---: |
| audited PF | 1.8251207019 | 1.8251207019 | 0.0000000000 |
| coverage | 1.0000 | 1.0000 | 0.0000 |
| uncovered orders | 0 | 0 | 0 |
| validations | 20920 | 12800 | -38.8% |
| uncached MILP boxes | 1960 | 1626 | -17.0% |
| subprocess seconds | 1038.2414 | 839.2221 | -19.2% |
| wall-clock seconds | 1458.5628 | 1100.4007 | -24.6% |

这个结果说明 adaptive budget control 是有潜力的：

- 同样的 audited PF；
- 同样的 100% coverage；
- 同样的 0 uncovered；
- 相比 fixed ranker180 显著减少 oracle cost。

但这个证据目前只有单窗口。因此它应该被表述为 focused diagnostic 或 next method direction，不能直接写成主实验结论。

## 科研贡献的合理表述

目前最稳的贡献点是：

1. 定义了一个 MILP-grounded box design 搜索问题设置，在 OR2023 order geometry 上用 exact feasibility oracle 比较箱型设计。
2. 提出了 certified learned candidate ordering：ML 只排序候选 move，最终解由 exact audit 认证。
3. 在 20 个 paired OR2023 test windows 上，方法没有 certified PF regression，保持 complete coverage，并降低 aggregate oracle cost。
4. 通过 hard-window 诊断发现，固定短 ranker cap 会导致预算错配，adaptive ranker/audit handoff 是下一步更强的方法方向。

不应该 claim：

- ML 替代 MILP feasibility。
- 当前结果证明 full OR2023 全量 superiority。
- 当前 baseline 是原论文私有数据和完整协议的严格复现。
- 每个窗口每个成本指标都下降。
- 当前 adaptive controller 已经是 broad main result。

## 当前科研判断

项目现在已经从“surrogate predictor 直接优化 PF”转向了更稳的方向：

- 直接用 predictor 过滤候选容易让搜索路径陷入局部最优，也很难保证最终 PF。
- certified ranker-audit 更符合 optimization reviewer 的预期：ML 只减少 exact oracle 的无效调用，最终质量由 exact oracle 认证。
- 目前 20-window evidence 已经有正向结果，但优势还不够大，适合作为 pilot 或 method validation。
- 真正有潜力拉开差距的方向是 adaptive budget control，尤其是在 hard windows 和更细粒度搜索空间下。

## 下一步实验建议

短期最应该做三件事：

1. 在 20-window 主集合上评估 `adaptive_3e-6` 或相近阈值。
   - 目标：验证 hard-window 改善是否能扩展到 broader paired windows。
   - 成功标准：无 PF regression，coverage 100%，aggregate cost 优于 fixed ranker180。

2. 做 budget-controller ablation。
   - 比较 fixed 180、fixed 300、fixed 600、fixed 900、marginal PF handoff。
   - 目的：证明不是“多跑 ranker”本身有效，而是 handoff policy 能更好分配 ranker 与 audit 的 exact oracle 预算。

3. 找到更能放大方法优势的数据或设置。
   - 如果 OR2023 尺寸本身偏整数，0.5 或 0.25 已经接近充分，0.1 的质量收益可能有限。
   - 方法优势应主要体现在 exact oracle 成本随候选空间增大而上升的场景。
   - 需要报告 order dimension granularity 和 candidate-space growth，而不是盲目追求更细 step。

## 汇报时的主线

建议组会或论文汇报按这个顺序讲：

1. 我们先修正问题定位：不是复现私有 baseline 数字，而是在 OR2023 MILP-grounded box design 上做可比实验。
2. 直接 surrogate predictor 不够安全，因为错过 move 会改变 local-search trajectory。
3. 因此方法转为 certified ranker-audit：ML 排序，MILP 认证。
4. 20-window 主结果证明该方法能保持 PF/coverage，并减少 aggregate oracle cost。
5. hard-window 诊断揭示 fixed cap 的不足。
6. adaptive budget control 是下一阶段创新点，有单窗口强证据，但还需要 broader validation。

## 关键证据文件

- 主 claim package：
  `BoxDesignSurrogateRL/docs/CERTIFIED_RANKER_AUDIT_PAPER_CLAIM_PACKAGE_20260705.md`
- cap-unified 20-window summary：
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_SUMMARY_20260706.md`
- cap-unified 20-window stats：
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_STATS_20260706.md`
- aggregate claim audit：
  `BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_AGGREGATE_CLAIM_AUDIT_20260706.md`
- adaptive budget comparison：
  `BoxDesignSurrogateRL/docs/ADAPTIVE_RANKER_BUDGET_SEED3_O400_COMPARISON_20260706.md`
- adaptive handoff controller：
  `BoxDesignSurrogateRL/docs/ADAPTIVE_RANKER_HANDOFF_CONTROLLER_20260705.md`
