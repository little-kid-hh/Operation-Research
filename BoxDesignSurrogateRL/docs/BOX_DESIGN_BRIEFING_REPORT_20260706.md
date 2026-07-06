# Box Design 项目汇报版总结

日期：2026-07-06

## 1. 当前可讲的核心结论

当前最稳的科研主张不是“机器学习完全替代 MILP”，而是：

> 在 OR2023 exact-MILP box design local search 中，学习型候选排序可以减少 exact MILP oracle 工作量；最终结果仍由 exact staged-greedy audit 认证，因此 PF 和 coverage 的比较仍然严格可比。

主结果来自 cap-unified 20-window paired evaluation：

| 指标 | 结果 |
| --- | ---: |
| paired windows | 20 |
| PF matched / improved / regressed | 19 / 1 / 0 |
| mean PF delta, ranker-audit minus exact | -0.0003110651 |
| min exact coverage | 1.0000 |
| min ranker-audit coverage | 1.0000 |
| total uncovered orders, exact / ranker-audit | 0 / 0 |
| MILP validations aggregate reduction | 11.12% |
| uncached MILP boxes aggregate reduction | 4.84% |
| Java/Gurobi subprocess time aggregate reduction | 5.57% |
| wall-clock time aggregate reduction | 5.87% |

这说明当前方法已经能支撑一个谨慎但严谨的 claim：在相同 exact objective、相同 MILP feasibility oracle、相同 initial conditions 和相同 test windows 下，learned candidate ordering 在不造成 certified PF regression、不损失 coverage 的情况下减少了 aggregate oracle cost。

## 2. 固定的问题定义

主论文问题应固定为 **OR2023 Exact-MILP Box Design**：

- 输入：OR2023 BSP unique order geometries。
- 决策：选择 `K=10` 个箱型维度。
- 可行性：order-box pair 必须通过 Java/Gurobi MILP loading oracle。
- 分配：每个 order 分配给体积最小的 MILP-feasible box。
- 目标：最小化 exact-MILP packaging factor，PF 越低越好。
- 有效性门槛：coverage 必须为 1.0000，uncovered orders 必须为 0。

当前不应把主实验说成“严格复现原论文私有数据结果”。更准确的表述是：我们复用了 box-sizing local-search 框架思想，但主结果是在 OR2023 数据和 exact MILP feasibility oracle 下重新定义的可比优化实验。

## 3. Baseline 合约

主 baseline 是 **exact staged greedy under the OR2023 MILP oracle**。

主比较保证：

- same OR2023 test split windows；
- same 100-order window size；
- same initial box set；
- same `K=10`；
- same schedule `0.25:1000`；
- same Java/Gurobi MILP oracle；
- same coverage repair rule；
- same final metric definitions。

因此，当前比较的变量只有搜索策略：exact staged greedy 会按 exact local-search 规则直接评估候选；ranker-audit 先用学习模型排序候选，再对有限候选调用同一个 exact MILP oracle，最后做 exact audit。

## 4. 当前方法：Certified Ranker-Audit

方法流程：

1. 生成和 exact staged baseline 相同的 local move candidates。
2. 用 assignment-aware HGBT ranker 对候选 move 排序。
3. 只对 top ranked bounded frontier 调用 Java/Gurobi MILP oracle。
4. ranker phase 得到一个候选 box set。
5. 从这个 box set 出发运行 exact staged-greedy audit。
6. 只报告 audit 后的 PF、coverage 和 cost。

这点非常关键：ML 没有替代最终可行性判断。ML 的作用是减少低价值 exact oracle queries，最终解仍由 exact oracle 认证。

## 5. 主实验结果解读

质量维度：

- `19 / 1 / 0` 表示 20 个 paired windows 中，ranker-audit 有 19 个窗口 PF 与 exact baseline 持平，1 个窗口 PF 更低，没有窗口 PF 更差。
- mean PF delta 为负数，说明平均上 ranker-audit 的 PF 略优于 exact baseline。
- coverage 两边都为 1.0000，uncovered orders 都为 0，因此 PF 比较没有因为漏覆盖而失真。

成本维度：

| metric | aggregate reduction | descriptive bootstrap 95% CI | windows reduced / tied / increased |
| --- | ---: | ---: | ---: |
| validations | 11.12% | [7.84%, 15.50%] | 19 / 0 / 1 |
| uncached boxes | 4.84% | [2.93%, 7.42%] | 18 / 1 / 1 |
| subprocess seconds | 5.57% | [3.75%, 8.17%] | 19 / 0 / 1 |
| elapsed seconds | 5.87% | [3.65%, 8.78%] | 19 / 0 / 1 |

这些结果应该被解释为 aggregate improvement，而不是 every-window dominance。严格要求每个窗口所有成本指标都下降的 claim 当前不成立；允许少量窗口成本增加的 aggregate claim 已通过当前审计。

## 6. Hard Window 和 Adaptive Budget 发现

最关键的 hard window 是 `seed3:test[400,500)`。

在 fixed ranker180 下，ranker-audit 的 PF 更好，但部分成本更高：

| variant | PF | coverage | validations | uncached boxes | subprocess s | elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact staged baseline | 1.8313420040 | 1.0000 | 20760 | 1919 | 1044.7523 | 1416.1872 |
| fixed ranker180 + audit | 1.8251207019 | 1.0000 | 20920 | 1960 | 1038.2414 | 1458.5628 |

这说明 fixed short ranker cap 可能导致预算分配不合理：ranker 找到了更好的 trajectory，但后续 audit 和候选验证成本没有被充分控制。

在同一个 hard window 上，`adaptive_3e-6` handoff 保持相同 audited PF 和 coverage，同时显著降低成本：

| metric | fixed ranker180 | adaptive_3e-6 | change vs fixed |
| --- | ---: | ---: | ---: |
| audited PF | 1.8251207019 | 1.8251207019 | 0.0000000000 |
| coverage | 1.0000 | 1.0000 | 0.0000 |
| uncovered orders | 0 | 0 | 0 |
| validations | 20920 | 12800 | -38.8% |
| uncached MILP boxes | 1960 | 1626 | -17.0% |
| subprocess seconds | 1038.2414 | 839.2221 | -19.2% |
| elapsed seconds | 1458.5628 | 1100.4007 | -24.6% |

这只是单窗口诊断，不能直接写成 broad main result。它目前最适合用于解释下一步创新方向：ranker 不只是排序候选，还需要一个自适应 handoff controller 来决定什么时候停止 ranker phase、什么时候交给 exact audit。

## 7. 当前仍在运行的验证

远端 Windows/Gurobi 机器正在跑 `adaptive_3e-6` 的 20-window broader validation。

当前状态：

- protocol：`adaptive_3e6_20window_20260706/manifest_frontier_20260706_192617`
- 已写入 manifest records：16 / 20
- summary：尚未生成
- 当前窗口：`seed4:test[100,200)`
- 当前结论处理：pending，不并入主结果

这次 run 完成后需要做三件事：

1. 拉回 summary、CSV、manifest。
2. 跑统计脚本和 claim audit。
3. 如果无 PF regression、coverage 仍为 1.0000，并且 aggregate cost 优于 ranker180，则把 adaptive controller 升级为新的主方法。

## 8. 可以发表的故事线

建议论文/汇报主线：

1. 细粒度 box design with exact loading constraints 的瓶颈在 exact MILP oracle calls。
2. 直接用 surrogate predictor 替代 feasibility 不够安全，因为它会改变 local-search trajectory，且最终可行性难以保证。
3. 我们把 ML 的角色收窄为 candidate ordering：模型只决定优先验证哪些 candidate，最终 feasibility 和 PF 仍由 exact MILP audit 认证。
4. 在 20 个 OR2023 paired test windows 上，certified ranker-audit 无 PF regression、无 uncovered orders，并降低 aggregate oracle cost。
5. hard-window 结果显示 fixed ranker cap 还不够好；adaptive budget handoff 是下一步主要创新点。

## 9. 当前不应过度 claim 的内容

不要 claim：

- ML 已经替代 MILP feasibility oracle。
- 当前已经证明 full OR2023 全量 superiority。
- 当前结果严格复现了原论文私有数据和全部实验条件。
- 每一个窗口、每一个成本指标都下降。
- adaptive controller 已经完成 broad validation。

可以 claim：

- 在当前 paired 20-window evidence 中，ranker-audit 没有 certified PF regression。
- 所有主实验窗口最终 coverage 都为 1.0000，uncovered orders 都为 0。
- aggregate oracle cost 有稳定下降。
- adaptive handoff 在 hard-window diagnostic 上显示出更强潜力，但仍需 broader validation。

## 10. 下一步行动

短期优先级：

1. 等远端 `adaptive_3e-6` 20-window run 完成并拉回结果。
2. 运行同一套统计和 claim audit。
3. 如果 adaptive 结果通过质量门槛，更新主 claim package，把方法从 fixed ranker-audit 升级为 adaptive certified ranker-audit。
4. 如果 adaptive 结果只在部分窗口改善，则保留 fixed ranker180 作为主方法，把 adaptive controller 写成 ablation/diagnostic。
5. 准备论文层面的 method section：明确 ML ranker、MILP oracle、exact audit、成本指标和失败条件。

## 11. 汇报用一句话

> 我们现在已经不把学习模型当作不可靠的 feasibility 替代品，而是把它放在 exact MILP 搜索前面做候选排序；最终解仍由 exact audit 认证。当前 20-window 结果显示 PF/coverage 不损失，同时减少 aggregate MILP oracle cost；下一步要验证 adaptive handoff 能否把这个加速幅度进一步放大。

## 12. 证据文件

- 主 claim package：`BoxDesignSurrogateRL/docs/CERTIFIED_RANKER_AUDIT_PAPER_CLAIM_PACKAGE_20260705.md`
- 项目阶段性长报告：`BoxDesignSurrogateRL/docs/BOX_DESIGN_PROJECT_SUMMARY_REPORT_20260706.md`
- 20-window 主 summary：`BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_SUMMARY_20260706.md`
- 20-window 统计：`BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_STATS_20260706.md`
- aggregate claim audit：`BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_AGGREGATE_CLAIM_AUDIT_20260706.md`
- hard-window adaptive budget comparison：`BoxDesignSurrogateRL/docs/ADAPTIVE_RANKER_BUDGET_SEED3_O400_COMPARISON_20260706.md`
- formal problem definition：`BoxDesignSurrogateRL/docs/FORMAL_PROBLEM.md`
