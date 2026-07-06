# Box Design 项目汇报稿（RL 主线修正版）

日期：2026-07-07

## 0. 重要纠正：项目主线仍然是 RL

本项目的研究目标没有改变：我们要做的是 **Kandula-style / RL-style box design**，也就是在箱型设计问题中学习一个能够选择箱型调整动作的策略。

前一版表述里把当前已有最稳的 `ranker + exact audit` 结果写成了“主方法”，这容易误导。更准确的关系是：

```text
最终研究主线：RL / learned policy for box design
当前已验证辅助结果：supervised ranker + exact audit 可以减少 MILP oracle cost
```

因此，汇报时应该这样定位：

- **RL 是项目主线**：状态、动作、奖励和原论文框架都围绕 RL-style box sizing。
- **Exact staged greedy 是当前严谨 baseline**：用于提供可比的 exact MILP 搜索参考。
- **HGBT ranker 不是 RL**：它是目前跑出稳定结果的监督学习辅助模块，用来说明“学习模型可以减少 exact oracle 查询”，也可以作为后续 RL 搜索中的候选筛选/动作优先级模块。
- **当前最终表格不是 RL policy 的最终胜利**：它是 ranker-audit 的已验证结果，应作为中间证据或辅助模块证据，而不是说项目目标已经从 RL 改成 ranker。

## 1. 一句话总结

我们现在应该汇报成两层结论：

> 项目主线是复现并扩展 RL-style box design：状态是箱型尺寸，动作是调整箱子维度，奖励来自 PF 改善。在当前阶段，我们先建立了 exact MILP baseline，并验证了一个 supervised ranker 辅助模块：它不替代 MILP，而是对候选动作排序，从而减少 MILP oracle 查询。这个 ranker 结果是支持 RL 主线的中间证据，不是最终把 RL 换掉。

所以明天不要说“我们已经不做 RL”。正确说法是：

> 我们的目标仍是 RL box design；当前最稳定的已完成实验是一个可认证的 learned candidate ordering 模块，它可以作为 RL 搜索中的加速/候选筛选组件。

## 2. 数据集是什么

我们使用的是 OR2023 BSP unique-order 数据。它是电商订单的三维装箱数据。

一个订单长这样：

```xml
<order id="165">
  <item id="0"><p>40</p><q>37</q><r>22</r></item>
  <item id="1"><p>40</p><q>37</q><r>22</r></item>
</order>
```

也就是说：

- 一个 order 里可以有 1 个或多个 items；
- 每个 item 有三个尺寸 `p/q/r`；
- 当前 XML 里的 item 尺寸都是整数，没有小数；
- 方向可以旋转，所以最终是否能装进某个箱子不能只靠简单长宽高比较，主实验里用 MILP 判断。

全量数据大概如下：

| 指标 | 数值 |
| --- | ---: |
| unique orders | 12,864 |
| total items | 20,632 |
| 平均每单 item 数 | 1.604 |
| 单 item orders | 7,905 |
| 多 item orders | 4,959 |
| 多 item 占比 | 38.55% |
| 每单最多 item 数 | 6 |
| 尺寸是否整数 | 全部是整数 |
| 尺寸范围 | 2 到 118 |
| 平均 order volume | 34,263.758 |
| median order volume | 16,470 |
| p90 order volume | 84,000 |
| max order volume | 864,000 |

item 数量分布：

| 每单 item 数 | order 数 |
| --- | ---: |
| 1 | 7,905 |
| 2 | 3,377 |
| 3 | 856 |
| 4 | 392 |
| 5 | 167 |
| 6 | 167 |

## 3. 当前训练和测试怎么来

当前实验不是直接用全量 12,864 个订单跑完整优化，因为 exact MILP 很慢。

我们从 OR2023 里固定抽样/划分出一个 2,500-order 子集：

| 部分 | orders | 当前用途 |
| --- | ---: | --- |
| reserved train | 1,500 | 预留，当前主结果没有重点使用 |
| development orders | 500 | 用来生成训练搜索记录和调方法 |
| held-out test orders | 500 | 最终比较 baseline 和 ranker-audit 辅助模块 |

为了汇报简单，可以说：

> 当前 ranker 的训练样本来自 300 个订单上跑 exact baseline 得到的搜索记录；最终测试使用另一组不重叠的 500 个测试订单。

注意：ranker 的训练样本不是原始 order 表的一行。训练样本是 baseline 搜索过程中产生的候选动作记录。

具体来说：

1. 在 300 个训练订单上跑 exact baseline。
2. baseline 每一步会生成约 60 个候选箱型调整动作。
3. MILP oracle 评估这些候选动作的真实效果。
4. 记录每个候选动作的特征和结果。
5. 用这些记录训练 ranker。

当前 ranker 训练数据规模：

| 指标 | 数值 |
| --- | ---: |
| 训练订单数 | 300 |
| exact search steps | 128 |
| candidate move records | 7,680 |

## 4. 我们优化的问题是什么

目标是设计 `K=10` 个箱型，使测试订单都能被装下，同时平均包装浪费尽量小。

每个候选箱型集合会被这样评估：

1. 对每个 order 和每个 box，调用 Java/Gurobi MILP oracle 判断是否能装下。
2. 每个 order 分配给体积最小的可行 box。
3. 计算 packaging factor。

Packaging factor 定义为：

```text
PF = mean assigned box volume / mean order volume
```

越低越好。

主实验有效性要求：

```text
coverage = 100%
uncovered orders = 0
```

如果一个方法 PF 看起来更低，但漏掉订单，这不能算胜利。

## 5. Baseline 是什么

当前主 baseline 是：

> exact staged greedy under the OR2023 MILP oracle

它不是原论文私有数据上的严格复现。

Baseline 配置：

| 项 | 配置 |
| --- | --- |
| dataset | OR2023 held-out test orders |
| box 数量 | `K=10` |
| 初始化 | KMeans initial boxes |
| 搜索动作 | 调整某个箱子的某个维度 |
| schedule | `0.25:1000` |
| feasibility | Java/Gurobi MILP oracle |
| orientation label | `label_6ori` |
| coverage repair | `geometric_expand` |
| 目标 | 先保证 coverage，再最小化 PF |

Baseline 每一步基本做的是：

```text
生成所有候选动作
对候选动作调用 MILP
选择 exact objective 最好的动作
```

它质量可靠，但 MILP 调用很贵。

## 6. 和原论文 / RL 的关系

我们早期实现过 Kandula-style 强化学习框架复现。

这个 RL 框架大概是：

| 项 | 配置 |
| --- | --- |
| 状态 | `K x 3` 个箱型尺寸 |
| K | 10 |
| 状态维度 | 30 |
| 动作 | 对某个箱子的某个维度 `+step` 或 `-step`，再加 resign action |
| 动作数 | `6K + 1 = 61` |
| 默认 step | paper-style `0.5` |
| reward | PF 下降给正 reward |
| policy | PPO actor-critic |

这条线是项目主线，但目前只能说是“算法框架复现”，不是严格数值复现原论文。

原因：

- 原论文使用私有数据，我们没有；
- 原论文的完整 online companion 超参数和实现细节不完整；
- 我们的数据是 OR2023 multi-item order geometry；
- 主实验使用 Java/Gurobi MILP oracle，和原论文 feasibility setting 不一样。

所以汇报时应这样说：

> 我们复现了原论文的状态/动作/奖励框架，并把它迁移到 OR2023 exact-MILP box design 问题上。由于原论文私有数据和完整实现不可得，当前不是数值复现，而是框架复现。为了让 RL 主线有严谨参照，我们先固定 exact staged greedy baseline；同时，当前 ranker-audit 结果说明 learned candidate ordering 可以减少 MILP oracle cost，后续可以并入 RL 搜索。

## 7. 当前已验证的辅助方法是什么

当前已验证的辅助方法是：

> certified ranker-audit

它不是 RL，也不是 feasibility predictor。它不是用来替代项目主线的，而是目前已经完成验证的学习型候选排序模块。

它是一个监督学习 ranker，模型是：

```text
HGBT, HistGradientBoostingTree
```

ranker 输入不是单个 order，而是：

```text
当前搜索状态 + 一个候选箱型调整动作
```

例如一条输入可以理解成：

```text
当前有 10 个箱子；
候选动作是：把第 5 个箱子的高度 +0.25；
ranker 判断这个动作值不值得优先用 MILP 检查。
```

ranker 输出是一个 priority score：

```text
这个候选动作值得优先检查的程度
```

每一步搜索大约有 60 个候选动作，因为：

```text
10 个箱子 x 3 个维度 x 加/减两个方向 = 60
```

ranker 给这 60 个动作分别打分，然后按分数排序。

这个模块和 RL 的关系可以这样讲：

> RL policy 最终也要在一堆箱型调整动作中选择下一步。Ranker 学到的是“哪些候选动作更值得优先检查”的信息，因此可以作为 RL 搜索里的候选动作筛选器、动作 prior，或者用于减少 RL/PAAS 阶段的 MILP oracle 调用。

## 8. Ranker 看哪些信息

ranker 的特征包括：

| 信息 | 例子 |
| --- | --- |
| 当前箱子尺寸 | 10 个 box 的长宽高、体积 |
| 候选动作 | 改哪个箱子、哪个维度、增大还是减小 |
| 改完后的箱子 | 新长宽高、新体积、体积变化 |
| 当前订单分配 | 当前有多少订单分配给被改的箱子 |
| 订单体积统计 | 这些订单的总体积、平均体积 |
| 潜在 assignment 改善 | 箱子变大后可能吸收多少原本分配给更大箱子的订单 |
| shrink 风险 | 箱子变小后可能影响多少当前已分配订单 |

这个 assignment-aware 信息很重要。

有些动作会让某个箱子变大，单看箱子体积似乎不好；但它可能让很多订单从更大的箱子转移到这个箱子，整体 PF 反而下降。ranker 要学习的就是这种搜索经验。

## 9. 当前 ranker-audit 辅助模块怎么运行

每一步流程：

```text
1. 生成和 baseline 相同的约 60 个候选动作
2. HGBT ranker 给每个候选动作打分
3. 按分数排序
4. 优先检查 top-ranked candidates
5. 被检查的候选动作仍然调用 MILP oracle
6. ranker phase 结束后，再跑 exact staged-greedy audit
7. 最终只报告 audit 后的 PF 和 coverage
```

当前主实验的 ranker budget 是：

```text
20,30,40,50
```

意思是：

```text
先检查 top 20；
不够再扩大到 top 30；
再不够扩大到 top 40；
最多到 top 50。
```

所以它不是固定只筛 30 个。

## 10. Baseline 和当前辅助模块的核心区别

| 方面 | exact staged baseline | ranker-audit 辅助模块 |
| --- | --- | --- |
| 候选动作集合 | 约 60 个 coordinate moves | 同样的约 60 个 coordinate moves |
| 候选顺序 | 直接 exact 评估 | 先由 HGBT ranker 排序 |
| MILP 调用 | 倾向检查更多候选 | 优先检查高分候选 |
| 可行性判断 | MILP | MILP |
| 最终质量认证 | exact search result | ranker search 后再 exact audit |
| ML 是否决定最终可行性 | 否 | 否 |
| 目标 | 降低 PF | 同样降低 PF，同时减少 oracle cost |

一句话：

> Baseline 是用 MILP 直接搜索；ranker-audit 辅助模块是用机器学习先排序候选动作，减少无效 MILP 查询，但最终仍由 MILP 认证。

## 11. 测试设置

最终测试使用 500 个 held-out test orders。

因为 MILP 很慢，我们没有一次性跑 500 个订单，而是按每 100 个订单一组评估：

```text
test orders 0-99
test orders 100-199
test orders 200-299
test orders 300-399
test orders 400-499
```

每组订单用 4 个不同初始箱型条件重复：

```text
5 组订单 x 4 个 initial seeds = 20 组 paired comparison
```

每一组 paired comparison 中，baseline 和 ranker-audit 辅助模块共享：

- 同一批订单；
- 同一组初始箱型；
- 同一个 `K=10`；
- 同一个 MILP oracle；
- 同一个目标函数；
- 同一个 coverage 要求。

所以这是公平的一一对应比较。

## 12. 当前已完成的 ranker-audit 指标对比

当前已完成结果基于 20 组 paired comparison：

| 指标 | Exact staged baseline | Ranker + exact audit | 变化 |
| --- | ---: | ---: | ---: |
| PF 持平 / 变好 / 变差 | - | 19 / 1 / 0 | 0 组变差 |
| coverage | 100% | 100% | 不变 |
| uncovered orders | 0 | 0 | 不变 |
| MILP validations | 273,960 | 243,490 | -11.12% |
| uncached MILP boxes | 25,948 | 24,693 | -4.84% |
| Java/Gurobi subprocess time | 14,360.755 s | 13,560.486 s | -5.57% |
| wall-clock time | 19,401.358 s | 18,262.778 s | -5.87% |

这说明：

- ranker-audit 没有造成 PF regression；
- coverage 保持 100%；
- uncovered orders 保持 0；
- MILP oracle 工作量和运行时间有 aggregate reduction。

当前优势主要体现在减少 exact oracle cost，而不是显著降低 PF。

这张表应该被讲成：

> 当前学习型候选排序模块的已验证结果。

不要讲成：

> 最终 RL policy 已经超过 baseline。

## 13. 指标怎么解释

| 指标 | 含义 | 越低/越高 |
| --- | --- | --- |
| PF | 平均分配箱体积 / 平均订单体积 | 越低越好 |
| coverage | 有至少一个可行箱子的订单比例 | 越高越好，主实验要求 100% |
| uncovered orders | 没有任何可行箱子的订单数 | 必须为 0 |
| MILP validations | 搜索过程中评估候选动作的次数 | 越低越省 |
| uncached MILP boxes | 没被缓存命中的新 box feasibility 查询 | 越低越好 |
| subprocess time | Java/Gurobi oracle 本身耗时 | 越低越好 |
| wall-clock time | 端到端总耗时 | 越低越好 |

其中 `uncached MILP boxes` 和 `subprocess time` 最能反映当前实现里的 exact oracle 成本。

## 14. 500 个订单一次性跑会不会优势更大

不能直接 claim。

直觉上，订单越多，每次 MILP candidate evaluation 越贵，ranker 减少无效候选检查的价值可能更大。

但也有风险：

- 订单更多后 assignment 更复杂；
- ranker 可能更容易漏掉关键 move；
- cache 和 Java/Gurobi subprocess 行为可能非线性变化；
- exact audit 的成本也可能上升。

所以严谨说法是：

> 当前结果证明了 100-order 批次上的 controlled paired evidence。500-order 一次性测试是否优势更明显，需要单独运行，不能直接外推。

## 15. Adaptive budget 的补充发现

有一个 hard case：`seed3:test[400,500)`。

在 fixed ranker180 下，ranker-audit 的 PF 更好，但部分成本反而上升。后来试了 adaptive handoff，在这个 hard case 上有更好结果：

| 指标 | fixed ranker180 | adaptive_3e-6 | 变化 |
| --- | ---: | ---: | ---: |
| audited PF | 1.8251207019 | 1.8251207019 | 不变 |
| coverage | 100% | 100% | 不变 |
| uncovered orders | 0 | 0 | 不变 |
| validations | 20,920 | 12,800 | -38.8% |
| uncached MILP boxes | 1,960 | 1,626 | -17.0% |
| subprocess time | 1,038.241 s | 839.222 s | -19.2% |
| wall-clock time | 1,458.563 s | 1,100.401 s | -24.6% |

但是这只是单个 hard case 的诊断结果，不能作为主结论。它适合作为下一步方向：

> ranker 不仅要排序候选动作，还要学习/设计什么时候停止 ranker phase、什么时候交给 exact audit。

## 16. 可以 claim 什么

当前可以讲：

1. OR2023 数据中约 39% 是 multi-item orders，主实验用 MILP oracle 判断真实 loading feasibility。
2. 项目目标仍是 RL-style box design；我们已经实现了状态、动作、奖励形式的框架复现。
3. 当前 ranker 不是 feasibility predictor，而是 supervised candidate ranker。
4. ranker 训练来自 300 个订单上 exact baseline 产生的 7,680 条 candidate move records。
5. 最终测试在不重叠的 500 个 held-out test orders 上完成。
6. 在 20 组 paired comparison 中，ranker + exact audit 没有 PF 变差，coverage 保持 100%，uncovered orders 为 0。
7. ranker-audit 辅助模块减少了 aggregate MILP validations、uncached MILP boxes、subprocess time 和 wall-clock time。

## 17. 不要 claim 什么

当前不要说：

- 我们严格复现了原论文全部实验。
- 当前 ranker 表格就是最终 RL 结果。
- ML 已经替代了 MILP。
- ranker 是预测 order-box feasibility。
- 每个测试组都更快。
- 500-order 一次性测试一定优势更大。
- adaptive handoff 已经完成完整 broad validation。
- 项目目标已经从 RL 改成 ranker。

更稳的表述是：

> 项目主线仍是 RL-style box design。当前最可靠的已验证结果是一个 supervised learned candidate ordering 辅助模块，它可以作为后续 RL/PAAS 搜索减少 MILP oracle cost 的组件。

## 18. 汇报时推荐讲法

可以按这个顺序讲：

1. 数据：OR2023 电商订单，每单 1 到 6 个 items，尺寸为整数，约 39% 是多 item。
2. 问题：设计 10 个箱型，所有订单都要能装下，同时 PF 尽量低。
3. RL 框架：状态是 10 个箱子的三维尺寸，动作是对某个维度加减，奖励来自 PF 改善。
4. Baseline：exact staged greedy，每一步用 MILP 检查候选动作，质量可靠但慢。
5. 早期尝试：直接用 ML feasibility predictor 替代 MILP 不够安全，容易改变搜索轨迹。
6. 当前辅助模块：HGBT ranker 对候选箱型调整动作排序，优先检查高分动作。
7. 保证：ranker-audit 结果的最终可行性和 PF 仍由 MILP + exact audit 认证。
8. 当前证据：20 组 paired comparison 中无 PF 变差，coverage 100%，MILP validations 降低 11.12%，wall-clock 降低 5.87%。
9. 下一步：把 learned candidate ordering / adaptive handoff 并入 RL/PAAS 主线，并在更大测试规模上验证。

## 19. 最短版本

如果只讲一页：

> 我们的问题是 OR2023 多 item 订单的箱型设计。数据里有 12,864 个 unique orders，约 39% 是多 item，item 尺寸都是整数。项目主线是 RL-style box design：状态是 10 个箱子的尺寸，动作是调整某个箱子的某个维度，奖励来自 PF 改善。为了建立严谨对照，我们使用 exact staged greedy 作为 baseline，每一步由 MILP 检查候选动作。当前已验证的学习模块是 HGBT candidate ranker，它不替代 MILP，而是对候选动作排序，优先检查更有价值的动作，最后再用 exact audit 认证结果。在 20 组 paired comparison 中，ranker-audit PF 没有变差，coverage 保持 100%，同时 MILP validations 减少 11.12%，wall-clock time 减少 5.87%。下一步是把这个 learned candidate ordering 机制整合回 RL/PAAS 主线。

## 20. 证据文档

- 数据画像：`BoxDesignSurrogateRL/docs/OR2023_DATA_PROFILE_20260706.md`
- 汇报简版：`BoxDesignSurrogateRL/docs/BOX_DESIGN_BRIEFING_REPORT_20260706.md`
- 主结果统计：`BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_STATS_20260706.md`
- 主结果 summary：`BoxDesignSurrogateRL/docs/ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_SEED4_RANKER180_TESTSPLIT_SUMMARY_20260706.md`
- claim package：`BoxDesignSurrogateRL/docs/CERTIFIED_RANKER_AUDIT_PAPER_CLAIM_PACKAGE_20260705.md`
