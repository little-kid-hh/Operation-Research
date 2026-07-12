# Kandula-style RL + 机器学习可行性判断：负结果报告

## 摘要结论

在当前 OR2023 多物品订单数据、六方向精确 MILP 可行性定义和统一实验口径下，复现的 Kandula-style 强化学习框架加机器学习可行性 surrogate，**没有超过更简单的监督学习 multiscale ranker**。

这份报告不声称“强化学习普遍不适合箱型设计”。当前证据支持的范围更窄：

> 已测试的 direct-action PPO、PPO rollout、PPO 候选并集、FQI 验证预算控制和二元步长控制，都没有在相同 MILP 最终验证条件下，给 frozen supervised ranker 带来独立的解质量或计算效率提升。

当前最强方法因此不是 RL，而是：

```text
多尺度动作枚举
→ 监督学习 ranker 排序
→ 少量候选调用 MILP
→ MILP 决定接受动作并保证最终可行性
```

## 1. 复现范围

### 1.1 复现了 Kandula 论文框架的哪些部分

当前实现复现的是算法结构：

1. 使用 KMeans 初始化 `K` 种箱型；
2. 状态包含当前箱型的三维尺寸；
3. 动作调整某个箱子的某一维；
4. 以 packaging factor（PF）改善为主要奖励；
5. 训练策略进行连续决策；
6. 使用策略概率或 rollout 辅助动作评分。

固定步长时，动作空间为 `6K + 1`：每个箱子的长、宽、高均可增减，加一个停止动作。`K=10` 时共 61 个动作。multiscale 扩展使用 `{2, 1, 0.5, 0.25}` 四种步长，共 240 个坐标动作。

### 1.2 为什么不能称为“与原论文严格一模一样”

这是 framework reproduction，不是原论文私有数据实验的 exact numerical reproduction：

- 原论文主要是 SKU/需求驱动的箱型设计；本项目使用 OR2023 多物品 order geometry。
- 数据集、划分、`K`、随机种子、训练轮数和 horizon 都是本地实验配置。
- 当前可发表问题中的可行性由 Java/Gurobi 六方向多物品 3D packing MILP 定义。
- 论文风格的 aggregate proxy 与精确 MILP 是两个不同优化问题，PF 不能混合比较。

准确表述应当是：**Kandula-style RL adapted to OR2023 exact-MILP box design**。

## 2. 固定的优化问题

给定 `K=10` 个箱型，每个订单分配给 MILP 证明可行且体积最小的箱子。搜索目标采用字典序：

```text
1. 最小化 uncovered orders；
2. 最小化 unknown MILP pairs；
3. 最小化 exact-MILP PF。
```

有效结果必须满足 100% coverage、零 uncovered、零 unknown。通过损失覆盖率得到的低 PF 不算胜利。

机器学习 feasibility model 预测一个 order 能否装入候选 box。在 surrogate RL 环境中，它会影响订单分配、PF、reward 和终止惩罚，因此不只是一个最后的过滤器。所有正式比较中的候选动作和最终解仍由同一个 MILP oracle 验证。

## 3. 比较的方法

### Exact staged baseline

枚举坐标动作，用 MILP 判断每个候选箱型集合的可行性与 PF，贪心接受字典序最优的改善动作。

### Supervised multiscale ranker

枚举 240 个 multiscale 动作，用历史 exact trace 训练的 HGBT ranker 排序，再由 MILP 从 shortlist 中选择动作。这是监督学习搜索，不是 RL。

### 测试过的 RL 变体

- 用 feasibility surrogate 和 PF reward 训练 direct-action PPO；
- 用 PPO rollout 对 ranker 候选做多步评分；
- 把 multiscale PPO 的 top 动作加入 ranker shortlist；
- 用 FQI 动态选择送入 MILP 的候选数量；
- 用 counterfactual terminal query cost 设计 coarse/fine step controller。

## 4. 实验结果

### 4.1 Matched-step PPO rollout：PF 不变，成本增加

在 100-order diagnostic exact smoke test 上：

| 方法 | PF | MILP validations | Uncached boxes | Subprocess | Wall |
|---|---:|---:|---:|---:|---:|
| Ranker-only | **1.986088** | **1000** | **275** | **111.04 s** | **133.03 s** |
| Ranker + matched PPO rollout | **1.986088** | 1291 | 306 | 134.13 s | 174.39 s |

两者动作序列完全相同。PPO 多做 291 次 exact rollout evaluation，但没有改变 PF 或搜索路径。该 smoke 仅作诊断，不作为 held-out 论文结果；它足以否决这一个 PPO 配置的独立贡献。

### 4.2 Multi-window PPO：改变动作，但终局没有改善

PPO 只使用 frozen train split 的 1,500 个订单训练，划分为 15 个互不重叠的 100-order 环境。在一个 exact development window 上：

| 方法 | PF | MILP validations | Uncached boxes | Subprocess | Wall |
|---|---:|---:|---:|---:|---:|
| Exact staged | 2.472933 | 3000 | 582 | 317.82 s | 406.07 s |
| Ranker-only | **2.472933** | **1000** | **492** | **264.34 s** | **314.49 s** |
| Ranker + PPO rollout | **2.472933** | 2215 | 531 | 293.05 s | 373.75 s |

RL 改变了 50 步中的 12 个动作，但这些变化主要是等价的宽/高调整，或暂时更差、随后追平的动作顺序。它比 ranker-only 多 1,215 次 rollout validation 和 59.26 秒 wall time，终局 PF 没有改善。

加入 order-distribution context 也没有解决问题。测试策略在五个 surrogate dev window 上坍缩为重复同一个动作，并且未通过 coverage screening，因此没有进入 exact 实验。

### 4.3 Multiscale PPO 候选并集：候选不同，但没有一个被 MILP 选中

配对五步 development smoke：

| 方法 | 五步后 PF | Validations | Uncached boxes | Wall |
|---|---:|---:|---:|---:|
| Ranker-only | **2.3566543532** | **50** | **113** | **77.80 s** |
| Ranker + PPO union | **2.3566543532** | 75 | 118 | 82.36 s |

PPO 每轮补充的五个候选与 ranker shortlist 的重合数为 0，说明它确实提出了新动作。但是这些动作经过 MILP 验证后没有一个胜出；两种方法最终选择完全相同的五个动作。候选“不同”不等于候选“有价值”。

### 4.4 FQI 验证预算：输给固定自适应规则

在 held-out training episode `[200,300)` 跑到收敛后：

| 方法 | Terminal PF | Validations | Uncached boxes | Subprocess | Wall |
|---|---:|---:|---:|---:|---:|
| Exact staged | 2.006760 | 21,300 | 1,963 | 840.91 s | 1,219.42 s |
| Fixed ranker budget | **2.006760** | **6,920** | **1,452** | **644.28 s** | **855.41 s** |
| FQI budget + audit | **2.006760** | 10,760 | 1,566 | 742.16 s | 1,032.25 s |

相对 fixed adaptive ranker，FQI：

- validations 增加 55.5%；
- uncached boxes 增加 7.9%；
- subprocess 增加 15.2%；
- wall time 增加 20.7%；
- PF 和 coverage 没有改善。

因此简单的 `10 → 30 → full audit` 规则更强。

### 4.5 Step-control RL：完美策略的理论收益也只有 1.03%

在训练模型前，使用 exact branch-and-follow 计算 coarse/fine 首步决策的 hindsight oracle 上界：

| Iteration | Fine queries | Coarse queries | Oracle choice |
|---:|---:|---:|---|
| 1 | **850** | 881 | fine |
| 50 | 739 | **725** | coarse |
| 100 | **651** | 653 | fine |
| 150 | 536 | **534** | coarse |
| 200 | 417 | **409** | coarse |

每一对分支都达到 PF `1.7253077469`、100% coverage 和相同 terminal boxes。完美 hindsight policy 使用 3,160 个 queries，只比永远选择 fine 少 33 个，即 1.03%。真实学习策略不可能超过 hindsight oracle，因此该 action definition 在训练前被否决。

## 5. 为什么这些 RL 方案失败

### 5.1 Surrogate 误差会改变整条优化轨迹

Feasibility predictor 决定订单分配、PF、reward 和 terminal penalty。False negative 会隐藏好动作；false positive 会让 surrogate 认为某个动作很好，但在 MILP 验证时失败。连续的小误差会改变 policy 访问的状态并形成 surrogate 特有的局部最优。

### 5.2 训练环境和正式推理的 transition model 不一致

PPO 在近似 feasibility 下学习转移与奖励，正式结果却按精确 MILP feasibility 决策。策略可能很好地优化 surrogate environment，但无法正确排序 exact-MILP 动作。

### 5.3 原始 box-only state 存在部分可观测性

相同的 `K × 3` box dimensions 在不同订单分布下具有不同价值。只观察箱型的共享 policy 不知道自己正在优化哪一批订单。加入简单 order context 是合理尝试，但当前 context variant 仍发生 action collapse。

### 5.4 很多固定步长动作在长期上可交换

固定 `0.25` 的 exact、ranker 和 FQI 收敛实验接受相同 multiset 的 354 个动作，并得到 byte-identical boxes，只是动作顺序不同。反事实 rank-1/rank-2 首动作也最终合并。底层动作本身没有终局差异时，RL rollout 无法凭空创造质量提升。

### 5.5 Supervised ranker 已经提取了大部分有效信号

Ranker 直接学习 exact search 历史上接受哪些候选。PPO 面对的是已经大幅压缩后的 shortlist。额外候选通常冗余或被 MILP 拒绝，而 rollout 自身还增加 exact evaluation。

### 5.6 部分 RL 决策变量根本没有足够收益空间

Coarse/fine controller 的负门控说明：训练前应先算完美 oracle policy 的上界。只有 1.03% 的 ceiling，不足以抵消预测错误、额外复杂度与时间噪声。

## 6. 当前真正成立的正结果：非 RL 的 multiscale ranker

### 6.1 有效的到底是什么

有效组件是 **assignment-aware HGBT candidate ranker + multiscale local search + exact MILP certification**，不是 PPO，也不是 feasibility predictor 直接替代 MILP。

完整流程如下：

```text
输入当前 10 个箱型和当前 exact MILP assignment
→ 枚举 10 × 3 × 2 × 4 = 240 个候选动作
→ HGBT 为每个候选动作预测“它像不像 exact search 会接受的动作”
→ 按预测分数排序
→ 先把小 shortlist 送入 MILP
→ MILP 重新计算 order-box feasibility、assignment、coverage 和 PF
→ 只接受 MILP 证明字典序改善的动作
→ 无改善时扩大 shortlist，并在终止前做 exact fallback/audit
```

因此 ranker 只负责**决定先验证谁**，不负责最终判定可行性，也无权直接接受动作。

### 6.2 Ranker 预测什么

模型是 HGBT accepted-move classifier。它的预测对象是一条候选 box move，例如：

```text
box 7 的 height 减少 0.5
```

输出是该候选动作被历史 exact greedy search 接受的倾向分数。随后按分数从高到低排列 240 个动作。它不是下面这种模型：

```text
输入一个 order 和一个 box → 输出能否装下
```

后者才是 feasibility predictor。两个模型的学习目标、输入粒度和算法作用都不同。

| 组件 | 输入 | 输出 | 当前结论 |
|---|---|---|---|
| Feasibility predictor | 一个 order + 一个 box | 装载可行概率 | 可构造快速 surrogate 环境，但未证明改善 exact 终局 |
| Candidate ranker | 当前 box-set + assignment + 一个 move | 该 move 值得优先验证的分数 | 三窗口 development 上有效 |
| PPO policy | 当前状态 | 61/241 个动作的概率 | 已测试配置未超过 ranker |
| MILP oracle | 一个 order + 一个 box | exact feasible / infeasible / unknown | 最终可行性与 PF 的权威判定 |

### 6.3 Ranker 的训练数据从哪里来

当前 frozen assignment-aware ranker 使用历史 exact candidate traces：

- 7,680 条候选动作记录；
- 128 个完整搜索状态组；
- 包含 step `0.5` 和 `0.25` 的 repaired exact traces；
- 标签为该动作是否被 exact search 接受；
- 模型为 HistGradientBoostingClassifier（HGBT）。

每条训练记录描述当前状态和一个候选动作，包括：

- 当前 PF、coverage、box-set 体积统计；
- 被调整箱子的当前尺寸和候选尺寸；
- 调整方向、步长、体积和表面积变化；
- 当前 MILP assignment 中有多少订单分配给该箱子；
- 这些订单的总体积、占比，以及 shrink 后可能失去可行性的订单；
- 候选箱可能从更大箱子捕获的订单统计。

Assignment-aware 特征只使用当前已知 assignment、订单几何和候选 box geometry，不读取候选动作的 MILP 标签，因此推理时可获得，不构成 label leakage。

当前模型没有使用 step `1.0` 和 `2.0` 的训练候选；multiscale 实验对这两个尺度属于模型迁移。该限制必须保留在论文中，后续可用 frozen protocol 补充 multiscale training traces，但不能反过来污染已完成的 development comparison。

### 6.4 为什么它有效

它解决了当前系统真正昂贵的环节：MILP candidate verification。240 个动作全部用 MILP 检查在运行时间上不可接受，而 HGBT 一次批量排序开销很小。只要好动作在 ranker 前部，搜索就能用较少 MILP query 找到改善；若 ranker 漏掉动作，逐级扩大 shortlist 和 terminal exact audit 负责安全兜底。

multiscale action space 同时解决固定细步长的路径长度问题。固定 `0.25` 需要重复执行大量可交换的小幅 shrink；允许 `2/1/0.5/0.25` 后，可以用大步快速移动，再用小步细化。监督 ranker 让这个从 60 扩大到 240 的动作空间仍可计算。

有效性来自两个互补部分：

```text
multiscale：提高可达到的解质量，并减少搜索轮数
ranker：减少每轮需要精确验证的候选
MILP audit：保持 coverage 和最终可行性口径
```

### 6.5 它相对什么 baseline 有效

三个互不重叠 development windows 上，相对 converged fine exact search：

| Aggregate metric | Fine exact | Multiscale ranker | Change |
|---|---:|---:|---:|
| Mean PF | 1.984467 | **1.888892** | **-4.82%** |
| Coverage | 3/3 为 100% | 3/3 为 100% | 持平 |
| Search rounds | 957 | **277** | **-71.1%** |
| Validations | 57,420 | **17,570** | **-69.4%** |
| Uncached boxes | 5,432 | **3,384** | **-37.7%** |
| Subprocess | 2,126.54 s | **1,896.49 s** | **-10.8%** |
| Wall time | 3,124.38 s | **2,275.63 s** | **-27.2%** |

这个结果说明 multiscale supervised candidate ordering 能让 240-action search 变得可运行，并进入更好的 PF basin。它不能被包装成 RL 结果。

这里的主表对照是 converged fixed-fine exact search。对 global staged exact baseline，单个 `[200,300)` window 上 multiscale ranker 的 PF 更低 2.79%，但 uncached boxes 多 8.8%、subprocess 多 41.7%、wall time 多 24.9%，属于质量/成本 trade-off，不是全面支配。因此最严谨的正 claim 是：它在三个 development windows 上同时优于 converged fine exact；尚不能声称它全面支配所有 staged exact 配置。

以上是三个 development windows 的配对描述统计，不是最终泛化结论。由于各 RL 变体已经在 development gate 失败，按预先约定的实验纪律，没有为了寻找有利数字而继续运行 untouched test。正式论文若采用 multiscale ranker 正结果，仍需冻结配置后完成独立测试与重复 timing evaluation。

## 7. 可以说什么，不能说什么

### 可以支持的结论

在当前 OR2023 exact-MILP 设置下，feasibility surrogate 能提供低成本的近似训练环境和候选信号，但尚未证明它能改善 exact end-to-end optimization。通过实验门控的是 supervised candidate ranking；已测试 RL 变体没有带来独立收益。

### 不能支持的结论

当前证据不能说明：

- 所有 RL 算法在所有箱型设计数据上都失败；
- Kandula 原论文在其私有数据上的结果错误；
- 设计完全不同的 state/action/model-based RL 一定无效；
- 单次 wall/subprocess 差异稳定。相同轨迹复跑出现超过 20% 的时间波动，正式时间 claim 必须做重复实验。

## 8. 项目决策建议

1. 将 supervised multiscale ranker 固定为当前最强方法。
2. 把现有 RL 结果作为 negative ablation，不作为正贡献。
3. 不再增加 direct-action PPO、PPO union、当前 budget FQI 或 binary first-step policy 的训练轮数。
4. 未来任何 RL action definition 必须先证明 hindsight oracle headroom 足够大。
5. 如果论文必须声称 RL contribution，当前证据不够，必须实质性改变决策问题，而不是继续调现有 PPO 参数。

## 复现实验索引

- `FORMAL_PROBLEM.md`
- `RANKER_RL_ROLLOUT_SMOKE_RESULTS_20260710.md`
- `MULTIWINDOW_RL_DEV_GATE_20260710.md`
- `BUDGET_FQI_CONVERGENCE_GATE_20260711.md`
- `MULTISCALE_PPO_UNION_SMOKE_20260712.md`
- `COUNTERFACTUAL_STEP_POLICY_GATE_20260712.md`
- `MULTISCALE_RANKER_GATE_20260711.md`
