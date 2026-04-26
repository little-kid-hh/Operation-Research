# Ensemble Baseline 工作流 README

## 目标

在当前 HybridSVM 体系中新增“强分类器集成基线”路线：

- 基学习器：`SVM + LR + RF + XGBoost(或GBDT回退)`
- 顶层融合：**可学习线性组合**（默认 logistic regression 作为 meta combiner）
- 目的：抬升 baseline，同时保持后续 hard-case mining / evolution 可复用


## 固定工作流（后续按此执行）

1. **先改 README / 设计文档**  
   - 先写清设计、接口、参数、评估口径、兼容策略。  
   - 未更新文档前，不进入代码实现。

2. **再按 README 写代码**  
   - 先实现最小可运行版本（MVP），再补全参数与持久化。  
   - 保持 `svm` 旧路径可用，不破坏现有流程。

3. **最后跑实验验证**  
   - 同一数据切分下对比 `svm` vs `ensemble`。  
   - 输出可追溯结果与配置（results/checkpoint）。


## 当前代码锚点

- 训练与特征：`HybridSVM/src/svm_train.py`
- 实验主流程：`HybridSVM/run_experiment.py`
- 评估比较：`HybridSVM/src/evaluate.py`
- hard-case / evolution 依赖 Stage 1 的概率与硬预测输出


## 方案总览

- 保留当前 `svm` baseline 作为默认路径（向后兼容）。
- 新增 `ensemble` baseline 路径：
  - 训练基学习器
  - 通过 OOF 生成 meta 特征（防泄漏）
  - 训练线性融合器
  - 产出测试集 `P(y=1)` 与硬标签
- 下游 Stage 2/3/4 统一消费 baseline 输出，不绑定 SVM。


## 分阶段实施计划

### 阶段 1：新增集成模块（设计与实现）

建议新增：`HybridSVM/src/ensemble_train.py`

职责：

- 基学习器构建与训练（SVM/LR/RF/XGB）
- OOF 概率矩阵构建（`StratifiedKFold`）
- 线性融合器训练（默认 `logreg`）
- 推理接口：`predict_proba` / `predict`
- 工件保存与加载（模型、特征名、融合器、配置）

输出应与现有 SVM 产物风格一致，便于接入 `run_experiment.py`。


### 阶段 2：接入 run_experiment Stage 1

在 `HybridSVM/run_experiment.py` 中增加 baseline 模式分支：

- `svm`：保持现有逻辑不变
- `ensemble`：走新模块，生成通用变量
  - `baseline_probs_test`
  - `baseline_pred_test`

并将 Stage 2/3/4 从 `svm_*` 变量逐步统一到 `baseline_*`（必要时保留别名保证兼容）。


### 阶段 3：参数与可复现性

新增 CLI 参数（建议）：

- `--baseline-mode {svm,ensemble}`
- `--ensemble-base-models svm,lr,rf,xgb`
- `--ensemble-meta-model logreg`
- `--ensemble-cv-folds`
- `--ensemble-random-state`

同步写入 checkpoint，支持 resume 一致性检查。


### 阶段 4：结果与报告扩展

`results.json` 增加 baseline 元信息：

- baseline 类型
- 基学习器列表
- 融合器类型（及可选系数）

评估沿用 `evaluate.py`，重点对比：

- Accuracy
- Recall
- AUC
- `TPR@FPR=1%`
- hard-case 规模与修正表现


### 阶段 5：依赖策略

- `xgboost` 作为可选依赖
- 若未安装，自动回退 sklearn GBDT，并在日志中明确提示


## 防泄漏与实验规范

1. 融合器训练只能用训练集 OOF 预测，不能直接用测试集训练。  
2. 对比实验必须固定相同切分（同一 `test_size/random_state`）。  
3. 所有关键参数需写入 checkpoint 和结果文件。  
4. 新路径不能影响既有 `svm` 模式结果。


## 验收标准

- `svm` 模式行为与当前版本一致（回归通过）。
- `ensemble` 模式可完整跑通 Stage 1~4 并产出完整工件。
- 在至少一个核心指标上（推荐 AUC 或 `TPR@FPR=1%`）相对 SVM 有可观提升。


## 里程碑建议

- M1：`ensemble_train.py` 最小可运行（仅 SVM+LR）  
- M2：加入 RF/XGB + OOF + meta 训练  
- M3：接入 run_experiment + checkpoint/results 扩展  
- M4：完整对比实验与文档沉淀


## 后续（第二条路线）

在本路线稳定后，再开展 `tableformer` 路线（单独文档与实验目录，不与本 README 混写）。
