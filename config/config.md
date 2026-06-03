# 配置与方法变动记录

从 2026-06-03 起，新的实验配置、方法改动和改动原因统一记录在本文件。

## 记录规则

- 每次只记录本次实际改动，不重复粘贴完整 `config.yaml`。
- 记录需要包含：版本名、基于哪个版本、改动内容、改动原因、预期观察指标。
- 实验结果仍可继续写入 `experiments.md`，本文件重点记录“跑之前改了什么和为什么改”。

## 2026-06-03 - V10-method-balanced-supcon

**基于版本:** `V10-no-adapter-lowC-minckpt` / 当前全脑-only主线

**目标:** 不继续做 batch/lr 等配置搜索，优先修正方法层面的 checkpoint 选择和跨数据集对比学习目标。

**改动内容:**

- `training.checkpoint_selection.metric = "balanced_auc"`
- `training.checkpoint_selection.gap_penalty = 0.25`
- checkpoint 保存和 early stop 从纯混合 `val_auc` 改为 `balanced_score`
- `balanced_score = mean(val_auc_77sets, val_auc_isle2024) - 0.25 * |val_auc_77sets - val_auc_isle2024|`
- 验证日志新增 `77`、`ISLE`、`Gap`、`Score`
- SupCon 新增 `dataset_id` 输入
- 同标签跨数据集正样本权重：`cross_dataset_positive_weight = 1.0`
- 同标签同数据集正样本权重：`same_dataset_positive_weight = 0.5`

**改动原因:**

- 之前 best checkpoint 只按混合 validation AUC 选择，可能选到某个数据集表现好、另一个数据集下降的 epoch。
- 当前论文目标是证明跨模态训练有效，因此模型选择指标应惩罚 77sets 与 ISLE 的 AUC gap。
- 原 SupCon 只按 label 构造正样本，没有区分同域和跨域；这不够贴合“跨模态共享预后表征”的目标。
- 跨数据集同标签样本应被更强拉近，同数据集同标签样本只作为弱正样本，降低队列内部模式主导对齐的风险。

**预期变化:**

- best epoch 可能不再是混合 `Val AUC` 最高的 epoch。
- 单数据集峰值 AUC 可能略降，但双数据集 mean AUC、最差数据集 AUC和 modality gap 应更稳定。
- 日志中的 `Score` 才是当前 checkpoint 选择依据，`Val AUC` 仅作为参考。

**主要观察指标:**

- 77sets pooled AUC 是否回到或超过 `0.7912`
- ISLE pooled AUC 是否超过 `0.7846`
- 双数据集 mean pooled AUC 是否超过 `0.7874`
- mean modality gap 是否下降到 `< 0.08`
