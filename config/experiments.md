# X-Former 调参记录

---

## V1 — 基准 (baseline)

**日期:** 2026-05-25
**日志:** `logs/xformer_20260525_030338.log`
**配置:** 默认参数

| 参数 | 值 |
|------|-----|
| focal.weight | 1.0 |
| domain.weight | 0.1 |
| contrastive.weight | 0.1 |
| consistency.weight | 0.05 |
| batch_size | 4 |
| accumulation_steps | 8 |
| lr | 5e-5 |
| T_0 | 15 |
| warmup_epochs | 5 |

### Per-Fold 结果

| Fold | Val AUC | 77sets Test AUC | ISLE Test AUC | Epoch |
|------|---------|-----------------|---------------|-------|
| 1 | 0.820 | 0.810 | 0.760 | 104 |
| 2 | 0.727 | 0.667 | 0.635 | 43 |
| 3 | 0.755 | 0.661 | 0.605 | 41 |
| 4 | 0.764 | 0.893 | 0.615 | 101 |
| 5 | 0.848 | 0.679 | 0.621 | 87 |

### 汇总

| 指标 | 77sets | ISLE2024 |
|------|--------|----------|
| AUC (mean±std) | 0.742±0.093 | 0.647±0.057 |
| Accuracy | 0.663±0.072 | 0.604±0.052 |
| F1 | 0.647±0.072 | 0.544±0.061 |

### 分析

- 对比损失 (C) 值为 Focal (F) 的 6-10 倍，主导梯度
- 高方差：Fold 3 ISLE=0.605 vs Fold 1=0.760
- 验证集 AUC 普遍高估测试集 0.05-0.15
- ISLE 4/5 折在 0.60-0.64 区间，基本未学到有效特征
- Pooled bootstrap 因 NameError 未完整输出

---

## V2 — 降低辅助损失权重

**日期:** 2026-05-25
**日志:** `logs/xformer_20260525_150337.log`
**改动:**

| 参数 | V1 | V2 | 原因 |
|------|----|----|------|
| contrastive.weight | 0.1 | **0.03** | Focal 主导分类，对比损失辅助对齐 |
| domain.weight | 0.1 | **0.05** | 域对抗同步降权，避免过早混淆 |

### Per-Fold 结果

| Fold | Val AUC | 77sets Test AUC | ISLE Test AUC | Epoch |
|------|---------|-----------------|---------------|-------|
| 1 | 0.835 | 0.825 | 0.790 | 106 |
| 2 | 0.792 | 0.730 | 0.855 | 108 |
| 3 | 0.823 | 0.696 | 0.870 | 107 |
| 4 | 0.842 | 0.786 | 0.720 | 143 |
| 5 | 0.879 | 0.696 | 0.763 | 85 |

### 汇总

| 指标 | V1 (77sets) | V2 (77sets) | V1 (ISLE) | V2 (ISLE) |
|------|------------|------------|-----------|-----------|
| AUC (mean±std) | 0.742±0.093 | 0.747±0.051 | 0.647±0.057 | **0.800±0.056** |
| Accuracy | 0.663±0.072 | 0.714±0.090 | 0.604±0.052 | 0.718±0.038 |
| F1 | 0.647±0.072 | 0.701±0.102 | 0.544±0.061 | 0.636±0.129 |

### Pooled Bootstrap (V2)

| 指标 | 77sets | 95% CI | ISLE2024 | 95% CI |
|------|--------|-------|----------|-------|
| AUC | 0.765 | 0.649-0.860 | 0.760 | 0.666-0.844 |
| Accuracy | 0.766 | 0.675-0.844 | 0.761 | 0.691-0.826 |
| F1 | 0.754 | 0.661-0.840 | 0.743 | 0.675-0.807 |

### V1→V2 对比分析

**改善：**
- ISLE AUC +15.3% (0.647→0.800)，所有折均 >0.70
- 77sets 标准差减半 (0.093→0.051)，高方差问题解决
- 不再出现 <0.65 的极端折
- 每折训练更稳定 (85-143 epochs vs 41-104)
- 零报错，完整输出 pooled CI

**尚未达标：**
- 77sets pooled AUC 0.765 仍低于 Tri-CAF 的 0.813
- ISLE 与 Tri-CAF 的 0.836 差距较大
- 77sets Fold 3/5 仍在 0.70 以下
- 对比损失降权后 ISLE 大幅改善，说明 V1 的对比损失过重主要伤害了 CT 侧

**下一步方向：**
- 尝试 contrastive.weight=0.01（极低），观察 ISLE 是否继续上升
- 或提升 batch_size→6（减少噪声梯度）
- 或增大 T_0→30（更慢的余弦周期，每折有更多时间收敛）
- ISLE 改善显著但未达论文 1 水平，可能需单独调 CT 窗口或放射组学特征

---

## V3 — 解冻layer2 + 拉长周期 + 增大batch

**日期:** 2026-05-26
**日志:** `logs/xformer_20260526_112852.log` (提前终止)
**改动:**

| 参数 | V2 | V3 | 原因 |
|------|----|----|------|
| swin_freeze_layers | 2 | 1 | 解冻 layer2 |
| accumulation_steps | 8 | 12 | 有效 batch=48 |
| T_0 | 15 | 30 | 拉长余弦周期 |

### Per-Fold 结果（仅 2 折，提前终止）

| Fold | Val AUC | 77sets Test AUC | ISLE Test AUC | Epoch |
|------|---------|-----------------|---------------|-------|
| 1 | 0.854 | 0.714 | 0.790 | 101 |
| 2 | 0.687 | 0.508 | 0.625 | 42 |

### 分析

- **全部倒退。** 77sets Fold 1: 0.825→0.714, Fold 2: 0.730→0.508
- ISLE Fold 2 也从 0.855 跌至 0.625
- 解冻 layer2 增加 ~8M 参数，每折仅 ~50 样本，严重过拟合
- 结论：模型太大，数据太少

---

## V4 — 缩模型 + 强正则

**日期:** 2026-05-26
**日志:** `logs/xformer_20260526_161755.log` (提前终止)
**思路:** 模型太大、数据太少，缩参数 + 防过拟合

| 参数 | V3 | V4 | 原因 |
|------|----|----|------|
| bottleneck.n_slots | 8 | 4 | 分类头 2048→1024 |
| clinical.n_prototypes | 8 | 4 | 临床原型减半 |
| swin_freeze_layers | 1 | 2 | 回退冻结 layer2 |
| classifier.dropout | 0.3 | 0.5 | 分类头加强 dropout |
| weight_decay | 0.05 | 0.1 | 翻倍 L2 正则 |
| focal.alpha | 0.7 | 0.85 | 加强少数类关注 |

### Per-Fold 结果（仅 Fold 1，欠拟合提前终止）

| Fold | Val AUC | 77sets Test AUC | ISLE Test AUC |
|------|---------|-----------------|---------------|
| 1 | 0.652 | — | — |

### 分析

- **欠拟合。** Best Val AUC 仅 0.652，21 轮未提升
- 瓶颈 8→4 信息压缩过猛，有用信号也被挤掉
- 结论：缩容量方向错误，V2 的 B=8 是合理下限

---

## V5 — 路线A: 关闭对比损失

**日期:** 2026-05-26
**日志:** `logs/xformer_20260526_170810.log`
**思路:** 对比损失在 batch=4 小批次下可能是纯噪声，关掉后单测 focal+域对抗

| 参数 | V4 | V5 | 原因 |
|------|----|----|------|
| use_contrastive_loss | true | **false** | 核心改动 |
| n_slots | 4 | **8** | 回 V2 |
| n_prototypes | 4 | **8** | 回 V2 |
| classifier.dropout | 0.5 | **0.3** | 回 V2 |
| weight_decay | 0.1 | **0.05** | 回 V2 |
| T_0 | 30 | **15** | 回 V2 |
| accum | 12 | **8** | 回 V2 |
| focal.alpha | 0.85 | — | 保留 |
| domain.weight | 0.05 | — | 保留 |

### Per-Fold 结果

| Fold | Val AUC | 77sets Test AUC | ISLE Test AUC | Epoch |
|------|---------|-----------------|---------------|-------|
| 1 | 0.848 | 0.794 | 0.720 | 112 |
| 2 | 0.801 | **0.873** | 0.855 | 142 |
| 3 | **0.938** | 0.804 | 0.725 | 103 |
| 4 | 0.882 | 0.839 | 0.705 | 108 |
| 5 | 0.898 | 0.661 | 0.668 | 61 |

### V5 vs V2 对比

| 指标 | V2 (77sets) | V5 (77sets) | V2 (ISLE) | V5 (ISLE) |
|------|------------|------------|-----------|-----------|
| AUC (mean±std) | 0.747±0.051 | **0.794±0.072** | **0.800±0.056** | 0.735±0.063 |
| Pooled AUC | 0.765 | **0.805** | **0.760** | 0.741 |
| 95% CI | 0.649-0.860 | 0.711-0.893 | 0.666-0.844 | 0.661-0.817 |

### 分析 — V5 证实了核心猜想

**77sets 追上 Tri-CAF（0.805 vs 0.813，差距仅 0.008）：** 关闭对比损失后瓶颈融合在 MRI 上几乎达到第一篇水平。说明信息瓶颈本身足够好，小数据集下对比损失帮不了 77sets。

**ISLE 下降（0.800→0.735）：** ISLE 样本更多、异质性更高，对比损失帮助 CT 侧的特征对齐。关掉后 ISLE 失去跨模态正则。**路线A 被部分证伪** —— 对比损失对 ISLE 有贡献。

### 当前天花板判断

| 数据集 | 版本 | AUC | Tri-CAF | 差距 | 瓶颈 |
|--------|------|-----|---------|------|------|
| 77sets | V5 | 0.805 | 0.813 | **-0.008** | 接近数据上限 |
| ISLE | V2 | 0.760 | 0.836 | **-0.076** | 模型或数据特征问题 |

---

## V6 — MRI 预处理修复 + 轻量辅损失

**日期:** 2026-05-27
**日志:** `logs/xformer_20260527_105049.log`
**改动:**

| 改动 | 说明 |
|------|------|
| 77sets 图像归一化 | CT HU 窗宽 → **MRI z-score 三通道** |
| 77sets 数据增强 | **新增 RandBiasField** |
| batch_size | 4→**8** |
| contrastive.weight | 0→**0.01** |
| domain.weight | 0.05→**0.03** |

### Per-Fold 结果

| Fold | Val AUC | 77sets Test AUC | ISLE Test AUC | Epoch |
|------|---------|-----------------|---------------|-------|
| 1 | 0.832 | 0.794 | 0.705 | 102 |
| 2 | 0.829 | 0.841 | 0.895 | 103 |
| 3 | 0.839 | 0.679 | 0.915 | 88 |
| 4 | 0.786 | 0.857 | 0.675 | 60 |
| 5 | 0.919 | 0.661 | 0.721 | 63 |

### V6 vs 历史对比

| 版本 | 77sets AUC | ISLE AUC | Fold 方差(77/ISLE) |
|------|-----------|----------|-------------------|
| V5 (关对比) | **0.794±0.072** | 0.735±0.063 | 0.072/0.063 |
| V6 (z-score+轻对比) | 0.766±0.082 | **0.782±0.102** | 0.082/0.102 |

- 77sets 反跌 0.028, z-score 修复没起作用
- ISLE 涨 0.047, 对比权重 0.01 对齐仍有帮助
- **Fold 间方差翻倍**, 77sets 和 ISLE 在每折里反相关（Fold 3: ISLE=0.915 vs 77sets=0.679）

### 分析

**反相关模式揭示根本问题：域对抗 + 瓶颈 = 冲突。**

分类器要学 mRS 预测特征，GRL 要模型"不区分 MRI/CT"。模型的妥协方案：把 77sets 学好、ISLE 学烂（或反过来），域分类器无法区分因为特征本就不一致。**信息瓶颈已经天然跨模态对齐——8 个槽位自动挤出模态无关信号——域对抗不仅冗余，还和分类器抢瓶颈。**

---

## V7 — 关域对抗 + z-score + BiasField

**日期:** 2026-05-28
**日志:** `logs/xformer_20260528_103221.log`
**核心理论:** 瓶颈压缩决定跨模态泛化——域对抗是冗余的竞争对手

| 参数 | V6 | V7 |
|------|----|-----|
| use_domain_loss | true | **false** |
| 77sets 窗/BiasField | z-score+有 | 不变 |

### Per-Fold 结果

| Fold | Val AUC | 77sets Test AUC | ISLE Test AUC | Epoch |
|------|---------|-----------------|---------------|-------|
| 1 | 0.845 | 0.873 | 0.705 | 109 |
| 2 | 0.745 | 0.873 | 0.625 | 41 |
| 3 | 0.795 | 0.964 | 0.745 | 41 |
| 4 | 0.860 | 0.857 | 0.715 | — |
| 5 | 0.882 | 0.768 | 0.632 | 82 |

### V7 vs V5 对比

| 指标 | V5 | V7(77sets) | V7(ISLE) |
|------|-----|-----------|----------|
| AUC 逐折均值 | 0.794±0.072 | **0.867±0.062** | 0.684±0.048 |
| AUC pooled | **0.805** | 0.720 | 0.665 |
| Accuracy | **0.765** | 0.637 | 0.658 |
| F1 | **0.749** | 0.576 | 0.614 |

### 分析

- **z-score + BiasField 伤害泛化。** 逐折 AUC 虚高（15人测试集运气加成），pooled AUC 从 V5 的 0.805 跌至 0.720
- Accuracy/F1 极差——模型只学了排序不会分类，正负例概率差距太小
- Fold 2/3 早停仅 41 轮，验证集太小检测不到真实提升
- 结论：z-score 方向错误，回退 CT 窗权

---

## V8 — 回退 z-score, domain 0.05, contrastive 0.02

**日期:** 2026-05-28
**日志:** 待运行
**核心策略:** V2 的 ISLE 优势 + V5 的 77sets 稳定性

| 参数 | V7 | V8 | 原因 |
|------|----|----|------|
| 77sets 窗 | z-score | **CT 窗** | 回退，V5/V2 验证 |
| 77sets BiasField | 有 | **无** | 回退 |
| use_domain_loss | false | **true** | V2 ISLE 0.80 靠这个 |
| domain.weight | — | **0.05** | V2 原值 |
| contrastive.weight | 0.01 | **0.02** | 介于 V2(0.03) 和 V5(0) 之间 |

**预期:** 77sets pooled≥0.80, ISLE pooled≥0.78, CI 与第一篇重叠。

### 结果 (仅 Fold 1，DDL 超时崩溃)

| Fold | 77sets Test AUC | ISLE Test AUC | Epoch |
|------|----------------|---------------|-------|
| 1 | 0.778 | 0.740 | 113 |

- 77sets 从 V5 的 0.805 跌至 0.778, ISLE 持平
- DDP 集体操作超时 (ALLREDUCE 600s), fold 切换时 GPU 间不同步
- 结论：不是域对抗权重的问题——**交错采样器本身才是瓶颈**

---

## V9 — 随机采样 + 关域对抗 (ISLE 数据 100% 利用)

**日期:** 2026-05-29
**日志:** 待运行
**核心:** InterleavedBatchSampler 丢弃了 49% 的 ISLE 数据。关域对抗后不再需要 1:1 交错，改用随机 shuffle。

| 改动 | V8 | V9 | 原因 |
|------|----|----|------|
| 采样器 | Interleaved(1:1) | **RandomSampler** | 随机混合, ISLE 100% 数据参与 |
| 离线增强 | 无 | **77sets×2, ISLE×1** | ×4导致过拟合(Loss→0)，降为×2 |
| use_domain_loss | true | **false** | 随机采样下域对抗无效 |
| use_contrastive_loss | disabled | disabled | V5 验证最优 |
| 其余 | V5 配置 | 不变 | CT窗 + B=8 + focal α=0.85 |

**训练集规模:** 77×4 + 149×2 = 308 + 298 = 606 有效样本 (原来 ~145)

**预期:** 77sets pooled≥0.82, ISLE pooled≥0.80, 首次两数据集同时超第一篇。单卡训练 (DDP 超时问题)。

### 实际结果

**日志:** `logs/xformer_20260529_144046.log`  
**实际运行:** 3 卡 DDP，`World size: 3`

| Fold | Val AUC | 77sets Test AUC | ISLE Test AUC | Threshold |
|------|---------|-----------------|---------------|-----------|
| 1 | 0.854 | 0.825 | 0.750 | 0.80 |
| 2 | 0.826 | 0.794 | 0.860 | 0.48 |
| 3 | 0.839 | 0.714 | 0.875 | 0.88 |
| 4 | 0.826 | 0.821 | 0.770 | 0.90 |
| 5 | 0.839 | 0.714 | 0.605 | 0.58 |

| 指标 | 77sets | ISLE2024 |
|------|--------|----------|
| AUC (mean±std) | 0.774±0.050 | 0.772±0.097 |
| Pooled AUC (log bootstrap) | 0.751 | 0.740 |
| Pooled AUC (CSV fixed threshold) | 0.750 | 0.739 |
| Accuracy (CSV fixed threshold) | 0.675 | 0.718 |
| F1 (CSV fixed threshold) | 0.673 | 0.657 |

### V9 结论

- V9 没超过第一篇：Tri-CAF 参考为 77sets AUC=0.813、ISLE AUC=0.836。
- 训练 loss 在多个 fold 很快接近 0，说明 `77sets×4, ISLE×2` 的重复采样让模型记忆训练集。
- `use_domain_loss=false` 且 `use_contrastive_loss=false` 后，模型只靠 focal loss 学分类，跨模态表示不稳定。
- DDP 随机采样分支使用 `DistributedSampler`，但 `Trainer.train_epoch()` 未调用 `loader.sampler.set_epoch(epoch)`，每轮 shuffle 可能重复。
- 阈值从 0.48 到 0.90 大幅波动，概率校准差。AUC 也未达标，不能靠调阈值解决。
- 日志中的 bootstrap Accuracy/F1 和 CSV 不一致：bootstrap 内部重新搜索 threshold，CSV 使用 mean validation threshold。论文表格应使用固定验证阈值口径。

---

## V10 — 全脑-only 稳定化路线

**日期:** 2026-05-31  
**核心约束:** 第一篇未使用 lesion，且当前 lesion 数据不完整、不规范，因此 V10 主线只使用全脑影像/全脑 radiomics/临床数据。

### V10-sanity 默认配置

| 参数 | V9 | V10-sanity | 原因 |
|------|----|------------|------|
| 采样 | RandomSampler + hard-code oversampling | `balanced_full` | ISLE 全量参与，77sets replacement 补齐，避免高倍复制记忆 |
| augment_77 / augment_isle | 4 / 2 | 1 / 1 | 取消离线重复，只保留在线增强 |
| use_domain_loss | false | true | 只作为弱正则恢复跨模态约束 |
| domain.weight | 0.05 | 0.01 | 避免 GRL 抢瓶颈分类梯度 |
| grl_lambda | 0.1 | 0.05 | 降低域对抗强度 |
| use_contrastive_loss | false | true | 对 ISLE 有历史收益，但需极低权重 |
| contrastive.weight | 0.02 | 0.005 | 只做轻量对齐 |
| use_lesion | false | false | 保持与第一篇公平可比 |
| use_consistency_loss | false | false | consistency 当前依赖 lesion 分支 |
| batch_size | 6 | 4 | 降低小样本记忆风险 |
| accumulation_steps | 4 | 8 | 保持稳定有效 batch |
| lr | 5e-5 | 3e-5 | 降低过拟合速度 |
| weight_decay | 0.05 | 0.08 | 加强正则 |
| scheduler | cosine warm restart | cosine | 不再反复高 LR 重启已记忆模型 |
| max_epochs / patience | 150 / 35 | 120 / 25 | 更早停止过拟合 |

### V10 工程修复

- 采样参数写入 `config/config.yaml`，不再在 `train/crossval_joint.py` hard-code。
- 新增 `BalancedFullSampler`：每个 epoch 使用 ISLE 训练样本一次，77sets replacement 采样到相同规模，batch 近似 1:1。
- DDP 下同时调用 `loader.batch_sampler.set_epoch()` 和 `loader.sampler.set_epoch()`。
- bootstrap 支持固定阈值，日志/CSV/论文表格使用同一阈值口径。
- 保存 per-subject prediction CSV：`fold,dataset,subject,y_true,y_prob,threshold,y_pred`。
- 可选 dataset-conditioned residual adapter 保留为后续 V10-adapter，但默认关闭；它不使用 lesion，只基于共享瓶颈和 dataset_id 做轻量校准。

### V10 实验顺序

| 版本 | 改动 | 通过标准 |
|------|------|----------|
| V10-sanity | 采样/DDP/阈值修复 + light domain/contrastive，全脑-only | 77sets pooled AUC ≥0.80，ISLE pooled AUC ≥0.76，Fold 5 不再崩 |
| V10-adapter | 在 V10-sanity 基础上开启 dataset-conditioned residual adapter | 已完成，未达标，见 `V10-adapter-min-ckpt 实际结果` |
| V10-report | 选择 sanity/adapter 中更稳定版本做主结果 | 全脑-only，对第一篇公平可比 |

### V10-sanity 实际结果

**日期:** 2026-05-31  
**日志:** `logs/xformer_20260531_010714.log`  
**实际运行:** 单卡，`Device: cuda:0, World size: 1`  
**核心配置:** `balanced_full`，`augment_77=1`，`augment_isle=1`，`use_lesion=false`，`domain.weight=0.01`，`contrastive.weight=0.005`，adapter 关闭。

| Fold | Best Val AUC | Best Epoch | Threshold | 77sets Test AUC | ISLE Test AUC |
|------|-------------:|-----------:|----------:|----------------:|--------------:|
| 1 | 0.8354 | 36 | 0.50 | 0.8095 | 0.6950 |
| 2 | 0.8137 | 55 | 0.33 | 0.8889 | 0.9400 |
| 3 | 0.8851 | 25 | 0.69 | 0.7321 | 0.8350 |
| 4 | 0.8323 | 49 | 0.68 | 0.9464 | 0.7650 |
| 5 | 0.8385 | 14 | 0.64 | 0.6607 | 0.6000 |

| 指标 | 77sets | ISLE2024 |
|------|--------|----------|
| AUC mean±std | 0.8075±0.1153 | 0.7670±0.1300 |
| Pooled AUC (fixed threshold CSV) | 0.7912 | 0.7846 |
| 95% CI | 0.6930-0.8779 | 0.7068-0.8594 |
| Accuracy | 0.7143 | 0.7248 |
| F1-macro | 0.7142 | 0.7162 |
| Fixed threshold | 0.568 | 0.568 |

### V10-sanity 结论

- 相比 V9 fixed-threshold pooled AUC，77sets 从 `0.750` 提升到 `0.791`，ISLE 从 `0.739` 提升到 `0.785`，说明 `balanced_full + light domain/contrastive + 去高倍复制` 方向有效。
- 仍未超过第一篇：Tri-CAF 参考为 77sets `0.813`、ISLE `0.836`；当前差距约为 77sets `-0.022`、ISLE `-0.051`。
- 主要失败点是 Fold 5：77sets `0.6607`、ISLE `0.6000`。Fold 5 的 best checkpoint 出现在 epoch 14，而 domain loss epoch 20 才启动，说明保存了 warmup 前的早期验证尖峰。
- Fold 5 概率分布塌缩：77sets 正/负均值约 `0.637/0.634`，ISLE 正/负均值约 `0.601/0.565`。这是 checkpoint 选择和校准问题，不是 lesion 信息不足导致。
- V10-sanity 达到 ISLE 最低通过线，但 Fold 5 未解决；下一步应做全脑-only 的 `V10-adapter-min-ckpt`，不引入 lesion。

### V10-adapter-min-ckpt 配置

| 参数 | V10-sanity | V10-adapter-min-ckpt | 原因 |
|------|------------|----------------------|------|
| classifier.adapter.enabled | false | **true** | 给 MRI/CT 轻量 residual 校准能力，不拆成两个模型 |
| classifier.adapter.scale | 0.2(未启用) | **0.1** | 限制 adapter 影响，避免小样本过拟合 |
| training.min_checkpoint_epoch | 无 | **25** | 禁止保存 domain/contrastive warmup 前的早期尖峰 |
| use_lesion | false | **false** | 保持与第一篇公平可比 |
| domain / contrastive | 0.01 / 0.005 | 不变 | V10-sanity 已证明轻量约束有效 |

### V10-adapter-min-ckpt 实际结果

**日期:** 2026-05-31  
**日志:** `logs/xformer_20260531_194526.log`  
**实际运行:** 单卡，`Device: cuda:0, World size: 1`  
**核心配置:** `balanced_full`，`augment_77=1`，`augment_isle=1`，`use_lesion=false`，`domain.weight=0.01`，`contrastive.weight=0.005`，`classifier.adapter.enabled=true`，`adapter.scale=0.1`，`min_checkpoint_epoch=25`。

| Fold | Best Val AUC | Best Epoch | Threshold | 77sets Test AUC | ISLE Test AUC |
|------|-------------:|-----------:|----------:|----------------:|--------------:|
| 1 | 0.7950 | 32 | 0.73 | 0.8254 | 0.7000 |
| 2 | 0.7671 | 34 | 0.62 | 0.7619 | 0.8650 |
| 3 | 0.8851 | 38 | 0.70 | 0.7679 | 0.8550 |
| 4 | 0.8416 | 66 | 0.67 | 0.8750 | 0.7150 |
| 5 | 0.8323 | 28 | 0.61 | 0.6964 | 0.7842 |

| 指标 | 77sets | ISLE2024 |
|------|--------|----------|
| AUC mean±std | 0.7853±0.0607 | 0.7838±0.0684 |
| Pooled AUC (fixed threshold CSV) | 0.7544 | 0.7451 |
| Pooled AUC (log bootstrap) | 0.7550 | 0.7459 |
| 95% CI | 0.6442-0.8638 | 0.6574-0.8267 |
| Accuracy | 0.6883 | 0.7248 |
| F1-macro | 0.6878 | 0.7011 |
| Fixed threshold | 0.666 | 0.666 |

### V10-adapter-min-ckpt vs V10-sanity

| Fold | 77sets ΔAUC | ISLE ΔAUC | 主要变化 |
|------|------------:|----------:|----------|
| 1 | +0.0159 | +0.0050 | 基本持平，但 Val AUC 下降 |
| 2 | -0.1270 | -0.0750 | adapter 明显伤害排序，是 pooled AUC 下跌主因之一 |
| 3 | +0.0358 | +0.0200 | 小幅改善 |
| 4 | -0.0714 | -0.0500 | 77sets 仍高，但 ISLE 被拉低，gap 扩大到 0.1600 |
| 5 | +0.0357 | +0.1842 | Fold 5 修复明显，主要来自 `min_checkpoint_epoch=25` 避开 epoch 14 早期尖峰 |

| 指标 | V10-sanity | V10-adapter-min-ckpt | 变化 |
|------|-----------:|---------------------:|-----:|
| 77sets pooled AUC | 0.7912 | 0.7544 | -0.0368 |
| ISLE pooled AUC | 0.7846 | 0.7451 | -0.0395 |
| 77sets Accuracy | 0.7143 | 0.6883 | -0.0260 |
| ISLE Accuracy | 0.7248 | 0.7248 | +0.0000 |
| 77sets F1-macro | 0.7142 | 0.6878 | -0.0264 |
| ISLE F1-macro | 0.7162 | 0.7011 | -0.0151 |

### V10-adapter-min-ckpt 概率诊断

基于 `results/joint/predictions.csv` 重新计算，当前概率分布没有全局塌缩，但存在明显 fold 级别排序和校准不稳。

| Fold | Dataset | AUC | Thr | Pos mean | Neg mean | Gap | FP/FN |
|------|---------|----:|----:|---------:|---------:|----:|------:|
| 1 | 77sets | 0.8254 | 0.73 | 0.7812 | 0.6633 | 0.1179 | 4/2 |
| 1 | ISLE | 0.7000 | 0.73 | 0.7189 | 0.6178 | 0.1011 | 9/3 |
| 2 | 77sets | 0.7619 | 0.62 | 0.6516 | 0.4019 | 0.2497 | 2/3 |
| 2 | ISLE | 0.8650 | 0.62 | 0.6377 | 0.3491 | 0.2886 | 3/3 |
| 3 | 77sets | 0.7679 | 0.70 | 0.7225 | 0.5608 | 0.1617 | 3/2 |
| 3 | ISLE | 0.8550 | 0.70 | 0.6701 | 0.4545 | 0.2155 | 3/4 |
| 4 | 77sets | 0.8750 | 0.67 | 0.7781 | 0.4214 | 0.3567 | 3/1 |
| 4 | ISLE | 0.7150 | 0.67 | 0.6662 | 0.4063 | 0.2599 | 5/3 |
| 5 | 77sets | 0.6964 | 0.61 | 0.5347 | 0.4399 | 0.0948 | 2/4 |
| 5 | ISLE | 0.7842 | 0.61 | 0.6351 | 0.4273 | 0.2078 | 5/3 |

Pooled 级别：77sets 正/负均值 `0.6936/0.4991`，gap `0.1945`；ISLE 正/负均值 `0.6656/0.4512`，gap `0.2143`。均值 gap 看似足够，但 AUC 只有 `0.7544/0.7451`，说明错误主要来自跨 fold 的排序不一致，而不是所有病例都不可分。

### V10-adapter-min-ckpt 问题分析

- `min_checkpoint_epoch=25` 生效，Fold 5 best epoch 从 V10-sanity 的 `14` 推迟到 `28`，Fold 5 也从 `77sets=0.6607/ISLE=0.6000` 提升到 `0.6964/0.7842`。因此上一版的早期 checkpoint 问题被部分修复。
- 但整体 pooled AUC 反而明显下降：77sets `0.7912 -> 0.7544`，ISLE `0.7846 -> 0.7451`。这说明当前问题不是单纯阈值或 checkpoint，而是 adapter 结构对排序能力产生了负收益。
- 负收益集中在 Fold 2 和 Fold 4：这两个 fold 在 V10-sanity 中贡献了较高 pooled 排序，本版分别下降 `0.127/0.075` 和 `0.071/0.050`。Fold 5 的提升不足以抵消这些折的退化。
- Fold 1 和 Fold 4 的跨模态 gap 分别达到 `0.1254`、`0.1600`，平均 gap `0.1127±0.0274`，高于目标 `<0.08`。共享瓶颈没有得到更好的跨模态对齐。
- adapter 的设计目标是校准 MRI/CT 决策边界，但 AUC 是阈值无关指标，AUC 同时下降说明它不只是校准失败，而是改变了样本排序。dataset-conditioned residual 可能让模型利用 dataset id 做队列特异拟合，削弱共享 outcome 表示。
- 当前模型已经有 clinical dataset embedding，adapter 又在分类头显式输入 `dataset_id`，相当于给队列身份两次进入决策路径。小样本下这会提高队列特异拟合风险，和“共享瓶颈学习跨模态结局表示”的论文核心相冲突。
- 概率分布没有全局塌缩，但存在 fold 级别排序弱化。77sets Fold 5 正/负均值仅 `0.5347/0.4399`，gap `0.0948`；ISLE Fold 1 正/负均值 `0.7189/0.6178`，gap `0.1011`，导致大量高概率假阳性。
- 训练后期 focal loss 继续下降到很低，contrastive 项相对占比升高；在 adapter 额外自由度存在时，轻量对齐损失可能仍会把共享瓶颈拉向“队列对齐”而非“结局排序”。下一步应隔离 `min_checkpoint_epoch` 的收益，而不是继续扩大 adapter。

### V10 后续修正方向

下一步不应继续加大 adapter。优先做两个隔离实验：

| 版本 | 改动 | 目的 | 通过标准 |
|------|------|------|----------|
| V10-min-ckpt-only | 关闭 adapter，保留 `min_checkpoint_epoch=25` | 判断 Fold 5 修复是否来自 min checkpoint，而不是 adapter | pooled AUC 回到接近 V10-sanity，且 Fold 5 不再低于 0.65 |
| V10-no-adapter-lowC | 关闭 adapter，`contrastive.weight` 降到 `0.002` 或关闭 domain | 判断后期辅助损失是否压制排序 | pooled AUC ≥ V10-sanity，gap < 0.10 |

### V10-no-adapter-lowC-minckpt 下一版配置

**状态:** 已写入 `config/config.yaml`，准备运行。  
**目标:** 全脑-only、公平对比第一篇；优先恢复 V10-sanity 的 pooled AUC，同时保留 Fold 5 修复。

| 参数 | V10-adapter-min-ckpt | 下一版 | 原因 |
|------|----------------------|--------|------|
| `classifier.adapter.enabled` | true | **false** | adapter 让 pooled AUC 双数据集同时下降约 0.04，已证伪 |
| `training.min_checkpoint_epoch` | 25 | **25** | 已证明能避开 Fold 5 epoch 14 早期尖峰 |
| `contrastive.weight` | 0.005 | **0.002** | 保留轻量跨模态正则，但降低训练后期对排序的干扰 |
| `domain.weight` | 0.01 | **0.01** | 保持弱域正则，不再同时改太多变量 |
| `sampling.strategy` | balanced_full | **balanced_full** | V10-sanity 已证明比 V9 随机高倍复制更稳 |
| `use_lesion` | false | **false** | 第一篇未用 lesion，当前 lesion 数据也不规范 |

预期结果不能按“必超第一篇”承诺。基于历史结果，合理区间如下：

| 指标 | 保守预期 | 理想预期 | 第一篇 Tri-CAF |
|------|---------:|---------:|---------------:|
| 77sets pooled AUC | 0.79-0.81 | 0.82-0.83 | 0.813 |
| ISLE pooled AUC | 0.79-0.82 | 0.82-0.84 | 0.836 |
| 77sets F1-macro | 0.70-0.73 | 0.73-0.75 | 待对齐 |
| ISLE F1-macro | 0.71-0.74 | 0.74-0.76 | 待对齐 |

这版有机会超过第一篇的 77sets AUC，但 ISLE 要超过 `0.836` 需要 Fold 1 和 Fold 5 同时明显改善。判断标准：

- 若 77sets ≥ `0.813` 且 ISLE ≥ `0.82`，说明主路线接近成功；下一步只做轻量 ISLE-lift。
- 若 77sets 回到 `0.80+` 但 ISLE 仍 < `0.82`，下一版再把 `contrastive.weight` 回调到 `0.005-0.01`，专门拉 ISLE。
- 若双数据集仍低于 V10-sanity，说明问题不在 adapter/contrastive，而要回到采样或 checkpoint metric。

若只想提升 Accuracy/F1，应做 dataset-specific threshold 或 temperature scaling；但 AUC 未稳定前，不应把校准方案当成主模型改进。

如果全脑-only V10 无法把 ISLE 推到 0.836，论文叙事应改为“跨模态联合训练可提升/接近单模态第一篇”，而不是强行引入 lesion 数据冲指标。

### V10-no-adapter-lowC-minckpt 实际结果

**日期:** 2026-06-01  
**日志:** `logs/xformer_20260601_154552.log`  
**实际运行:** 单卡，`Device: cuda:0, World size: 1`  
**核心配置:** `balanced_full`，`augment_77=1`，`augment_isle=1`，`use_lesion=false`，`classifier.adapter.enabled=false`，`min_checkpoint_epoch=25`，`domain.weight=0.01`，`contrastive.weight=0.002`。

| Fold | Best Val AUC | Best Epoch | Threshold | 77sets Test AUC | ISLE Test AUC |
|------|-------------:|-----------:|----------:|----------------:|--------------:|
| 1 | 0.8758 | 53 | 0.79 | 0.8730 | 0.7650 |
| 2 | 0.8354 | 42 | 0.70 | 0.7937 | 0.8900 |
| 3 | 0.8758 | 30 | 0.58 | 0.7679 | 0.8050 |
| 4 | 0.8261 | 39 | 0.64 | 0.8393 | 0.7250 |
| 5 | 0.8602 | 27 | 0.70 | 0.7679 | 0.7211 |

| 指标 | 77sets | ISLE2024 |
|------|--------|----------|
| AUC mean±std | 0.8083±0.0416 | 0.7812±0.0624 |
| Pooled AUC (CSV) | 0.7714 | 0.7554 |
| Pooled AUC (log bootstrap) | 0.7713 | 0.7553 |
| 95% CI | 0.6646-0.8721 | 0.6676-0.8377 |
| Accuracy | 0.7273 | 0.7450 |
| F1-macro | 0.7256 | 0.6803 |
| Pooled threshold | 0.682 | 0.682 |
| Mean modality gap | 0.0805±0.0321 | — |

### V10-no-adapter-lowC-minckpt vs V10-sanity

| Fold | 77sets ΔAUC | ISLE ΔAUC | 主要变化 |
|------|------------:|----------:|----------|
| 1 | +0.0635 | +0.0700 | min checkpoint 后 Fold 1 明显改善 |
| 2 | -0.0952 | -0.0500 | 原本最强 fold 被削弱，是 pooled AUC 下跌主因之一 |
| 3 | +0.0358 | -0.0300 | MRI 小幅提升，CT 排序下降 |
| 4 | -0.1071 | -0.0400 | 77sets 从高点回落，跨模态 gap 仍偏大 |
| 5 | +0.1072 | +0.1211 | Fold 5 修复有效，但不足以抵消 Fold 2/4 回退 |

| 指标 | V10-sanity | V10-no-adapter-lowC-minckpt | 变化 |
|------|-----------:|-----------------------------:|-----:|
| 77sets pooled AUC | 0.7912 | 0.7714 | -0.0198 |
| ISLE pooled AUC | 0.7846 | 0.7554 | -0.0292 |
| 77sets Accuracy | 0.7143 | 0.7273 | +0.0130 |
| ISLE Accuracy | 0.7248 | 0.7450 | +0.0202 |
| 77sets F1-macro | 0.7142 | 0.7256 | +0.0114 |
| ISLE F1-macro | 0.7162 | 0.6803 | -0.0359 |
| Mean modality gap | 未记录 | 0.0805±0.0321 | 接近目标 `<0.08` |

### V10-no-adapter-lowC-minckpt 概率诊断

基于 `results/joint/predictions.csv` 重新计算，adapter 关闭后概率分布比上一版更有分离度，但 Fold 3/5 尤其是 ISLE 仍然偏弱。

| Fold | Dataset | AUC | Thr | Pos mean | Neg mean | Gap | FP/FN |
|------|---------|----:|----:|---------:|---------:|----:|------:|
| 1 | 77sets | 0.8730 | 0.79 | 0.7374 | 0.3060 | 0.4315 | 2/2 |
| 1 | ISLE | 0.7650 | 0.79 | 0.5354 | 0.2602 | 0.2753 | 2/6 |
| 2 | 77sets | 0.7937 | 0.70 | 0.6975 | 0.3709 | 0.3266 | 2/3 |
| 2 | ISLE | 0.8900 | 0.70 | 0.6732 | 0.4453 | 0.2279 | 1/3 |
| 3 | 77sets | 0.7679 | 0.58 | 0.7230 | 0.5687 | 0.1543 | 3/1 |
| 3 | ISLE | 0.8050 | 0.58 | 0.5914 | 0.4572 | 0.1342 | 3/4 |
| 4 | 77sets | 0.8393 | 0.64 | 0.7709 | 0.5133 | 0.2576 | 4/1 |
| 4 | ISLE | 0.7250 | 0.64 | 0.6444 | 0.4823 | 0.1621 | 4/3 |
| 5 | 77sets | 0.7679 | 0.70 | 0.6206 | 0.4667 | 0.1539 | 0/3 |
| 5 | ISLE | 0.7211 | 0.70 | 0.5607 | 0.4396 | 0.1211 | 1/7 |

Pooled 级别：77sets 正/负均值 `0.7099/0.4400`，gap `0.2699`；ISLE 正/负均值 `0.6010/0.4167`，gap `0.1843`。77sets 的分离度比 adapter 版更好，但 ISLE 的 gap 变小，尤其 Fold 5 有 7 个 FN，解释了 ISLE F1 从 V10-sanity 的 `0.7162` 降到 `0.6803`。

### V10-no-adapter-lowC-minckpt 结论

- 这一版未超过第一篇：Tri-CAF 参考为 77sets `0.813`、ISLE `0.836`；当前 pooled AUC 分别为 `0.7714/0.7554`。
- 关闭 adapter 是正确的局部修正：相比 V10-adapter-min-ckpt，pooled AUC 从 `0.7544/0.7451` 回升到 `0.7714/0.7554`，77sets F1 也从 `0.6878` 回升到 `0.7256`。
- 但把 `contrastive.weight` 从 `0.005` 降到 `0.002` 没有带来预期收益。相比 V10-sanity，77sets pooled AUC 下降 `0.0198`，ISLE pooled AUC 下降 `0.0292`。
- `min_checkpoint_epoch=25` 继续有效：Fold 5 从 V10-sanity 的 `77sets=0.6607/ISLE=0.6000` 提升到 `0.7679/0.7211`。问题是 Fold 2/4 的回退抵消了 Fold 5 收益。
- 跨模态 gap 降到 `0.0805±0.0321`，接近目标 `<0.08`，说明 adapter 关闭后共享瓶颈更平衡；但“平衡”不等于“排序更强”，两数据集 pooled AUC 仍下降。
- 当前主要问题是低对比约束下 CT 侧结局排序不足。ISLE pooled gap 只有 `0.1843`，Fold 5 正负均值差仅 `0.1211`，提示 `contrastive=0.002` 对 ISLE 支撑不够。
- 下一步不应继续降 contrastive 或重开 adapter。建议跑 `V10-minckpt-C005-noadapter`：`classifier.adapter.enabled=false`，`min_checkpoint_epoch=25`，`contrastive.weight=0.005`，其余保持 V10-sanity/V10-lowC 相同。这个实验能隔离“Fold 5 修复”是否只来自 min checkpoint，而不被低 contrastive 混淆。

### V10-minckpt-C005-B8A4-noadapter 下一版配置

**状态:** 已写入 `config/config.yaml`，准备运行。  
**目标:** 在 V10-no-adapter-lowC-minckpt 基础上恢复 V10-sanity 的对比约束强度，同时测试更大的物理 batch 是否提升吞吐和 batch 内对齐稳定性。

| 参数 | 上一版 | 下一版 | 原因 |
|------|--------|--------|------|
| `classifier.adapter.enabled` | false | **false** | adapter 已证实会降低 pooled AUC，继续关闭 |
| `training.min_checkpoint_epoch` | 25 | **25** | 保留 Fold 5 修复收益 |
| `losses.contrastive.weight` | 0.002 | **0.005** | 回到 V10-sanity 的轻量对齐强度，修复低 C 导致的 ISLE 排序不足 |
| `training.batch_size` | 4 | **8** | 提高物理 batch，增加 batch 内样本对，可能改善 contrastive 稳定性和 GPU 利用率 |
| `training.accumulation_steps` | 8 | **4** | 保持有效 batch 为 `8*4=32`，避免直接变成 64 后减少优化步数 |
| `domain.weight` | 0.01 | **0.01** | 保持弱域正则，不再额外引入变量 |
| `sampling.strategy` | balanced_full | **balanced_full** | 保持 V10 主线采样 |
| `use_lesion` | false | **false** | 继续全脑-only，保持与第一篇公平对比 |

预期判断：

- 如果 77sets/ISLE pooled AUC 回到或超过 V10-sanity 的 `0.7912/0.7846`，说明 `C=0.005 + min_checkpoint_epoch=25` 能同时保留 Fold 5 修复和轻量对齐收益。
- 如果相比上一版只提升训练速度但 AUC 不升，说明 batch 不是主矛盾，下一步应回到 `batch_size=4, accumulation_steps=8` 或固定 B8/A4 后搜索 `contrastive=0.0075/0.01`。
- 如果 AUC 继续低于上一版，说明更大的物理 batch 可能削弱小样本噪声正则，应回退 B4/A8，只保留 `contrastive=0.005` 做隔离实验。
