# X-Former: Cross-Modal Information Bottleneck Network for Stroke Outcome Prediction

第二篇论文：跨模态信息瓶颈网络，预测卒中 90 天 mRS 结局（二分类）。

## 核心创新

用**信息瓶颈（Information Bottleneck）**替代级联注意力融合：通过少量可学习的"瓶颈槽位"强制三种模态（影像、临床、影像组学）竞争有限的表示容量，天然学到模态无关的特征。联合训练 77sets（MRI N=77）+ ISLE2024（CT N=149）共 226 例。

### 与第一篇（Tri-CAF）的关键区别

| | Tri-CAF | X-Former |
|---|---|---|
| 训练范式 | 独立训练 | 联合训练 |
| 图像编码 | Swin | Swin + Perceiver (K=32) |
| 临床编码 | 分组 MLP | 可学习原型 (M=8) |
| 融合机制 | 级联交叉注意力 | 信息瓶颈 (B=8, L=4) |
| 模态对齐 | 无 | 对抗域混淆 + 监督对比 |

## 项目结构

```
Second/
├── config/
│   └── config.yaml           # 统一配置文件（所有超参数）
├── model/
│   ├── image_encoder.py      # Swin-UNETR + Perceiver 压缩
│   ├── clinical_encoder.py   # 原型临床编码器
│   ├── radiomics_encoder.py  # 多源影像组学编码器
│   ├── bottleneck_fusion.py  # 信息瓶颈融合（核心创新）
│   └── x_former.py           # 主模型组装
├── data_provider/
│   ├── dataset_77sets.py     # 77sets MRI 数据集
│   ├── dataset_isle2024.py   # ISLE2024 CT 数据集
│   └── joint_sampler.py      # 联合数据集 + 交错批次采样器
├── losses/
│   ├── focal_loss.py         # Focal Loss (alpha=0.7, gamma=2.0)
│   ├── contrastive_loss.py   # 监督对比损失 (Khosla 2020)
│   ├── consistency_loss.py   # JS 散度一致性损失
│   └── domain_loss.py        # 梯度反转层 (GRL) + 域对抗损失
├── train/
│   ├── trainer.py            # 训练循环（混合精度, 梯度累积, 早停）
│   └── crossval_joint.py     # 5 折交叉验证主脚本
├── test/
│   └── evaluate.py           # 评估模块（Bootstrap CI, 绘图）
├── utils/
│   ├── metrics.py            # 指标计算（AUC/Acc/F1/Precision/Recall）
│   ├── bootstrap.py          # Bootstrap 95% CI (1000 次重采样)
│   └── plotting.py           # 600 DPI 论文图（ROC, 混淆矩阵, 消融柱状图）
├── datasets/
│   ├── 77sets/               # MRI 队列 (N=77)
│   │   ├── Clinical/clinical.csv
│   │   ├── Image/*.nii.gz
│   │   └── Radiomics/*.csv
│   └── ISLE2024/             # CT 队列 (N=149)
│       ├── Clinical/clinical_isle_2024_brain.csv
│       ├── Image_Brain/*.nii.gz
│       ├── Radiomics_Brain/*.csv
│       └── Radiomics_Lesion/*.csv
├── pretrained/               # 放置预训练 Swin 权重 (model_swinvit.pt)
├── results/                  # 训练输出（自动生成）
├── logs/                     # 训练日志（自动生成）
├── run.sh                    # 一键启动脚本
└── README.md
```

## 快速开始

### 1. 预训练权重（可选）

将 `model_swinvit.pt` 放入 `pretrained/`，然后在 `config/config.yaml` 中设置：

```yaml
model:
  image:
    swin_pretrained: true
```

无预训练权重也可直接训练（从头初始化 Swin）。

### 2. 训练命令

```bash
# 前台训练（Ctrl+C 停止）
bash run.sh

# 后台训练（关终端不中断）
bash run.sh --nohup

# 多卡 DDP 后台训练
bash run.sh --ddp --gpus "0,1,2" --nohup

# 监控训练
tail -f logs/xformer_*.log
watch -n 1 nvidia-smi
```

### 3. 消融实验

编辑 `config/config.yaml` 中 `ablation` 部分的开关：

```yaml
ablation:
  use_bottleneck: false        # 去掉信息瓶颈
  use_perceiver: false         # 去掉 Perceiver (用 GAP)
  use_domain_loss: false       # 去掉域对抗
  use_contrastive_loss: false  # 去掉对比损失
  use_consistency_loss: false  # 去掉一致性损失
  use_lesion: false            # 去掉病灶影像组学
  use_proto_clinical: false    # MLP 替代原型编码器
```

然后正常运行 `bash run.sh`。

### 4. 结果输出

`results/joint/` 包含：
- `roc_curves/roc_77sets.tif` — ROC 曲线
- `roc_curves/roc_ISLE2024.tif`
- `confmat/confmat_77sets.tif` — 混淆矩阵
- `confmat/confmat_ISLE2024.tif`
- `metrics_77sets.csv` — 77sets 各指标 + 95% CI
- `metrics_ISLE2024.csv` — ISLE2024 各指标 + 95% CI
- `fold_results.json` — 各折详细结果

## 评估指标

- AUC, Accuracy, F1-macro, Precision, Recall
- 95% CI（1000 Bootstrap）
- 不含 MCC

## 图表规范

- 600 DPI, TIF+PNG 双格式
- Times New Roman 字体
- ColorBrewer Set2 色盲友好配色

## 显存预估 (RTX 3090 24GB)

| 配置 | Batch Size | 显存 | 单折时间 |
|---|---|---|---|
| 单卡 | 4 (accum=8) | ~18GB | ~2-3h |
| 3卡 DDP | 4/卡 | ~18GB/卡 | ~1h |
| 无 Perceiver | 8 | ~14GB | ~1.5h |

## FAQ

**Q: 如何中断恢复？**  
A: 目前不支持断点续训。每个 fold 独立训练，可手动从指定 fold 开始。

**Q: 显存不足怎么办？**  
A: 减小 `batch_size` 并增大 `accumulation_steps`（有效 batch = batch_size × accumulation_steps）。

**Q: 如何只用单个数据集？**  
A: 设 `ablation.joint_training: false`，模型会独立训练两个数据集。
